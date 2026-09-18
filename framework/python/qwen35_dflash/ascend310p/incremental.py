"""Functional, explicit-state AIR graphs with selectable GDR verification.

No Python bridge/cache object survives an invocation. Verify computes acceptance
and commits inside one OM: Chunk recomputes the accepted prefix, while MTP gathers
its FP32 per-row state bank. Chunk also retains raw first-pass discard outputs.
Capsules and MTP banks remain internal; only selected states become cache inputs.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .contracts import AirGraphSpec, CustomOpExportSpec
from .incremental_plan import (
    ABI, MTP_ABI, ATTENTION_EXPORT_POLICY, DRAFT_LENGTH_POLICY,
    VERIFY_STATE_OUTPUT_POLICY, MTP_STATE_OUTPUT_POLICY, VERIFY_GDR_ROUTES,
    verify_discard_descriptors,
)

CHUNK_ABI = ABI


def conv_chunk(x: Tensor, state: Tensor, weight: Tensor, bias: Tensor | None):
    """The branch's causal-conv formula, including every valid-prefix state."""
    rows = x.shape[-1]
    history = torch.cat((state, x), dim=-1).to(weight.dtype)
    output = F.silu(F.conv1d(history, weight.unsqueeze(1), bias, groups=x.shape[1]))
    # Window i is history[..., i+1:i+1+K], the state after input row i.
    # TorchAir cannot lower aten.unfold. Stack K shifted slices instead of
    # creating an overlapping view; K is the static convolution width (4).
    bank = torch.stack(
        [history[..., offset + 1 : offset + 1 + rows]
         for offset in range(state.shape[-1])],
        dim=-1,
    )
    return output[..., -rows:].to(x.dtype), bank.permute(0, 2, 1, 3).contiguous().to(
        x.dtype
    )


def prefix_state(bank: Tensor, rows: Tensor) -> Tensor:
    return torch.index_select(bank, 1, rows.to(torch.long) - 1).squeeze(1).contiguous()


def accepted_prefix_length(input_ids: Tensor, top1: Tensor, valid_rows: Tensor) -> Tensor:
    """Count matching proposals before the first mismatch, ignoring padding.

    Use the INT32 Cumsum/ReduceSum route from the quant AIR transaction graph:
    TorchAir has no amin/min.dim/cumprod converter on the receiver toolchain.
    The scan contains at most 15 bits, so integer accumulation is exact. Keep
    the public accepted_count INT64 ABI and accepted+1 GDR commit unchanged.
    """
    indices = torch.arange(input_ids.shape[1] - 1, device=input_ids.device)
    proposal_count = valid_rows.to(torch.long) - 1
    within_requested = indices[None, :] < proposal_count[:, None]
    mismatch = (input_ids[:, 1:] != top1[:, :-1]) & within_requested
    cumulative_mismatches = torch.cumsum(
        mismatch.to(torch.int32), dim=1, dtype=torch.int32
    )
    accepted_mask = within_requested & cumulative_mismatches.eq(0)
    return accepted_mask.to(torch.int32).sum(dim=1, dtype=torch.int32).to(torch.long)


def copy_cache_rows(cache: Tensor, dim: int, positions: Tensor, values: Tensor) -> Tensor:
    """Replace complete cache rows at distinct positions without mutating input.

    Every caller constructs consecutive positions within the locked capacity.
    Repeating the row indices makes scatter equivalent to index_copy here;
    TorchAir lowers scatter.src to ScatterElements, while index_copy has no GE
    converter on the receiver route.
    """
    index_shape = [1] * cache.ndim
    index_shape[dim] = -1
    # Materialize the static head/channel repeats with Tile. This is the quant
    # AIR branch's cache-index route; dynamic BroadcastTo shape inputs can fail
    # receiver ATC auto-tiling even when the inferred index shape is correct.
    repeats = list(values.shape)
    repeats[dim] = 1
    indices = positions.reshape(index_shape).repeat(*repeats)
    return torch.scatter(cache, dim, indices, values)


def update_paged(
    cache: Tensor,
    values: Tensor,
    positions: Tensor,
    *,
    cache_update: Callable | None = None,
    aligned_prefill: bool = False,
) -> Tensor:
    """Write rows in the receiver's [blocks,H*D/16,64,16] layout.

    Padded rows can overwrite only uncommitted slots. Capacity includes a
    private 64-row scratch tail, so even the final short gear stays in bounds.
    The supplied operation determines aliasing; the AIR frontend is functional,
    while the native eager receiver writes in place.
    """
    blocks, width, block_size, tile = cache.shape
    if cache_update is not None:
        # CacheUpdate consumes the receiver's paged layout directly. Only the
        # new rows need packing; never flatten/transpose the complete cache.
        updates = values.reshape(values.shape[1], width, tile).contiguous()
        if aligned_prefill:
            # C++ Prefill starts at 0 and advances by 64. The final short
            # chunk still writes a complete physical gear into scratch slots;
            # only valid_rows become visible. This call never crosses a page.
            if updates.shape[0] != block_size:
                raise ValueError("aligned CacheUpdate prefill requires one complete page")
            target_block = (positions[:1] // block_size).to(torch.int32)
            offset = torch.zeros((), dtype=torch.int32, device=positions.device)
            return cache_update(cache, updates, target_block, offset)
        # Decode/verify can start at any offset. Use the receiver's supported
        # single-row calls rather than assuming multi-row cross-page support.
        # The output dataflow chains the writes before fused attention.
        for row in range(updates.shape[0]):
            position = positions[row]
            target_block = (position // block_size).reshape(1).to(torch.int32)
            offset = (position % block_size).to(torch.int32)
            cache = cache_update(cache, updates[row : row + 1], target_block, offset)
        return cache
    # Explicit CPU/Tensor reference used by contract tests. Production NPU
    # factories always supply CacheUpdate and must fail if it is unavailable.
    rows = cache.permute(0, 2, 1, 3).reshape(blocks * block_size, width, tile)
    values = values.reshape(values.shape[1], width, tile)
    rows = copy_cache_rows(rows, 0, positions, values)
    return (
        rows.reshape(blocks, block_size, width, tile).permute(0, 2, 1, 3).contiguous()
    )


class AirTargetAttention(nn.Module):
    """Receiver projections/RoPE/fused attention with explicit KV outputs."""

    def __init__(
        self, base: nn.Module, operation: Callable, rotary: Callable,
        *, cache_update: Callable | None = None, aligned_prefill: bool = False,
    ):
        super().__init__()
        self.base, self.operation, self.apply_rotary = base, operation, rotary
        self.cache_update, self.aligned_prefill = cache_update, aligned_prefill

    def forward(self, x, key_cache, value_cache, positions, mask):
        base = self.base
        shape = x.shape[:-1]
        view = (*shape, -1, base.head_dim)
        query, gate = torch.chunk(
            base.q_proj(x).view(*shape, -1, base.head_dim * 2), 2, dim=-1
        )
        query = base.q_norm(query.reshape(view))
        key = base.k_norm(base.k_proj(x).view(view))
        value = base.v_proj(x).view(view)
        cosine, sine = base.rotary_emb(x, positions.unsqueeze(0))
        query, key = self.apply_rotary(
            query.transpose(1, 2), key.transpose(1, 2), cosine, sine
        )
        key_cache = update_paged(
            key_cache, key.transpose(1, 2), positions,
            cache_update=self.cache_update, aligned_prefill=self.aligned_prefill,
        )
        value_cache = update_paged(
            value_cache, value, positions,
            cache_update=self.cache_update, aligned_prefill=self.aligned_prefill,
        )
        query = query.contiguous()
        query_shape = query.shape
        query_nz = base.transform_nd_2_nz(query).reshape(
            1, base.num_heads * base.head_dim // 16, shape[1], 16
        )
        output = self.operation(
            query=query_nz,
            key=[key_cache],
            value=[value_cache],
            # The receiver frontend exposes lengths as SymInt[], and GE puts
            # them in INT64 all_seq_lengths_q. Follow the quant AIR static
            # route: use physical capacity and the runtime causal/prefix mask.
            # pse_shift is an optional FP16 bias, never a sequence-length slot.
            all_seq_lengths_q=[base.kv_max_len],
            actual_seq_lengths_q=[shape[1]],
            actual_seq_lengths_kv=[base.kv_max_len],
            block_table=base.block_table,
            num_heads=base.num_heads,
            num_key_value_heads=base.num_key_value_heads,
            block_size=base.block_size,
            input_layout="BNSD",
            scale_value=base.scaling,
            inner_precise=2,
            atten_mask=mask.to(torch.float16),
        )
        output = (
            base.transform_nz_2_nd(output.reshape(query_shape))
            .transpose(1, 2)
            .contiguous()
            .reshape(*shape, -1)
        )
        return (
            base.o_proj(output * torch.sigmoid(gate.reshape(*shape, -1))),
            key_cache,
            value_cache,
        )


class AirGdn(nn.Module):
    def __init__(self, base: nn.Module, operation: Callable, *, mtp: bool = False):
        super().__init__()
        self.base, self.operation = base, operation
        self.mtp = mtp

    def forward(self, x, conv, recurrent, valid_rows):
        base = self.base
        batch, rows, _ = x.shape
        mixed, bank = conv_chunk(
            base.in_proj_qkv(x).transpose(1, 2),
            conv,
            base.conv1d.weight.squeeze(1),
            base.conv1d.bias,
        )
        query, key, value = torch.split(
            mixed.transpose(1, 2), [base.key_dim, base.key_dim, base.value_dim], dim=-1
        )
        query = query.reshape(batch, rows, -1, base.head_k_dim)
        key = key.reshape(batch, rows, -1, base.head_k_dim)
        value = value.reshape(batch, rows, -1, base.head_v_dim).contiguous()
        repeat = base.num_v_heads // base.num_k_heads
        if repeat > 1:
            query, key = (
                query.repeat_interleave(repeat, dim=2),
                key.repeat_interleave(repeat, dim=2),
            )
        beta = base.in_proj_b(x).sigmoid()
        g = -base.A_log.float().exp() * F.softplus(
            base.in_proj_a(x).float() + base.dt_bias
        )
        if recurrent.dtype != torch.float32:
            raise TypeError("GDR recurrent cache must be FP32")
        initial = recurrent
        if self.mtp:
            # The graph boundary carries an already committed scalar state.
            # Seed the input bank, then select slot 0 with INT8 zero. accepted_tokens is
            # NOT this round's acceptance count, which is only known at the head.
            initial_bank = initial.unsqueeze(1).repeat(1, rows, 1, 1, 1)
            selected_slot = torch.zeros(batch, dtype=torch.int8, device=x.device)
            output, recurrent_bank = self.operation(
                query, key, value, g, beta, initial_bank, selected_slot,
                chunk_size=64, output_final_state=True, use_qk_l2norm_in_kernel=True,
            )
            if recurrent_bank.dtype != torch.float32 or recurrent_bank.shape != initial_bank.shape:
                raise ValueError("GDR MTP must return a full FP32 state bank")
            final = prefix_state(recurrent_bank, valid_rows)
            committed_final = final
            capsule = (bank, recurrent_bank)
        else:
            output, final = self.operation(
                query,
                key,
                value,
                g=g,
                beta=beta,
                effective_length=valid_rows,
                chunk_size=1 if rows == 1 else 64,
                initial_state=initial,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            if final.dtype != torch.float32:
                raise TypeError("GDR recurrent output must be FP32")
            committed_final = final
            capsule = (query, key, value, g, beta, initial, bank)
        z = base.in_proj_z(x).reshape(-1, base.head_v_dim)
        output = base.norm(output.reshape(-1, base.head_v_dim), z).reshape(
            batch, rows, -1
        )
        return (
            base.out_proj(output),
            prefix_state(bank, valid_rows),
            committed_final,
            capsule,
            final,
        )


class TargetRowsGraph(nn.Module):
    def __init__(
        self,
        target,
        *,
        rows: int,
        verify: bool,
        feature_layers: tuple[int, ...],
        gdr: Callable,
        attention: Callable,
        rotary: Callable,
        cache_update: Callable | None = None,
        verify_gdr: str = "chunk",
        gdr_mtp: Callable | None = None,
    ):
        super().__init__()
        model = target.dflash_execution_model
        self.body = model.language_model
        self.embedding = (
            getattr(target, "_target_quantized_embedding", None)
            or target.get_input_embeddings()
        )
        # The bridge's public embedding getters retain the FP16 checkpoint
        # modules for Draft. Target must use the execution model's W8A8 head.
        self.head = getattr(model, "lm_head", None)
        if not isinstance(self.head, nn.Module):
            raise TypeError("incremental Target requires execution-model lm_head")
        self.rows, self.verify, self.feature_layers = rows, verify, feature_layers
        if verify_gdr not in VERIFY_GDR_ROUTES:
            raise ValueError("verify_gdr must be chunk or mtp")
        self.mtp = verify and verify_gdr == "mtp"
        if self.mtp and not callable(gdr_mtp):
            raise ValueError("mtp verification requires GDR MTP; no fallback is permitted")
        self.cache_capacity = target.kv_cache_max_len
        self.blocks = nn.ModuleList(
            [
                AirGdn(layer.linear_attn, gdr_mtp if self.mtp else gdr, mtp=self.mtp)
                if layer.block_type == "linear_attention"
                else AirTargetAttention(
                    layer.self_attn, attention, rotary,
                    cache_update=cache_update, aligned_prefill=rows == 64,
                )
                for layer in self.body.layers
            ]
        )
        self.linear_indices = tuple(
            i for i, block in enumerate(self.blocks) if isinstance(block, AirGdn)
        )
        self.commit = (MtpCommitGraph(len(self.linear_indices)) if self.mtp
                       else TargetCommitGraph(gdr, len(self.linear_indices)))

    def forward(self, input_ids, start_position, valid_rows, *state):
        positions = start_position + torch.arange(
            self.rows, dtype=torch.long, device=input_ids.device
        )
        logical_end = start_position + valid_rows.to(torch.long)
        columns = torch.arange(self.cache_capacity, device=input_ids.device)
        visible = (columns[None, :] <= positions[:, None]) & (
            columns[None, :] < logical_end
        )
        mask = torch.where(visible, 0.0, float("-inf"))[None, None]
        hidden = self.embedding(input_ids).to(torch.float16)
        row_valid = (
            torch.arange(self.rows, device=input_ids.device) < valid_rows.to(torch.long)
        )[:, None]
        next_state, capsules, features, verify_discard = [], [], [], []
        for index, (layer, block) in enumerate(zip(self.body.layers, self.blocks)):
            normalized = layer.input_layernorm(hidden)
            if isinstance(block, AirGdn):
                mixed, conv, recurrent, capsule, raw_final = block(
                    normalized, state[2 * index], state[2 * index + 1], valid_rows
                )
                next_state.extend((conv, recurrent))
                if self.verify:
                    capsules.extend(capsule)
                    # Keep the native FP32 output live at the OM boundary so
                    # GE gives both first-pass GDR outputs real storage.
                    # This state includes unaccepted proposals: never commit it.
                    if not self.mtp:
                        verify_discard.append(raw_final)
            else:
                mixed, key, value = block(
                    normalized,
                    state[2 * index],
                    state[2 * index + 1],
                    positions,
                    mask,
                )
                next_state.extend((key, value))
            hidden = hidden + mixed
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            # Receiver GDR padding outputs are not part of its valid-row
            # contract. Sanitize them before a later layer writes paged KV.
            hidden = torch.where(row_valid[None], hidden, torch.zeros_like(hidden))
            if index in self.feature_layers:
                features.append(hidden.clone())
        hidden = self.body.norm(hidden)
        # Prefill/decode applies the full-vocabulary head to one real row only.
        head_rows = (
            hidden
            if self.verify
            else torch.index_select(hidden, 1, valid_rows.to(torch.long) - 1)
        )
        top1 = torch.argmax(self.head(head_rows), dim=-1)
        acceptance_output = ()
        if self.verify:
            accepted = accepted_prefix_length(input_ids, top1, valid_rows)
            committed = self.commit((accepted + 1).to(torch.int16), *capsules)
            for offset, layer_index in enumerate(self.linear_indices):
                next_state[2 * layer_index : 2 * layer_index + 2] = committed[
                    2 * offset : 2 * offset + 2
                ]
            acceptance_output = (accepted,)
        # One Draft input gear serves prompt chunks and committed verify rows.
        feature_output = ()
        if features:
            features = torch.cat(features, dim=-1)
            if self.rows < 64:
                features = F.pad(features, (0, 0, 0, 64 - self.rows))
            feature_output = (features,)
        return (top1, *acceptance_output, *feature_output, *next_state, *verify_discard)


class MtpCommitGraph(nn.Module):
    """Select slot a after processing anchor + a accepted proposals."""

    def __init__(self, layers: int):
        super().__init__()
        self.layers = layers

    def forward(self, committed_rows, *capsules):
        result = []
        for index in range(self.layers):
            conv_bank, recurrent_bank = capsules[2 * index : 2 * index + 2]
            result.extend((prefix_state(conv_bank, committed_rows),
                           prefix_state(recurrent_bank, committed_rows)))
        return tuple(result)


class TargetCommitGraph(nn.Module):
    def __init__(self, operation: Callable, layers: int):
        super().__init__()
        self.operation, self.layers = operation, layers

    def forward(self, committed_rows, *capsules):
        result = []
        for index in range(self.layers):
            query, key, value, g, beta, initial, bank = capsules[
                index * 7 : index * 7 + 7
            ]
            _, final = self.operation(
                query,
                key,
                value,
                g=g,
                beta=beta,
                effective_length=committed_rows,
                chunk_size=64,
                initial_state=initial,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            if final.dtype != torch.float32:
                raise TypeError("GDR committed recurrent state must be FP32")
            result.extend((prefix_state(bank, committed_rows), final))
        return tuple(result)


class DraftContextGraph(nn.Module):
    """Append only committed Target features; transient noise never enters KV."""

    def __init__(self, draft, rows=None):
        super().__init__()
        self.draft, self.rows = draft, rows

    def forward(self, features, start_position, *state):
        draft = self.draft
        rows = features.shape[1] if self.rows is None else self.rows
        projected = draft.hidden_norm(draft.fc(features))
        positions = start_position + torch.arange(
            rows, dtype=torch.long, device=features.device
        )
        cosine, sine = draft.rotary(positions[None], projected.dtype)
        cosine, sine = cosine[:, None], sine[:, None]
        result = []
        from .quant_factory import _rotate_half

        for index, layer in enumerate(draft.layers):
            base, config = layer.self_attn, draft.config
            key = base.k_norm(
                base.k_proj(projected).reshape(
                    1, rows, config.num_key_value_heads, config.head_dim
                )
            ).transpose(1, 2)
            key = key * cosine + _rotate_half(key) * sine
            value = (
                base.v_proj(projected)
                .reshape(1, rows, config.num_key_value_heads, config.head_dim)
                .transpose(1, 2)
            )
            result.extend(
                (
                    copy_cache_rows(state[2 * index], 2, positions, key),
                    copy_cache_rows(state[2 * index + 1], 2, positions, value),
                )
            )
        return tuple(result)


class DraftProposeGraph(nn.Module):
    def __init__(self, draft, embedding, head):
        super().__init__()
        self.draft, self.embedding, self.head = draft, embedding, head

    def forward(self, anchor, context_length, proposal_count, *state):
        draft, config = self.draft, self.draft.config
        block_ids = torch.cat(
            (
                anchor.reshape(1, 1),
                torch.full(
                    (1, config.block_size - 1),
                    config.mask_token_id,
                    dtype=torch.long,
                    device=anchor.device,
                ),
            ),
            dim=1,
        )
        hidden = self.embedding(block_ids) * config.input_embedding_scale
        offsets = torch.arange(config.block_size, device=anchor.device)
        positions = context_length + offsets
        cosine, sine = draft.rotary(positions[None], hidden.dtype)
        capacity = state[0].shape[2]
        context_positions = torch.arange(capacity, device=anchor.device)
        key_positions = torch.cat((context_positions, positions))
        valid = torch.cat(
            (
                context_positions < context_length,
                # A short native block contains anchor + K masks. Hidden
                # rows beyond K must never become attention keys, including
                # in the final non-causal layer.
                offsets <= proposal_count.to(torch.long),
            )
        )
        distance = positions[:, None] - key_positions[None, :]
        for index, layer in enumerate(draft.layers):
            base = layer.self_attn
            normalized = layer.input_layernorm(hidden)
            query = base.q_norm(
                base.q_proj(normalized).reshape(
                    1, config.block_size, config.num_attention_heads, config.head_dim
                )
            ).transpose(1, 2)
            key = base.k_norm(
                base.k_proj(normalized).reshape(
                    1, config.block_size, config.num_key_value_heads, config.head_dim
                )
            ).transpose(1, 2)
            value = (
                base.v_proj(normalized)
                .reshape(
                    1, config.block_size, config.num_key_value_heads, config.head_dim
                )
                .transpose(1, 2)
            )
            query, key = base.ops.rotary(query, key, cosine, sine)
            key, value = (
                torch.cat((state[2 * index], key), dim=2),
                torch.cat((state[2 * index + 1], value), dim=2),
            )
            mask = valid[None, :].expand(config.block_size, -1)
            if base.is_causal:
                mask = mask & (distance >= 0)
            if base.sliding_window is not None:
                mask = mask & (distance < base.sliding_window)
                if not base.is_causal:
                    mask = mask & (-distance < base.sliding_window)
            mixed = base.ops.attention(
                query,
                key,
                value,
                mask[None, None],
                base.scale,
                config.num_key_value_groups,
            )
            mixed = (
                mixed.transpose(1, 2)
                .contiguous()
                .reshape(1, config.block_size, config.query_width)
            )
            hidden = hidden + base.o_proj(mixed)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        top1 = draft.ops.top1(draft.norm(hidden)[:, 1:], self.head.weight)
        active = offsets[1:] <= proposal_count.to(torch.long)
        return (torch.where(active[None], top1, torch.zeros_like(top1)),)


def copy_draft_cache_rows(cache, positions, values, row_update=None):
    """Functional whole-row writes without a full-cache transpose or index Tile.

    BHCD is already contiguous by row. Each [batch,head,position] selects one
    D-wide row; indices are unique and include padded uncommitted scratch rows.
    The production callable is the existing functional NPU ScatterNdUpdate.
    """
    batch, heads, capacity, dim = cache.shape
    head_base = torch.arange(batch * heads, dtype=torch.int32, device=cache.device) * capacity
    indices = (head_base[:, None] + positions.to(torch.int32)[None, :]).reshape(-1, 1)
    flat = cache.reshape(-1, dim)
    updates = values.reshape(-1, dim)
    if row_update is None:
        if cache.device.type != "cpu":
            raise RuntimeError("NPU Draft requires a functional whole-row cache update")
        # CPU oracle only. No index_copy fallback is exported on the NPU route.
        result = torch.index_copy(flat, 0, indices[:, 0].long(), updates)
    else:
        result = row_update(flat, indices, updates)
    return result.reshape_as(cache)


class PackedDraftLayer(nn.Module):
    """Pack projection output channels once; retain no original K/V/gate/up."""

    def __init__(self, layer, *, consume_source=False):
        super().__init__()
        base = layer.self_attn
        self.input_norm = layer.input_layernorm
        self.post_norm = layer.post_attention_layernorm
        self.q_proj, self.q_norm, self.k_norm = base.q_proj, base.q_norm, base.k_norm
        self.o_proj, self.down_proj = base.o_proj, layer.mlp.down_proj
        from models.dflash_v1.draft_quantization import GroupQuantLinear
        self.quantized = isinstance(base.k_proj, GroupQuantLinear)
        if self.quantized:
            self.kv_linear = GroupQuantLinear.concatenate(base.k_proj, base.v_proj)
        else:
            self.kv_weight = nn.Parameter(torch.cat(
                (base.k_proj.weight.detach(), base.v_proj.weight.detach()), dim=0
            ), requires_grad=False)
        if consume_source:
            del base.k_proj, base.v_proj
        if self.quantized:
            self.gate_up_linear = GroupQuantLinear.concatenate(layer.mlp.gate_proj, layer.mlp.up_proj)
        else:
            self.gate_up_weight = nn.Parameter(torch.cat(
                (layer.mlp.gate_proj.weight.detach(), layer.mlp.up_proj.weight.detach()), dim=0
            ), requires_grad=False)
        if consume_source:
            del layer.mlp.gate_proj, layer.mlp.up_proj
        self.ops, self.scale = base.ops, base.scale
        self.is_causal, self.sliding_window = base.is_causal, base.sliding_window

    def project_kv(self, value):
        return self.kv_linear(value) if self.quantized else self.ops.linear(value, self.kv_weight)

    def project_gate_up(self, value):
        return self.gate_up_linear(value) if self.quantized else self.ops.linear(value, self.gate_up_weight)


class DraftGraph(nn.Module):
    """Single 16/64-gear OM with shared context/noise K/V projections."""

    def __init__(self, draft, embedding, head, *, row_update=None, consume_source=False,
                 feature_layers=None):
        super().__init__()
        self.config, self.ops = draft.config, draft.ops
        self.fc, self.hidden_norm, self.rotary = draft.fc, draft.hidden_norm, draft.rotary
        self.norm, self.embedding, self.head = draft.norm, embedding, head
        # Production consumes one pair at a time after checkpoint validation,
        # avoiding simultaneous retention of every packed and original weight.
        self.layers = nn.ModuleList(
            PackedDraftLayer(layer, consume_source=consume_source) for layer in draft.layers
        )
        self.row_update = row_update
        source_layers = tuple(feature_layers or draft.config.target_layer_ids)
        if not set(draft.config.target_layer_ids).issubset(source_layers):
            raise ValueError("Target feature output is missing a selected Draft layer")
        self.feature_slots = tuple(source_layers.index(i) for i in draft.config.target_layer_ids)
        self.select_features = source_layers != tuple(draft.config.target_layer_ids)
        # Do not retain draft/context/propose: that would also register the
        # unpacked parameters, doubling their weight storage in the exported OM.

    def forward(self, features, start_position, valid_rows, anchor, proposal_count, *state):
        config = self.config
        if self.select_features:
            features = torch.cat(tuple(features[..., i * config.hidden_size:(i + 1) * config.hidden_size]
                                       for i in self.feature_slots), dim=-1)
        rows = features.shape[1]
        context_offsets = torch.arange(rows, device=features.device)
        visible = context_offsets < valid_rows.to(torch.long)
        features = torch.where(visible[None, :, None], features, torch.zeros_like(features))
        projected = self.hidden_norm(self.fc(features))
        context_positions = start_position + context_offsets
        context_cos, context_sin = self.rotary(context_positions[None], projected.dtype)
        context_cos, context_sin = context_cos[:, None], context_sin[:, None]
        block_ids = torch.cat((anchor.reshape(1, 1), torch.full(
            (1, config.block_size - 1), config.mask_token_id,
            dtype=torch.long, device=anchor.device,
        )), dim=1)
        hidden = self.embedding(block_ids) * config.input_embedding_scale
        offsets = torch.arange(config.block_size, device=anchor.device)
        context_length = start_position + valid_rows.to(torch.long)
        positions = context_length + offsets
        cosine, sine = self.rotary(positions[None], hidden.dtype)
        cache_positions = torch.arange(state[0].shape[2], device=anchor.device)
        distance = positions[:, None] - torch.cat((cache_positions, positions))[None, :]
        valid = torch.cat((cache_positions < context_length, offsets <= proposal_count.to(torch.long)))
        updated = []
        from .quant_factory import _rotate_half

        for index, layer in enumerate(self.layers):
            normalized = layer.input_norm(hidden)
            # Same weights, different input rows. One projection instead of
            # rereading K/V weights separately for committed and proposal rows.
            kv = layer.project_kv(torch.cat((projected, normalized), dim=1))
            key_all, value_all = kv.split(config.key_value_width, dim=-1)
            key_context = layer.k_norm(key_all[:, :rows].reshape(
                1, rows, config.num_key_value_heads, config.head_dim
            )).transpose(1, 2)
            key_context = key_context * context_cos + _rotate_half(key_context) * context_sin
            value_context = value_all[:, :rows].reshape(
                1, rows, config.num_key_value_heads, config.head_dim
            ).transpose(1, 2)
            cached_key = copy_draft_cache_rows(state[2 * index], context_positions, key_context, self.row_update)
            cached_value = copy_draft_cache_rows(state[2 * index + 1], context_positions, value_context, self.row_update)
            updated.extend((cached_key, cached_value))
            query = layer.q_norm(layer.q_proj(normalized).reshape(
                1, config.block_size, config.num_attention_heads, config.head_dim
            )).transpose(1, 2)
            key = layer.k_norm(key_all[:, rows:].reshape(
                1, config.block_size, config.num_key_value_heads, config.head_dim
            )).transpose(1, 2)
            value = value_all[:, rows:].reshape(
                1, config.block_size, config.num_key_value_heads, config.head_dim
            ).transpose(1, 2)
            query, key = layer.ops.rotary(query, key, cosine, sine)
            key, value = torch.cat((cached_key, key), dim=2), torch.cat((cached_value, value), dim=2)
            mask = valid[None, :].expand(config.block_size, -1)
            if layer.is_causal:
                mask = mask & (distance >= 0)
            if layer.sliding_window is not None:
                mask = mask & (distance < layer.sliding_window)
                if not layer.is_causal:
                    mask = mask & (-distance < layer.sliding_window)
            mixed = layer.ops.attention(query, key, value, mask[None, None],
                                        layer.scale, config.num_key_value_groups)
            mixed = mixed.transpose(1, 2).contiguous().reshape(1, config.block_size, config.query_width)
            hidden = hidden + layer.o_proj(mixed)
            gate, up = layer.project_gate_up(layer.post_norm(hidden)).split(
                config.intermediate_size, dim=-1
            )
            hidden = hidden + layer.down_proj(layer.ops.swiglu(gate, up))
        top1 = self.ops.top1(self.norm(hidden)[:, 1:], self.head.weight)
        active = offsets[1:] <= proposal_count.to(torch.long)
        return (torch.where(active[None], top1, torch.zeros_like(top1)), *updated)


def tensor_spec(name, tensor):
    return {
        "name": name,
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
    }


def incremental_graph_specs(
    target,
    draft,
    *,
    capacity: int,
    metadata: dict,
    gdr: Callable,
    attention: Callable,
    rotary: Callable,
    cache_update: Callable | None = None,
    draft_row_update: Callable | None = None,
    target_feature_layers: tuple[int, ...] | None = None,
    custom_ops: tuple[CustomOpExportSpec, ...] = (),
    include_ordinary_decode: bool = True,
    verify_gdr: str = "chunk",
    gdr_mtp: Callable | None = None,
):
    """Build static Target graphs and a Draft with finite 16/64 context gears."""
    if verify_gdr not in VERIFY_GDR_ROUTES:
        raise ValueError("verify_gdr must be chunk or mtp")
    if verify_gdr == "mtp" and not callable(gdr_mtp):
        raise ValueError("mtp verification requires GDR MTP; no fallback is permitted")
    device = target.requested_device
    feature_layers = tuple(target_feature_layers or draft.config.target_layer_ids)
    if tuple(sorted(set(feature_layers))) != feature_layers or not set(draft.config.target_layer_ids).issubset(feature_layers):
        raise ValueError("Target feature layers must be a sorted unique superset of Draft feature layers")
    feature_width = len(feature_layers) * draft.config.hidden_size
    state = tuple(t for pair in target._fresh_hybrid_cache(batch_size=1) for t in pair)
    names, gdn_names, kv_names, capsules, capsule_names = [], [], [], [], []
    layers = target.dflash_execution_model.language_model.layers
    state = tuple(
        tensor.float() if i % 2 == 1 and layers[i // 2].block_type == "linear_attention"
        else tensor for i, tensor in enumerate(state)
    )
    for index, layer in enumerate(layers):
        linear = layer.block_type == "linear_attention"
        pair = [
            f"t{index}_{suffix}"
            for suffix in (("conv", "recurrent") if linear else ("key", "value"))
        ]
        names.extend(pair)
        (gdn_names if linear else kv_names).extend(pair)
        if linear:
            base = layer.linear_attn
            shapes = ((1, 16, base.num_v_heads, base.head_k_dim),) * 2 + (
                (1, 16, base.num_v_heads, base.head_v_dim),
                (1, 16, base.num_v_heads),
                (1, 16, base.num_v_heads),
                tuple(state[2 * index + 1].shape),
                (1, 16, base.conv_dim, base.conv_kernel_size),
            )
            capsule_layout = list(zip(
                ("q", "k", "v", "g", "beta", "initial", "conv_bank"),
                shapes,
                (
                    torch.float16,
                    torch.float16,
                    torch.float16,
                    torch.float32,
                    torch.float16,
                    torch.float32,
                    torch.float16,
                ),
            ))
            if verify_gdr == "mtp":
                capsule_layout = [
                    ("conv_bank", (1, 16, base.conv_dim, base.conv_kernel_size), torch.float16),
                    ("recurrent_bank", (1, 16, *state[2 * index + 1].shape[1:]), torch.float32),
                ]
            for suffix, shape, dtype in capsule_layout:
                capsule_names.append(f"c{index}_{suffix}")
                # Shape documentation only: capsules are internal graph values.
                capsules.append(torch.empty(shape, dtype=dtype, device="meta"))
    start = torch.zeros(1, dtype=torch.long, device=device)
    valid = torch.ones(1, dtype=torch.int16, device=device)
    draft_names = tuple(
        f"d{i}_{kind}" for i in range(len(draft.layers)) for kind in ("key", "value")
    )
    draft_state = tuple(
        torch.zeros(
            (
                1,
                draft.config.num_key_value_heads,
                target.kv_cache_max_len,
                draft.config.head_dim,
            ),
            dtype=torch.float16,
            device=device,
        )
        for _ in draft_names
    )
    contract = {
        "abi": CHUNK_ABI if verify_gdr == "chunk" else MTP_ABI,
        "verify_gdr": verify_gdr,
        "capacity": capacity,
        "cache_capacity": target.kv_cache_max_len,
        "block_size": 16,
        "prefill_rows": 64,
        "target_states": [tensor_spec(n, t) for n, t in zip(names, state)],
        "draft_states": [tensor_spec(n, t) for n, t in zip(draft_names, draft_state)],
        "gdn_states": gdn_names,
        "kv_states": kv_names,
        "capsules": [tensor_spec(n, t) for n, t in zip(capsule_names, capsules)],
        "vocab_size": draft.config.vocab_size,
        "feature_width": feature_width,
        "target_feature_layers": list(feature_layers),
        "draft_feature_layers": list(draft.config.target_layer_ids),
        "draft_quantization": getattr(draft, "draft_quantization", "fp16"),
        "recurrent_state_dtype": "float32",
        "state_policy": ("in-graph-acceptance-two-pass-gdr-atomic-fp32-state-output"
                         if verify_gdr == "chunk" else "in-graph-mtp-bank-select-fp32-recurrent"),
        "verify_state_output_policy": (VERIFY_STATE_OUTPUT_POLICY if verify_gdr == "chunk"
                                       else MTP_STATE_OUTPUT_POLICY),
        "attention_export": ATTENTION_EXPORT_POLICY,
        "draft_length_policy": DRAFT_LENGTH_POLICY,
        "single_row_policy": "ordinary_decode1_chunk1; speculative_fallback_verify16_valid1",
        "target_kv_update": (
            "CacheUpdate_paged_aligned_prefill_per_row_decode_verify"
            if cache_update is not None else "functional_scatter_reference"
        ),
        "draft_kv_update": ("ScatterNdUpdate_dense_rows" if draft_row_update is not None
                            else "IndexCopy_CPU_reference"),
        "draft_execution_policy": "packed_context_noise_kv_gate_up_grouped_query_v1",
        "commit_capsules": "internal_to_target_verify_not_external_OM_IO",
    }
    contract["draft_context_rows"] = 16
    contract["draft_context_gears"] = [16, 64]
    contract["draft_prefill_policy"] = "single_draft16_64_gears"
    contract["verify_discard_states"] = verify_discard_descriptors(contract)
    common = {**metadata, "draft_quantization": contract["draft_quantization"], "incremental_contract": contract}
    specs = []

    def add(name, model, args, inputs, outputs, output_tensors, ops=()):
        meta = {
            **common,
            "tensor_abi": {
                "inputs": [tensor_spec(n, t) for n, t in zip(inputs, args)],
                "outputs": [tensor_spec(n, t) for n, t in zip(outputs, output_tensors)],
            },
        }
        if name == "draft":
            # tensor_abi records allocation maxima. Only this one AIR axis is
            # symbolic; ACL selects a finite gear before each execution.
            meta["dynamic_input_axes"] = {"features": [1]}
        if ops:
            meta["custom_op_export_contracts"] = [
                {"torch_target": op.torch_target, "ge_op_type": op.ge_op_type,
                 "minimum_occurrences": op.minimum_occurrences,
                 "preservation": "one registered GE operator; no Tensor decomposition"}
                for op in ops
            ]
        if not ops:
            meta.pop("custom_op_export_contract", None)
            meta.pop("custom_op_export_contracts", None)
        if not ops or name == "draft":
            meta.pop("standard_op_export_contracts", None)
        specs.append(
            AirGraphSpec(
                name=name,
                role=name.replace("_", "-"),
                model=model,
                example_args=tuple(args),
                input_names=tuple(inputs),
                output_names=tuple(outputs),
                metadata=meta,
                custom_ops=ops,
                dynamic=name == "draft",
            )
        )

    for name, rows, verify in (
        ("target_prefill", 64, False),
        ("target_decode", 1, False),
        ("target_verify", 16, True),
    ):
        if rows == 1 and not include_ordinary_decode:
            continue
        ids = torch.zeros((1, rows), dtype=torch.long, device=device)
        feature_tensor = torch.zeros(
            (1, 64, feature_width), dtype=torch.float16, device=device
        )
        top1 = torch.zeros((1, rows if verify else 1), dtype=torch.long, device=device)
        out_names, out_tensors = ["target_top1"], [top1]
        if verify:
            out_names.append("accepted_count")
            out_tensors.append(start)
        if rows != 1:
            out_names.append("features")
            out_tensors.append(feature_tensor)
        selected_names = names
        out_names += selected_names
        state_by_name = dict(zip(names, state))
        out_tensors += [state_by_name[n] for n in selected_names]
        if verify:
            for tensor in contract["verify_discard_states"]:
                out_names.append(tensor["name"])
                out_tensors.append(torch.empty(
                    tensor["shape"], dtype=torch.float32, device="meta"
                ))
        graph = TargetRowsGraph(
            target,
            rows=rows,
            verify=verify,
            feature_layers=feature_layers if rows != 1 else (),
            gdr=gdr,
            attention=attention,
            rotary=rotary,
            cache_update=cache_update,
            verify_gdr=verify_gdr,
            gdr_mtp=gdr_mtp,
        )
        # Every layer calls input/post norm, plus GDN gated norm or attention
        # Q/K norms; all gears also call the final norm (105 for the 4B Target).
        norm_count = 2 * len(layers) + len(gdn_names) // 2 + len(kv_names) + 1
        graph_ops = tuple(
            replace(op, minimum_occurrences=norm_count)
            if op.torch_op == "npu::adn_rms_norm" else
            replace(op, minimum_occurrences=len(kv_names) * (1 if rows == 64 else rows))
            if cache_update is not None and op.ge_op_type == "CacheUpdate"
            else replace(op, minimum_occurrences=len(gdn_names) // 2)
            if op.torch_op == "npu::npu_gated_delta_rule_mtp"
            else op for op in custom_ops
            if not (op.torch_op == "npu::npu_gated_delta_rule_mtp" and not (verify and verify_gdr == "mtp"))
            and not (op.torch_op == "npu::npu_chunk_gated_delta_rule" and verify and verify_gdr == "mtp")
            and op.torch_op != "npu::npu_scatter_nd_update"
        )
        add(
            name,
            graph,
            (ids, start, valid, *state),
            ("input_ids", "start_position", "valid_rows", *names),
            out_names,
            out_tensors,
            graph_ops,
        )
    embedding = (
        target.get_input_embeddings()
    )  # Draft uses the authoritative FP16 embedding.
    features = torch.zeros(
        (1, 64, feature_width), dtype=torch.float16, device=device
    )
    # Context projection: one norm + one K norm per layer. Proposal: four
    # norms per layer + final norm. Both branches are live in the single OM.
    draft_ops = tuple(
        replace(op, minimum_occurrences=5 * len(draft.layers) + 2)
        if op.torch_op == "npu::adn_rms_norm" else
        replace(op, minimum_occurrences=2 * len(draft.layers))
        for op in custom_ops if op.torch_op == "npu::adn_rms_norm"
        or (draft_row_update is not None and op.torch_op == "npu::npu_scatter_nd_update")
    )
    draft_graph = DraftGraph(draft, embedding, target.get_output_embeddings(),
                             row_update=draft_row_update, consume_source=True, feature_layers=feature_layers)
    from models.dflash_v1.draft_quantization import GroupQuantLinear
    from models.dflash_v1.weight_quant_matmul import TORCH_OP, GE_OP
    native_linears = sum(isinstance(m, GroupQuantLinear) and m.matmul_backend == "weight_quant"
                         for m in draft_graph.modules())
    if native_linears:
        draft_ops += (CustomOpExportSpec(TORCH_OP, GE_OP, minimum_occurrences=native_linears),)
    add(
        "draft",
        draft_graph,
        (features, start, valid, start.clone(),
         torch.full_like(valid, 15), *draft_state),
        ("features", "start_position", "valid_rows", "anchor", "proposal_count", *draft_names),
        ("draft_top1", *draft_names),
        (torch.zeros((1, 15), dtype=torch.long, device=device), *draft_state),
        draft_ops,
    )
    from .draft_constants import expose_draft_constants
    specs[-1] = expose_draft_constants(specs[-1])
    # Constants belong only to Draft, but the complete bundle contract names
    # them so the host can validate the exact ordered inputs before loading.
    contract["draft_constants"] = specs[-1].metadata.get("constant_tensors", [])
    return tuple(specs)
