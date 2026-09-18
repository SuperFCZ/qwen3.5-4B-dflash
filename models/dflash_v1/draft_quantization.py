"""Pinned compressed-tensors DFlash imports and device-resident W8/W4 linears.

These are the published RTN W8 and GPTQ W4 checkpoints, not a requantization
of the older six-layer draft. Only CPU loading repacks bits; inference uses
device Tensor operations and never keeps a dense copy of a Linear weight.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .dflash_config import Qwen35DFlashConfig
from .dflash_weights import require_official_dflash_checkpoint, sha256_file


DRAFT_QUANTIZATIONS = ("fp16", "w8a16", "w4a16")
DRAFT_QUANT_MATMUL_BACKENDS = ("weight_quant", "dequant")
QUANTIZED_DRAFTS = {
    "w8a16": {
        "repository": "naveenrajk/Qwen3.5-4B-DFlash-W8A16",
        "revision": "b4bf9f0f7f79bd8589d34b457e136f644c122180",
        "config_sha256": "ee0a79b5a20020cb13ef7d44533ce3a94153144ed437f5742892396d4c6cad4f",
        "model_sha256": "8b83f06aeb2a10e9b7e4df16622d5b04b849fa8e8d04b9cea7437f6c3c346790",
        "model_bytes": 545870208,
        "num_bits": 8,
        "algorithm": "RTN",
    },
    "w4a16": {
        "repository": "nota-ai/Qwen3.5-4B-DFlash-GPTQ-W4A16",
        "revision": "c5fb290e47e30c81d06e48b0495ec06f2560dd4e",
        "config_sha256": "8de18870190cb2ad8a7085677a7095abb79358d047dcc9aae002384bdce98361",
        "model_sha256": "385e50472c9b8ec31599db0b7c5150cf836489b8656f0276237be78337307caf",
        "model_bytes": 277172584,
        "num_bits": 4,
        "algorithm": "GPTQ",
    },
}


def validate_quantization_config(raw: dict[str, Any], variant: str) -> int:
    if variant not in QUANTIZED_DRAFTS:
        raise ValueError(f"unsupported quantized Draft variant: {variant}")
    quant = raw.get("quantization_config", {})
    groups = quant.get("config_groups", {})
    if (quant.get("quant_method") != "compressed-tensors"
            or quant.get("format") != "pack-quantized"
            or quant.get("quantization_status") != "compressed"
            or len(groups) != 1 or quant.get("ignore") not in ([], ["lm_head"])):
        raise ValueError("Draft requires compressed-tensors pack-quantized Linear weights")
    group = next(iter(groups.values()))
    weights = group.get("weights", {})
    bits = QUANTIZED_DRAFTS[variant]["num_bits"]
    if (group.get("targets") != ["Linear"]
            or group.get("input_activations") is not None
            or group.get("output_activations") is not None
            or weights.get("num_bits") != bits
            or weights.get("group_size") != 128
            or weights.get("strategy") != "group"
            or weights.get("symmetric") is not True
            or weights.get("type") != "int"
            or weights.get("dynamic") is not False
            or weights.get("actorder") is not None):
        raise ValueError(f"Draft {variant} requires symmetric group=128 weight-only quantization")
    return bits


def unpack_packed_int32(packed: Tensor, bits: int, shape: tuple[int, int]) -> Tensor:
    """Decode compressed-tensors offset-binary, low bits first along K.

    The stored code is q + 2**(bits-1), not a signed two's-complement nibble.
    Masking after an arithmetic right shift also handles negative int32 words.
    """
    if bits not in (4, 8) or packed.dtype != torch.int32:
        raise ValueError("expected INT32 packed W4/W8")
    rows, cols = shape
    factor = 32 // bits
    if tuple(packed.shape) != (rows, (cols + factor - 1) // factor):
        raise ValueError("packed weight shape differs from weight_shape")
    shifts = torch.arange(factor, device=packed.device, dtype=torch.int32) * bits
    codes = (packed.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)
    return (codes.reshape(rows, -1)[:, :cols] - (1 << (bits - 1))).to(torch.int8)


def pack_device_weight(q: Tensor, bits: int) -> Tensor:
    if q.dtype != torch.int8 or q.ndim != 2 or bits not in (4, 8):
        raise ValueError("expected a rank-2 signed quantized weight")
    if bits == 8:
        return q.contiguous()
    if q.shape[1] % 2 or torch.any((q < -8) | (q > 7)).item():
        raise ValueError("W4 requires an even K and signed codes in [-8,7]")
    codes = q.to(torch.int16) + 8
    return (codes[:, 0::2] + 16 * codes[:, 1::2]).to(torch.uint8).contiguous()


class GroupQuantLinear(nn.Module):
    """Compressed resident weights; FP16 activations and output.

    ``weight_quant`` consumes INT8 codes and group scales through CANN's
    WeightQuantBatchMatmulV2. W4 stays packed at rest and unpacks to INT8
    transiently: 310P does not expose native INT4 weights through this API.
    ``dequant`` is the explicit FP16-dequantization oracle for A/B checks.
    Neither route retains a dense FP16 weight; compiled workspace must be
    measured on the target device.
    """

    def __init__(self, packed: Tensor, scale: Tensor, *, bits: int,
                 in_features: int, ops: Any, matmul_backend: str = "dequant") -> None:
        super().__init__()
        if bits not in (4, 8) or in_features <= 0 or in_features % 128:
            raise ValueError("quantized Linear requires W4/W8 and K divisible by 128")
        expected_dtype = torch.int8 if bits == 8 else torch.uint8
        if (packed.ndim != 2 or packed.dtype != expected_dtype
                or packed.shape[1] != in_features * bits // 8
                or tuple(scale.shape) != (packed.shape[0], in_features // 128)
                or scale.dtype != torch.float16 or scale.device != packed.device):
            raise ValueError("quantized Linear buffer shape, dtype or device mismatch")
        self.in_features = in_features
        self.out_features = packed.shape[0]
        self.bits = bits
        self.group_size = 128
        self.ops = ops
        if matmul_backend not in DRAFT_QUANT_MATMUL_BACKENDS:
            raise ValueError("draft_quant_matmul must be weight_quant or dequant")
        self.matmul_backend = matmul_backend
        self.register_buffer("qweight", packed)
        self.register_buffer("scales", scale)

    def dequantize(self) -> Tensor:
        values = self.qweight.to(torch.float16)
        if self.bits == 4:
            high = torch.floor(values * (1.0 / 16.0))
            low = values - high * 16.0
            values = torch.stack((low, high), dim=-1).reshape(self.out_features, self.in_features) - 8.0
        values = values.reshape(self.out_features, self.in_features // 128, 128)
        return (values * self.scales.unsqueeze(-1)).reshape(self.out_features, self.in_features)

    def integer_weight(self) -> Tensor:
        """Return signed [N,K] codes without applying scales or retaining a copy."""
        if self.bits == 8:
            return self.qweight
        # All byte arithmetic is exact in FP16. Cast each half BEFORE stacking
        # to avoid materializing a full [N,K] FP16 matrix for the native path.
        values = self.qweight.to(torch.float16)
        high = torch.floor(values * (1.0 / 16.0))
        low = (values - high * 16.0 - 8.0).to(torch.int8)
        high = (high - 8.0).to(torch.int8)
        return torch.stack((low, high), dim=-1).reshape(self.out_features, self.in_features)

    @classmethod
    def concatenate(cls, left, right):
        if not isinstance(right, cls) or (left.bits, left.in_features, left.matmul_backend) != (right.bits, right.in_features, right.matmul_backend):
            raise ValueError("packed projections need matching quantization and input dimensions")
        return cls(torch.cat((left.qweight, right.qweight), dim=0),
                   torch.cat((left.scales, right.scales), dim=0), bits=left.bits,
                   in_features=left.in_features, ops=left.ops, matmul_backend=left.matmul_backend)

    def forward(self, value: Tensor) -> Tensor:
        if value.dtype != torch.float16:
            raise ValueError("W8A16/W4A16 Draft activations must be float16")
        if self.matmul_backend == "weight_quant":
            from .weight_quant_matmul import weight_quant_linear
            return weight_quant_linear(value, self.integer_weight(), self.scales)
        return self.ops.linear(value, self.dequantize())


def audit_quantized_tensors(root: Path, config: Qwen35DFlashConfig, bits: int) -> dict[str, int]:
    from safetensors import safe_open

    expected = {}
    linear_count = resident_bytes = max_dense_bytes = 0
    for name, shape in config.required_tensor_shapes().items():
        if len(shape) == 1:
            expected[name] = (shape, "BF16")
            resident_bytes += shape[0] * 2
        else:
            n, k = shape
            if k % 128:
                raise ValueError("Draft K must be divisible by group_size=128")
            prefix = name.removesuffix("weight")
            expected[prefix + "weight_packed"] = ((n, k * bits // 32), "I32")
            expected[prefix + "weight_scale"] = ((n, k // 128), "BF16")
            expected[prefix + "weight_shape"] = ((2,), "I64")
            linear_count += 1
            resident_bytes += n * k * bits // 8 + n * (k // 128) * 2
            max_dense_bytes = max(max_dense_bytes, n * k * 2)
    with safe_open(root / "model.safetensors", framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(expected):
            raise ValueError("quantized Draft tensor names differ from the checkpoint contract")
        for name, (shape, dtype) in expected.items():
            view = handle.get_slice(name)
            if tuple(view.get_shape()) != shape or view.get_dtype() != dtype:
                raise ValueError(f"quantized Draft tensor shape/dtype mismatch: {name}")
            if name.endswith("weight_shape"):
                key = name.removesuffix("weight_shape") + "weight"
                if tuple(handle.get_tensor(name).tolist()) != config.required_tensor_shapes()[key]:
                    raise ValueError(f"quantized Draft logical shape mismatch: {name}")
    return {"linear_count": linear_count, "actual_tensor_count": len(expected),
            "resident_weight_bytes": resident_bytes, "largest_dequantized_linear_bytes": max_dense_bytes}


def require_draft_checkpoint(model_dir: str | Path, variant: str = "fp16") -> dict[str, Any]:
    if variant == "fp16":
        return require_official_dflash_checkpoint(model_dir, verify_model_hash=True)
    if variant not in QUANTIZED_DRAFTS:
        raise ValueError(f"unsupported Draft quantization: {variant}")
    root = Path(model_dir).expanduser().resolve()
    lock = QUANTIZED_DRAFTS[variant]
    for name, key in (("config.json", "config_sha256"), ("model.safetensors", "model_sha256")):
        path = root / name
        if not path.is_file() or path.is_symlink() or sha256_file(path) != lock[key]:
            raise ValueError(f"{variant} checkpoint hash mismatch: {path}")
    if (root / "model.safetensors").stat().st_size != lock["model_bytes"]:
        raise ValueError("quantized Draft checkpoint size mismatch")
    raw = json.loads((root / "config.json").read_text())
    config = Qwen35DFlashConfig.from_dict(raw)
    bits = validate_quantization_config(raw, variant)
    tensor_audit = audit_quantized_tensors(root, config, bits)
    return {"status": "PASS", "variant": variant, "source": dict(lock),
            "config_sha256": lock["config_sha256"], "model_sha256": lock["model_sha256"],
            "model_bytes": lock["model_bytes"], "parameter_count": config.parameter_count,
            "config": config.to_dict(), "eos_token_id": raw["eos_token_id"], **tensor_audit}


@torch.no_grad()
def load_quantized_draft(model_class: type[nn.Module], model_dir: str | Path, *,
                         variant: str, ops: Any, device: str | torch.device,
                         dtype: torch.dtype) -> nn.Module:
    from safetensors import safe_open
    from .modeling_dflash import DFlashRotaryEmbedding

    if dtype != torch.float16:
        raise ValueError("quantized Draft supports float16 activations only")
    audit = require_draft_checkpoint(model_dir, variant)
    root = Path(model_dir).expanduser().resolve()
    config = Qwen35DFlashConfig.from_pretrained(root)
    bits = QUANTIZED_DRAFTS[variant]["num_bits"]
    backend = getattr(ops, "quant_matmul_backend", None) or (
        "weight_quant" if str(device).startswith("npu") else "dequant")
    if backend not in DRAFT_QUANT_MATMUL_BACKENDS:
        raise ValueError("draft_quant_matmul must be weight_quant or dequant")
    if backend == "weight_quant":
        from .weight_quant_matmul import require_weight_quant_matmul
        require_weight_quant_matmul()
    # Meta construction avoids ever allocating a full FP16 draft alongside the
    # compressed buffers. Each Linear is replaced before device placement.
    model = model_class(config, ops=ops, device="meta", dtype=dtype)
    with safe_open(root / "model.safetensors", framework="pt", device="cpu") as handle:
        for name, shape in config.required_tensor_shapes().items():
            if len(shape) == 1:
                module_path, attribute = name.rsplit(".", 1)
                value = handle.get_tensor(name).to(dtype=dtype, device=device)
                if not torch.isfinite(value).all().item():
                    raise ValueError(f"nonfinite Draft norm: {name}")
                setattr(model.get_submodule(module_path), attribute, nn.Parameter(value, requires_grad=False))
            else:
                path = name.removesuffix(".weight")
                q = unpack_packed_int32(handle.get_tensor(path + ".weight_packed"), bits, shape)
                packed = pack_device_weight(q, bits).to(device=device)
                scales = handle.get_tensor(path + ".weight_scale").to(dtype=dtype)
                if not torch.isfinite(scales).all().item() or not (scales > 0).all().item():
                    raise ValueError(f"nonfinite or nonpositive Draft scale: {path}")
                module = GroupQuantLinear(packed, scales.to(device=device), bits=bits,
                                          in_features=shape[1], ops=model.ops, matmul_backend=backend)
                if "." in path:
                    parent, attribute = path.rsplit(".", 1)
                    setattr(model.get_submodule(parent), attribute, module)
                else:
                    setattr(model, path, module)
    model.rotary = DFlashRotaryEmbedding(config, device=device)
    model.draft_quantization = variant
    model.draft_quantization_audit = {
        **audit, "execution": ("cann-weight-quant-batchmatmul-v2" if backend == "weight_quant"
                                else "device-group-dequant-fp16-matmul"),
        "matmul_backend": backend, "group_size": 128,
        "matmul_weight_dtype": "int8" if backend == "weight_quant" else "float16",
        "w4_unpack": "transient-int8" if bits == 4 and backend == "weight_quant" else None,
        "inner_precise": 0 if backend == "weight_quant" else None,
        "persistent_dense_linear_weights": False, "activation_dtype": "float16",
        "new_custom_operators": [], "device": str(device),
    }
    return model.eval()


def configure_target_for_draft(target: nn.Module, config: Qwen35DFlashConfig) -> None:
    """Bind selected feature rows per model instance, without changing Target math."""
    from .dflash_target_features import DFlashTargetFeatureSpec

    spec = DFlashTargetFeatureSpec.from_draft_config(config)
    for module in target.modules():
        existing = getattr(module, "dflash_target_feature_spec", None)
        if existing is not None and existing != spec:
            raise ValueError("Target is already bound to a different Draft feature contract")
        module.dflash_target_feature_spec = spec


def require_loaded_quantized_draft(draft: nn.Module) -> None:
    """Keep the scheduler's checkpoint gate when selecting a five-layer draft."""
    variant = getattr(draft, "draft_quantization", None)
    audit = getattr(draft, "draft_quantization_audit", {})
    if variant not in QUANTIZED_DRAFTS:
        raise ValueError("unknown loaded quantized Draft")
    lock = QUANTIZED_DRAFTS[variant]
    if (audit.get("status") != "PASS" or audit.get("variant") != variant
            or audit.get("source") != lock
            or audit.get("model_sha256") != lock["model_sha256"]
            or audit.get("config_sha256") != lock["config_sha256"]
            or audit.get("config") != draft.config.to_dict()):
        raise ValueError("quantized Draft has no matching pinned checkpoint audit")
    linears = [module for module in draft.modules() if isinstance(module, GroupQuantLinear)]
    if len(linears) != 36 or any(module.bits != lock["num_bits"] for module in linears):
        raise ValueError("quantized Draft Linear coverage differs from checkpoint")
