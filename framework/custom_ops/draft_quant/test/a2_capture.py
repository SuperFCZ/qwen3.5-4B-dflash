#!/usr/bin/env python3
"""Capture A2 gate/up+down or A4 all projections during a frozen native Draft replay.

This imports the production model without editing it. The replay stops after
the last down projection, so no Target decoder or vocabulary head is loaded.
These are native eager activations, not claimed to be internal full-OM dumps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from a2_common import (REPO, SHAPES, record, replay_context_rows,
                       sha256, snapshot_contract, write_json, scope_identity)
import a4_contract

sys.path[:0] = [str(REPO / "framework/python"), str(REPO)]


def require_draft_row_update(torch_module):
    """Check the production overload's registration; replay tests NPU execution."""
    message = ("Draft capture requires callable torch.ops.npu.npu_scatter_nd_update.default "
               "for the production functional whole-row cache update. Load the receiver "
               "torch_npu extension that registers this operator; no fallback is used.")
    try:
        operation = torch_module.ops.npu.npu_scatter_nd_update.default
    except AttributeError as error:
        raise RuntimeError(message) from error
    if not callable(operation):
        raise RuntimeError(message)
    return operation


def frozen_inputs(report_path, config, feature_layers):
    import numpy as np
    report_path = report_path.resolve()
    report = json.loads(report_path.read_text())
    if report.get("fake_acl") is not False:
        raise ValueError("capture requires a real AscendCL frozen Draft replay report")
    directory = Path(report["input_directory"]).resolve()
    contract = directory / "contract.txt"
    om_hash, specs = snapshot_contract(contract.read_text())
    if om_hash != report["draft_om_sha256"]:
        raise ValueError("snapshot OM identity differs from replay report")
    names = ["features", "start_position", "valid_rows", "anchor", "proposal_count"]
    names += [f"d{i}_{kind}" for i in range(5) for kind in ("key", "value")]
    arrays = []
    for name in names:
        spec, path = specs[name], directory / f"{name}.bin"
        if path.stat().st_size != spec["bytes"] or sha256(path) != report["snapshot_sha256"][name]:
            raise ValueError(f"frozen snapshot bytes/hash differ: {name}")
        dtype = {"float16": "<f2", "int64": "<i8", "int16": "<i2"}.get(spec["dtype"])
        if dtype is None:
            raise ValueError(f"unsupported replay input dtype: {name}")
        arrays.append(np.fromfile(path, dtype=dtype).reshape(spec["shape"]))
    features, start, valid, anchor, count, *state = arrays
    layers = tuple(feature_layers or config.target_layer_ids)
    if tuple(sorted(set(layers))) != layers or not set(config.target_layer_ids).issubset(layers):
        raise ValueError("--feature-layers must match the sorted Target feature-layer order")
    if (features.dtype != np.float16 or features.ndim != 3 or features.shape[0] != 1 or
            features.shape[1] not in (16, 64) or features.shape[2] != config.hidden_size * len(layers)):
        raise ValueError("frozen feature shape differs; pass the original export's --feature-layers")
    for name, value, dtype in (("start", start, np.int64), ("anchor", anchor, np.int64),
                               ("valid", valid, np.int16), ("proposal_count", count, np.int16)):
        if value.shape != (1,) or value.dtype != dtype:
            raise ValueError(f"invalid replay scalar ABI: {name}")
    if (not 0 <= int(anchor[0]) < config.vocab_size or not 1 <= int(count[0]) <= 15 or
            not 1 <= int(valid[0]) <= features.shape[1] or int(start[0]) < 0):
        raise ValueError("invalid replay scalar values")
    storage_rows = features.shape[1]
    context_rows = replay_context_rows(int(valid[0]), storage_rows,
                                       report.get("kv_output_audit", {}).get("context_rows"))
    features = np.ascontiguousarray(features[:, :context_rows])
    arrays[0] = features
    capacity = state[0].shape[2] if state[0].ndim == 4 else 0
    expected = (1, config.num_key_value_heads, capacity, config.head_dim)
    if capacity <= 0 or int(start[0]) + features.shape[1] > capacity:
        raise ValueError("snapshot cache cannot hold the context gear")
    if any(a.shape != expected or a.dtype != np.float16 or not np.isfinite(a).all() for a in state):
        raise ValueError("snapshot cache shape/dtype/finite check failed")
    if not np.isfinite(features[:, :int(valid[0])]).all():
        raise ValueError("nonfinite visible frozen features")
    source = {"report": str(report_path), "report_sha256": sha256(report_path),
              "contract_sha256": sha256(contract), "draft_om_sha256": om_hash,
              "snapshot_sha256": {n: report["snapshot_sha256"][n] for n in names},
              "replay_input_sha256": {name: hashlib.sha256(a.tobytes()).hexdigest() for name, a in zip(names, arrays)},
              "feature_layers": list(layers), "context_rows": context_rows,
              "feature_storage_rows": storage_rows,
              "draft_static64_om_sha256": report.get("draft_static64_om_sha256"),
              "start_position": int(start[0]), "valid_rows": int(valid[0]),
              "anchor": int(anchor[0]), "proposal_count": int(count[0]),
              "full_om_internal_activation_parity": "NOT_RUN"}
    return names, arrays, layers, source


def embedding_rows(directory, key, ids, config):
    import torch
    from safetensors import safe_open
    directory = directory.resolve()
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        shard = json.loads(index.read_text())["weight_map"][key]
        path = (directory / shard).resolve()
    else:
        path = directory / "model.safetensors"
    if not path.is_relative_to(directory):
        raise ValueError("embedding shard escapes checkpoint")
    with safe_open(path, framework="pt", device="cpu") as handle:
        tensor = handle.get_slice(key)
        if tensor.get_shape() != [config.vocab_size, config.hidden_size]:
            raise ValueError("Target embedding must match Draft vocabulary and hidden size")
        # Slice before materialization: only anchor/mask rows are needed.
        rows = torch.cat([tensor[i:i + 1].to(torch.float16) for i in ids], dim=0)
    if not torch.isfinite(rows).all().item():
        raise ValueError("nonfinite Target embedding rows")
    return rows, {"checkpoint": str(directory), "tensor": key, "token_ids": ids,
                  "shard": str(path), "shard_sha256": sha256(path),
                  "rows_fp16_sha256": hashlib.sha256(rows.numpy().tobytes()).hexdigest()}


def capture(args):
    scope = getattr(args, "scope", "a2")
    abi, case_names = scope_identity(scope)
    if os.environ.get("ASCEND310P_SIMULATION_ONLY") == "1":
        raise ValueError(f"{scope.upper()} capture requires a real NPU")
    import torch
    import torch_npu
    from models.dflash_v1.draft_quantization import load_quantized_draft
    from models.dflash_v1.modeling_dflash import DFlashDraftModel
    from qwen35_dflash.ascend310p.incremental import DraftGraph
    from qwen35_dflash.ascend310p.quant_factory import AirDFlashOps
    from qwen35_dflash.ascend310p.utils import require_run_output
    from qwen35_dflash.ascend310p.weight_prepack import pack_int8_nz

    root = require_run_output(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    manifest = {"abi": abi, "status": "RUNNING", "capture_runtime": "native NPU DraftGraph replay",
                "cpu_fallback": False, "capture_repeat_equal": False, "cases": [],
                "native_om_parity": "NOT_RUN", "full_draft_validation": "NOT_RUN"}
    write_json(root / "manifest.json", manifest)
    handles = []
    try:
        device = f"npu:{args.device_id}"
        torch.npu.set_device(device)
        device_name = torch.npu.get_device_name(args.device_id)
        if "310P" not in device_name.upper():
            raise ValueError(f"A2 expects Ascend 310P, got {device_name}")
        row_update = require_draft_row_update(torch)
        ops = AirDFlashOps(quant_matmul_backend="weight_quant")
        draft = load_quantized_draft(DFlashDraftModel, args.draft_dir, variant="w8a16", ops=ops,
                                     device=device, dtype=torch.float16)
        config = draft.config
        if (config.hidden_size, config.intermediate_size, config.num_hidden_layers, config.block_size) != (2560, 9728, 5, 16):
            raise ValueError("A2 requires the pinned five-layer W8 Draft")
        names, arrays, layers, source = frozen_inputs(args.replay_report, config, args.feature_layers)
        if scope == "a4" and (getattr(args, "context_rows", None) not in (16, 64) or
                              args.context_rows != source["context_rows"]):
            raise ValueError("A4 --context-rows must match the actual frozen replay gear; no padding/replacement")
        rows, embedding_source = embedding_rows(args.target_dir, args.embedding_key,
                                                 [source["anchor"], config.mask_token_id], config)

        class FrozenEmbedding(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("rows", rows.to(device))

            def forward(self, ids):
                if not ((ids == source["anchor"]) | (ids == config.mask_token_id)).all().item():
                    raise ValueError("unexpected token in frozen replay")
                return self.rows[(ids != source["anchor"]).long()]

        # The final down hook exits before the vocabulary head is touched.
        graph = DraftGraph(draft, FrozenEmbedding(), torch.nn.Identity(),
                           row_update=row_update, consume_source=True, feature_layers=layers).eval()
        manifest.update(checkpoint=draft.draft_quantization_audit, source=source,
                        context_rows=source["context_rows"],
                        embedding=embedding_source,
                        environment={"torch": str(torch.__version__), "torch_npu": str(torch_npu.__version__),
                                     "device_id": args.device_id, "device_name": device_name},
                        sources={str(p.relative_to(REPO)): sha256(p) for p in (
                            Path(__file__), Path(__file__).with_name("a4_contract.py"),
                            Path(__file__).with_name("a2_common.py"), REPO / "models/dflash_v1/draft_quantization.py",
                            REPO / "models/dflash_v1/weight_quant_matmul.py",
                            REPO / "framework/python/qwen35_dflash/ascend310p/incremental.py")})
        inputs = tuple(torch.from_numpy(a.copy()).to(device) for a in arrays)
        current_repeat, seen = 0, []

        class CaptureComplete(Exception):
            pass

        def save(case, name, tensor):
            path = root / case["name"] / name
            path.parent.mkdir(exist_ok=True)
            tensor.detach().cpu().contiguous().numpy().tofile(path)
            case["files"][name] = record(path, root)

        def hook_for(descriptor):
            case = dict(descriptor, files={})
            m, k, n = (case[d] for d in ("m", "k", "n"))
            manifest["cases"].append(case)

            def hook(module, values, result):
                if (tuple(values[0].shape) not in ((m, k), (1, m, k)) or
                        tuple(result.shape) not in ((m, n), (1, m, n))):
                    raise ValueError(f"captured projection row/feature shape differs: {case['name']}")
                x = values[0].detach().reshape(m, k).contiguous()
                y = result.detach().reshape(m, n).contiguous()
                if x.dtype != torch.float16 or y.dtype != torch.float16 or not torch.isfinite(x).all().item() or not torch.isfinite(y).all().item():
                    raise ValueError(f"invalid real activation/output: {case['name']}")
                if module.bits != 8 or module.matmul_backend != "weight_quant":
                    raise ValueError("capture cannot use dequant/CPU fallback")
                if current_repeat == 0:
                    q, s = module.integer_weight().cpu().contiguous(), module.scales.cpu().contiguous()
                    if tuple(q.shape) != (n, k) or tuple(s.shape) != (n, k // 128):
                        raise ValueError("captured checkpoint shape differs")
                    save(case, "x.bin", x)
                    save(case, "q_nk.bin", q)
                    save(case, "w_nz.bin", pack_int8_nz(q))
                    save(case, "s_gn.bin", s.t().contiguous())
                elif hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() != case["files"]["x.bin"]["sha256"]:
                    raise ValueError(f"activation capture drift: {case['name']}")
                save(case, f"eager-{current_repeat}.bin", y)
                if current_repeat and case["files"]["eager-0.bin"]["sha256"] != case["files"]["eager-1.bin"]["sha256"]:
                    raise ValueError(f"native eager repeat drift: {case['name']}")
                seen.append(case["name"])
                if case["name"] == "layer-4-down":
                    raise CaptureComplete()
            return hook

        if scope == "a4":
            targets = a4_contract.capture_targets(graph, source["context_rows"])
        else:
            targets = []
            for layer, module in enumerate(graph.layers):
                for kind, projection in (("gate_up", module.gate_up_linear), ("down", module.down_proj)):
                    m, k, n = SHAPES[kind]
                    targets.append((projection, dict(name=f"layer-{layer}-{kind}", layer=layer, projection=kind,
                        m=m, k=k, n=n, group_size=128, layout="nz_int8_v1")))
        for module, descriptor in targets:
            handles.append(module.register_forward_hook(hook_for(descriptor)))
        with torch.inference_mode():
            for current_repeat in range(2):
                seen.clear()
                try:
                    graph(*inputs)
                except CaptureComplete:
                    pass
                torch.npu.synchronize()
                if tuple(seen) != case_names:
                    raise ValueError("incomplete or repeated projection capture")
                for name, value in zip(names, inputs):
                    if hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest() != source["replay_input_sha256"][name]:
                        raise ValueError(f"replay modified frozen input: {name}")
        manifest.update(status="PASS", capture_repeat_equal=True)
    except Exception as error:
        manifest.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for handle in handles:
            handle.remove()
        write_json(root / "manifest.json", manifest)
    print(f"Captured {len(case_names)} {scope.upper()} real projections: {root / 'manifest.json'}; OM/custom validation NOT_RUN", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("a2", "a4"), default="a2")
    parser.add_argument("--context-rows", type=int, choices=(16, 64), help="required frozen gear for A4")
    parser.add_argument("--draft-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True, help="original Target checkpoint containing embeddings")
    parser.add_argument("--embedding-key", default="model.language_model.embed_tokens.weight")
    parser.add_argument("--replay-report", type=Path, required=True, help="debug_draft_om.py private.json/shared.json")
    parser.add_argument("--feature-layers", type=lambda value: tuple(map(int, value.split(","))),
                        help="original Target export's feature order; defaults to W8 Draft's selected layers")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory below AI_RUN_DIR")
    parser.add_argument("--device-id", type=int, default=0)
    capture(parser.parse_args())


if __name__ == "__main__":
    main()
