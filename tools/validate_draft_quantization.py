"""Real-checkpoint CPU reference gate; does not claim NPU or OM validation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.dflash_v1.draft_quantization import GroupQuantLinear
from models.dflash_v1.modeling_dflash import DFlashDraftModel, WeightOnlyLinear


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-dir", required=True)
    parser.add_argument("--draft-quantization", required=True, choices=("w8a16", "w4a16"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if "AI_RUN_DIR" in os.environ:
        output.relative_to(Path(os.environ["AI_RUN_DIR"]).resolve())
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(args.threads)
    model = DFlashDraftModel.from_pretrained(args.draft_dir, dtype=torch.float16,
                                            draft_quantization=args.draft_quantization)
    torch.manual_seed(703)
    config = model.config
    hidden = torch.randn(1, 3, config.feature_size, dtype=torch.float16) * 0.1
    noise = torch.randn(1, 16, config.hidden_size, dtype=torch.float16) * 0.1
    positions = torch.arange(19).view(1, -1)
    with torch.inference_mode():
        result = model(hidden, noise, positions)
        cache = model.new_kv_cache(max_length=64)
        model.forward_cached(hidden[:, :2], noise, torch.arange(18).view(1, -1), cache)
        cached = model.forward_cached(hidden[:, 2:], noise, torch.arange(2, 19).view(1, -1), cache)
    # Independent decoder: view the original little-endian packed words as
    # bytes, then unpack two nibbles. No production bit-unpack or dequant helper.
    linears = []
    with safe_open(Path(args.draft_dir) / "model.safetensors", framework="pt", device="cpu") as source:
        for name, module in list(model.named_modules()):
            if not isinstance(module, GroupQuantLinear):
                continue
            words = source.get_tensor(name + ".weight_packed").numpy().astype('<i4')
            codes = words.view(np.uint8).reshape(module.out_features, -1)
            if module.bits == 4:
                codes = np.stack((codes & 15, codes >> 4), axis=-1).reshape(module.out_features, -1)
            q = codes[:, :module.in_features].astype(np.int16) - (1 << (module.bits - 1))
            scale = source.get_tensor(name + ".weight_scale").float().numpy()
            weight = (q.reshape(module.out_features, -1, 128).astype(np.float32)
                      * scale[..., None]).reshape(q.shape).astype(np.float16)
            reference = torch.from_numpy(weight)
            torch.testing.assert_close(module.dequantize(), reference, rtol=0, atol=0)
            dense = WeightOnlyLinear(module.in_features, module.out_features, model.ops, dtype=torch.float16)
            with torch.no_grad(): dense.weight.copy_(reference)
            parent, attribute = name.rsplit('.', 1) if '.' in name else ('', name)
            setattr(model.get_submodule(parent), attribute, dense)
            linears.append(name)
    with torch.inference_mode():
        oracle = model(hidden, noise, positions)
    torch.testing.assert_close(result, oracle, rtol=0, atol=0)
    torch.testing.assert_close(cached, oracle, rtol=0, atol=0)
    if not torch.isfinite(result).all():
        raise RuntimeError("nonfinite real Draft output")
    payload = {
        "status": "PASS", "scope": "real quantized Draft CPU reference and incremental cache",
        "cpu_fallback": True, "torch": str(torch.__version__),
        "checkpoint": model.draft_quantization_audit,
        "independent_linear_comparisons": linears,
        "linear_max_abs_error": 0.0, "draft_hidden_max_abs_error": 0.0,
        "cached_hidden_max_abs_error": 0.0, "draft_layers": len(model.layers),
        "cached_context_rows": cache.committed_length, "output_shape": list(result.shape),
        "full_target_generation": "PENDING", "npu": "PENDING", "om": "PENDING",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({k:v for k,v in payload.items() if k not in ('checkpoint', 'independent_linear_comparisons')}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
