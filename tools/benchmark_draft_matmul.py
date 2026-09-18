#!/usr/bin/env python3
"""NPU-only A/B probe of grouped Draft MatMul, without loading model weights."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "framework/python"), str(REPO)]

import torch

from models.dflash_v1.draft_quantization import GroupQuantLinear, pack_device_weight
from models.dflash_v1.weight_quant_matmul import preflight_weight_quant_matmul
from qwen35_dflash.ascend310p.utils import atomic_write_json, require_run_output

# M,K,N from the five-layer published checkpoints; KV includes context+noise.
SHAPES = {"gate_up": (16, 2560, 19456), "q": (16, 2560, 4096),
          "kv": (80, 2560, 2048), "down": (16, 9728, 2560), "fc": (64, 12800, 2560)}


class DequantOracleOps:
    @staticmethod
    def linear(value, weight):
        return torch.nn.functional.linear(value, weight)


@torch.inference_mode()
def measure(layer, value, warmup, repetitions):
    for _ in range(warmup):
        layer(value)
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    base_bytes = torch.npu.memory_allocated()
    times, outputs = [], []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        output = layer(value)
        torch.npu.synchronize()
        times.append((time.perf_counter_ns() - start) / 1e6)
        outputs.append(output.cpu())  # transfer outside the measured interval
        del output
    return {
        "median_ms": statistics.median(times), "calls_ms": times,
        "torch_allocator_extra_peak_bytes": torch.npu.max_memory_allocated() - base_bytes,
        "repeat_drift": any(not torch.equal(outputs[0], item) for item in outputs[1:]),
    }, outputs[0]


def comparison(reference, current):
    if not bool(torch.isfinite(current).all() and torch.isfinite(reference).all()):
        return {"finite": False, "bitwise_equal": False, "mismatched_values": None,
                "max_abs_error": None, "mean_abs_error": None, "relative_l2_error": None}
    delta = (reference.float() - current.float()).abs()
    return {"bitwise_equal": torch.equal(reference, current),
            "mismatched_values": int(torch.count_nonzero(reference != current)),
            "max_abs_error": float(delta.max()), "mean_abs_error": float(delta.mean()),
            "relative_l2_error": float(torch.linalg.vector_norm(delta) /
                                        torch.linalg.vector_norm(reference.float()).clamp_min(1e-30)),
            "finite": bool(torch.isfinite(current).all())}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--device-id", type=int, default=0)
    cli.add_argument("--bits", type=int, nargs="+", choices=(4, 8), default=[4, 8])
    cli.add_argument("--projection", nargs="+", choices=tuple(SHAPES), default=["gate_up"])
    cli.add_argument("--warmup", type=int, default=1)
    cli.add_argument("--repetitions", type=int, default=3)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.warmup < 0 or args.repetitions < 1:
        cli.error("warmup must be >=0 and repetitions >=1")
    path = require_run_output(args.output)
    if path.exists():
        cli.error("use a new output path")
    report = {"schema_version": 1, "status": "RUNNING", "cpu_fallback": False,
              "scope": "synthetic same-input eager NPU linear A/B; not OM or full-Draft validation",
              "warmup": args.warmup, "repetitions": args.repetitions,
              "activation_dtype": "float16", "group_size": 128, "inner_precise": 0,
              "memory_scope": "PyTorch allocator only; CANN workspace/OM peak requires profiling", "cases": []}
    try:
        import torch_npu
        device = f"npu:{args.device_id}"
        torch.npu.set_device(device)
        report["environment"] = {"device": device, "device_name": torch.npu.get_device_name(args.device_id),
                                  "torch": str(torch.__version__), "torch_npu": str(torch_npu.__version__)}
        report["preflight"] = preflight_weight_quant_matmul(device)
        print("Bits | Projection | Shape M,K,N | Dequant ms | WeightQuant ms | Speedup | Max abs error", flush=True)
        for bits in dict.fromkeys(args.bits):
            for name in dict.fromkeys(args.projection):
                m, k, n = SHAPES[name]
                generator = torch.Generator().manual_seed(781 + bits)
                q = torch.randint(-(1 << (bits-1)), 1 << (bits-1), (n, k), dtype=torch.int8, generator=generator)
                packed = pack_device_weight(q, bits).to(device)
                scales = (torch.rand(n, k // 128, generator=generator) * .008 + .002).half().to(device)
                value = torch.randn(m, k, generator=generator).half().to(device)
                layer = GroupQuantLinear(packed, scales, bits=bits, in_features=k,
                                         ops=DequantOracleOps(), matmul_backend="dequant")
                del q
                baseline, golden = measure(layer, value, args.warmup, args.repetitions)
                layer.matmul_backend = "weight_quant"
                native, actual = measure(layer, value, args.warmup, args.repetitions)
                errors = comparison(golden, actual)
                if not errors["finite"]:
                    raise RuntimeError("matmul returned non-finite values")
                row = {"bits": bits, "projection": name, "mkn": [m, k, n],
                       "dequant": baseline, "weight_quant": native, "difference": errors,
                       "speedup": baseline["median_ms"] / native["median_ms"]}
                report["cases"].append(row)
                atomic_write_json(path, report)
                print(f"{bits} | {name} | {m},{k},{n} | {baseline['median_ms']:.3f} | "
                      f"{native['median_ms']:.3f} | {row['speedup']:.2f}x | {errors['max_abs_error']:.6g}", flush=True)
                del layer, packed, scales, value, golden, actual
                gc.collect()
                torch.npu.empty_cache()
        report["status"] = "MEASURED"
        report["accuracy_gate"] = "not established; inspect differences and compare full Draft tokens/acceptance"
    except Exception as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        atomic_write_json(path, report)
        cli.exit(1, f"draft-matmul: {error}\nReport: {path}\n")
    atomic_write_json(path, report)
    print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
