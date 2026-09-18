#!/usr/bin/env python3
"""Compile synthetic W4/W8 Draft linears through the production AIR path.

No checkpoint, Target model or C++ runner is loaded. This tests ATC graph
compatibility, not OM execution, accuracy or performance.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "framework/python"), str(REPO)]

import torch

from models.dflash_v1.draft_quantization import GroupQuantLinear, pack_device_weight
from models.dflash_v1.weight_quant_matmul import TORCH_OP, GE_OP, require_weight_quant_matmul
from qwen35_dflash.ascend310p.contracts import AirGraphSpec, CustomOpExportSpec
from qwen35_dflash.ascend310p.compiler import compile_air_bundle, resolve_atc_executable, validate_soc_version
from qwen35_dflash.ascend310p.exporter import export_air_bundle
from qwen35_dflash.ascend310p.utils import atomic_write_json, require_run_output

# K,N. X's M axis is dynamic with the same 16/64 ATC gears as Draft features.
SHAPES = {"tiny": (256, 64), "gate_up": (2560, 19456), "q": (2560, 4096),
          "kv": (2560, 2048), "down": (9728, 2560), "fc": (12800, 2560)}


class ProbeLinear(torch.nn.Module):
    def __init__(self, packed, scales, bits, k):
        super().__init__()
        self.n, self.k, self.bits = packed.shape[0], k, bits
        self.linear = GroupQuantLinear(packed, scales, bits=bits, in_features=k,
                                      ops=None, matmul_backend="weight_quant")
        # Just as in DraftConstantInputs, weights are flat runtime inputs, not
        # Const nodes that ATC can expand/fold into a dense FP16 checkpoint.
        self.linear.qweight = packed.new_empty(0)
        self.linear.scales = scales.new_empty(0)

    def forward(self, x, qweight, scales):
        return torch.func.functional_call(self.linear, {
            "qweight": qweight.view(self.n, self.k * self.bits // 8),
            "scales": scales.view(self.n, self.k // 128),
        }, (x,))


def make_spec(bits, projection, device):
    k, n = SHAPES[projection]
    generator = torch.Generator().manual_seed(812 + bits)
    q = torch.randint(-(1 << (bits - 1)), 1 << (bits - 1), (n, k),
                      generator=generator, dtype=torch.int8)
    packed = pack_device_weight(q, bits).to(device)
    scales = ((1 + torch.arange(n * (k // 128)).reshape(n, k // 128) % 4).half() / 32).to(device)
    x = (torch.randn(16, k, generator=generator).half() / 16).to(device)
    model = ProbeLinear(packed, scales, bits, k).eval()
    args = (x, packed.view(-1), scales.view(-1))
    names = ("x", "qweight", "scales")
    signature = [{"name": name, "shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch.")}
                 for name, t in zip(names, args)]
    return AirGraphSpec(name="weight_quant_probe", role="diagnostic", model=model, example_args=args,
                        input_names=names, output_names=("y",), dynamic=True,
                        custom_ops=(CustomOpExportSpec(TORCH_OP, GE_OP),),
                        metadata={"tensor_abi": {"inputs": signature}, "dynamic_input_axes": {"x": [0]},
                                  "bits": bits, "projection": projection, "group_size": 128})


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-dir", type=Path, required=True)
    cli.add_argument("--atc", required=True)
    cli.add_argument("--soc-version", required=True)
    cli.add_argument("--device-id", type=int, default=0)
    cli.add_argument("--bits", nargs="+", choices=(4, 8), type=int, default=[4, 8])
    cli.add_argument("--projection", nargs="+", choices=tuple(SHAPES), default=["tiny"])
    return cli


def main(argv=None):
    cli = parser(); args = cli.parse_args(argv)
    root = require_run_output(args.output_dir)
    if root.exists():
        cli.error("use a new output directory; existing probe evidence is retained")
    atc = resolve_atc_executable(args.atc)
    soc = validate_soc_version(args.soc_version)
    root.mkdir(parents=True)
    report = {"schema_version": 1, "status": "RUNNING", "cpu_fallback": False,
              "scope": "synthetic AIR and ATC compilation only; no OM execution or speed claim",
              "soc_version": soc, "cases": []}
    try:
        import torch_npu
        import torchair
        device = f"npu:{args.device_id}"
        torch.npu.set_device(device)
        require_weight_quant_matmul()
        report["environment"] = {"torch": str(torch.__version__), "torch_npu": str(torch_npu.__version__),
                                 "device": device, "device_name": torch.npu.get_device_name(args.device_id)}
        for projection in dict.fromkeys(args.projection):
            for bits in dict.fromkeys(args.bits):
                case = {"bits": bits, "projection": projection, "status": "RUNNING"}
                report["cases"].append(case)
                print(f"[matmul-atc] W{bits}A16 {projection} START", flush=True)
                try:
                    spec = make_spec(bits, projection, device)
                    directory = root / f"w{bits}a16-{projection}"
                    air = export_air_bundle(lambda _: (spec,), {}, directory)
                    case["air_manifest"] = air["manifest_path"]
                    case["layout"] = air["graphs"][0]["runtime_input_abi"]["weight_quant_layout"]
                    result = compile_air_bundle(air["manifest_path"], atc_bin=atc, soc_version=soc,
                                               extra_args=["--precision_mode=must_keep_origin_dtype", "--deterministic=0"])
                    case.update(status="PASS", deployment_manifest=result["manifest_path"])
                except Exception as error:
                    case.update(status="FAIL", error=f"{type(error).__name__}: {error}")
                finally:
                    spec = None
                    torch._dynamo.reset(); gc.collect(); torch.npu.empty_cache()
                print(f"[matmul-atc] W{bits}A16 {projection} {case['status']}", flush=True)
                if case["status"] == "FAIL": print(case["error"], flush=True)
                atomic_write_json(root / "summary.json", report)
        report["status"] = "PASS" if all(c["status"] == "PASS" for c in report["cases"]) else "FAIL"
    except Exception as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        print(report["error"], flush=True)
    atomic_write_json(root / "summary.json", report)
    print(f"Report: {root / 'summary.json'}", flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
