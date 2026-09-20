#!/usr/bin/env python3
"""Compile synthetic W4/W8 linears to isolate CANN template support.

No checkpoint, Target model or C++ runner is loaded. This tests ATC graph
compatibility, not OM execution, accuracy or performance.
Group 0 is a synthetic per-channel control, never a checkpoint conversion.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import itertools
from pathlib import Path
import sys
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "framework/python"), str(REPO)]

import torch

from models.dflash_v1.draft_quantization import pack_device_weight
from models.dflash_v1.weight_quant_matmul import TORCH_OP, GE_OP, require_weight_quant_matmul
from qwen35_dflash.ascend310p.contracts import AirGraphSpec, CustomOpExportSpec
from qwen35_dflash.ascend310p.compiler import compile_air_bundle, resolve_atc_executable, validate_soc_version
from qwen35_dflash.ascend310p.exporter import export_air_bundle
from qwen35_dflash.ascend310p.utils import atomic_write_json, require_run_output

# K,N. X's M axis is dynamic with the same 16/64 ATC gears as Draft features.
SHAPES = {"tiny": (256, 64), "gate_up": (2560, 19456), "q": (2560, 4096),
          "kv": (2560, 2048), "down": (9728, 2560), "fc": (12800, 2560)}


class ProbeLinear(torch.nn.Module):
    def __init__(self, n, k, bits, group_size, weight_layout):
        super().__init__()
        self.n, self.k, self.bits = n, k, bits
        self.group_size, self.weight_layout = group_size, weight_layout
        self.groups = k // group_size if group_size else 1

    def forward(self, x, qweight, scales):
        # Inputs stay live. KN controls are prepacked in that physical order;
        # they do not depend on ATC folding a Transpose/contiguous pair.
        # This W4 unpack is the same exact byte arithmetic as GroupQuantLinear.
        if self.bits == 4:
            values = qweight.to(torch.float16)
            high = torch.floor(values * (1.0 / 16.0))
            low = (values - high * 16.0 - 8.0).to(torch.int8)
            high = (high - 8.0).to(torch.int8)
            qweight = torch.stack((low, high), dim=-1)
        weight = (qweight.reshape(self.n, self.k).t() if self.weight_layout == "nk"
                  else qweight.reshape(self.k, self.n))
        # CANN 9.0.0 on the receiver rejects [1,N] in per-channel mode.
        # [N] is unambiguous in both the Python API and the GE tiler.
        scale = (scales.reshape(self.n) if self.group_size == 0
                 else scales.reshape(self.n, self.groups).t())
        return torch.ops.npu.npu_weight_quant_batchmatmul.default(
            x, weight, scale,
            antiquant_group_size=self.group_size, inner_precise=0,
        )


class ConstantProbeLinear(torch.nn.Module):
    def __init__(self, weight, scales):
        super().__init__()
        self.register_buffer("qweight", weight)
        self.register_buffer("scales", scales)

    def forward(self, x):
        from models.dflash_v1.weight_quant_matmul import weight_quant_linear
        return weight_quant_linear(x, self.qweight, self.scales)


def make_spec(bits, projection, device, group_size=128, weight_layout="nk", weight_format="nz", *,
              prepack_dir=None, static_rows=None, immutable_nd=False):
    if (group_size not in (0, 128) or weight_layout not in ("nk", "kn")
            or weight_format not in ("nd", "nz") or (weight_format == "nz" and weight_layout != "nk")):
        raise ValueError("probe controls require group_size=0/128, weight_format=nd/nz, "
                         "weight_layout=nk/kn; NZ requires NK on 310P")
    if static_rows not in (None, 16, 64):
        raise ValueError("static probe rows must be 16 or 64")
    if immutable_nd and prepack_dir is not None:
        raise ValueError("the ND constant control cannot also replace the weight with NZ Const")
    k, n = SHAPES[projection]
    generator = torch.Generator().manual_seed(812 + bits)
    q = torch.randint(-(1 << (bits - 1)), 1 << (bits - 1), (n, k),
                      generator=generator, dtype=torch.int8)
    packed = pack_device_weight(q if weight_layout == "nk" else q.t().contiguous(), bits).to(device)
    groups = k // group_size if group_size else 1
    scales = ((1 + torch.arange(n * groups).reshape(n, groups) % 4).half() / 32).to(device)
    x = (torch.randn(static_rows or 16, k, generator=generator).half() / 16).to(device)
    model = ProbeLinear(n, k, bits, group_size, weight_layout).eval()
    args = (x, packed.view(-1), scales.view(-1))
    names = ("x", "qweight", "scales")
    signature = [{"name": name, "shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch.")}
                 for name, t in zip(names, args)]
    spec = AirGraphSpec(name="weight_quant_probe", role="diagnostic", model=model, example_args=args,
                        input_names=names, output_names=("y",), dynamic=static_rows is None,
                        custom_ops=(CustomOpExportSpec(TORCH_OP, GE_OP),),
                        metadata={"tensor_abi": {"inputs": signature}, "dynamic_input_axes": {"x": [0]},
                                  "bits": bits, "projection": projection, "group_size": group_size,
                                  "synthetic_control": group_size == 0,
                                  "weight_quant_probe": {"group_size": group_size, "weight_layout": weight_layout,
                                                         "weight_format": weight_format}})
    if static_rows is not None:
        spec = replace(spec, metadata={key: value for key, value in spec.metadata.items()
                                       if key != "dynamic_input_axes"})
    if immutable_nd:
        if (bits, group_size, weight_layout, weight_format) != (8, 128, "nk", "nz"):
            raise ValueError("ND constant control requires W8/group128/NK/NZ")
        # Same immutable q/scales, native math and only-x ABI as the prepack
        # case, but let the existing TransData remain in AIR. This diagnoses
        # the Const boundary; it does not promise ATC will fold TransData.
        metadata = {key: value for key, value in spec.metadata.items() if key != "weight_quant_probe"}
        metadata.update(tensor_abi={"inputs": signature[:1]}, synthetic_nd_constant=True)
        spec = replace(spec, model=ConstantProbeLinear(packed, scales), example_args=(x,),
                       input_names=("x",), metadata=metadata)
    if prepack_dir is not None:
        if (bits, group_size, weight_layout, weight_format) != (8, 128, "nk", "nz"):
            raise ValueError("offline constant probe requires W8/group128/NK/NZ")
        import hashlib
        from qwen35_dflash.ascend310p.weight_prepack import PREPACK_POLICY, pack_int8_nz
        from qwen35_dflash.ascend310p.utils import file_record
        directory = require_run_output(prepack_dir)
        directory.mkdir(parents=True, exist_ok=False)
        carrier = pack_int8_nz(q)
        path = directory / "weight.nz.bin"
        carrier.numpy().tofile(path)
        record = {**file_record(path, relative_to=directory), "dtype": "int8", "format": "FRACTAL_NZ",
                  "logical_shape": list(q.shape), "storage_shape": list(carrier.shape),
                  "logical_sha256": hashlib.sha256(q.numpy().tobytes()).hexdigest()}
        cache = atomic_write_json(directory / "manifest.json", {"schema_version": 1, "policy": PREPACK_POLICY,
            "status": "PASS", "weight_count": 1, "weights": [record], "scope": "synthetic offline probe"})
        # Use the production fixed group/layout policy. Only x is public; the
        # weight uses the same immutable binding and AIR save hook as Draft.
        metadata = {k: v for k, v in spec.metadata.items() if k != "weight_quant_probe"}
        metadata.update(draft_weight_storage=PREPACK_POLICY, draft_quantization="w8a16",
                        draft_weight_prepack_manifest=str(cache),
                        tensor_abi={"inputs": signature[:1]}, synthetic_prepack=True)
        spec = replace(spec, model=ConstantProbeLinear(packed, scales), example_args=(x,),
                       input_names=("x",), metadata=metadata)
    return spec


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-dir", type=Path, required=True)
    cli.add_argument("--atc", required=True)
    cli.add_argument("--soc-version", required=True)
    cli.add_argument("--device-id", type=int, default=0)
    cli.add_argument("--bits", nargs="+", choices=(4, 8), type=int, default=[4, 8])
    cli.add_argument("--projection", nargs="+", choices=tuple(SHAPES), default=["tiny"])
    cli.add_argument("--group-size", nargs="+", choices=(0, 128), type=int, default=[128],
                     help="128: checkpoint grouping; 0: synthetic per-channel support control only")
    cli.add_argument("--weight-layout", nargs="+", choices=("nk", "kn"), default=["nk"],
                     help="physical weight axes: NK with transpose_weight=true, or KN with false")
    cli.add_argument("--weight-format", choices=("nz", "nd"), default="nz",
                     help="nz: 310P TransData + WeightNz path; nd: explicit negative/control path")
    cli.add_argument("--prepack-weights", action="store_true",
                     help="test offline W8 NZ constants; requires --bits 8 --group-size 128 --weight-layout nk --weight-format nz")
    cli.add_argument("--diagnose-prepack", action="store_true",
                     help="tiny-only: capture ATC shapes for prepacked dynamic, prepacked static M16, "
                          "and ND-Const + TransData dynamic controls; requires --prepack-weights")
    return cli


def main(argv=None):
    cli = parser(); args = cli.parse_args(argv)
    if args.weight_format == "nz" and "kn" in args.weight_layout:
        cli.error("310P NZ requires --weight-layout nk; use --weight-format nd for KN controls")
    if args.prepack_weights and (args.bits != [8] or args.group_size != [128]
                                or args.weight_layout != ["nk"] or args.weight_format != "nz"):
        cli.error("--prepack-weights requires --bits 8 --group-size 128 --weight-layout nk --weight-format nz")
    if args.diagnose_prepack and (not args.prepack_weights or args.projection != ["tiny"]):
        cli.error("--diagnose-prepack requires --prepack-weights --bits 8 --projection tiny")
    root = require_run_output(args.output_dir)
    if root.exists():
        cli.error("use a new output directory; existing probe evidence is retained")
    atc = resolve_atc_executable(args.atc)
    soc = validate_soc_version(args.soc_version)
    root.mkdir(parents=True)
    report = {"schema_version": 1, "status": "RUNNING", "cpu_fallback": False,
              "scope": "synthetic AIR and ATC compilation only; no OM execution or speed claim",
              "soc_version": soc, "execution_status": "NOT_RUN", "cases": []}
    try:
        import torch_npu
        import torchair
        device = f"npu:{args.device_id}"
        torch.npu.set_device(device)
        require_weight_quant_matmul()
        report["environment"] = {"torch": str(torch.__version__), "torch_npu": str(torch_npu.__version__),
                                 "device": device, "device_name": torch.npu.get_device_name(args.device_id)}
        controls = ("prepacked-dynamic", "prepacked-static16", "ndconst-dynamic") if args.diagnose_prepack else ("default",)
        combinations = itertools.product(dict.fromkeys(args.projection), dict.fromkeys(args.bits),
                                         dict.fromkeys(args.group_size), dict.fromkeys(args.weight_layout), controls)
        for projection, bits, group_size, layout, control in combinations:
            prepacked = args.prepack_weights and control != "ndconst-dynamic"
            case = {"bits": bits, "projection": projection, "group_size": group_size,
                    "weight_layout": layout, "weight_format": args.weight_format,
                    "offline_weight_prepack": prepacked, "control": control,
                    "synthetic_control": group_size == 0, "execution_status": "NOT_RUN",
                    "status": "RUNNING", "phase": "prepare"}
            report["cases"].append(case)
            name = f"w{bits}a16-{projection}-g{group_size}-{layout}-{args.weight_format}"
            if control != "default":
                name += "-" + control
            case["name"] = name
            print(f"[matmul-atc] {name} START", flush=True)
            diagnostics = None
            try:
                options = {"prepack_dir": root / (name + "-offline")} if prepacked else {}
                if control == "prepacked-static16":
                    options["static_rows"] = 16
                if control == "ndconst-dynamic":
                    options["immutable_nd"] = True
                spec = make_spec(bits, projection, device, group_size, layout, args.weight_format, **options)
                directory = root / name
                case["phase"] = "export"
                air = export_air_bundle(lambda _: (spec,), {}, directory)
                case["air_manifest"] = air["manifest_path"]
                case["layout"] = air["graphs"][0]["runtime_input_abi"]["weight_quant_layout"]
                case["phase"] = "compile"
                compile_options = {}
                if args.diagnose_prepack:
                    from qwen35_dflash.ascend310p.atc_diagnostics import AtcShapeDiagnostics
                    diagnostics = AtcShapeDiagnostics(root / (name + "-diagnostics"))
                    compile_options["runner"] = diagnostics
                result = compile_air_bundle(air["manifest_path"], atc_bin=atc, soc_version=soc,
                                           extra_args=["--precision_mode=must_keep_origin_dtype", "--deterministic=0"],
                                           **compile_options)
                case.update(status="PASS", phase="complete", deployment_manifest=result["manifest_path"])
            except Exception as error:
                trace = root / f"{name}-error.txt"
                trace.write_text(traceback.format_exc(), encoding="utf-8")
                case.update(status="FAIL", error=f"{type(error).__name__}: {error}", traceback=str(trace))
            finally:
                if diagnostics is not None:
                    case["shape_diagnostics"] = diagnostics.collect()
                    case["shape_diagnostics_path"] = str(diagnostics.report_path)
                spec = None
                torch._dynamo.reset(); gc.collect(); torch.npu.empty_cache()
            print(f"[matmul-atc] {name} {case['status']} phase={case['phase']}", flush=True)
            if case["status"] == "FAIL": print(case["error"], flush=True)
            atomic_write_json(root / "summary.json", report)
        report["status"] = "PASS" if all(c["status"] == "PASS" for c in report["cases"]) else "FAIL"
    except Exception as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        print(report["error"], flush=True)
    atomic_write_json(root / "summary.json", report)
    print("\n| Bits | Projection | Group size | Weight layout | Weight format | Status | Phase |")
    print("|---:|---|---:|---|---|---|---|")
    for case in report["cases"]:
        print(f"| {case['bits']} | {case['projection']} | {case['group_size']} | "
              f"{case['weight_layout'].upper()} | {case['weight_format'].upper()} | "
              f"{case['status']} | {case['phase']} |")
    print("Group 0 is a synthetic per-channel control; checkpoint grouping is unchanged.")
    print("PASS means AIR/ATC compilation only. OM execution, numerical parity and latency are NOT_RUN.")
    if args.diagnose_prepack:
        from qwen35_dflash.ascend310p.atc_diagnostics import diagnostic_summary
        text = "\n\n".join(diagnostic_summary(case) for case in report["cases"])
        text += ("\n\nControls isolate immutable Const representation and dynamic-gear expansion. "
                 "A passing ND constant control does not prove compile-time folding or faster OM execution.\n")
        (root / "diagnostics.txt").write_text(text, encoding="utf-8")
        print(text)
        print(f"Compact diagnostics: {root / 'diagnostics.txt'}")
    print(f"Report: {root / 'summary.json'}", flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
