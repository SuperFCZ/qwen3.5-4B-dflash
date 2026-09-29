#!/usr/bin/env python3
"""Export/compile ten static native WeightQuant OMs from an A2 real-data bundle."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys

from a2_common import (ABI, REPO, case_shape, checked_file, load_bundle, record, sha256,
                       validate_layout, validate_native_graph, write_json)

sys.path[:0] = [str(REPO / "framework/python"), str(REPO)]


def export(args):
    if os.environ.get("ASCEND310P_SIMULATION_ONLY") == "1":
        raise ValueError("A2 export requires the real CANN/TorchAir environment")
    import numpy as np
    import torch
    import torch_npu
    from models.dflash_v1.weight_quant_matmul import TORCH_OP, GE_OP, require_weight_quant_matmul
    from qwen35_dflash.ascend310p.contracts import AirGraphSpec, CustomOpExportSpec
    from qwen35_dflash.ascend310p.compiler import compile_air_bundle, resolve_atc_executable
    from qwen35_dflash.ascend310p.exporter import export_air_bundle
    from qwen35_dflash.ascend310p.utils import require_run_output
    from qwen35_dflash.ascend310p.weight_prepack import PREPACK_POLICY

    bundle_path = args.bundle.resolve()
    bundle = load_bundle(bundle_path)
    root = require_run_output(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    device = f"npu:{args.device_id}"
    torch.npu.set_device(device)
    if "310P" not in torch.npu.get_device_name(args.device_id).upper():
        raise ValueError("A2 native export requires Ascend 310P")
    require_weight_quant_matmul()
    atc = resolve_atc_executable(args.atc)
    report = {"abi": ABI, "status": "RUNNING", "bundle_sha256": sha256(bundle_path),
              "bundle": str(bundle_path), "soc_version": "Ascend310P3", "cases": [],
              "scope": "static native WeightQuant OMs with identical real offline-NZ weights/scales",
              "native_om_execution": "NOT_RUN", "full_draft_validation": "NOT_RUN",
              "environment": {"torch": str(torch.__version__), "torch_npu": str(torch_npu.__version__)}}
    report_path = root / "native-om.json"
    write_json(report_path, report)

    class NativeLinear(torch.nn.Module):
        def __init__(self, q, scales):
            super().__init__()
            self.register_buffer("qweight", q)
            self.register_buffer("scales", scales)

        def forward(self, x):
            from models.dflash_v1.weight_quant_matmul import weight_quant_linear
            return weight_quant_linear(x, self.qweight, self.scales)

    for case in bundle["cases"]:
        entry = {"name": case["name"], "status": "RUNNING", "phase": "prepare",
                 "input_hashes": {name: item["sha256"] for name, item in case["files"].items()}}
        report["cases"].append(entry)
        model = x = q = scales = spec = None
        try:
            print(f"A2 native export: {case['name']}", flush=True)
            validate_layout(bundle_path.parent, case)
            m, k, n = case_shape(case)
            directory = root / case["name"]
            cache = directory / "prepack"
            cache.mkdir(parents=True)
            nz = cache / "weight.nz.bin"
            shutil.copyfile(checked_file(bundle_path.parent, case["files"]["w_nz.bin"]), nz)
            weight_record = dict(record(nz, cache), dtype="int8", format="FRACTAL_NZ",
                                 logical_shape=[n, k], storage_shape=[k // 32, n // 16, 16, 32],
                                 logical_sha256=case["files"]["q_nk.bin"]["sha256"])
            write_json(cache / "manifest.json", {"schema_version": 1, "policy": PREPACK_POLICY,
                       "status": "PASS", "weight_count": 1, "weights": [weight_record]})

            def tensor(name, dtype, shape):
                path = checked_file(bundle_path.parent, case["files"][name])
                return torch.from_numpy(np.fromfile(path, dtype=dtype).reshape(shape).copy())

            q = tensor("q_nk.bin", "i1", (n, k)).to(device)
            # Restore NG contiguous storage before the production GN view.
            scales = tensor("s_gn.bin", "<f2", (k // 128, n)).t().contiguous().to(device)
            x = tensor("x.bin", "<f2", (m, k)).to(device)
            model = NativeLinear(q, scales).eval()
            with torch.inference_mode():
                actual = model(x).cpu().contiguous().numpy().tobytes()
            if actual != checked_file(bundle_path.parent, case["files"]["eager-0.bin"]).read_bytes():
                raise ValueError("standalone native eager differs from captured projection")
            spec = AirGraphSpec(name="weight_quant_reference", role="diagnostic", model=model,
                                example_args=(x,), input_names=("x",), output_names=("y",), dynamic=False,
                                custom_ops=(CustomOpExportSpec(TORCH_OP, GE_OP),), metadata={
                                    "draft_weight_storage": PREPACK_POLICY, "draft_quantization": "w8a16",
                                    "draft_weight_prepack_manifest": str(cache / "manifest.json"),
                                    "tensor_abi": {"inputs": [{"name": "x", "dtype": "float16", "shape": [m, k]}]},
                                    "a2_case": case["name"], "a2_bundle_sha256": report["bundle_sha256"],
                                    "synthetic_prepack": False})
            entry["phase"] = "export"
            air = export_air_bundle(lambda _: (spec,), {}, directory / "artifacts")
            validate_native_graph(air["graphs"][0], case, report["bundle_sha256"])
            entry["phase"] = "atc"
            compiled = compile_air_bundle(air["manifest_path"], atc_bin=atc, soc_version="Ascend310P3",
                                          extra_args=["--precision_mode=must_keep_origin_dtype", "--deterministic=0"])
            graph = compiled["graphs"][0]
            validate_native_graph(graph, case, report["bundle_sha256"])
            om_path = checked_file(Path(compiled["manifest_path"]).parent, graph["om"])
            entry.update(status="PASS", phase="complete", om=record(om_path, root),
                         deployment=record(Path(compiled["manifest_path"]), root),
                         air_manifest=record(Path(air["manifest_path"]), root),
                         atc_command=graph["atc_command"], offline_weight=weight_record)
        except Exception as error:
            entry.update(status="FAIL", error=f"{type(error).__name__}: {error}")
            print(f"FAIL {case['name']}: {entry['error']}", flush=True)
        finally:
            model = x = q = scales = spec = None
            torch._dynamo.reset()
            gc.collect()
            torch.npu.empty_cache()
            write_json(report_path, report)
    # Reject a changing data bundle, even if every compiler returned success.
    load_bundle(bundle_path)
    report["status"] = ("PASS" if sha256(bundle_path) == report["bundle_sha256"] and
                         all(c["status"] == "PASS" for c in report["cases"]) else "FAIL")
    write_json(report_path, report)
    print(f"{report['status']}: native OM compilation; execution NOT_RUN; {report_path}", flush=True)
    return 0 if report["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory below AI_RUN_DIR")
    parser.add_argument("--atc", default="atc")
    parser.add_argument("--device-id", type=int, default=0)
    return export(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
