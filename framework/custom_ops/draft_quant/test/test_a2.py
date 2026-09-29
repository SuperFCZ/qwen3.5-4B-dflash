"""CPU-only tests of A2 evidence gates; no CANN or NPU acceptance claims."""
import copy
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

from a2_common import (ABI, CASE_NAMES, REPO, SHAPES, case_shape, checked_file,
                       load_bundle, record, replay_context_rows, snapshot_contract, validate_execution)
from run_a2 import compare_outputs, validate_om_manifest


class EvidenceTests(unittest.TestCase):
    def test_case_contract_rejects_cartesian_product_and_wrong_identity(self):
        case = dict(name="layer-4-down", layer=4, projection="down", m=16, k=9728, n=2560,
                    group_size=128, layout="nz_int8_v1")
        self.assertEqual(case_shape(case), SHAPES["down"])
        for key, value in (("m", 64), ("k", 2560), ("n", 19456), ("layer", 5),
                           ("group_size", 64), ("projection", "q"), ("name", "layer-3-down")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                case_shape(dict(case, **{key: value}))

    def test_files_are_bound_to_contents_size_and_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "x.bin"
            path.write_bytes(b"\x00\x3c")
            item = record(path, root)
            self.assertEqual(checked_file(root, item, 2), path.resolve())
            with self.assertRaises(ValueError):
                checked_file(root, item, 4)
            path.write_bytes(b"\x01\x3c")
            with self.assertRaises(ValueError):
                checked_file(root, item, 2)
            sub = root / "sub"
            sub.mkdir()
            (sub / "escape.bin").symlink_to(path)
            with self.assertRaises(ValueError):
                checked_file(sub, dict(item, path="escape.bin"))

    def test_native_om_gate_cannot_be_replaced_by_eager_or_cpu_agreement(self):
        zero = struct.pack("<6H", *([0] * 6))
        changed = struct.pack("<6H", 1, 0, 0, 0, 0, 0)
        shape = (2, 128, 3)
        self.assertEqual(compare_outputs([zero, zero], [changed, changed], zero, shape)["status"], "FAIL")
        self.assertEqual(compare_outputs([zero, zero], [zero, zero], changed, shape)["status"], "PASS")
        self.assertEqual(compare_outputs([zero, changed], [zero, zero], zero, shape)["status"], "FAIL")
        self.assertEqual(compare_outputs([zero, zero], [zero, changed], zero, shape)["status"], "FAIL")
        signed_zero = struct.pack("<6H", 0x8000, 0, 0, 0, 0, 0)
        self.assertEqual(compare_outputs([signed_zero] * 2, [zero] * 2, zero, shape)["status"], "FAIL")
        nonfinite = struct.pack("<6H", 0x7C00, 0, 0, 0, 0, 0)
        self.assertEqual(compare_outputs([nonfinite] * 2, [nonfinite] * 2, zero, shape)["status"], "FAIL")

    def test_snapshot_uses_recorded_shape_with_optional_static_gear(self):
        prefix = "qwen35-draft-replay-inputs-v1\n" + "a" * 64 + "\n256 0 15\n"
        tensors = "123 456 \nfeatures float16 81920 1 16 2560\nanchor int64 8 1\n"
        for gear in ("", "static64 " + "b" * 64 + "\n"):
            digest, specs = snapshot_contract(prefix + gear + tensors)
            self.assertEqual(digest, "a" * 64)
            self.assertEqual(specs["features"]["shape"], [1, 16, 2560])
            with self.assertRaises(ValueError):
                snapshot_contract(prefix + gear + tensors.replace("81920", "81922"))
            with self.assertRaises(ValueError):
                snapshot_contract(prefix + gear + tensors + "anchor int64 8 1\n")

    def test_replay_selects_active_gear_from_larger_snapshot_storage(self):
        self.assertEqual(replay_context_rows(1, 64, 16), 16)
        self.assertEqual(replay_context_rows(16, 64, 16), 16)
        self.assertEqual(replay_context_rows(17, 64, 64), 64)
        self.assertEqual(replay_context_rows(64, 64, 64), 64)
        for values in ((16, 64, 64), (17, 16, 64), (0, 64, 16), (65, 64, 64), (16, 32, 16)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replay_context_rows(*values)

    def test_execution_rejects_stale_shape_device_and_incomplete_measurements(self):
        report = dict(status="PASS", runtime="AscendCL native OM", cpu_fallback=False,
                      input_readonly=True, guards_intact=True, repetitions=2, io_validated=True,
                      m=16, k=9728, n=2560, device_id=0, workspace_bytes=4096,
                      tracked_device_allocation_bytes=8192,
                      timing=dict(status="MEASURED", warmup=3, repetitions=10,
                                  execute_sync={"samples_ms": [1.0] * 10}))
        validate_execution(report, SHAPES["down"], 0, "AscendCL native OM", 3, 10)
        for key, value in (("device_id", 1), ("n", 64), ("io_validated", False),
                           ("status", "RUNNING"), ("guards_intact", False), ("cpu_fallback", True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_execution(dict(report, **{key: value}), SHAPES["down"], 0, "AscendCL native OM", 3, 10)
        broken = copy.deepcopy(report)
        broken["timing"]["execute_sync"]["samples_ms"].pop()
        with self.assertRaises(ValueError):
            validate_execution(broken, SHAPES["down"], 0, "AscendCL native OM", 3, 10)

    def test_partial_or_synthetic_capture_cannot_pass_as_a2(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            for bundle in ({"abi": ABI, "status": "PASS", "cases": []},
                           {"abi": ABI, "status": "PASS", "capture_runtime": "synthetic"}):
                path.write_text(json.dumps(bundle))
                with self.assertRaises(ValueError):
                    load_bundle(path)

    def test_om_bound_to_all_input_hashes_and_exported_const(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            bundle_path = root / "manifest.json"
            bundle_path.write_text("{}")
            from a2_common import sha256
            bundle_hash = sha256(bundle_path)
            source = {"files": {"x.bin": {"sha256": "x"}, "w_nz.bin": {"sha256": "nz"},
                                "q_nk.bin": {"sha256": "nk"}}}
            cases = []
            for layer in range(5):
                for kind, (m, k, n) in SHAPES.items():
                    cases.append(dict(source, name=f"layer-{layer}-{kind}", layer=layer,
                                      projection=kind, m=m, k=k, n=n, group_size=128, layout="nz_int8_v1"))
            bundle = {"cases": cases}
            report = {"abi": ABI, "status": "PASS", "soc_version": "Ascend310P3",
                      "bundle_sha256": bundle_hash, "cases": []}
            om, air = root / "test-only.om", root / "air.json"
            om.write_bytes(b"CPU test artifact; never executed as an OM")
            air.write_text("{}")
            for case in cases:
                m, k, n = case_shape(case)
                graph = {"name": "weight_quant_reference", "input_names": ["x"], "output_names": ["y"],
                         "om": record(om, root), "metadata": {"a2_case": case["name"],
                             "a2_bundle_sha256": bundle_hash,
                             "tensor_abi": {"inputs": [{"name": "x", "dtype": "float16", "shape": [m, k]}]}},
                         "runtime_input_abi": {"weight_quant_layout": {"status": "PASS", "node_count": 1,
                             "inserted_transdata": [], "prepack": {"status": "PASS", "node_count": 1,
                                 "constants": [{"logical_sha256": "nk", "storage_sha256": "nz",
                                                "logical_shape": [n, k], "storage_shape": [k // 32, n // 16, 16, 32],
                                                "roundtrip": "BIT_EXACT"}]}}}}
                deployment = root / (case["name"] + ".json")
                deployment.write_text(json.dumps({"status": "PASS", "graphs": [graph]}))
                report["cases"].append({"name": case["name"], "status": "PASS",
                    "input_hashes": {"x.bin": "x", "w_nz.bin": "nz", "q_nk.bin": "nk"},
                    "om": record(om, root), "deployment": record(deployment, root), "air_manifest": record(air, root),
                    "offline_weight": {"sha256": "nz", "logical_sha256": "nk"}})
            path = root / "native-om.json"
            path.write_text(json.dumps(report))
            validate_om_manifest(path, bundle_path, bundle)
            report["cases"][4]["input_hashes"]["x.bin"] = "different-real-activation"
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                validate_om_manifest(path, bundle_path, bundle)
            report["cases"][4]["input_hashes"]["x.bin"] = "x"
            # Even a consistently re-hashed deployment must carry the right Const.
            graph["runtime_input_abi"]["weight_quant_layout"]["prepack"]["constants"][0]["storage_sha256"] = "other-weight"
            deployment.write_text(json.dumps({"status": "PASS", "graphs": [graph]}))
            report["cases"][-1]["deployment"] = record(deployment, root)
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                validate_om_manifest(path, bundle_path, bundle)


class RunnerSyntaxTests(unittest.TestCase):
    def test_runner_cpp_syntax_with_explicit_cpu_declaration_stubs(self):
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        here = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # These declarations only catch C++ syntax/refactoring errors.
            # They do NOT establish CANN header/ABI compatibility.
            (root / "extras.h").write_text('''#include <cstddef>
extern "C" const char *aclGetRecentErrMsg();
extern "C" int aclrtMemset(void *, std::size_t, int, std::size_t);
''')
            (root / "aclnn_d_flash_group_quant_linear.h").write_text('''
#pragma once
#include "acl/acl.h"
struct aclTensor; struct aclOpExecutor;
enum aclFormat { ACL_FORMAT_ND, ACL_FORMAT_FRACTAL_NZ };
aclTensor *aclCreateTensor(const int64_t *, uint64_t, aclDataType, const int64_t *, int64_t,
                           aclFormat, const int64_t *, uint64_t, void *);
aclError aclDestroyTensor(aclTensor *);
aclError aclnnDFlashGroupQuantLinearGetWorkspaceSize(const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
aclError aclnnDFlashGroupQuantLinear(void *, uint64_t, aclOpExecutor *, aclrtStream);
''')
            for source in ("main.cpp", "native_om.cpp"):
                result = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-fsyntax-only",
                                         "-I", str(root), "-I", str(REPO / "framework/runtime/cpp/tests/fake_acl"),
                                         "-include", str(root / "extras.h"), str(here / source)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
