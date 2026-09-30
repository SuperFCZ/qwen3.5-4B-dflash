"""Host-side A3 launch/configuration and evidence rejection tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from a3_launch import (CONFIG, POLICY, PREFIX, configure_build, core_limit,
                       launch_evidence, load_build_config)
from a2_common import CASE_NAMES, SHAPES, write_json
from compare_a3 import compatible_runs, timings
from run_a3 import collect_launches


def launch_record(k=2560, n=19456, available=8, limit=0):
    return {"version": 1, "policy": POLICY, "m": 16, "k": k, "n": n,
            "tile_n": 64, "tile_k": 256 if k == 256 else 128, "available_cores": available,
            "core_limit": limit, "block_dim": min(available, n // 64, limit or available),
            "n_tiles": n // 64, "system_workspace_bytes": 2097152}


class A3Tests(unittest.TestCase):
    def test_build_cap_is_recorded_without_changing_tracked_default(self):
        original = CONFIG.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header, report = root / "config.h", root / "build.json"
            for limit in (0, 1, 3):
                configure_build(header, report, limit)
                config = load_build_config(report)
                self.assertEqual(config["core_limit"], limit)
                self.assertIn(f"CORE_LIMIT {limit}U", header.read_text())
                self.assertIn("op_kernel/d_flash_group_quant_linear.cpp", config["source_sha256"])
            self.assertEqual(CONFIG.read_bytes(), original)
            header.write_text(header.read_text().replace("3U", "1U"))
            with self.assertRaisesRegex(ValueError, "header changed"):
                load_build_config(report)
            with self.assertRaises(ValueError):
                configure_build(CONFIG, report, 1)
        for value in (-1, 65536, "01", "auto", "1; exit", True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                core_limit(value)

    def test_launch_evidence_describes_balanced_complete_disjoint_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "runner.log"
            for k, n, available, limit in ((256, 64, 8, 0), (512, 128, 8, 0),
                                           (2560, 19456, 8, 0), (9728, 2560, 3, 0),
                                           (2560, 19456, 8, 1), (9728, 2560, 8, 3)):
                item = launch_record(k, n, available, limit)
                log.write_text("SDK diagnostic\n" + PREFIX + json.dumps(item) + "\n")
                result = launch_evidence(log, (16, k, n), limit, 2097152)
                counts, columns = [], []
                for owner in result["assignments"]:
                    count = owner["tile_count"]
                    counts.append(count)
                    owned = [owner["first_column"] + i * owner["tile_stride_columns"] for i in range(count)]
                    self.assertEqual(owned[-1], owner["last_tile_column"])
                    columns.extend(owned)
                self.assertEqual(sorted(columns), list(range(0, n, 64)))
                self.assertLessEqual(max(counts) - min(counts), 1)

    def test_stale_or_changed_launch_cannot_pass_as_multicore(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "runner.log"
            good = launch_record()
            log.write_text("ACLNN completed; guards/inputs intact\n")
            with self.assertRaisesRegex(ValueError, "missing A3"):
                launch_evidence(log, (16, 2560, 19456), 0, 2097152)
            for key, value in (("block_dim", 1), ("available_cores", 0), ("core_limit", 1),
                               ("tile_k", 256), ("n", 64), ("n_tiles", 40), ("block_dim", "8"),
                               ("system_workspace_bytes", 2097153)):
                broken = dict(good, **{key: value})
                log.write_text(PREFIX + json.dumps(broken) + "\n")
                with self.subTest(key=key), self.assertRaises(ValueError):
                    launch_evidence(log, (16, 2560, 19456), 0, 2097152)
            log.write_text(PREFIX + json.dumps(good) + "\n" + PREFIX + json.dumps(dict(good, block_dim=4)))
            with self.assertRaisesRegex(ValueError, "plan changed"):
                launch_evidence(log, (16, 2560, 19456), 0, 2097152)

    def test_real_multicore_gate_rejects_single_core_device_and_allows_explicit_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a1, real = root / "a1", root / "real"
            a1.mkdir(); real.mkdir()
            cases = [{"name": f"c{i}"} for i in range(81)]
            workload = a1 / "workload"
            workload.mkdir()
            for case in cases:
                dest = workload / case["name"]
                dest.mkdir()
                write_json(dest / "execution.json", {"workspace_bytes": 2097152})
            write_json(a1 / "suite.json", {"status": "PASS", "workloads": [
                {"name": "workload", "mkn": [16, 256, 64], "custom_cases": cases}]})
            executions = []
            for name in CASE_NAMES:
                kind = name.split("-", 2)[2]
                m, k, n = SHAPES[kind]
                item = {"m": m, "k": k, "n": n, "workspace_bytes": 2097152, "timing": {}}
                executions.append({"name": name, "executions": {"custom": item, "native_om": item}})
            write_json(real / "suite.json", {"status": "PASS", "cases": executions})
            with patch("run_a3.launch_evidence", return_value={"block_dim": 1}):
                with self.assertRaisesRegex(ValueError, "at least two cores"):
                    collect_launches(a1, real, 0)
                evidence, _ = collect_launches(a1, real, 1)
                self.assertEqual(len(evidence["real"]), 10)
            with patch("run_a3.launch_evidence", return_value={"block_dim": 8}):
                evidence, _ = collect_launches(a1, real, 0)
                self.assertEqual(len(evidence["a1"]), 81)
            write_json(real / "suite.json", {"status": "FAIL", "cases": executions})
            with self.assertRaisesRegex(ValueError, "numerical gates"):
                collect_launches(a1, real, 0)

    def test_timing_comparison_requires_same_inputs_and_code(self):
        single = {"bundle_sha256": "same-capture", "native_om_manifest_sha256": "same-native",
                  "device_id": 0, "build": {"source_sha256": {"kernel": "same-code"}, "core_limit": 1}}
        multi = copy.deepcopy(single)
        multi["build"]["core_limit"] = 0
        compatible_runs(single, multi)
        for key, value in (("bundle_sha256", "other-capture"), ("native_om_manifest_sha256", "other-OM"), ("device_id", 1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                compatible_runs(single, dict(multi, **{key: value}))
        multi["build"]["source_sha256"]["kernel"] = "other-code"
        with self.assertRaises(ValueError):
            compatible_runs(single, multi)
        timing = {"status": "MEASURED", "warmup": 3, "repetitions": 10,
                  "execute_sync": {"samples_ms": list(range(1, 11))}}
        measured = timings({"timing": timing})
        self.assertEqual(measured["median_ms"], 5.5)
        self.assertEqual(measured["p95_ms"], 10)
        timing["execute_sync"]["samples_ms"].pop()
        with self.assertRaises(ValueError):
            timings({"timing": timing})


if __name__ == "__main__":
    unittest.main()
