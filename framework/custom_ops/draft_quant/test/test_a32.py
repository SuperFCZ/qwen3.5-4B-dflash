"""A3.2 variant identity and frozen gate/up regressions; no device execution."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from a2_common import CASE_NAMES, SHAPES, sha256, write_json
from a3_launch import CONFIG, PREFIX, configure_build, launch_evidence, load_build_config, pipeline_plan
from compare_a3 import compatible_runs
from compare_a32 import matched_variants
from test_a3 import launch_record


def launch(k, n, mode):
    plan = pipeline_plan(k, n, mode)
    return dict(launch_record(k, n, available=7), version=3, dequant_mode="batched", **plan,
                matmul_ub_bytes=256 * 1024 - plan["user_ub_bytes"])


def aggregate(mode):
    return {"bundle_sha256": "same-frozen-capture", "native_om_manifest_sha256": "same-OM", "device_id": 0,
            "stage": "A3.2", "timing_protocol": "continuous-v1",
            "build": {"abi": "dflash-group-quant-linear-build-v3", "core_limit": 0,
                      "dequant_mode": "batched", "pipeline_mode": mode, "source_sha256": {"kernel": "same-source"}},
            "launches": {"real": [{"name": name, "launch": launch(*SHAPES[name.split("-", 2)[2]][1:], mode)}
                                  for name in CASE_NAMES]}}


class A32Tests(unittest.TestCase):
    def test_pipeline_configuration_requires_matching_headers_and_batched_math(self):
        original = CONFIG.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            host, kernel, report = root / "host.h", root / "kernel.h", root / "build.json"
            for mode, number in (("serial", 0), ("prefetch", 1)):
                configure_build(host, report, 0, "batched", kernel, mode)
                self.assertEqual(host.read_bytes(), kernel.read_bytes())
                self.assertIn(f"PIPELINE_MODE {number}U", kernel.read_text())
                self.assertEqual(load_build_config(report)["pipeline_mode"], mode)
            self.assertEqual(CONFIG.read_bytes(), original)
            for mode, pipeline in (("legacy", "prefetch"), ("batched", "unknown")):
                with self.assertRaisesRegex(ValueError, "prefetch requires batched"):
                    configure_build(host, report, 0, mode, kernel, pipeline)
            kernel.write_text(kernel.read_text().replace("PIPELINE_MODE 1U", "PIPELINE_MODE 0U"))
            with self.assertRaisesRegex(ValueError, "host/kernel"):
                load_build_config(report)
            # Even consistent edited header hashes must agree with the mode.
            host.write_bytes(kernel.read_bytes())
            config = json.loads(report.read_text()); config["header_sha256"] = sha256(host)
            write_json(report, config)
            with self.assertRaisesRegex(ValueError, "pipeline mode"):
                load_build_config(report)

    def test_launch_policy_keeps_gate_up_and_whole_k_serial(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "runner.log"
            for k, n in ((256, 64), (256, 256), (512, 64), (1024, 256), (2560, 19456), (9728, 2560)):
                for mode in ("serial", "prefetch"):
                    item = launch(k, n, mode)
                    log.write_text(PREFIX + json.dumps(item))
                    evidence = launch_evidence(log, (16, k, n), 0, 2097152, "batched", mode)
                    enabled = mode == "prefetch" and k in (512, 1024, 9728)
                    self.assertEqual(evidence["raw_banks"], 2 if enabled else 1)
                    self.assertEqual(evidence["user_ub_bytes"], 33024 if enabled else (49408 if k == 256 else 24704))
                    self.assertEqual(evidence["selected_pipeline"], "raw-prefetch-v1" if enabled else "serial-v1")
            for key, value in (("raw_banks", 1), ("selected_pipeline", "serial-v1"), ("user_ub_bytes", 24704),
                               ("pipeline_mode", "serial"), ("matmul_ub_bytes", 1), ("matmul_ub_bytes", 0)):
                bad = dict(launch(9728, 2560, "prefetch"), **{key: value})
                log.write_text(PREFIX + json.dumps(bad))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    launch_evidence(log, (16, 9728, 2560), 0, 2097152, "batched", "prefetch")
            stale = dict(launch_record(9728, 2560), version=2, dequant_mode="batched")
            log.write_text(PREFIX + json.dumps(stale))
            for mode in ("serial", "prefetch"):
                with self.assertRaisesRegex(ValueError, "pipeline mode"):
                    launch_evidence(log, (16, 9728, 2560), 0, 2097152, "batched", mode)

    def test_comparison_only_changes_pipeline_with_all_gate_up_controls(self):
        before, after = aggregate("serial"), aggregate("prefetch")
        matched_variants(before, after)
        with self.assertRaisesRegex(ValueError, "same pipeline"):
            compatible_runs(before, after)
        for path, value in ((("build", "core_limit"), 1), (("build", "dequant_mode"), "legacy"),
                            (("build", "pipeline_mode"), "serial"), (("stage",), "A3.1"),
                            (("timing_protocol",), "checked-v1"), (("bundle_sha256",), "other-capture"),
                            (("build", "source_sha256", "kernel"), "other-source")):
            bad = copy.deepcopy(after)
            parent = bad
            for key in path[:-1]: parent = parent[key]
            parent[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                matched_variants(before, bad)
        for index, key, value in ((0, "raw_banks", 2), (0, "selected_pipeline", "raw-prefetch-v1"),
                                  (0, "matmul_ub_bytes", 200000), (1, "raw_banks", 1),
                                  (1, "block_dim", 6), (1, "matmul_ub_bytes", 200000)):
            bad = copy.deepcopy(after); bad["launches"]["real"][index]["launch"][key] = value
            with self.subTest(index=index, key=key), self.assertRaises(ValueError):
                matched_variants(before, bad)
        bad = copy.deepcopy(after); bad["launches"]["real"] = bad["launches"]["real"][1::2]
        with self.assertRaisesRegex(ValueError, "all five"):
            matched_variants(before, bad)

    def test_server_rejects_unsupported_mode_before_build(self):
        script = CONFIG.parent.parent / "run_server.sh"
        for mode, pipeline in (("legacy", "prefetch"), ("batched", "invalid"), ("legacy", "serial")):
            env = dict(os.environ, DFLASH_SUITE="a32", DFLASH_DEQUANT_MODE=mode, DFLASH_PIPELINE_MODE=pipeline)
            result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("batched" if mode == "legacy" else "serial or prefetch", result.stderr)


if __name__ == "__main__":
    unittest.main()
