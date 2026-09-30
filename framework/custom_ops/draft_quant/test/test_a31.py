"""A3.1 host regressions. CPU callbacks/models do not execute a NPU."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from a2_common import REPO, record, sha256, validate_execution, write_json
from a3_launch import configure_build, launch_evidence, load_build_config, PREFIX
from compare_a3 import compatible_runs
from compare_a31 import matched_variants
from compare_a31 import run as compare_variants
from profile_a31 import prepare as prepare_profile
from test_a3 import launch_record


class A31Tests(unittest.TestCase):
    def test_comparator_checks_tail_bytes_even_with_consistent_hashes(self):
        # Small tensor fixture exercises the comparator, not workload/NPU acceptance.
        baseline = {"bundle_sha256": "same", "native_om_manifest_sha256": "same", "device_id": 0,
                    "timing_protocol": "continuous-v1", "build": {"core_limit": 0, "dequant_mode": "legacy",
                    "source_sha256": {"kernel": "same"}}, "launches": {"real": [{"launch": dict(launch_record(512, 128), dequant_mode="legacy")}]}}
        candidate = copy.deepcopy(baseline)
        candidate["build"]["dequant_mode"] = "batched"
        candidate["launches"]["real"][0]["launch"]["dequant_mode"] = "batched"
        timing = dict(status="MEASURED", protocol="continuous-v1", warmup=3, repetitions=10,
                      execute_sync={"samples_ms": [1.] * 10}, prepare={"samples_ms": [.01] * 10},
                      per_timed_call_readback=False, per_timed_call_poison=False,
                      correctness_before_calls=2, correctness_after_calls=1, timed_tail_checked=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            built = []
            for name in ("before", "after"):
                dest = root / name; dest.mkdir()
                summary = dest / "suite.json"; summary.write_text("CPU fixture only")
                case = {"name": "test-projection", "status": "PASS", "executions": {}, "outputs": {}, "bracket_outputs": {}}
                for kind in ("custom", "native_om"):
                    case["executions"][kind] = dict(status="PASS", runtime="AscendCL ACLNN" if kind == "custom" else "AscendCL native OM",
                        cpu_fallback=False, input_readonly=True, guards_intact=True, io_validated=True, repetitions=2,
                        m=16, k=512, n=128, device_id=0, workspace_bytes=2097152, tracked_device_allocation_bytes=2101248,
                        op="DFlashGroupQuantLinear", tile_n=64, tile_k=128, timing=copy.deepcopy(timing))
                    files = []
                    for label in ("actual-0", "actual-1", "benchmark-last", "postcheck"):
                        path = dest / (kind + "-" + label + ".bin")
                        path.write_bytes(b"\x00\x3c" * (16 * 128))
                        files.append(record(path, dest))
                    case["outputs"][kind], case["bracket_outputs"][kind] = files[:2], files[2:]
                built.append(({"cases": [case]}, dest, summary))
            before, after = built
            replies = [(baseline, before[0], before[1]), (candidate, after[0], after[1])]
            with patch("compare_a31.load_run", side_effect=replies), redirect_stdout(io.StringIO()):
                self.assertEqual(compare_variants(before[2], after[2], root / "ok.json"), 0)
            tail = after[1] / "custom-postcheck.bin"
            tail.write_bytes(b"\x00\x40" + tail.read_bytes()[2:])
            after[0]["cases"][0]["bracket_outputs"]["custom"][1] = record(tail, after[1])
            with patch("compare_a31.load_run", side_effect=replies), redirect_stdout(io.StringIO()):
                self.assertEqual(compare_variants(before[2], after[2], root / "bad.json"), 1)

    def test_profile_preparation_is_separate_and_preserves_accepted_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            build = root / "build with spaces"
            runner = build / "test/dflash_group_quant_linear_test"
            runner.parent.mkdir(parents=True); runner.write_bytes(b"CPU fixture, never executed")
            vendor = build / "opp/vendors/customize/bin/set_env.bash"
            vendor.parent.mkdir(parents=True); vendor.write_text("# CPU fixture\n")
            source = root / "real/layer-0-gate_up/custom"
            source.mkdir(parents=True)
            capture = root / "capture"; capture.mkdir()
            files = {}
            for name in ("x.bin", "w_nz.bin", "s_gn.bin"):
                path = capture / name; path.write_bytes(b"hash-bound test input")
                files[name] = record(path, capture); (source / name).symlink_to(path)
            manifest = capture / "manifest.json"
            write_json(manifest, {"cases": [{"name": "layer-0-gate_up", "files": files}]})
            write_json(root / "real/suite.json", {"status": "PASS", "bundle": str(manifest), "bundle_sha256": sha256(manifest),
                "runners": {"custom": {"path": str(runner), "sha256": sha256(runner)}}})
            write_json(source / "execution.json", {"status": "PASS", "runtime": "AscendCL ACLNN", "m": 16,
                "k": 2560, "n": 19456, "device_id": 0, "timing": {"protocol": "continuous-v1"}})
            write_json(source / "command.json", [str(runner), "0", str(source), "16", "2560", "19456", "3", "10", "--continuous"])
            original = (source / "command.json").read_bytes()
            command, launcher = prepare_profile(source, root / "new profile")
            self.assertNotEqual(command[2], str(source))
            self.assertEqual((source / "command.json").read_bytes(), original)
            checked = subprocess.run(["bash", "-n", str(launcher)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            (capture / "w_nz.bin").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "bytes/hash"):
                prepare_profile(source, root / "rejected profile")

    def test_generated_host_and_kernel_modes_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            host, kernel, report = root / "host.h", root / "kernel.h", root / "build.json"
            for mode, number in (("legacy", 0), ("batched", 1)):
                configure_build(host, report, 0, mode, kernel)
                self.assertEqual(host.read_bytes(), kernel.read_bytes())
                self.assertIn(f"DEQUANT_MODE {number}U", kernel.read_text())
                self.assertEqual(load_build_config(report)["dequant_mode"], mode)
            kernel.write_text(kernel.read_text().replace("DEQUANT_MODE 1U", "DEQUANT_MODE 0U"))
            with self.assertRaisesRegex(ValueError, "host/kernel"):
                load_build_config(report)

    def test_launch_rejects_wrong_variant_and_old_unlabelled_opp(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "runner.log"
            launch = dict(launch_record(), version=2, dequant_mode="batched")
            log.write_text(PREFIX + json.dumps(launch))
            self.assertEqual(launch_evidence(log, (16, 2560, 19456), 0, 2097152, "batched")["dequant_mode"], "batched")
            with self.assertRaisesRegex(ValueError, "dequantization mode"):
                launch_evidence(log, (16, 2560, 19456), 0, 2097152, "legacy")
            log.write_text(PREFIX + json.dumps(launch_record()))
            with self.assertRaisesRegex(ValueError, "dequantization mode"):
                launch_evidence(log, (16, 2560, 19456), 0, 2097152, "batched")

    def test_comparison_rejects_mixed_timing_cores_and_wrong_direction(self):
        baseline = {"bundle_sha256": "same", "native_om_manifest_sha256": "same", "device_id": 0,
                    "timing_protocol": "continuous-v1", "build": {"core_limit": 0, "dequant_mode": "legacy",
                    "source_sha256": {"kernel": "same"}}, "launches": {"real": [{"launch": dict(launch_record(), dequant_mode="legacy")}]}}
        candidate = copy.deepcopy(baseline)
        candidate["build"]["dequant_mode"] = "batched"
        candidate["launches"]["real"][0]["launch"]["dequant_mode"] = "batched"
        matched_variants(baseline, candidate)
        with self.assertRaisesRegex(ValueError, "same dequantization mode"):
            compatible_runs(baseline, candidate)
        for which, value in (("timing_protocol", "checked-v1"), ("device_id", 1)):
            with self.subTest(which=which), self.assertRaises(ValueError):
                matched_variants(baseline, dict(candidate, **{which: value}))
        with self.assertRaisesRegex(ValueError, "baseline must be legacy"):
            matched_variants(candidate, baseline)
        candidate["launches"]["real"][0]["launch"]["block_dim"] = 4
        with self.assertRaisesRegex(ValueError, "launch differs"):
            matched_variants(baseline, candidate)

    def test_continuous_report_requires_bracket_checks_and_protocol(self):
        report = dict(status="PASS", runtime="AscendCL native OM", cpu_fallback=False,
                      input_readonly=True, guards_intact=True, io_validated=True, repetitions=2,
                      m=16, k=9728, n=2560, device_id=0, workspace_bytes=4096, tracked_device_allocation_bytes=8192,
                      timing=dict(status="MEASURED", protocol="continuous-v1", warmup=3, repetitions=10,
                                  execute_sync={"samples_ms": [1.] * 10}, per_timed_call_readback=False,
                                  per_timed_call_poison=False, correctness_before_calls=2,
                                  correctness_after_calls=1, timed_tail_checked=True))
        def validate(value):
            validate_execution(value, (16, 9728, 2560), 0, "AscendCL native OM", 3, 10, "continuous-v1")
        validate(report)
        for key, value in (("protocol", "checked-v1"), ("per_timed_call_readback", True),
                           ("per_timed_call_poison", True), ("correctness_after_calls", 0), ("timed_tail_checked", False)):
            broken = copy.deepcopy(report); broken["timing"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(broken)

    def test_real_runner_scheduler_has_no_io_between_continuous_calls(self):
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        source = Path(__file__).with_name("benchmark_schedule_model.cpp")
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "schedule"
            compiled = subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                                       "-I", str(REPO / "framework/runtime/cpp/tests/fake_acl"), str(source), "-o", str(binary)],
                                      capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            executed = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)


if __name__ == "__main__":
    unittest.main()
