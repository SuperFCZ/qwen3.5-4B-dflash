"""Regression for the receiver's a1.ovznlDEL kernel JSON lookup failure."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import opp_preflight as check


class OppPreflightTests(unittest.TestCase):
    def test_reported_directory_reproduces_cann_wrong_json_path(self):
        root = Path("/home/w00949577/z50058744/qwen3.5-4B-dflash/framework/custom_ops/draft_quant/.build")
        binary = root / "a1.ovznlDEL/opp/vendors/customize/op_impl/ai_core/tbe/kernel/ascend310p" / \
            check.OP_KERNEL / "DFlashGroupQuantLinear_2cf67ac22eafa0ecc89a845d1f0e845a.o"
        self.assertEqual(check.cann_json_path(binary), root / "a1.json")
        with self.assertRaisesRegex(ValueError, "first '.o'"):
            check.check_binary_path(binary)

    def test_hyphen_prefix_accepts_every_alphanumeric_initial(self):
        for initial in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
            with self.subTest(initial=initial):
                root = Path(f"/tmp/dflash/.build/a1-{initial}vznlDEL")
                self.assertEqual(check.check_root(root)["status"], "PASS")

    def test_parent_or_json_prefix_is_also_rejected(self):
        for root in ("/tmp/.operator/repo/.build/a1-safe", "/tmp/project.json-cache/.build/a1-safe"):
            with self.subTest(root=root), self.assertRaisesRegex(ValueError, "path collision"):
                check.check_root(Path(root))

    def test_safe_alias_to_unsafe_real_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual = root / "a1.older"
            actual.mkdir()
            alias = root / "a1-safe"
            alias.symlink_to(actual, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "path collision"):
                check.check_root(alias)

    def test_installed_pair_missing_and_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "a1-ovznlDEL/opp"
            with self.assertRaisesRegex(ValueError, "no installed"):
                check.check_install(root)
            kernel_dir = root / "vendors/customize/op_impl/ai_core/tbe/kernel/ascend310p" / check.OP_KERNEL
            kernel_dir.mkdir(parents=True)
            binary = kernel_dir / "DFlashGroupQuantLinear_test.o"
            binary.write_bytes(b"test-only-object")  # Not a compiled device binary.
            with self.assertRaises(FileNotFoundError): check.check_install(root)
            metadata = binary.with_suffix(".json")
            metadata.write_text("not JSON")
            with self.assertRaises(ValueError): check.check_install(root)
            metadata.write_text("[]")
            with self.assertRaisesRegex(ValueError, "must be an object"):
                check.check_install(root)
            metadata.write_text('{"binFileName":"DFlashGroupQuantLinear_test"}')
            report = check.check_install(root)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["npu_status"], "NOT_RUN")
            self.assertEqual(report["kernels"][0]["json"], str(metadata))
            self.assertEqual(report["kernels"][0]["binary_bytes"], len(b"test-only-object"))
            binary.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "empty kernel"):
                check.check_install(root)

    def test_cli_records_failure_before_device_execution(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            report_path = Path(directory) / "preflight.json"
            with patch("sys.argv", ["opp_preflight.py", "--root", "/tmp/a1.ovznlDEL",
                                    "--report", str(report_path)]):
                self.assertEqual(check.main(), 1)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["npu_status"], "NOT_RUN")
            self.assertIn("/tmp/a1.json", report["error"])


if __name__ == "__main__":
    unittest.main()
