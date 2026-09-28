"""Local CPU tests for fixture layout and the checker; no Ascend execution."""
import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

import reference as ref


def checker_fixture(root, native_transform=None, custom_transform=None):
    """Synthetic records for checker tests ONLY; never physical device evidence."""
    ref.prepare(root)
    manifest = ref.load_cases(root)
    native = {"status": "PASS", "cpu_fallback": False, "group_size": 128, "inner_precise": 0,
              "reference_policy": ref.NATIVE_POLICY,
              "manifest_sha256": ref.digest((root / "manifest.json").read_bytes()),
              "environment": {"device_id": 0}, "cases": []}
    for case in manifest["cases"]:
        folder = root / case["name"]
        raw = (folder / "cpu.bin").read_bytes()
        golden = native_transform(case["name"], raw) if native_transform else raw
        actual = custom_transform(case["name"], raw) if custom_transform else raw
        for repeat in range(ref.REPETITIONS):
            (folder / f"native-{repeat}.bin").write_bytes(golden)
            (folder / f"actual-{repeat}.bin").write_bytes(actual)
        native["cases"].append({"name": case["name"], "output_sha256": [ref.digest(golden)] * 2})
        ref.write_json(folder / "execution.json", {"status": "PASS", "runtime": "AscendCL ACLNN",
            "op": "DFlashGroupQuantLinear", "cpu_fallback": False, "device_id": 0,
            "input_readonly": True, "guards_intact": True, "repetitions": 2})
    ref.write_json(root / "native-eager.json", native)


def one_bit_error(name, raw):
    return bytes([raw[0] ^ 1]) + raw[1:] if name == "group_nz_boundaries" else raw


class ReferenceTests(unittest.TestCase):
    def test_nz_byte_permutation_against_physical_loop(self):
        q = [(n * 71 + k * 23) % 256 - 128 for n in range(ref.N) for k in range(ref.K)]
        expected = bytes(q[(n1 * 16 + n0) * ref.K + k1 * 32 + k0] & 255
                         for k1 in range(8) for n1 in range(4) for n0 in range(16) for k0 in range(32))
        self.assertEqual(ref.pack_nz(q), expected)
        self.assertEqual(ref.unpack_nz(expected), q)
        self.assertEqual(set(ref.nz_offset(n, k) for n in range(ref.N) for k in range(ref.K)),
                         set(range(ref.N * ref.K)))
        self.assertIn(-128, q)
        self.assertIn(127, q)

    def test_scale_transpose_preserves_all_half_bits(self):
        # Include signed zeros, subnormals, NaN payload and infinities here:
        # transposition is a byte operation, independent of input validation.
        bits = [0, 0x8000, 1, 0x7C00, 0xFC00, 0x7E13] + list(range(122))
        raw = struct.pack("<128H", *bits)
        expected = struct.pack("<128H", *(bits[n * 2 + g] for g in range(2) for n in range(64)))
        self.assertEqual(ref.scales_to_gn(raw), expected)
        self.assertEqual(ref.scales_to_ng(expected), raw)
        with self.assertRaises(ValueError):
            ref.scales_to_ng(expected[:-1])

    def test_gn_ng_mixup_reproduces_reported_native_failures(self):
        # Receiver report after the ACLNN metadata fix: custom matches CPU on all six fixtures.
        # Reinterpreting contiguous GN as NG reproduces every reported native
        # mismatch count, error bound and first pair of FP16 bits on CPU.
        observed = {
            "group_nz_boundaries": (764, 0.08935546875, 2048, "0xab08", "0xacb0"),
            "dense_signed_391": (1022, 2.21240234375, 30784, "0xbe65", "0xbcfc"),
            "dense_signed_817": (1024, 2.126953125, 30230, "0x3ae3", "0x39be"),
            "second_group_only": (763, 0.61962890625, 2048, "0x2840", "0x2660"),
            "rounding_probe": (1023, 11.69140625, 35742, "0x3e0a", "0x40ca"),
        }
        for name, xraw, q, s_gn, _ in ref.cases():
            with self.subTest(case=name):
                correct = ref.cpu_reference(xraw, q, s_gn)
                wrong = ref.cpu_reference(xraw, q, ref.scales_to_gn(s_gn))
                delta = ref.compare(wrong, correct)
                if name == "signed_zero":
                    self.assertTrue(delta["bitwise_equal"])
                else:
                    self.assertEqual((delta["bit_mismatches"], delta["max_abs_error"], delta["max_ulp"],
                                      delta["first_difference"]["expected_bits"],
                                      delta["first_difference"]["actual_bits"]), observed[name])
                # NG storage is re-viewed as NG by CANN, then per-group scale
                # transpose creates the original GN bytes for its kernel.
                fixed_gn = ref.scales_to_gn(ref.scales_to_ng(s_gn))
                self.assertEqual(fixed_gn, s_gn)

    def test_exact_fixtures_match_independent_integer_dot(self):
        for name, xraw, q, scales_raw, exact in ref.cases():
            if not exact:
                continue
            with self.subTest(case=name):
                x = ref.read_half(xraw, ref.M * ref.K)
                scales = ref.read_half(scales_raw, 2 * ref.N)
                xu = [int(v * 128) for v in x]
                su = [int(v * 32) for v in scales]
                # Every possible partial sum is bounded by 256 * 128 * 4;
                # integer accumulation gives an independent exact oracle.
                expected = [sum(xu[m * ref.K + k] * q[n * ref.K + k] * su[(k // 128) * ref.N + n]
                                for k in range(ref.K)) / 4096
                            for m in range(ref.M) for n in range(ref.N)]
                self.assertEqual(ref.cpu_reference(xraw, q, scales_raw), ref.half_bytes(expected))

    def test_wrong_group_scale_is_detected_at_128(self):
        _, xraw, q, scales_raw, _ = next(c for c in ref.cases() if c[0] == "group_nz_boundaries")
        wrong = scales_raw[:ref.N * 2] * 2
        actual = ref.cpu_reference(xraw, q, wrong)
        expected = ref.cpu_reference(xraw, q, scales_raw)
        self.assertEqual(actual[:10 * ref.N * 2], expected[:10 * ref.N * 2])
        self.assertNotEqual(actual[10 * ref.N * 2:], expected[10 * ref.N * 2:])

    def test_dequantization_rounds_to_half_before_dot(self):
        x = [0.0] * (ref.M * ref.K)
        x[0] = ref.half(1.1)
        q = [0] * (ref.N * ref.K)
        q[0] = -127
        scale = ref.half(0.002)
        actual = ref.read_half(ref.cpu_reference(ref.half_bytes(x), q,
                               ref.half_bytes([scale] * (2 * ref.N))), ref.M * ref.N)[0]
        self.assertEqual(actual, ref.half(x[0] * ref.half(q[0] * scale)))
        self.assertNotEqual(actual, ref.half(x[0] * q[0] * scale))

    def test_comparison_is_bitwise_and_reports_ulp(self):
        positive_zero = bytes(ref.M * ref.N * 2)
        negative_zero = b"\x00\x80" + positive_zero[2:]
        result = ref.compare(positive_zero, negative_zero)
        self.assertFalse(result["bitwise_equal"])
        self.assertEqual(result["bit_mismatches"], 1)
        self.assertEqual(result["max_abs_error"], 0)
        self.assertEqual(result["first_difference"]["index"], [0, 0])
        for one, next_one in ((0x3C00, 0x3C01), (0xBC00, 0xBC01)):
            expected = struct.pack("<H", one) + positive_zero[2:]
            actual = struct.pack("<H", next_one) + positive_zero[2:]
            self.assertEqual(ref.compare(expected, actual)["max_ulp"], 1)
        for special in (0x7E00, 0x7C00, 0xFC00):
            raw = struct.pack("<H", special) + positive_zero[2:]
            self.assertFalse(ref.compare(raw, raw)["finite"])
        with self.assertRaises(ValueError):
            ref.compare(positive_zero, positive_zero[:-1])

    def test_invalid_scale_and_carrier_rejected(self):
        x = bytes(ref.M * ref.K * 2)
        q = [0] * (ref.N * ref.K)
        for value in (0.0, -0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ref.cpu_reference(x, q, ref.half_bytes([value] * (2 * ref.N)))
        with self.assertRaises(ValueError):
            ref.pack_nz([128] * (ref.N * ref.K))
        with self.assertRaises(ValueError):
            ref.unpack_nz(bytes(ref.N * ref.K - 1))

    def test_cpu_gate_keeps_rounding_probe_diagnostic(self):
        self.assertEqual(ref.cpu_gate({"finite": True, "bitwise_equal": True}, True), "PASS")
        self.assertEqual(ref.cpu_gate({"finite": True, "bitwise_equal": False}, True), "FAIL")
        self.assertEqual(ref.cpu_gate({"finite": True, "bitwise_equal": False}, False), "DIAGNOSTIC_ONLY")
        self.assertEqual(ref.cpu_gate({"finite": False, "bitwise_equal": True}, False), "FAIL")

    def test_fixture_hash_lock_and_missing_device_evidence(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory) / "data"
            ref.prepare(root)
            manifest = ref.load_cases(root)
            self.assertEqual(len(manifest["cases"]), 6)
            with self.assertRaises(FileNotFoundError):
                ref.check(root)  # CPU goldens alone can never pass the checker.
            case_file = root / "group_nz_boundaries" / "w_nz.bin"
            raw = bytearray(case_file.read_bytes())
            raw[0] ^= 1
            case_file.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "fixture changed"):
                ref.load_cases(root)

    def test_checker_rejects_wrong_custom_output_with_native_evidence(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory) / "data"
            checker_fixture(root, custom_transform=lambda name, raw:
                            b"\x00\x7e" + raw[2:] if name == "rounding_probe" else raw)
            self.assertFalse(ref.check(root))
            result = json.loads((root / "comparison.json").read_text())
            self.assertEqual(result["native_om_parity"], "NOT_RUN")
            self.assertEqual(result["cases"][-1]["status"], "FAIL")

    def test_checker_identifies_bad_native_reference(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory) / "data"
            checker_fixture(root, native_transform=one_bit_error)
            self.assertFalse(ref.check(root))
            result = json.loads((root / "comparison.json").read_text())
            comparison = result["cases"][1]["comparisons"][0]
            self.assertEqual(comparison["cpu"]["bit_mismatches"], 0)
            self.assertEqual(comparison["native_vs_cpu"]["bit_mismatches"], 1)
            self.assertEqual(comparison["custom_cpu_gate"], "PASS")
            self.assertEqual(comparison["native_cpu_gate"], "FAIL")
            self.assertEqual(comparison["failed_checks"], ["native_reference_vs_cpu", "custom_vs_native"])

    def test_checker_rejects_shared_custom_native_error(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory) / "data"
            checker_fixture(root, native_transform=one_bit_error, custom_transform=one_bit_error)
            self.assertFalse(ref.check(root))
            result = json.loads((root / "comparison.json").read_text())
            comparison = result["cases"][1]["comparisons"][0]
            self.assertEqual(comparison["native_eager"]["bit_mismatches"], 0)
            self.assertEqual(comparison["failed_checks"], ["custom_vs_cpu", "native_reference_vs_cpu"])

    def test_checker_allows_cpu_rounding_difference_only_for_diagnostic_case(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory) / "data"
            def rounding_difference(name, raw):
                return bytes([raw[0] ^ 1]) + raw[1:] if name == "rounding_probe" else raw
            checker_fixture(root, native_transform=rounding_difference, custom_transform=rounding_difference)
            self.assertTrue(ref.check(root))
            result = json.loads((root / "comparison.json").read_text())
            comparison = result["cases"][-1]["comparisons"][0]
            self.assertEqual(comparison["native_cpu_gate"], "DIAGNOSTIC_ONLY")
            self.assertEqual(comparison["native_eager"]["bit_mismatches"], 0)

    def test_checker_rejects_old_native_scale_policy(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory) / "data"
            checker_fixture(root)
            report_path = root / "native-eager.json"
            native = json.loads(report_path.read_text())
            del native["reference_policy"]
            ref.write_json(report_path, native)
            with self.assertRaisesRegex(ValueError, "stale native reference"):
                ref.check(root)


if __name__ == "__main__":
    unittest.main()
