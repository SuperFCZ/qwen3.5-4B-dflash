"""CPU fixture/identity checks for expanded N/K; never device evidence."""
import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

import reference as ref
from test_reference import checker_fixture


class A1ReferenceTests(unittest.TestCase):
    def test_fixed_shape_matrix_and_tile_plan(self):
        self.assertEqual(len(ref.A1_WORKLOADS), 9)
        self.assertEqual({(s.m, s.k, s.n) for s in ref.A1_WORKLOADS},
                         {(16, k, n) for k in (256, 512, 1024) for n in (64, 128, 256)})
        for shape in ref.A1_WORKLOADS:
            self.assertEqual(shape.tile_k, 256 if shape.k == 256 else 128)
            self.assertEqual(shape.groups, shape.k // 128)
        for m, k, n in ((32, 256, 64), (16, 384, 64), (16, 512, 80), (16, -256, 64), (16, 256.0, 64)):
            with self.assertRaises(ValueError): ref.Workload(m, k, n)

    def test_all_shapes_nz_and_scale_byte_permutations(self):
        for shape in ref.A1_WORKLOADS:
            with self.subTest(shape=shape.name):
                q = [(n * 73 + k * 19 + (k // 128) * 31) % 256 - 128
                     for n in range(shape.n) for k in range(shape.k)]
                expected = bytes(q[(n1 * 16 + n0) * shape.k + k1 * 32 + k0] & 255
                                 for k1 in range(shape.k // 32) for n1 in range(shape.n // 16)
                                 for n0 in range(16) for k0 in range(32))
                self.assertEqual(ref.pack_nz(q, shape), expected)
                self.assertEqual(ref.unpack_nz(expected, shape), q)
                bits = [(i * 137) & 0xFFFF for i in range(shape.groups * shape.n)]
                raw_ng = struct.pack(f"<{len(bits)}H", *bits)
                expected_gn = struct.pack(f"<{len(bits)}H", *(bits[n * shape.groups + g]
                                         for g in range(shape.groups) for n in range(shape.n)))
                self.assertEqual(ref.scales_to_gn(raw_ng, shape), expected_gn)
                self.assertEqual(ref.scales_to_ng(expected_gn, shape), raw_ng)

    def test_all_group_boundaries_and_fp32_cancellation(self):
        for shape in ref.A1_WORKLOADS:
            if shape == ref.TINY: continue
            with self.subTest(shape=shape.name):
                fixtures = {case[0]: case for case in ref.cases(shape)}
                covered = set()
                for name, (_, x, _, _, _) in fixtures.items():
                    if name.startswith("all_group_boundaries"):
                        covered.update(i % shape.k for i, value in enumerate(ref.read_half(x, shape.m * shape.k))
                                       if value != 0)
                self.assertTrue({g * 128 + d for g in range(1, shape.groups) for d in (-1, 0)} <= covered)
                self.assertIn(shape.k - 1, covered)
                _, x, q, scales, exact = fixtures["fp32_accumulator_cancellation"]
                self.assertTrue(exact)
                expected = ref.half_bytes([(-1 if m % 2 else 1) * (n % 7 + 1) / 64
                                           for m in range(shape.m) for n in range(shape.n)])
                self.assertEqual(ref.cpu_reference(x, q, scales, shape), expected)
                with self.assertRaises(OverflowError):
                    ref.half(1024 * 127 + 1 / 64)  # A premature FP16 partial result cannot represent this.
                for name, group in (("second_group_only", 1), ("last_group_only", shape.groups - 1)):
                    values = ref.read_half(fixtures[name][1], shape.m * shape.k)
                    self.assertEqual({(i % shape.k) // 128 for i, v in enumerate(values) if v != 0}, {group})

    def test_scales_distinguish_all_groups_and_n_tiles(self):
        shape = ref.Workload(k=1024, n=256)
        _, _, _, raw, _ = next(ref.cases(shape))
        scales = ref.read_half(raw, shape.groups * shape.n)
        self.assertEqual(len({scales[g * shape.n] for g in range(shape.groups)}), shape.groups)
        self.assertEqual(len({scales[n] for n in (0, 64, 128, 192)}), 4)

    def test_expanded_manifest_and_output_coordinates(self):
        shape = ref.Workload(k=512, n=128)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory) / "data"
            ref.prepare(root, shape)
            manifest = ref.load_cases(root)
            self.assertEqual(manifest["tile"], [16, 64, 128])
            self.assertEqual((manifest["n_tiles"], manifest["k_tiles"]), (2, 4))
            raw = bytes(shape.m * shape.n * 2)
            actual = raw[:-2] + b"\x01\x00"
            self.assertEqual(ref.compare(raw, actual, shape)["first_difference"]["index"], [15, 127])
            with self.assertRaises(ValueError): ref.compare(raw, actual[:2048], shape)
            manifest["k"] = 1024
            ref.write_json(root / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "tile/accumulation"):
                ref.load_cases(root)

    def test_expanded_checker_requires_matching_runner_identity(self):
        shape = ref.Workload(k=256, n=128)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory) / "data"
            checker_fixture(root, shape=shape)
            self.assertTrue(ref.check(root))  # Synthetic checker test, not an NPU run.
            path = root / "signed_zero" / "execution.json"
            evidence = json.loads(path.read_text())
            evidence["n"] = 64
            ref.write_json(path, evidence)
            with self.assertRaisesRegex(ValueError, "execution evidence"):
                ref.check(root)


if __name__ == "__main__":
    unittest.main()
