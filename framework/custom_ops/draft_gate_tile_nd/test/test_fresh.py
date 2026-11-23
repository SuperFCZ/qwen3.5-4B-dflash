from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from fresh_inputs import synthetic_operands
from fresh_run import data_files
from layout import M, K, N


class FreshStart(unittest.TestCase):
    def test_dyadic_reference_and_nz_layout(self):
        m, k, n = 3, 256, 64
        data = synthetic_operands(m, k, n)
        self.assertEqual(data, synthetic_operands(m, k, n))
        logits = np.frombuffer(data["golden.bin"], "<f2").reshape(m, n)
        x = np.frombuffer(data["x.bin"], "<f2").reshape(m, k)
        gn = np.frombuffer(data["s_gn.bin"], "<f2").reshape(k // 128, n)
        nz = np.frombuffer(data["w_nz.bin"], "i1").reshape(k // 32, n // 16, 16, 32)
        q = nz.transpose(1, 2, 0, 3).reshape(n, k)
        w16 = q.astype(np.float16) * np.repeat(gn.T, 128, axis=1)
        ref = (x.astype(np.float64) @ w16.T.astype(np.float64)).astype("<f2")
        self.assertEqual(logits.tobytes(), ref.tobytes())
        self.assertTrue((logits[:, ::2] > 0).all() and (logits[:, 1::2] < 0).all())

    def test_synthetic_is_fresh_and_checks_complete_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files, scope = data_files(root, None)
            self.assertEqual(scope, "SYNTHETIC_SEEDED_C16_GATE")
            self.assertEqual(
                {name: i["bytes"] for name, i in files.items()},
                {
                    "x.bin": 2 * M * K,
                    "w_nz.bin": K * N,
                    "s_gn.bin": 2 * (K // 128) * N,
                    "golden.bin": 2 * M * N,
                },
            )
            with self.assertRaisesRegex(ValueError, "unexpected C16 Layer0"):
                data_files(root, root / "inputs")
            with self.assertRaises(FileExistsError):
                data_files(root, None)

    def test_foreign_so_is_denied(self):
        from fresh_dso import check_maps

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "maps"
            p.write_text("111000-112000 r-xp 00000000 08:11 123 /home/other/libcust_opapi.so\n")
            with self.assertRaisesRegex(ValueError, "unbound OM DSO"):
                check_maps(p, {})
