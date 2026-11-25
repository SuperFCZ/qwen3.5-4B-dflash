from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from recipes import CONTROL, OPS, gate, packed, header, candidates
from kernels import device
from layout import once

sys.path.insert(0, str(OPS / "draft_q_bandwidth/test"))
from test_q import q_cpu_headers


def cpu_headers(d):
    inc = q_cpu_headers(d)
    h = inc / "kernel_operator.h"
    s = h.read_text()
    a = s.index("    const bool stageA =")
    b = s.index("    assert(params.blockCount>0", a)
    s = s[:a] + """    using namespace transfer_panel;
    const bool stageA=false;
    constexpr unsigned localK=Activation<RowsCount>::LocalK;
    const bool gather=params.blockCount==RowsCount && params.srcStride==K/16-1 && params.dstStride==0;
    const bool scatter=params.blockCount==localK/16 && params.srcStride==0 && params.dstStride==RowsCount-1;
    assert(sizeof(T)==2 && params.blockLen==1 && (gather || scatter));
""" + s[b:]
    h.write_text(s)
    h = inc / "lib/matmul_intf.h"
    s = once(
        h.read_text(),
        "tiling_.singleM==16 && tiling_.orgKa==2560",
        "tiling_.singleM==transfer_panel::RowsCount && tiling_.orgKa==transfer_panel::PanelK",
    )
    s = once(s, "tiling_.singleN == 64);", "tiling_.singleN == transfer_panel::Columns);")
    h.write_text(once(s, "globalIndex % tiling_.orgN / 64", "globalIndex % tiling_.orgN / tiling_.singleN"))
    return inc


class Devices(unittest.TestCase):
    def test_all_candidates_and_consumers(self):
        rng = np.random.default_rng(325664)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            inc = cpu_headers(d)
            file = d / "impl.cpp"
            exe = d / "run"
            tested = 0
            for key, base in CONTROL.items():
                m, k, n = base.shape
                nk = rng.integers(-128, 128, (n, k), np.int8)
                gn = rng.uniform(0.006, 0.02, (k // 128, n)).astype("<f2")
                nz = nk.reshape(n // 16, 16, k // 32, 32).transpose(2, 0, 1, 3).tobytes()
                x = np.zeros((m, k), "<f2")
                acc = np.zeros((m, n), np.float32)
                for row in range(m):
                    for group in range(k // 128):
                        kk = group * 128 + (row * 13 + group * 5) % 128
                        x[row, kk] = (-1 if (row + group) % 2 else 1) * (1 + row / 83)
                        acc[row] += np.float32(x[row, kk]) * (nk[:, kk].astype("<f2") * gn[group]).astype(
                            np.float32
                        )
                x.tofile(d / "nd")
                x.reshape(m, k // 16, 16).transpose(1, 0, 2).tofile(d / "packed")
                acc.astype("<f2").tofile(d / "gold")
                for r in candidates(key):
                    with self.subTest(mode=r.name):
                        tested += 1
                        w, s = packed.packed_weights(nz, gn.tobytes(), n, k, r.width, r.panel)
                        (d / "codes").write_bytes(w)
                        (d / "scales").write_bytes(s)
                        (d / "transfer_spec.h").write_bytes(header(r))
                        src = device(r)
                        file.write_text(src)
                        calls = (n // r.width) * (k // r.panel)
                        fences = (2 * calls + n // r.width) if key == "down" else (3 * calls)
                        acopies = {"row": m, "gather": (k if r.keep_a else r.panel) // 16, "packed": 1}[
                            r.activation
                        ]
                        cc = [
                            "c++",
                            "-std=c++17",
                            "-O2",
                            "-ffp-contract=off",
                            "-include",
                            str(d / "transfer_spec.h"),
                            "-I" + str(inc),
                            "-I" + str(d),
                            "-I" + str(gate.HERE),
                            "-DTEST_ENTRY=" + r.spec.stem,
                            f"-DTEST_A_COPIES={acopies}",
                            f"-DTEST_FENCES={fences}",
                            f"-DTEST_AHEAD={int(r.ahead)}",
                            str(HERE / "test/portable.cpp"),
                            "-o",
                            str(exe),
                        ]
                        subprocess.run(cc, check=True)
                        for fail in sorted({-1, 0, min(1, calls // 8 - 1), calls // 8 - 1}):
                            subprocess.run([exe, d, str(fail)], check=True)
                        # Specifically exercise W1 reuse after a prefetched piece and the second reader of paired raw.
                        removal = {
                            "down-next-w1": (
                                "Pair<HardEvent::MTE3_V>(stored); // last UB->B1 reader of W0",
                                "W16 overwritten",
                            ),
                            "o-n16-raw2": ("Pair<HardEvent::V_MTE2>(rawConsumed);", "raw overwrite"),
                        }.get(r.name)
                        if removal:
                            file.write_text(
                                once(src, removal[0], "/* removed handoff for CPU fault check */")
                            )
                            subprocess.run(cc, check=True)
                            p = subprocess.run([exe, d, "-1"], capture_output=True, text=True, cwd=d)
                            self.assertNotEqual(p.returncode, 0)
                            self.assertIn(removal[1], p.stderr)
            self.assertEqual(tested, 24)


if __name__ == "__main__":
    unittest.main()
