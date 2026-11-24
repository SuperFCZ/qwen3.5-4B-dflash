import copy
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import q_panel as q
from q_timing import select
from layout import once

sys.path.insert(0, str(q.gate.HERE / "test"))
from test_panels import cpu_headers


def q_cpu_headers(root):
    inc = cpu_headers(root)
    p = inc / "kernel_operator.h"
    s = p.read_text()
    s = once(
        s,
        "if(sizeof(T)==2 && params.blockCount==16)\n        assert(params.blockLen==1 && params.srcStride==159 && params.dstStride==0);",
        "if(sizeof(T)==2 && params.blockCount==160)\n        assert(params.blockLen==1 && params.srcStride==0 && params.dstStride==15);",
    )
    # Track the cross-pipeline last INT8 reader even across a vector-only barrier.
    s = once(s, "struct SyncModel {", "struct SyncModel {\n    std::vector<Range> rawReaders;")
    s = once(
        s,
        'Require(cpuSync.vectorIssued == event.fence, "wrong Vector to MTE2 fence generation");',
        'Require(cpuSync.vectorIssued == event.fence, "wrong Vector to MTE2 fence generation");cpuSync.rawReaders.clear();',
    )
    s = once(s, "++cpuMetrics.fullBarriers;", "++cpuMetrics.fullBarriers; cpuSync.rawReaders.clear();")
    s = once(
        s,
        "static_assert(sizeof(D) == 2 && sizeof(S) == 1);",
        "static_assert(sizeof(D) == 2 && sizeof(S) == 1); cpuSync.rawReaders.push_back(Bytes(src.data, mask*repeats));",
    )
    s = once(
        s,
        "    if (cpuSync.l1B.bytes && Overlaps(target,cpuSync.l1B)) {",
        '    if constexpr(sizeof(T)==1) for(auto reader:cpuSync.rawReaders) Require(!Overlaps(target,reader),"raw overwrite before Vector to MTE2 fence");\n    if (cpuSync.l1B.bytes && Overlaps(target,cpuSync.l1B)) {',
    )
    p.write_text(s)
    return inc


class TestQ(unittest.TestCase):
    def test_delayed_dma_kernel(self):
        k, n, m = q.K, q.N, q.M
        rng = np.random.default_rng(909)
        nk = rng.integers(-128, 128, size=(n, k), dtype=np.int8)
        nz = nk.reshape(n // 16, 16, k // 32, 32).transpose(2, 0, 1, 3).tobytes()
        gn = rng.uniform(0.006, 0.03, (k // 128, n)).astype("<f2")
        codes, scale, _ = q.gate.pack_cube_panels(nz, gn.tobytes(), k, k=k, n=n)
        x = np.zeros((m, k), "<f2")
        acc = np.zeros((m, n), np.float32)
        for row in range(m):
            for group in range(k // 128):
                kk = group * 128 + (row * 13 + group * 7) % 128
                x[row, kk] = (-1 if (row + group) % 2 else 1) * (1 + row / 31)
                w = nk[:, kk].astype(np.float16) * gn[group]
                acc[row] += np.float32(x[row, kk]) * w.astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inc = q_cpu_headers(root)
            (root / "codes").write_bytes(codes)
            (root / "scales").write_bytes(scale)
            x.tofile(root / "x")
            acc.astype("<f2").tofile(root / "expected")
            for mode in (q.PANEL, q.PREFETCH):
                (root / "q_spec.h").write_bytes(q.header(mode))
                command = [
                    "c++",
                    "-std=c++17",
                    "-O2",
                    "-ffp-contract=off",
                    "-I" + str(inc),
                    "-I" + str(root),
                    "-I" + str(HERE / "op_kernel"),
                    "-I" + str(q.gate.HERE),
                    str(HERE / "test/portable_q.cpp"),
                    "-o",
                    str(root / "portable"),
                ]
                subprocess.run(command, check=True)
                subprocess.run([root / "portable", root], check=True)
                subprocess.run([root / "portable", root, "fail-first-iterate"], check=True)
                if mode == q.PREFETCH:
                    source = (HERE / "op_kernel" / (q.STEM + ".cpp")).read_text()
                    for anchor, msg in (
                        ("if (stripe>0) Pair<HardEvent::V_MTE2>(rawConsumed);", "raw overwrite"),
                        (
                            "if (stripe==1 || stripe==2) Pair<HardEvent::MTE3_V>(weightStored);",
                            "W16 overwritten",
                        ),
                    ):
                        (root / (q.STEM + ".cpp")).write_text(
                            once(source, anchor, "// injected missing producer/consumer dependency")
                        )
                        subprocess.run(command, check=True)
                        bad = subprocess.run(
                            [root / "portable", root], capture_output=True, text=True, cwd=root
                        )
                        self.assertNotEqual(bad.returncode, 0)
                        self.assertIn(msg, bad.stderr)

    def test_resource_and_selection(self):
        offsets = {f["name"]: 24 + f["offset"] for f in q.schema()}
        for mode, raw_banks, scratch in ((q.PANEL, 1, 46592), (q.PREFETCH, 2, 5632)):
            b = bytearray(224)
            struct.pack_into("<4IQ", b, 0, q.M, q.K, q.N, q.K, scratch)
            for k, v in dict(
                singleCoreM=16,
                singleCoreN=64,
                singleCoreK=2560,
                baseM=16,
                baseN=64,
                baseK=256,
                stepKa=10,
                stepKb=10,
                depthA1=10,
                depthB1=10,
                dbL0A=2,
                dbL0B=2,
                dbL0C=1,
                transLength=2048,
            ).items():
                struct.pack_into("<I", b, offsets[k], v)
            platform = dict(
                availableCores=8,
                blockDim=8,
                coreLimit=0,
                workspace=2097152,
                capacity=dict(UB=262144, L1=1048576, L0A=65536, L0B=65536, L0C=262144),
            )
            query = dict(
                platform=platform,
                tiling_sha256=q.sha(b),
                schedule=dict(
                    tiling_hex=b.hex(),
                    baseline_assignments=[list(range(i, 64, 8)) for i in range(8)],
                    resources=dict(
                        raw_banks=raw_banks,
                        tile_k=2560,
                        codes_bytes=40960 * raw_banks,
                        scale_bytes=2560,
                        weight_bytes=163840,
                        user_ub=262144 - scratch,
                        scratch=scratch,
                        workspace=2097152,
                        l1_used=819200,
                        l0a_used=16384,
                        l0b_used=65536,
                        l0c_used=4096,
                    ),
                ),
            )
            self.assertEqual(q.plan(query, mode, query)["iterates_per_n64"], 1)
            bad = copy.deepcopy(query)
            bad["schedule"]["resources"]["weight_bytes"] -= 1
            with self.assertRaisesRegex(ValueError, "allocations"):
                q.plan(bad, mode, query)
        self.assertEqual(q.BYTES, 10862592)
        self.assertAlmostEqual(q.FLOOR_MS, 0.05304)
        stats = {
            q.CONTROL + "__" + q.PANEL: {"decision": "GO"},
            q.CONTROL + "__" + q.PREFETCH: {"decision": "GO"},
            q.PANEL + "__" + q.PREFETCH: {"decision": "NO_GO"},
        }
        self.assertEqual(select(stats), q.PANEL)
        stats[q.PANEL + "__" + q.PREFETCH]["decision"] = "GO"
        self.assertEqual(select(stats), q.PREFETCH)


if __name__ == "__main__":
    unittest.main()
