import csv
import os
from pathlib import Path
import struct
import sys
import subprocess
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import packed_panel as p
from packed_run import charged_span
from air import protobuf_classes
from layout import once

sys.path.insert(0, str(p.f.q.HERE / "test"))
from test_q import q_cpu_headers


class Packed(unittest.TestCase):
    def test_actual_contiguous_a_and_wide_kernel(self):
        rng = np.random.default_rng(640320)
        m, k, n = p.M, p.K, p.N
        nk = rng.integers(-128, 128, (n, k), dtype=np.int8)
        gn = rng.uniform(0.007, 0.021, (k // 128, n)).astype("<f2")
        nz = nk.reshape(n // 16, 16, k // 32, 32).transpose(2, 0, 1, 3).tobytes()
        x = np.zeros((m, k), "<f2")
        expected = np.zeros((m, n), np.float32)
        for i in range(m):
            for g in range(k // 128):
                col = g * 128 + (
                    127 if g in (9, 19, 99) else 0 if g in (0, 10, 20) else (15 * i + 13 * g) % 128
                )
                x[i, col] = (-1 if (i + g) % 2 else 1) * (1 + i / 47)
                expected[i] += np.float32(x[i, col]) * (gn[g] * nk[:, col].astype(np.float16)).astype(
                    np.float32
                )
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            inc = q_cpu_headers(d)
            h = inc / "lib/matmul_intf.h"
            s = once(
                h.read_text(),
                "tiling_.singleM==16 && tiling_.orgKa==2560",
                "tiling_.singleM==64 && (tiling_.orgKa==1280 || tiling_.orgKa==2560)",
            )
            s = once(s, "tiling_.singleN == 64);", "(tiling_.singleN==64 || tiling_.singleN==320));")
            h.write_text(
                once(s, "globalIndex % tiling_.orgN / 64", "globalIndex % tiling_.orgN / tiling_.singleN")
            )
            x.reshape(m, k // 16, 16).transpose(1, 0, 2).tofile(d / "x")
            expected.astype("<f2").tofile(d / "expected")
            for width in (64, 320):
                pk = p.QUERY[width][0]
                weights, scales = p.packed_weights(nz, gn.tobytes(), n, k, width, pk)
                (d / "w").write_bytes(weights)
                (d / "s").write_bytes(scales)
                (d / "packed_spec.h").write_bytes(p.header(width))
                (d / (p.STEM + ".cpp")).write_text(p.device_source())
                ex = d / "run"
                cc = [
                    "c++",
                    "-std=c++17",
                    "-O2",
                    "-ffp-contract=off",
                    "-I" + str(inc),
                    "-I" + str(d),
                    "-I" + str(p.f.q.gate.HERE),
                    str(HERE / "test/portable_packed.cpp"),
                    "-o",
                    str(ex),
                ]
                subprocess.run(cc, check=True)
                for args in ([], ["0"], [str(k // pk - 1)]):
                    subprocess.run([ex, d, *args], check=True)

    def test_read_real_wide_scheduler(self):
        w = 320
        pk, bk, sa, sb, da, db = p.QUERY[w]
        b = bytearray(224)
        sc = 145152
        struct.pack_into("<4IQ", b, 0, 64, p.K, p.N, pk, sc)
        off = {f["name"]: 24 + f["offset"] for f in p.f.schema()}
        for k, v in dict(
            singleCoreM=64,
            singleCoreN=w,
            singleCoreK=pk,
            baseM=64,
            baseN=w,
            baseK=bk,
            stepKa=sa,
            stepKb=sb,
            depthA1=da,
            depthB1=db,
            dbL0A=2,
            dbL0B=2,
            dbL0C=1,
            isBias=0,
            transLength=40960,
        ).items():
            struct.pack_into("<I", b, off[k], v)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            lib = d / "private/lib.so"
            lib.parent.mkdir()
            lib.write_bytes(b"bound")
            pkg = {"opp": {"install": str(lib.parent), "preflight": {"files": [p.c.identity(lib)]}}}
            plat = dict(
                host_dso=str(lib),
                blockDim=8,
                availableCores=8,
                coreLimit=0,
                workspace=2097152,
                capacity=dict(UB=262144, L1=1048576, L0A=65536, L0B=65536, L0C=262144),
            )
            res = dict(
                tile_k=1280,
                raw_banks=1,
                user_ub=116992,
                scratch=sc,
                codes_bytes=20480,
                scale_bytes=6400,
                weight_bytes=81920,
                workspace=2097152,
                l1_used=1040384,
                l0a_used=8192,
                l0b_used=40960,
                l0c_used=81920,
            )
            sch = dict(
                tile_n=320,
                n_tiles=8,
                baseline_assignments=[[i] for i in range(8)],
                candidate_assignments=[[i] for i in range(8)],
                resources=res,
                tiling_bytes=len(b),
                tiling_hex=b.hex(),
                variant="A",
            )
            import json

            orig = {"platform": plat, "schedule": dict(sch, tile_n=64, n_tiles=40)}
            (d / "p").write_text(json.dumps(plat) + "\n")
            (d / "s").write_text(json.dumps(sch) + "\n")
            self.assertEqual(p.actual_plan(d, pkg, d / "p", d / "s", orig, w)["schedule"]["n_tiles"], 8)
            (d / "s").write_text(json.dumps(dict(sch, tile_n=64)) + "\n")
            with self.assertRaisesRegex(ValueError, "scheduler/resource"):
                p.actual_plan(d, pkg, d / "p", d / "s", orig, w)

    @unittest.skipUnless(os.environ.get("DFLASH_GE_DESCRIPTOR"), "CANN GE descriptor required")
    def test_real_air_pack_and_charged_span(self):
        types = protobuf_classes(Path(os.environ["DFLASH_GE_DESCRIPTOR"]).read_bytes())
        m, k, n = 64, 256, 640
        rng = np.random.default_rng(25)
        nz = rng.integers(-128, 128, (n, k), np.int8).tobytes()
        gn = rng.normal(size=(2, n)).astype("<f2").tobytes()
        msg = p.f.model_proto(types, nz, gn, projection_shape=(m, k, n))
        w, s = p.packed_weights(nz, gn, n, k, 320, 128)
        p.add_pack_graph(msg, 320, 128, w, s, m=m, k=k, n=n)
        table = {x.name: x for x in msg.graph[0].op}
        inputs = {"x": rng.integers(0, 65535, size=(m, k), dtype=np.uint16)}
        for row in msg.graph[0].op:
            if row.type == "Const" and row.name.startswith("fc_a_"):
                inputs[row.name] = np.frombuffer(row.attr["value"].t.data, "<i4")
            if row.type in ("Reshape", "Transpose"):
                a, b = (inputs[e.split(":")[0]] for e in row.input)
                inputs[row.name] = a.reshape(tuple(b)) if row.type == "Reshape" else a.transpose(tuple(b))
        actual = inputs["fc_a_pack"].copy().reshape(k // 16, m, 16)
        np.testing.assert_array_equal(actual.transpose(1, 0, 2).reshape(m, k), inputs["x"])
        self.assertEqual(table[p.f.OLD_OP].input[0], "fc_a_pack:0")
        self.assertEqual(table["weight"].attr["value"].t.data, w)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            path = d / "timed-tasks.csv"
            with path.open("w", newline="") as fp:
                a = csv.writer(fp)
                a.writerow(("module", "op", "call", "start", "end"))
                for i in range(7, 37):
                    a.writerow(("Unmapped", "tr", i, 100 * i, 100 * i + 5))
                    a.writerow(("fc", "proj", i, 100 * i + 7, 100 * i + 15))
            y = charged_span(d, dict(prep_nodes=["tr"], selected=dict(node="proj")), dict(attribution="TEST"))
            self.assertEqual((y["median_us"], y["pack_task_median_us"]), (15, 5))
            self.assertEqual(
                charged_span(
                    d, dict(prep_nodes=["missing"], selected=dict(node="proj")), dict(attribution="TEST")
                )["status"],
                "UNRESOLVED",
            )
