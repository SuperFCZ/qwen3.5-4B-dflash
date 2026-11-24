import copy
import importlib.util
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import fc_panel as f
from fc_source import replay_input
from layout import once
from air import protobuf_classes

sys.path.insert(0, str(f.q.HERE / "test"))
from test_q import q_cpu_headers


def cpu_headers(root):
    inc = q_cpu_headers(root)
    p = inc / "kernel_operator.h"
    s = p.read_text()
    s = once(
        s,
        "if(sizeof(T)==2 && params.blockCount==160)\n        assert(params.blockLen==1 && params.srcStride==0 && params.dstStride==15);",
        """if(sizeof(T)==2 && (params.blockCount==800 || params.blockCount==160))
        assert(params.blockLen==1 && params.srcStride==0 && params.dstStride==(params.blockCount==800?15:63));""",
    )
    # A row scatter has disjoint aligned 32B destinations. Index these delayed writes by address;
    # keep the simulator's unretired Cube consumer and overlapping DMA checks.
    s = once(s, "Range l1B{0,0};", "Range l1A{0,0}; Range l1B{0,0};")
    s = once(
        s,
        "if constexpr(P==TPosition::B1) cpuSync.l1B",
        "if constexpr(P==TPosition::A1) cpuSync.l1A=Bytes(buffer.template Get<uint8_t>(bytes).data,bytes);\n        if constexpr(P==TPosition::B1) cpuSync.l1B",
    )
    key = "for (const auto &task : cpuSync.pending) Require(!Overlaps(target, task.target)"
    s = once(
        s,
        key,
        """if(cpuSync.l1A.bytes && Overlaps(target,cpuSync.l1A)) {
        Require(target.bytes==32 && (target.begin-cpuSync.l1A.begin)%32==0,"misaligned A1 scatter");
        const auto it=cpuSync.writes.find(target.begin);
        Require(it==cpuSync.writes.end() || it->second.sequence<=cpuSync.dmaDone,"overlapping pending A1 writes");
    } else """ + key,
    )
    p.write_text(s)
    p = inc / "lib/matmul_intf.h"
    p.write_text(
        once(
            p.read_text(),
            "tiling_.singleM==16 && tiling_.orgKa==2560",
            "(tiling_.singleM==16 || tiling_.singleM==64) && tiling_.orgKa==2560",
        )
    )
    return inc


class FC(unittest.TestCase):
    def test_real_device_program_in_delayed_cpu_model(self):
        k, n = f.K, f.N
        rng = np.random.default_rng(128002560)
        w = rng.integers(-128, 128, (n, k), np.int8)
        gn = rng.uniform(0.006, 0.025, (k // 128, n)).astype("<f2")
        nz = w.reshape(n // 16, 16, k // 32, 32).transpose(2, 0, 1, 3).tobytes()
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            inc = cpu_headers(d)
            for m in f.GEARS:
                x = np.zeros((m, k), "<f2")
                total = np.zeros((m, n), np.float32)
                for row in range(m):
                    for g in range(k // 128):
                        col = g * 128 + (
                            127 if g in (19, 49, 99) else 0 if g in (0, 20, 50) else (row * 19 + g * 13) % 128
                        )
                        x[row, col] = (-1 if (row + g) % 2 else 1) * (1 + row / 32)
                        total[row] += np.float32(x[row, col]) * (gn[g] * w[:, col].astype(np.float16)).astype(
                            np.float32
                        )
                x.tofile(d / f"x{m}")
                total.astype("<f2").tofile(d / f"expected{m}")
            for mode in (f.LOCAL, f.GLOBAL):
                pk = f.panel_k(mode)
                codes, scales, _ = f.q.gate.pack_cube_panels(nz, gn.tobytes(), pk, k=k, n=n)
                (d / "codes").write_bytes(codes)
                (d / "scales").write_bytes(scales)
                (d / "fc_spec.h").write_bytes(f.header(mode))
                cpp = (HERE / "op_kernel" / (f.STEM + ".cpp")).read_text()
                source = d / (f.STEM + ".cpp")
                source.write_text(cpp)
                exe = d / "kernel"
                cc = [
                    "c++",
                    "-std=c++17",
                    "-O2",
                    "-ffp-contract=off",
                    "-I" + str(inc),
                    "-I" + str(d),
                    "-I" + str(f.q.gate.HERE),
                    str(HERE / "test/portable_fc.cpp"),
                    "-o",
                    str(exe),
                ]
                subprocess.run(cc, check=True)
                for m in f.GEARS:
                    for faults in ([], ["0"], [str(k // pk - 1)], [str(k // pk)]):
                        subprocess.run([exe, d, str(m), *faults], check=True)
                source.write_text(
                    once(
                        cpp,
                        "if (piece + 1 < Pieces) Pair<HardEvent::MTE3_V>(stored);",
                        "// injected missing W16 fence",
                    )
                )
                subprocess.run(cc, check=True)
                bad = subprocess.run([exe, d, "16"], cwd=d, capture_output=True, text=True)
                self.assertNotEqual(bad.returncode, 0)
                self.assertIn("W16 overwritten", bad.stderr)

    def test_sdk_shapes_and_memory(self):
        self.assertEqual([f.logical_bytes(m) for m in f.GEARS], [33771520, 35246080])
        offsets = {r["name"]: 24 + r["offset"] for r in f.schema()}
        for m, mode, l1, trans in (
            (16, f.LOCAL, 884736, 2048),
            (64, f.LOCAL, 1048576, 8192),
            (16, f.GLOBAL, 966656, 41472),
            (64, f.GLOBAL, 950272, 34816),
        ):
            pk = f.panel_k(mode)
            a, b, da, db = f.SDK[m, mode]
            raw = bytearray(224)
            scratch = 46592 if mode == f.LOCAL else 145152
            struct.pack_into("<4IQ", raw, 0, m, f.K, f.N, pk, scratch)
            for k, v in dict(
                singleCoreM=m,
                singleCoreN=64,
                singleCoreK=pk,
                baseM=m,
                baseN=64,
                baseK=256,
                stepKa=a,
                stepKb=b,
                depthA1=da,
                depthB1=db,
                dbL0A=2,
                dbL0B=2,
                dbL0C=1,
                isBias=0,
                transLength=trans,
            ).items():
                struct.pack_into("<I", raw, offsets[k], v)
            p = dict(
                availableCores=8,
                blockDim=8,
                coreLimit=0,
                workspace=2097152,
                capacity=dict(UB=262144, L1=1048576, L0A=65536, L0B=65536, L0C=262144),
            )
            codes = 40960 if mode == f.LOCAL else 20480
            resources = dict(
                raw_banks=1,
                tile_k=pk,
                codes_bytes=codes,
                weight_bytes=codes * 4,
                scale_bytes=pk,
                user_ub=262144 - scratch,
                scratch=scratch,
                l1_used=l1,
                l0a_used=m * 1024,
                l0b_used=65536,
                l0c_used=m * 256,
                workspace=2097152,
            )
            query = dict(
                platform=p,
                schedule=dict(
                    tiling_hex=raw.hex(),
                    resources=resources,
                    baseline_assignments=[list(range(i, 40, 8)) for i in range(8)],
                ),
                tiling_sha256=f.q.sha(raw),
            )
            self.assertEqual(f.plan(query, query, m, mode)["iterates_per_n64"], f.K // pk)
            bad = copy.deepcopy(query)
            bad["schedule"]["resources"]["l1_used"] -= 1
            with self.assertRaisesRegex(ValueError, "allocation"):
                f.plan(bad, query, m, mode)

    def test_gear_specific_pairing_and_counter_join(self):
        import csv
        from run_fc import benchmark
        from fc_runtime import pipeline_times

        def replay(path, mode, files, timed, i):
            m = files["gear"]
            value = (
                1.2
                if mode == f.CONTROL
                else (0.25 if mode == f.LOCAL else 0.4) if m == 16 else (0.6 if mode == f.LOCAL else 0.35)
            )
            return dict(
                round=i,
                mode=mode,
                folder=str(path),
                bit_mismatches=0,
                launch_authorized=True,
                profiled=False,
                validation={"timing": {"samples_ms": [value] * 30}},
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for m, winner in ((16, f.LOCAL), (64, f.GLOBAL)):
                out = benchmark(root / str(m), m, replay, {"gear": m})
                self.assertEqual(out["selected"], winner)
                self.assertEqual(len(out["runs"]), 18)
                for pair in out["statistics"].values():
                    self.assertTrue(all(r["count"] == 90 for r in pair["variants"].values()))
            p = root / "profiler"
            p.mkdir()
            summary = p / "op_summary_0.csv"
            with (root / "timed-tasks.csv").open("w", newline="") as fp, summary.open("w", newline="") as fq:
                a = csv.writer(fp)
                b = csv.writer(fq)
                a.writerow(("module", "idx", "call", "task_id", "stream", "op", "kind", "start"))
                b.writerow(
                    (
                        "Op Name",
                        "OP Type",
                        "Task Duration(us)",
                        "Stream ID",
                        "Task ID",
                        "Task Start Time(us)",
                        "vec_time(us)",
                    )
                )
                for i in reversed(range(38)):
                    b.writerow(("proj_fc", f.OP, 8, 1, 4, 100 * i, 1000 if i not in range(7, 37) else 5))
                for i in range(7, 37):
                    a.writerow(("fc", i, i, 4, 1, "proj_fc", f.OP, 100 * i))
            got = pipeline_times(root, {"op_summary": f.c.identity(summary)})
            self.assertEqual(got["joined"], 30)
            self.assertEqual(got["medians"]["vec_time(us)"], 5)

    @unittest.skipUnless(
        os.environ.get("DFLASH_GE_DESCRIPTOR") and importlib.util.find_spec("torch"),
        "CPU GE frontend and descriptor",
    )
    def test_actual_masked_feature_graph_bitwise(self):
        import torch
        from ge_frontend import Graph

        types = protobuf_classes(Path(os.environ["DFLASH_GE_DESCRIPTOR"]).read_bytes())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for m in f.GEARS:
                graph = Graph(types, "mask")
                features = graph.data("features", torch.float16, (1, m, 12800), 0)
                rows = graph.data("valid_rows", torch.int16, (1,), 1)
                mask = torch.arange(m) < rows.to(torch.int64)
                x = torch.where(mask[None, :, None], features, torch.zeros_like(features)).reshape(m, 12800)
                a = np.random.default_rng(42).normal(size=(m, 12800)).astype("<f2")
                a[0, 0] = np.float16(-0.0)
                a.tofile(root / "features")
                files = {"features": f.c.identity(root / "features")}
                for count in (m, m - 1):
                    np.array([count], "<i2").tofile(root / "valid")
                    files["valid_rows"] = f.c.identity(root / "valid")
                    result, valid = replay_input(graph.graph, x.name, files, m, 12800)
                    self.assertEqual(valid, count)
                    wanted = a.copy()
                    wanted[count:] = 0
                    np.testing.assert_array_equal(wanted.view("<u2"), result.view("<u2"))
