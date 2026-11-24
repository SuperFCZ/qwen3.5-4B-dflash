import copy
import importlib.util
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import kv_panel as kv
from run_kv import select, numeric_rejection
from layout import once
from o_source import extract
from air import protobuf_classes

sys.path.insert(0, str(kv.OPS / "draft_q_bandwidth/test"))
from test_q import q_cpu_headers


class TestKV(unittest.TestCase):
    def test_kernel_all_outputs_and_staging(self):
        k, n = kv.K, kv.N
        rng = np.random.default_rng(2048310)
        weights = rng.integers(-128, 128, (n, k), dtype=np.int8)
        gn = rng.uniform(0.006, 0.03, (k // 128, n)).astype("<f2")
        nz = weights.reshape(n // 16, 16, k // 32, 32).transpose(2, 0, 1, 3).tobytes()
        w, s, _ = kv.gate.pack_cube_panels(nz, gn.tobytes(), k, k=k, n=n)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            inc = q_cpu_headers(d)
            h = inc / "kernel_operator.h"
            hs = once(
                h.read_text(),
                "sizeof(T)==2 && params.blockCount==80;",
                "sizeof(T)==2 && params.blockCount==80 && params.blockLen==8;",
            )
            hs = once(
                hs,
                "sizeof(T)==2 && params.blockCount==160",
                "sizeof(T)==2 && (params.blockCount==80 || params.blockCount==160)",
            )
            h.write_text(
                once(hs, "params.dstStride==15);", "(params.dstStride==31 || params.dstStride==79));")
            )
            mat = inc / "lib/matmul_intf.h"
            v = once(
                mat.read_text(),
                "tiling_.singleM==16 && tiling_.orgKa==2560",
                "(tiling_.singleM==32 || tiling_.singleM==80) && (tiling_.orgKa==1280 || tiling_.orgKa==2560)",
            )
            v = once(
                v,
                "tiling_.singleN == 64);",
                "(tiling_.singleN==64 || tiling_.singleN==128 || tiling_.singleN==256));",
            )
            mat.write_text(
                once(v, "globalIndex % tiling_.orgN / 64", "globalIndex % tiling_.orgN / tiling_.singleN")
            )
            executables = []
            for mode in (*kv.CANDIDATES, *kv.WIDE):
                (d / (kv.STEM + ".cpp")).write_text(kv.device_source(mode))
                (d / "kv_spec.h").write_bytes(kv.header(mode))
                # Quoted includes must resolve against the same generated header as device packaging.
                (d / "kv_activation.h").write_bytes((HERE / "kv_activation.h").read_bytes())
                exe = d / mode
                cc = [
                    "c++",
                    "-std=c++17",
                    "-ffp-contract=off",
                    "-O2",
                    "-I" + str(inc),
                    "-I" + str(d),
                    "-I" + str(kv.gate.HERE),
                    str(HERE / "test/portable_kv.cpp"),
                    "-o",
                    str(exe),
                ]
                subprocess.run(cc, check=True)
                executables.append(exe)
            for m in (32, 80):
                x = np.zeros((m, k), "<f2")
                acc = np.zeros((m, n), np.float32)
                for row in range(m):
                    for group in range(k // 128):
                        col = group * 128 + (11 * row + 5 * group) % 128
                        x[row, col] = (-1 if (row + group) % 2 else 1) * (1 + row / 83)
                        acc[row] += np.float32(x[row, col]) * (
                            weights[:, col].astype(np.float16) * gn[group]
                        ).astype(np.float32)
                x.tofile(d / "nd")
                x.reshape(m, k // 16, 16).transpose(1, 0, 2).tofile(d / "packed")
                acc.astype("<f2").tofile(d / "gold")
                for exe in executables:
                    if exe.name in kv.WIDE:
                        width = kv.WIDE[exe.name]
                        codes, scales = kv.packed_panel.packed_weights(nz, gn.tobytes(), n, k, width, 1280)
                    else:
                        codes, scales = w, s
                    (d / "w").write_bytes(codes)
                    (d / "s").write_bytes(scales)
                    for fail in (-1, 0, 1 if exe.name in kv.WIDE else 3):
                        subprocess.run([exe, d, str(m), str(fail)], check=True)

    def test_resource_contract_and_floor(self):
        for gear, values, one_pass in (
            (16, (256, 10, 10, 10, 10, 983040), 27.44),
            (64, (160, 4, 2, 8, 4, 1024000), 29.60),
        ):
            m, k, n = kv.shape(gear)
            bk, sa, sb, da, db, l1 = values
            raw = bytearray(224)
            struct.pack_into("<4IQ", raw, 0, m, k, n, k, 46592)
            fields = {f["name"]: 24 + f["offset"] for f in kv.schema()}
            for key, value in dict(
                singleCoreM=m,
                singleCoreN=64,
                singleCoreK=k,
                baseM=m,
                baseN=64,
                baseK=bk,
                stepKa=sa,
                stepKb=sb,
                depthA1=da,
                depthB1=db,
                dbL0A=2,
                dbL0B=2,
                dbL0C=1,
                transLength=2 * m * 64,
            ).items():
                struct.pack_into("<I", raw, fields[key], value)
            p = dict(
                availableCores=8,
                blockDim=8,
                coreLimit=0,
                workspace=2097152,
                capacity=dict(UB=262144, L1=1048576, L0A=65536, L0B=65536, L0C=262144),
            )
            resources = dict(
                raw_banks=1,
                codes_bytes=40960,
                scale_bytes=2560,
                weight_bytes=163840,
                user_ub=215552,
                scratch=46592,
                tile_k=k,
                workspace=p["workspace"],
                l1_used=l1,
                l0a_used=4 * m * bk,
                l0b_used=256 * bk,
                l0c_used=256 * m,
            )
            q = dict(
                platform=p,
                tiling_sha256=kv.sha(raw),
                schedule=dict(
                    tiling_hex=raw.hex(),
                    resources=resources,
                    baseline_assignments=[list(range(i, 32, 8)) for i in range(8)],
                ),
            )
            for mode in kv.CANDIDATES:
                self.assertEqual(kv.plan(q, q, gear, mode)["iterates_per_n64"], 1)
            wrong = copy.deepcopy(q)
            wrong["schedule"]["resources"]["l1_used"] -= 32
            with self.assertRaisesRegex(ValueError, "allocation"):
                kv.plan(wrong, q, gear, kv.DIRECT)
            self.assertAlmostEqual(kv.logical_bytes(gear) / 204800, one_pass)

    def test_wide_sdk_resources_and_coverage(self):
        # Results of the host SDK SetShape / SetOrgShape / GetTiling for both public geometries.
        cases = ((16, 128, 901120), (64, 128, 1007616), (16, 256, 950272), (64, 256, 1028096))
        for gear, width, l1 in cases:
            m, k, n = kv.shape(gear)
            mode = next(v for v in kv.WIDE if kv.WIDE[v] == width)
            bk, sa, sb, da, db = kv.WIDE_SDK[m, width]
            sc = 262144 - (5 * 20480 + 20 * width + 8192)
            raw = bytearray(224)
            struct.pack_into("<4IQ", raw, 0, m, k, n, 1280, sc)
            offsets = {f["name"]: 24 + f["offset"] for f in kv.schema()}
            for name, value in dict(
                singleCoreM=m,
                singleCoreN=width,
                singleCoreK=1280,
                baseM=m,
                baseN=width,
                baseK=bk,
                stepKa=sa,
                stepKb=sb,
                depthA1=da,
                depthB1=db,
                dbL0A=2,
                dbL0B=2,
                dbL0C=1,
                transLength=2 * m * width,
            ).items():
                struct.pack_into("<I", raw, offsets[name], value)
            p = dict(
                blockDim=8,
                availableCores=8,
                coreLimit=0,
                workspace=2097152,
                capacity=dict(UB=262144, L1=1048576, L0A=65536, L0B=65536, L0C=262144),
            )
            resources = dict(
                user_ub=262144 - sc,
                scratch=sc,
                workspace=p["workspace"],
                tile_k=1280,
                raw_banks=1,
                codes_bytes=20480,
                weight_bytes=81920,
                scale_bytes=20 * width,
                l1_used=l1,
                l0a_used=4 * m * bk,
                l0b_used=4 * width * bk,
                l0c_used=4 * m * width,
            )
            sched = dict(
                tiling_hex=raw.hex(),
                resources=resources,
                baseline_assignments=[list(range(i, n // width, 8)) for i in range(8)],
            )
            query = dict(platform=p, schedule=sched, tiling_sha256=kv.sha(raw))
            self.assertEqual(kv.plan(query, query, gear, mode)["iterates_per_tile"], 2)
            sched["baseline_assignments"][-1].clear()
            with self.assertRaisesRegex(ValueError, "ownership"):
                kv.plan(query, query, gear, mode)

    def test_deployment_selection_requires_direct_pairs(self):
        outcomes = {
            (kv.RAW, kv.DIRECT): "GO",
            (kv.RAW, kv.STAGED): "NO_GO",
            (kv.RAW, kv.PACKED): "GO",
            (kv.DIRECT, kv.PACKED): "NO_GO",
        }

        def measure(a, b):
            return {"decision": outcomes[a, b]}

        self.assertEqual(select(kv.CANDIDATES, measure), kv.DIRECT)
        outcomes[kv.DIRECT, kv.PACKED] = "GO"
        self.assertEqual(select(kv.CANDIDATES, measure), kv.PACKED)
        self.assertEqual(select(kv.CANDIDATES, lambda a, b: {"decision": "NO_GO"}), kv.RAW)
        self.assertEqual(select(tuple(kv.WIDE), lambda a, b: {"decision": "NO_GO"}, kv.DIRECT), kv.DIRECT)

    def test_quarantine_distinguishes_device_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "measured"
            d.mkdir()
            kv.c.write(d / "mismatches.json", {"y-actual-0.bin": {"count": 10, "first_index": 32}})
            log = d / "runner.log"
            log.write_text("FAIL: input modified\n")
            self.assertIsNone(numeric_rejection(d.parent))
            log.write_text("FAIL: frozen Golden mismatch: y\n")
            self.assertEqual(numeric_rejection(d.parent)["case"], "measured")

    @unittest.skipUnless(os.environ.get("DFLASH_GE_DESCRIPTOR"), "CANN GE descriptor")
    def test_public_graph_input_and_offline_weights(self):
        types = protobuf_classes(Path(os.environ["DFLASH_GE_DESCRIPTOR"]).read_bytes())
        m, k, n = 32, 256, 64
        rng = np.random.default_rng(822)
        nz = rng.integers(-128, 128, (n, k), dtype=np.int8).tobytes()
        gn = rng.uniform(0.01, 0.5, (k // 128, n)).astype("<f2").tobytes()
        w, s, _ = kv.gate.pack_cube_panels(nz, gn, k, k=k, n=n)
        for mode in kv.CANDIDATES:
            proto = kv.model_proto(types, nz, gn, projection_shape=(m, k, n))
            kv.packed_panel.add_pack_graph(
                proto,
                64,
                k,
                w,
                s,
                m=m,
                k=k,
                n=n,
                op_type=kv.OP,
                prefix="kv_a",
                pack_activation=mode == kv.PACKED,
            )
            tab = {node.name: node for node in proto.graph[0].op}
            self.assertEqual(list(tab["x"].output_desc[0].shape.dim), [m, k])
            self.assertEqual(list(tab["weight"].output_desc[0].shape.dim), [1, 1, k * 64])
            self.assertEqual(tab["weight"].attr["value"].t.data, w)
            self.assertEqual(tab["scale"].attr["value"].t.data, s)
            op = tab[kv.OLD_OP]
            self.assertEqual(op.type, kv.OP)
            self.assertEqual(op.input[0], "kv_a_pack:0" if mode == kv.PACKED else "x:0")
            self.assertEqual(
                list(op.input_desc[0].shape.dim), [k // 16, m, 16] if mode == kv.PACKED else [m, k]
            )
            self.assertEqual("kv_a_pack" in tab, mode == kv.PACKED)

    @unittest.skipUnless(
        os.environ.get("DFLASH_GE_DESCRIPTOR")
        and importlib.util.find_spec("torch")
        and importlib.util.find_spec("transformers"),
        "CPU frontend and CANN GE descriptor",
    )
    def test_replay_both_row_groups_and_cache(self):
        import torch

        sys.path.insert(0, str(kv.OPS / "draft_full_panels/test"))
        from test_full import fixture, execute, fe

        types = protobuf_classes(Path(os.environ["DFLASH_GE_DESCRIPTOR"]).read_bytes())
        torch.manual_seed(31032)
        torch.set_num_threads(1)
        draft, ops = fixture()
        net = fe.export_net(draft, ops)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            tables = {}
            for name in ("embedding", "head"):
                file = d / name
                torch.randn(net.config.vocab_size, net.config.hidden_size).half().numpy().tofile(file)
                tables[name] = {"fp16": kv.c.identity(file)}
            for gear in (16, 64):
                folder = d / str(gear)
                folder.mkdir()
                graph, abi = fe.trace_graph(types, net, ops, {"target": tables}, gear, 96)
                original_inputs = {}
                vals = []
                scalars = dict(anchor=42, start_position=4, valid_rows=gear - 1, proposal_count=15)
                for inp in abi["inputs"]:
                    name = inp["name"]
                    v = (
                        torch.tensor([scalars[name]], dtype=getattr(torch, inp["dtype"]))
                        if name in scalars
                        else torch.randn(inp["shape"]).half()
                    )
                    v.numpy().tofile(folder / name)
                    original_inputs[name] = kv.c.identity(folder / name)
                    vals.append(v)
                row = graph.projections[kv.ROLE]
                data = extract(
                    folder,
                    types,
                    graph.model,
                    dict(abi=abi, projections=graph.projections, modules=graph.owners),
                    tables["embedding"]["fp16"],
                    dict(files=original_inputs, scalars=scalars),
                    shape=tuple(row[x] for x in ("m", "k", "n")),
                    role=kv.ROLE,
                    roles=("fc", kv.ROLE),
                )
                end = graph.graph.op[-1]
                edges = list(graph.graph.output[1:3])
                del end.input[:]
                end.input.extend([data["x_node"] + ":0", data["kv_node"] + ":0", *edges])
                expected = execute(graph, vals)
                prefix = types[0].FromString(kv.c.verify(data["reference_air"]).read_bytes()).graph[0]
                inputs = [
                    torch.from_numpy(
                        np.fromfile(kv.c.verify(data["replays"]["measured"][t["name"]]), t["dtype"]).reshape(
                            t["shape"]
                        )
                    )
                    for t in data["reference_abi"]["inputs"]
                ]
                actual = execute(SimpleNamespace(graph=prefix), inputs)
                self.assertEqual(len(actual), 4)
                for a, b in zip(actual, expected):
                    self.assertTrue(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))
                self.assertEqual(
                    {op.name for op in prefix.op if op.type == kv.OLD_OP}, {"proj_fc", "proj_layer0_kv"}
                )


if __name__ == "__main__":
    unittest.main()
