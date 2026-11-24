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
import o_panel as o
from layout import once
from o_runtime import abi_guard
from o_source import extract
from air import protobuf_classes

sys.path.insert(0, str(o.q.HERE / "test"))
from test_q import q_cpu_headers


class TestO(unittest.TestCase):
    def test_actual_schedule_and_banks(self):
        m, k, n = o.SHAPE
        rng = np.random.default_rng(40962560)
        nk = rng.integers(-128, 128, (n, k), dtype=np.int8)
        gn = rng.uniform(0.008, 0.025, (k // 128, n)).astype("<f2")
        nz = nk.reshape(n // 16, 16, k // 32, 32).transpose(2, 0, 1, 3).tobytes()
        codes, scales, _ = o.q.gate.pack_cube_panels(nz, gn.tobytes(), k, k=k, n=n)
        x = np.zeros((m, k), "<f2")
        acc = np.zeros((m, n), np.float32)
        for i in range(m):
            for group in range(k // 128):
                kk = 128 * group + (127 if group == 15 else 0 if group == 16 else (13 * i + 19 * group) % 128)
                x[i, kk] = (-1 if (group + i) % 2 else 1) * (1 + i / 31)
                acc[i] += np.float32(x[i, kk]) * (gn[group] * nk[:, kk].astype(np.float16)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            inc = q_cpu_headers(d)
            for file, a, b in (
                (inc / "kernel_operator.h", "params.blockCount==160", "params.blockCount==256"),
                (inc / "lib/matmul_intf.h", "tiling_.orgKa==2560", "tiling_.orgKa==4096"),
            ):
                file.write_text(once(file.read_text(), a, b))
            (d / "codes").write_bytes(codes)
            (d / "scales").write_bytes(scales)
            x.tofile(d / "x")
            acc.astype("<f2").tofile(d / "expected")
            impl = d / (o.STEM + ".cpp")
            impl.write_text(o.device_source())
            exe = d / "test"
            cc = [
                "c++",
                "-std=c++17",
                "-O2",
                "-ffp-contract=off",
                "-I" + str(inc),
                "-I" + str(d),
                "-I" + str(HERE),
                "-I" + str(o.q.gate.HERE),
                str(HERE / "test/portable_o.cpp"),
                "-o",
                str(exe),
            ]
            subprocess.run(cc, check=True)
            for flags in ([], ["0"], ["1"]):
                subprocess.run([exe, d, *flags], check=True)
            for anchor, message in (
                ("if (stripe + 1 < Pieces) Pair<HardEvent::MTE3_V>(weightStored);", "W16 overwritten"),
                ("Pair<HardEvent::V_MTE2>(rawConsumed);", "raw overwrite"),
            ):
                impl.write_text(once(o.device_source(), anchor, "// omitted dependency in CPU fault test"))
                subprocess.run(cc, check=True)
                bad = subprocess.run([exe, d], capture_output=True, text=True, cwd=d)
                self.assertNotEqual(bad.returncode, 0)
                self.assertIn(message, bad.stderr)

    def test_full_k_plan(self):
        r = dict(
            raw_banks=1,
            tile_k=4096,
            codes_bytes=32768,
            scale_bytes=4096,
            weight_bytes=131072,
            user_ub=176128,
            scratch=86016,
            workspace=2097152,
            l1_used=851968,
            l0a_used=16384,
            l0b_used=65536,
            l0c_used=4096,
        )
        offsets = {f["name"]: 24 + f["offset"] for f in o.schema()}
        raw = bytearray(224)
        struct.pack_into("<4IQ", raw, 0, *o.SHAPE, 4096, 86016)
        cube = dict(
            singleCoreM=16,
            singleCoreN=64,
            singleCoreK=4096,
            baseM=16,
            baseN=64,
            baseK=256,
            stepKa=16,
            stepKb=1,
            depthA1=16,
            depthB1=2,
            dbL0A=2,
            dbL0B=2,
            dbL0C=1,
            transLength=2048,
        )
        for k, v in cube.items():
            struct.pack_into("<I", raw, offsets[k], v)
        p = dict(
            availableCores=8,
            blockDim=8,
            coreLimit=0,
            workspace=2097152,
            capacity=dict(UB=262144, L1=1048576, L0A=65536, L0B=65536, L0C=262144),
        )
        query = dict(
            platform=p,
            tiling_sha256=o.q.sha(raw),
            schedule=dict(
                tiling_hex=raw.hex(),
                resources=r,
                baseline_assignments=[list(range(i, 40, 8)) for i in range(8)],
            ),
        )
        self.assertEqual(o.plan(query, o.PANEL, query)["iterates_per_n64"], 1)
        bad = copy.deepcopy(query)
        bad["schedule"]["resources"]["l1_used"] = 655360
        with self.assertRaisesRegex(ValueError, "resource drift"):
            o.plan(bad, o.PANEL, query)
        self.assertEqual(o.BYTES, 10862592)
        self.assertAlmostEqual(o.FLOOR_MS, 0.05304)
        self.assertIn("{16,4096}", abi_guard(o.ABI))
        with self.assertRaisesRegex(ValueError, "runner ABI"):
            abi_guard(dict(inputs=[dict(name='bad"name', shape=[1], dtype="float16")], outputs=[]))

    @unittest.skipUnless(
        os.environ.get("DFLASH_GE_DESCRIPTOR")
        and importlib.util.find_spec("torch")
        and importlib.util.find_spec("transformers"),
        "CPU frontend and CANN GE descriptor",
    )
    def test_attention_cut_matches_uncut_five_layer_forward(self):
        import torch
        from types import SimpleNamespace

        sys.path.insert(0, str(o.OPS / "draft_full_panels/test"))
        from test_full import fixture, execute, fe

        types = protobuf_classes(Path(os.environ["DFLASH_GE_DESCRIPTOR"]).read_bytes())
        torch.manual_seed(1556)
        torch.set_num_threads(1)
        net, ops = fixture()
        net = fe.export_net(net, ops)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            target = {}
            for name in ("embedding", "head"):
                f = d / name
                torch.randn(net.config.vocab_size, net.config.hidden_size).half().numpy().tofile(f)
                target[name] = {"fp16": o.c.identity(f)}
            graph, abi = fe.trace_graph(types, net, ops, {"target": target}, 16, 96)
            files = {}
            args = []
            scalars = dict(anchor=42, start_position=7, valid_rows=15, proposal_count=12)
            for t in abi["inputs"]:
                n = t["name"]
                value = (
                    torch.tensor([scalars[n]], dtype=getattr(torch, t["dtype"]))
                    if n in scalars
                    else torch.randn(t["shape"]).half()
                )
                path = d / (n + ".bin")
                value.numpy().tofile(path)
                args.append(value)
                files[n] = o.c.identity(path)
            out = d / "prefix"
            out.mkdir()
            rows = graph.projections["layer0.o"]
            src = extract(
                out,
                types,
                graph.model,
                dict(abi=abi, projections=graph.projections, modules=graph.owners),
                target["embedding"]["fp16"],
                dict(files=files, scalars=scalars),
                shape=tuple(rows[k] for k in ("m", "k", "n")),
            )
            prefix = types[0].FromString(o.c.verify(src["reference_air"]).read_bytes()).graph[0]
            # Tap the uncut original rather than computing the O input through the prefix under test.
            end = graph.graph.op[-1]
            kv = list(graph.graph.output[1:3])
            del end.input[:]
            end.input.extend([src["x_node"] + ":0", src["o_node"] + ":0", *kv])
            original = execute(graph, args)
            spec = src["reference_abi"]
            xx = [
                torch.from_numpy(
                    np.fromfile(o.c.verify(src["replays"]["measured"][t["name"]]), t["dtype"]).reshape(
                        t["shape"]
                    )
                )
                for t in spec["inputs"]
            ]
            got = execute(SimpleNamespace(graph=prefix), xx)
            self.assertEqual(len(got), 4)
            for a, b in zip(original, got):
                self.assertTrue(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))
            self.assertEqual(
                {n.name for n in prefix.op if n.type == o.OLD_OP}, {r["node"] for r in src["sites"].values()}
            )
            self.assertFalse(any(n.type == "FileConstant" for n in prefix.op))


if __name__ == "__main__":
    unittest.main()
