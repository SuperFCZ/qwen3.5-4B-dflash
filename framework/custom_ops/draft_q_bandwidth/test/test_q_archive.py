import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import q_panel as q
from q_source import extract
from air import protobuf_classes


@unittest.skipUnless(
    os.environ.get("DFLASH_GE_DESCRIPTOR"), "set DFLASH_GE_DESCRIPTOR to the local CANN ge_ir descriptor"
)
class TestPrefix(unittest.TestCase):
    def test_device_prefix_uses_real_rows_and_untouched_projection_bytes(self):
        types = protobuf_classes(Path(os.environ["DFLASH_GE_DESCRIPTOR"]).read_bytes())
        m, k, n = q.SHAPE
        rg = np.random.default_rng(42)
        codes = rg.integers(-128, 128, size=(k // 32, n // 16, 16, 32), dtype=np.int8).tobytes()
        gn = rg.uniform(0.01, 0.02, (k // 128, n)).astype("<f2").tobytes()
        original = q.model_proto(types, codes, gn, projection_shape=q.SHAPE)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out = root / "out"
            out.mkdir()
            rows = (rg.integers(-48, 48, size=(64, k)) / 16).astype("<f2")
            rows.tofile(root / "embed")
            model = types[0]()
            model.name = "fixture"
            g = model.graph.add(name="fixture")

            def node(name, t, dt, shape, args=()):
                op = g.op.add(name=name, type=t, id=len(g.op))
                op.input.extend(x + ":0" for x in args)
                desc = op.output_desc.add(name="y", dtype=dt, layout="ND")
                desc.shape.dim.extend(shape)
                return op

            def const(name, array, dt):
                op = node(name, "Const", dt, array.shape)
                op.attr["value"].t.desc.CopyFrom(op.output_desc[0])
                op.attr["value"].t.data = array.tobytes()

            node("anchor", "Data", 8, [1])
            const("to2", np.array([1, 1], "<i4"), 7)
            node("anchor2", "Reshape", 8, [1, 1], ["anchor", "to2"])
            const("mask", np.full((1, 15), 12, "<i8"), 8)
            op = node("ids", "ConcatD", 8, [1, 16], ["anchor2", "mask"])
            op.attr["concat_dim"].i = 1
            tab = node("table", "FileConstant", 2, rows.shape)
            tab.attr["dtype"].dt = 1
            tab.attr["file_path"].s = str(root / "embed").encode()
            const("axis", np.array(0, "<i4"), 7)
            op = node("lookup", "GatherV2", 2, [1, m, k], ["table", "ids", "axis"])
            op.attr["batch_dims"].i = 0
            const("toQ", np.array([m, k], "<i4"), 7)
            node("x", "Reshape", 2, [m, k], ["lookup", "toQ"])
            for op in original.graph[0].op:
                if op.type in ("Const", q.OLD_OP):
                    g.op.add().CopyFrom(op)
            node("irrelevant", "Abs", 2, [m, k], ["x"])
            meta = {
                "projections": {
                    q.ROLE: dict(
                        m=m,
                        k=k,
                        n=n,
                        node=q.OLD_OP,
                        weight="weight",
                        scale="scale",
                        nz_sha256=q.sha(codes),
                        gn_sha256=q.sha(gn),
                    )
                }
            }
            r = extract(out, types, model, meta, q.c.identity(root / "embed"), np.array([42], "<i8"))
            self.assertEqual(r["selected_token_ids"], [[42] + [12] * 15])
            self.assertEqual(q.c.verify(r["embedding"]).read_bytes(), rows[[42] + [12] * 15].tobytes())
            self.assertEqual(len(set(r["stress_token_ids"][0])), 16)
            msg = types[0].FromString(q.c.verify(r["reference_air"]).read_bytes())
            graph = msg.graph[0]
            ops = {o.name: o for o in graph.op}
            self.assertEqual(list(graph.input), ["lookup:0"])
            self.assertEqual(list(graph.output), ["x:0", q.OLD_OP + ":0"])
            self.assertEqual(ops["lookup"].type, "Data")
            self.assertNotIn("table", ops)
            self.assertNotIn("irrelevant", ops)
            self.assertEqual(ops["weight"].attr["value"].t.data, codes)
            self.assertEqual(ops["scale"].attr["value"].t.data, gn)
            bad = dict(meta["projections"][q.ROLE], nz_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "different checkpoint bytes"):
                extract(
                    out,
                    types,
                    model,
                    {"projections": {q.ROLE: bad}},
                    q.c.identity(root / "embed"),
                    np.array([42], "<i8"),
                )

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
        "requires the CPU model frontend",
    )
    def test_prefix_matches_original_layer0_frontend(self):
        import torch
        from model_frontend import ExportOps, ExternalEmbedding, TorchDFlashOps, ExportLinear
        from ge_frontend import Graph

        sys.path.insert(0, str(q.OPS / "draft_full_panels/test"))
        from test_full import execute

        torch.manual_seed(412)
        torch.set_num_threads(2)
        types = protobuf_classes(Path(os.environ["DFLASH_GE_DESCRIPTOR"]).read_bytes())
        with tempfile.TemporaryDirectory() as root:
            d = Path(root)
            table = d / "table.fp16"
            torch.randn(64, q.K).half().numpy().tofile(table)
            w = torch.randint(-127, 128, (q.N, q.K), dtype=torch.int8)
            s = torch.rand(q.N, q.K // 128).half() * 0.02
            g = Graph(types, "original_q")
            ops = ExportOps()
            ops.g = g
            anchor = g.data("anchor", torch.int64, (1,), 0)
            ids = torch.cat((anchor.reshape(1, 1), torch.full((1, 15), 21, dtype=torch.long)), 1)
            emb = ExternalEmbedding(ops)
            emb.weight = g.external(table, (64, q.K), "target_embedding")
            x = TorchDFlashOps.rms_norm(ops, emb(ids) * 1.0, torch.rand(q.K).half(), 1e-6)
            y = ExportLinear(ops, q.ROLE, w, s)(x)
            g.outputs((x.reshape(q.M, q.K), y.reshape(q.M, q.N)), ("x", "y"))
            result = extract(
                d, types, g.model, {"projections": g.projections}, q.c.identity(table), np.array([42], "<i8")
            )
            from types import SimpleNamespace

            recovered = SimpleNamespace(
                graph=types[0].FromString(q.c.verify(result["reference_air"]).read_bytes()).graph[0]
            )
            with torch.inference_mode():
                expected = execute(g, [torch.tensor([42])])
                observed = execute(
                    recovered,
                    [
                        torch.from_numpy(
                            np.fromfile(q.c.verify(result["embedding"]), "<f2").reshape(1, q.M, q.K)
                        )
                    ],
                )
            self.assertEqual(len(observed), 2)
            for a, b in zip(expected, observed):
                self.assertTrue(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))


if __name__ == "__main__":
    unittest.main()
