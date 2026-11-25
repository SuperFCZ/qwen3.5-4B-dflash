import copy
import importlib.util
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from recipes import CONTROL, OPS, down, q, c, validate, candidates
from atc_transfer import make_air, plan
from run_transfer import choose, fresh_pair, boundaries
from gate_input import cut_reference, single_graph
from air import protobuf_classes


class Contracts(unittest.TestCase):
    @unittest.skipUnless(os.getenv("DFLASH_GE_DESCRIPTOR"), "CANN GE descriptor")
    def test_public_nd_through_optional_activation_pack(self):
        import numpy as np

        m, k, n = 16, 256, 256
        op = CONTROL["kv64"].spec.op
        types = protobuf_classes(Path(os.environ["DFLASH_GE_DESCRIPTOR"]).read_bytes())
        rng = np.random.default_rng(310)
        nk = rng.integers(-127, 128, size=(n, k), dtype=np.int8)
        nz = nk.reshape(n // 16, 16, k // 32, 32).transpose(2, 0, 1, 3).tobytes()
        gn = rng.standard_normal((k // 128, n)).astype("<f2").tobytes()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            files = {}
            for name, data in (("w_nz.bin", nz), ("s_gn.bin", gn)):
                f = folder / name
                f.write_bytes(data)
                files[name] = c.identity(f)
            for paid in (False, True):
                r = SimpleNamespace(
                    shape=(m, k, n),
                    width=128,
                    panel=128,
                    role="kv",
                    gear=64,
                    name="test",
                    activation="packed" if paid else "row",
                    layout=(128, 128, paid),
                    spec=SimpleNamespace(op=op),
                )
                site = folder / r.activation
                site.mkdir()
                air, dims, (w, s) = make_air(
                    site, types, r, SimpleNamespace(layout=(64, k, False)), {}, files
                )
                graph = types[0].FromString(air.read_bytes()).graph[0]
                nodes = {v.name: v for v in graph.op}
                kv = nodes["proj_layer0_kv"]
                self.assertEqual(kv.type, op)
                self.assertEqual(list(kv.input[1:]), ["weight:0", "scale:0"])
                self.assertEqual(list(graph.output), ["proj_layer0_kv:0"])
                self.assertEqual(list(nodes["NetOutput"].input), list(graph.output))
                self.assertEqual(tuple(nodes["x"].output_desc[0].shape.dim), (m, k))
                self.assertEqual(tuple(kv.input_desc[0].shape.dim), (k // 16, m, 16) if paid else (m, k))
                self.assertEqual(sum(v.type == "Transpose" for v in graph.op), int(paid))
                self.assertEqual(tuple(nodes["weight"].output_desc[0].shape.dim), dims[0])
                self.assertEqual(
                    (nodes["weight"].attr["value"].t.data, nodes["scale"].attr["value"].t.data), (w, s)
                )
                # Same-layout transfers must consume the unchanged AIR, including the original producer.
                alias = site / "reuse"
                alias.mkdir()
                linked, _, _ = make_air(alias, types, r, r, {"air": c.identity(air)}, files)
                self.assertTrue(linked.samefile(air))

    def test_registry_and_boundaries(self):
        names = set()
        for key, b in CONTROL.items():
            choices = candidates(key)
            edges = boundaries(b.shape, [b, *choices])
            self.assertLessEqual(len(edges), b.shape[0])
            for r in choices:
                validate(r)
                self.assertNotIn(r.name, names)
                names.add(r.name)
                for cut in (r.base_k, r.chunk, r.panel):
                    if cut < b.shape[1]:
                        self.assertIn((cut - 1, cut), edges)
        self.assertEqual(len(names), 24)

    def test_actual_resources_and_same_cube(self):
        b = CONTROL["o"]
        r = candidates("o")[0]

        def query(r):
            raw = bytearray(224)
            m, k, n = r.shape
            bk = r.base_k
            w = r.width
            scratch = 262144 - r.ub
            struct.pack_into("<4IQ", raw, 0, m, k, n, r.panel, scratch)
            offsets = {f["name"]: 24 + f["offset"] for f in q.schema()}
            for key, v in dict(
                baseM=m,
                baseN=w,
                baseK=bk,
                singleCoreM=m,
                singleCoreN=w,
                singleCoreK=r.panel,
                stepKa=16,
                stepKb=1,
                depthA1=16,
                depthB1=2,
                dbL0A=2,
                dbL0B=2,
                dbL0C=1,
                transLength=2048,
            ).items():
                struct.pack_into("<I", raw, offsets[key], v)
            p = dict(
                blockDim=8,
                availableCores=8,
                coreLimit=0,
                workspace=2097152,
                capacity=dict(UB=262144, L1=1048576, L0A=65536, L0B=65536, L0C=262144),
            )
            res = dict(
                tile_k=r.panel,
                raw_banks=1,
                user_ub=r.ub,
                scratch=scratch,
                codes_bytes=32768 * r.packet,
                scale_bytes=4096,
                weight_bytes=131072,
                l1_used=851968,
                l0a_used=16384,
                l0b_used=65536,
                l0c_used=4096,
                workspace=p["workspace"],
            )
            s = dict(
                resources=res,
                tiling_hex=raw.hex(),
                baseline_assignments=[list(range(i, 40, 8)) for i in range(8)],
            )
            return dict(platform=p, schedule=s, tiling_sha256=q.sha(raw))

        original = query(b)
        good = query(r)
        self.assertEqual(plan(good, original, r, b)["resources"]["scratch"], 53248)
        bad = copy.deepcopy(good)
        raw = bytearray.fromhex(bad["schedule"]["tiling_hex"])
        field = next(f for f in q.schema() if f["name"] == "stepKa")
        struct.pack_into("<I", raw, 24 + field["offset"], 8)
        bad["schedule"]["tiling_hex"] = raw.hex()
        bad["tiling_sha256"] = q.sha(raw)
        with self.assertRaisesRegex(ValueError, "Cube tiling"):
            plan(bad, original, r, b)

    def test_pair_and_promotion(self):
        def launch(d, mode, files, timed, i):
            samples = [1.0] * 30 if mode == "inc" else [0.89] * 30
            return dict(
                folder=str(d),
                mode=mode,
                round=i,
                profiled=False,
                launch_authorized=True,
                bit_mismatches=0,
                validation=dict(timing=dict(samples_ms=samples)),
            )

        with tempfile.TemporaryDirectory() as tmp:
            result = fresh_pair(Path(tmp), "inc", "next", launch, {}, CONTROL["gate"].shape, True)
        self.assertEqual([x["mode"] for x in result["runs"]], ["next", "inc", "inc", "next", "next", "inc"])
        self.assertEqual(result["statistics"]["variants"]["next"]["count"], 90)
        seen = []

        def pair(a, b):
            seen.append((a, b))
            return {"decision": "NO_GO" if (a, b) == ("a", "c") or b == "b" else "GO"}

        self.assertEqual(choose("inc", ["a", "b", "c"], pair), "a")
        self.assertEqual(seen, [("inc", "a"), ("inc", "b"), ("inc", "c"), ("a", "c")])

    @unittest.skipUnless(
        os.getenv("DFLASH_GE_DESCRIPTOR")
        and importlib.util.find_spec("torch")
        and importlib.util.find_spec("transformers"),
        "CPU frontend + CANN GE descriptor",
    )
    def test_down_uses_original_swiglu_and_same_packed_control(self):
        import torch

        sys.path.insert(0, str(OPS / "draft_full_panels/test"))
        from test_full import fixture, execute, fe

        torch.manual_seed(3109728)
        torch.set_num_threads(1)
        net, ops = fixture()
        net = fe.export_net(net, ops)
        types = protobuf_classes(Path(os.environ["DFLASH_GE_DESCRIPTOR"]).read_bytes())
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            tab = {}
            for name in ("embedding", "head"):
                f = d / name
                torch.randn(net.config.vocab_size, net.config.hidden_size).half().numpy().tofile(f)
                tab[name] = dict(fp16=c.identity(f))
            G, abi = fe.trace_graph(types, net, ops, dict(target=tab), 16, 96)
            graph = G.model.graph[0]
            row = G.projections["layer0.down"]
            shape = tuple(row[key] for key in ("m", "k", "n"))
            m, k, n = shape
            gate = G.projections["layer0.gate_up"]["node"]
            src = cut_reference(
                d,
                types,
                graph,
                dict(modules=G.owners),
                row,
                {gate: ("gate_y", [m, 2 * k])},
                shape,
                "layer0.down",
                "layer0.swiglu",
                "Mul",
                "down_reference",
            )
            small = types[0].FromString(c.verify(src["reference_air"]).read_bytes()).graph[0]
            self.assertEqual([a.name for a in small.op if a.type == "Data"], ["gate_y"])
            end = graph.op[-1]
            del end.input[:]
            end.input.extend([a + ":0" for a in (gate, src["x_node"], row["node"])])
            scalars = dict(anchor=42, start_position=5, valid_rows=15, proposal_count=15)
            xs = [
                (
                    torch.tensor([scalars[t["name"]]], dtype=getattr(torch, t["dtype"]))
                    if t["name"] in scalars
                    else torch.randn(t["shape"]).half()
                )
                for t in abi["inputs"]
            ]
            gy, x, y = execute(G, xs)
            rx, ry = execute(SimpleNamespace(graph=small), [gy])
            for v, w in ((x, rx), (y, ry)):
                self.assertTrue(torch.equal(v.view(torch.uint8), w.view(torch.uint8)))
            proj, codes, scales = (G.ops[row[key]] for key in ("node", "weight", "scale"))
            fe.pack_projection(proj, codes, scales, row, down.OP, 128)
            a, _ = single_graph(
                d, types, G.model, row, src["weights"], shape=shape, op_type=down.OP, panel=128, name="down"
            )
            sole = types[0].FromString(c.verify(a).read_bytes()).graph[0]
            self.assertEqual([x.type for x in sole.op], ["Data", "Const", "Const", down.OP, "NetOutput"])
            (result,) = execute(SimpleNamespace(graph=sole), [rx])
            self.assertTrue(torch.equal(result.view(torch.uint8), y.view(torch.uint8)))


if __name__ == "__main__":
    unittest.main()
