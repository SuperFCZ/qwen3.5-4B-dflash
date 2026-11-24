from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from selection import OLD_OP, OPS, SINGLE, c, packed, selections, sites
from rewrite import digest, payload, transform
from compile_all import actual_sites, locate
from breakdown import aggregate, W8_TOTAL
from air import protobuf_classes


class CompleteDraft(unittest.TestCase):
    def test_selected_modes(self):
        for gear in (16, 64):
            s = selections(gear)
            self.assertEqual(len(sites(s)), 26)
            self.assertEqual(s["kv"].m, gear + 16)
            self.assertEqual(s["fc"].m, gear)
            self.assertEqual(s["gate_up"].mode, "panel-k2560-w2")
            self.assertEqual(s["down"].mode, "l1-k4864-w1")
            self.assertEqual(s["q"].mode, "q-k2560-w2-next-n")
            self.assertEqual(s["o"].mode, "o-k4096-v2048-next-n")
            self.assertEqual(s["fc"].pack_x, gear == 64)
            self.assertEqual(s["kv"].columns, 64 if gear == 16 else 128)
            self.assertEqual(s["fc"].columns, 64 if gear == 16 else 160)
            for row in SINGLE[gear].values():
                for pin in (row.result, row.om, row.text):
                    self.assertEqual(len(bytes.fromhex(pin)), 32)

    def test_const_probe_validates_all_bytes_and_wraps(self):
        b = bytes(range(250)) * 3
        wrong = b[:-1] + b"!"
        item = payload(b)
        self.assertEqual(locate(wrong + b, item, cursor=len(b) + 50), len(b))
        with self.assertRaisesRegex(ValueError, "constant payload"):
            locate(wrong, item)

    def test_projection_inventory_and_module_coverage(self):
        s = sites(selections(64))
        inv = {k: dict(node=k + "_export", op=v.op) for k, v in s.items()}
        graph = dict(inventory=inv)
        nodes = [
            {
                "name": k + "_device",
                "type": v.op,
                "attributes": [
                    {"key": "_datadump_original_op_names", "value": {"list": {"s": [inv[k]["node"]]}}}
                ],
            }
            for k, v in s.items()
        ]
        bound, _ = actual_sites(nodes, graph)
        actual = {k: dict(node=r["name"], op=r["type"]) for k, r in bound.items()}
        with self.assertRaisesRegex(ValueError, "unbound"):
            actual_sites(nodes + [dict(name="fallback", type=OLD_OP, attributes=[])], graph)
        rows = []
        for i in range(7, 37):
            for role, node in actual.items():
                rows.append(
                    dict(idx=len(rows), call=i, module=role, op=node["node"], start="100.10", end="102.10")
                )
            rows.append(
                dict(idx=len(rows), call=i, module="NOTIFY_WAIT", op="fc_device", start="100", end="800")
            )
            rows.append(
                dict(
                    idx=len(rows),
                    call=i,
                    module="Fused/layer0.k_norm,layer1.q_norm",
                    op="f",
                    start="102",
                    end="106",
                )
            )
        parts, calls = aggregate(rows, actual, with_fc_layout=True)
        groups = {r["component"]: r for r in parts}
        self.assertEqual(groups[W8_TOTAL]["median_ms"], 0.052)
        self.assertEqual(groups["Q"]["tasks"], 150)
        self.assertEqual(groups["Norm"]["median_ms"], 0.004)
        self.assertEqual(groups["NOTIFY_WAIT"]["median_ms"], 0.7)
        self.assertIsNone(groups["FC activation layout"]["median_ms"])
        rows.pop(0)
        rest, _ = aggregate(rows, actual)
        self.assertEqual(next(x for x in rest if x["component"] == W8_TOTAL)["complete_calls"], 29)
        self.assertIsNone(next(x for x in rest if x["component"] == W8_TOTAL)["p95_ms"])

    @unittest.skipUnless(
        os.getenv("DFLASH_GE_DESCRIPTOR")
        and importlib.util.find_spec("torch")
        and importlib.util.find_spec("transformers"),
        "CPU model frontend and CANN GE descriptor",
    )
    def test_all_26_integrated_in_full_five_layer_graph(self):
        import torch

        sys.path.insert(0, str(OPS / "draft_full_panels/test"))
        from test_full import fixture, execute, fe

        torch.manual_seed(31026)
        torch.set_num_threads(1)
        types = protobuf_classes(Path(os.environ["DFLASH_GE_DESCRIPTOR"]).read_bytes())
        net, ops = fixture()
        net = fe.export_net(net, ops)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            ext = {}
            for kind in ("embedding", "head"):
                file = d / kind
                torch.randn(net.config.vocab_size, net.config.hidden_size).half().numpy().tofile(file)
                ext[kind] = c.identity(file)
            for gear in (16, 64):
                graph, abi = fe.trace_graph(
                    types, net, ops, dict(target={k: dict(fp16=v) for k, v in ext.items()}), gear, 96
                )
                inputs = []
                for t in abi["inputs"]:
                    nums = dict(anchor=42, start_position=4, valid_rows=gear - 2, proposal_count=15)
                    inputs.append(
                        torch.tensor([nums[t["name"]]], dtype=getattr(torch, t["dtype"]))
                        if t["name"] in nums
                        else torch.randn(t["shape"]).half()
                    )
                raw = execute(graph, inputs)
                specs = {}
                assets = {}
                old = []
                for kind, prod in selections(gear).items():
                    name = "fc" if kind == "fc" else "layer0." + kind
                    row = graph.projections[name]
                    m, k, n = (row[a] for a in ("m", "k", "n"))
                    s = replace(prod, m=m, k=k, n=n, columns=(32 if prod.pack_x else 64), panel=128)
                    specs[kind] = s
                    w, sc = (graph.ops[row[a]].attr["value"].t.data for a in ("weight", "scale"))
                    w, sc = packed.packed_weights(w, sc, n, k, s.columns, s.panel)
                    assets[kind] = dict(layer0_constants=[digest(w), digest(sc)])
                for role, row in graph.projections.items():
                    if role.endswith((".gate_up", ".down")):
                        node, w, sc = (graph.ops[row[a]] for a in ("node", "weight", "scale"))
                        s = specs[role.split(".")[-1]]
                        dw, ds = fe.pack_projection(node, w, sc, row, s.op, s.panel)
                        old.append(
                            dict(role=role, type=s.op, panel_k=s.panel, packed_sha256=dw, scale_sha256=ds)
                        )
                ctrl = execute(graph, inputs)
                index = dict(projections=graph.projections, modules=graph.owners, replacements=old, abi=abi)
                info = transform(graph.model, index, specs, assets, ext)
                candidate = execute(
                    SimpleNamespace(graph=graph.model.graph[0]),
                    inputs,
                    panel_types=tuple(s.op for s in specs.values()),
                )
                self.assertEqual((len(info["inventory"]), len(info["changed_nodes"])), (26, 48))
                self.assertEqual(len(info["layout_nodes"]), 4 if gear == 64 else 0)
                self.assertFalse(any(op.type == OLD_OP for op in graph.model.graph[0].op))
                self.assertEqual(len(candidate), 11)
                for a, b, v in zip(raw, ctrl, candidate):
                    self.assertTrue(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))
                    self.assertTrue(torch.equal(a.view(torch.uint8), v.view(torch.uint8)))


if __name__ == "__main__":
    unittest.main()
