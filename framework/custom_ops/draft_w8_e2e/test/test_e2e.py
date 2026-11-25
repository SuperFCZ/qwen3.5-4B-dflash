import copy
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from winners import c, selected, UNION, SELECT, OPS
from history import TARGET_SHA, write_candidate_plan, parse_plan
from features import insert_selector, slots
from measure import compare_generation, DOCS
from packed_panel import nd_desc
from air import protobuf_classes


def plan_text():
    out = ["qwen35-dflash-chunk-v4", "capacity 2048 248320", "draft_prefill_policy static_draft16_64_oms"]

    def ts(mark, name, dtype, shape):
        return " ".join(map(str, (mark, name, dtype, len(shape), *shape)))

    state = [(f"d{i}_{t}", "float16", [1, 8, 2112, 128]) for i in range(5) for t in ("key", "value")]
    feature = ("features", "float16", [1, 64, 28160])
    for name in ("target_prefill", "target_decode", "target_verify", "draft"):
        out.append(f'graph {name} "/gone/{name}.om" ' + TARGET_SHA.get(name, "0" * 64))
        if name == "draft":
            out.append('static_gear64 "/gone/draft64.om" ' + "1" * 64)
            out += [
                ts("I", *v)
                for v in [
                    feature,
                    ("start_position", "int64", [1]),
                    ("valid_rows", "int16", [1]),
                    ("anchor", "int64", [1]),
                    ("proposal_count", "int16", [1]),
                    *state,
                ]
            ]
            out += [ts("O", *v) for v in [("draft_top1", "int64", [1, 15]), *state]]
        else:
            out.extend([ts("I", "ids", "int64", [1, 16]), ts("O", *feature)])
        out += ["end"]
    return "\n".join(out + ["done", ""])


class Integrated(unittest.TestCase):
    def test_all_winners(self):
        for gear in (16, 64):
            s = selected(gear)
            self.assertEqual(len(s), 6)
            self.assertEqual(
                (s["gate_up"].mode, s["down"].mode, s["q"].columns),
                ("panel-k2560-w2-next-n", "down-next-w2", 128),
            )
            self.assertEqual(s["fc"].mode, "fc64-raw2" if gear == 64 else "fc-a1-k2560-next")
            self.assertEqual(s["kv"].m, gear + 16)
        self.assertEqual(slots(UNION, SELECT), (0, 2, 5, 8, 10))

    def test_candidate_plan_excludes_ordinary_and_checks_abi(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            old = d / "prior"
            old.write_text(plan_text())
            g = parse_plan(old)
            h = dict(plan=c.identity(old))

            def file(name):
                p = d / name
                p.write_text(name)
                return c.identity(p)

            ts = {
                k: dict(om=file(k), abi=v["abi"], target_layers=list(UNION))
                for k, v in g.items()
                if k in TARGET_SHA
            }
            qs = {gear: dict(om=file(str(gear)), abi=copy.deepcopy(g["draft"]["abi"])) for gear in (16, 64)}
            qs[16]["abi"]["inputs"][0]["shape"][1] = 16
            rp = write_candidate_plan(d / "new", h, ts, qs)
            new = c.verify(rp).read_text()
            self.assertNotIn("graph target_decode", new)
            self.assertNotIn("/gone", new)
            self.assertEqual(new.count("graph "), 3)
            self.assertIn("static_gear64", new)
            ts["target_verify"]["target_layers"] = list(reversed(UNION))
            with self.assertRaisesRegex(ValueError, "layer order"):
                write_candidate_plan(d / "bad", h, ts, qs)
            old.write_text(plan_text().replace("capacity 2048", "capacity 512"))
            with self.assertRaisesRegex(ValueError, "2048"):
                parse_plan(old)

    @unittest.skipUnless(os.getenv("DFLASH_GE_DESCRIPTOR"), "CANN GE descriptor")
    def test_public_features_and_bitwise_replay(self):
        types = protobuf_classes(Path(os.environ["DFLASH_GE_DESCRIPTOR"]).read_bytes())
        for gear in (16, 64):
            msg = types[0]()
            g = msg.graph.add(name="draft")
            d = g.op.add(name="features", type="Data")
            d.attr["index"].i = 0
            nd_desc(d.output_desc.add(), "y", 2, [1, gear, 12])
            d.input_desc.add().CopyFrom(d.output_desc[0])
            g.input.append("features:0")
            end = g.op.add(type="NetOutput", name="NetOutput")
            end.input.append("features:0")
            end.input_desc.add().CopyFrom(d.output_desc[0])
            abi = {"inputs": [dict(name="features", dtype="float16", shape=[1, gear, 12])]}
            a, _ = insert_selector(g, abi, gear, 4, (1, 3, 6, 8, 11), (1, 6, 11))
            self.assertEqual(a["inputs"][0]["shape"], [1, gear, 20])
            self.assertEqual(abi["inputs"][0]["shape"], [1, gear, 12])
            x = np.arange(gear * 20, dtype="<u2").reshape(1, gear, 20)
            table = {}
            for i, n in enumerate(g.op):
                self.assertEqual(n.id, i)
                v = [table[e.split(":")[0]] for e in n.input]
                if n.type == "Data":
                    z = x
                elif n.type == "Const":
                    z = np.frombuffer(n.attr["value"].t.data, "<i4")
                elif n.type == "Slice":
                    z = v[0][tuple(slice(l, l + r) for l, r in zip(v[1], v[2]))]
                elif n.type == "ConcatD":
                    z = np.concatenate(v, axis=n.attr["concat_dim"].i)
                else:
                    z = v[0]
                table[n.name] = z
            np.testing.assert_array_equal(
                z, x.reshape(1, gear, 5, 4)[:, :, [0, 2, 4], :].reshape(1, gear, 12)
            )

    def test_comparison_tracks_tokens_stops_and_rounds(self):
        a = {
            "generated_token_ids": [11, 22],
            "stop_reason": "max_new_tokens",
            "rounds": [{"proposed_token_ids": [22]}],
        }
        b = copy.deepcopy(a)
        b["rounds"] = []
        j = compare_generation(
            {"measurements": [a, a, a]}, {"measurements": [a, b, dict(b, generated_token_ids=[11, 3])]}
        )
        self.assertEqual(
            (j["matching_repetitions"], j["round_traces_matched"], j["differences"][0]["first_index"]),
            (2, 1, 1),
        )
        self.assertEqual((DOCS["all"]["decode_ms"], DOCS["all"]["gen_tok_s"]), (3193.72, 29.16))


if __name__ == "__main__":
    unittest.main()
