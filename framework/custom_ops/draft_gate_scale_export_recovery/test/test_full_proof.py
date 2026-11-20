import copy
from contextlib import ExitStack
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from archive import c
from graph_contract import compare_graphs
from full_proof import compare_parts
from recover_full import STAGES, DEVICE, new_stage, run
from types import SimpleNamespace


def attr(k, v):
    return {"key": k, "value": v}


def model(root, tag, alias, label, image):
    root.joinpath(tag).mkdir()
    g = {
        "name": alias,
        "type": "Const",
        "attr": [
            attr("value", {"t": {"data": "01110", "desc": {"shape": {"dim": [1, 4]}, "dtype": "DT_FLOAT16"}}})
        ],
    }
    fmt = [attr("origin_format_for_int", {"i": v}) for v in ((0, 2) if tag == "raw" else (2, 0))]
    op = {
        "name": "stable",
        "type": "Add",
        "input": ["x:0", alias + ":0"],
        "input_desc": [{"attr": fmt}],
        "attr": [
            attr("fusion_failed", {"i": label}),
            attr("_kernel_bin_id", {"s": "prefix" + image["sha256"]}),
            attr("_tbeKernelSize", {"i": image["bytes"]}),
            attr("_kernelname", {"s": image["symbol"]}),
        ],
    }
    nodes = [
        g,
        op,
        {"name": "x", "type": "Data", "output_desc": [{"dtype": "DT_HALF", "shape": {"dim": [1, 4]}}]},
    ]
    p = root / tag / "graph.json"
    p.write_text(json.dumps({"op": nodes}))
    return {"dump": c.identity(p), "opp_root": str(root / tag)}, nodes


class GraphContract(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.d = Path(tmp.name).resolve()
        self.i = {"symbol": "entry", "bytes": 100, "sha256": "f" * 64}
        self.j = dict(self.i, bytes=90, sha256="e" * 64)
        self.a, self.left = model(self.d, "raw", "dynamic_const_123_55", 29, self.i)
        self.b, self.right = model(self.d, "new", "dynamic_const_678_88", 13, self.j)

    def check(self):
        return compare_graphs(self.a, self.b, {"stable": self.i}, {"stable": self.j})

    def save(self, model, rows):
        p = Path(model["dump"]["path"])
        p.write_text(json.dumps({"op": rows}))
        model["dump"] = c.identity(p)

    def test_protected_name_payload_and_scope(self):
        report = self.check()
        self.assertEqual(report["node_count"], 3)
        self.assertEqual(report["fusion_scope_id_changes"], {"stable": [29, 13]})
        self.assertEqual(report["generated_constants"], {"dynamic_const_678_88": "dynamic_const_123_55"})
        self.assertEqual(set(report["const_records"]), {"dynamic_const_123_55"})

    def test_no_wildcards_for_real_shape_and_dtype_or_actual_bits(self):
        for tag in ("shape", "dtype", "input", "data", "custom", "fusion", "prefix", "size", "duplicate"):
            rows = copy.deepcopy(self.right)
            if tag == "shape":
                rows[2]["output_desc"][0]["shape"]["dim"] = [-1]
            if tag == "dtype":
                rows[2]["output_desc"][0]["dtype"] = "DT_FLOAT"
            if tag == "input":
                rows[1]["input"].reverse()
            if tag == "data":
                rows[0]["attr"][0]["value"]["t"]["data"] = "01111"
            if tag == "custom":
                rows[1]["attr"].append(attr("new_flag", {"i": 123}))
            if tag == "fusion":
                rows[1]["attr"][0]["value"] = {"i": -1}
            if tag == "prefix":
                rows[1]["attr"][1]["value"]["s"] = "newprefix" + self.j["sha256"]
            if tag == "size":
                rows[1]["attr"][2]["value"]["i"] = 99
            if tag == "duplicate":
                rows[1]["input_desc"][0]["attr"].pop()
            self.save(self.b, rows)
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                self.check()

    def test_cannot_merge_or_split_failed_fusion_scopes(self):
        for m, ns, label in ((self.a, self.left, 29), (self.b, self.right, 17)):
            ns.append(dict(name="second", type="Add", attr=[attr("fusion_failed", {"i": label})]))
            self.save(m, ns)
        with self.assertRaisesRegex(ValueError, "membership"):
            self.check()

    def test_literal_constant_string_is_not_node_reference(self):
        for m, ns in ((self.a, self.left), (self.b, self.right)):
            ns[0]["attr"][0]["value"]["t"]["data"] = ns[0]["name"]
            self.save(m, ns)
        with self.assertRaisesRegex(ValueError, "data"):
            self.check()


class Admission(unittest.TestCase):
    def test_immutable_task_io_and_weight_partitions(self):
        left = {k: (128, str(k)) for k in (8, 0, 1, 3, 2)}
        for k in left:
            right = dict(left)
            right[k] = (128, "changed")
            with self.subTest(k=k), patch("full_proof.part_data", side_effect=[left, right]):
                if k in (1, 2, 8):
                    with self.assertRaises(ValueError):
                        compare_parts({}, {})
                else:
                    self.assertFalse(compare_parts({}, {})[str(k)]["exact_equal"])

    def test_explicit_stages_and_no_archived_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            with new_stage(root, "admit-c16") as out:
                self.assertTrue(out.is_dir())
            with self.assertRaises(ValueError):
                with new_stage(root, "admit-c16"):
                    self.fail("reran archive")
            for s in STAGES:
                with self.subTest(s=s), self.assertRaisesRegex(ValueError, "allow-device"):
                    run(SimpleNamespace(stage=s, allow_device=s not in DEVICE, root=root))

    def test_offline_admit_reuses_C16_and_never_calls_compile_or_runner(self):
        import recover_full as flow

        ctx = {}
        chain = {}
        failed = {"models": {"16": {"om": "actual"}}, "packages": {"16": {"opp": "bound"}}}
        for g in ("16", "64"):
            chain[f"c{g}-benchmark"] = {"folder": "/archive/" + g, "statistics": {"g": g}}
        with ExitStack() as s:
            s.enter_context(patch.object(flow.prior, "build_witness"))
            rt = s.enter_context(
                patch.object(
                    flow,
                    "runtime",
                    side_effect=lambda p, c, h, r, g, t, launch: (
                        r["statistics"] if not launch else self.fail("NPU")
                    ),
                )
            )
            s.enter_context(patch.object(flow, "compile_model", side_effect=AssertionError("compiled C16")))
            s.enter_context(patch.object(flow, "check_full", return_value={"cpu": "pass"}))
            out = {}
            flow.perform("admit-c16", Path("/unused"), ctx, chain, out, failed)
            self.assertEqual(out["model"], failed["models"]["16"])
            self.assertEqual(rt.call_count, 2)
