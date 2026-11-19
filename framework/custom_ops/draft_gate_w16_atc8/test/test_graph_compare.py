"""Real mode-1 JSON shape; values and permutations from the O34Eaf terminal diagnostic."""

import copy
import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from w16_common import FIRST, GATE, SERIAL, c
from w16_graph import protected_graph


def attr(key, value):
    return {"key": key, "value": value}


def descriptor(shape, layout="ND"):
    return dict(
        dtype="DT_FLOAT16",
        layout=layout,
        shape={"dim": shape},
        attr=[
            attr("origin_shape_range", None),
            attr("origin_format_for_int", {"i": 2}),
            attr("origin_shape", {"list": {"i": shape[:], "val_type": "VT_LIST_INT"}}),
            attr("origin_format_for_int", {"i": 2}),
            attr("origin_shape_initialized", {"b": True}),
            attr("origin_shape_initialized", {"b": False}),
        ],
    )


class CompiledGraphTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.raw, self.new = self.base / "raw/opp", self.base / "double-w-serial/opp"
        for root in (self.raw, self.new):
            (root / "vendors/customize").mkdir(parents=True)
        self.roots = dict(raw_opp_root=self.raw, new_opp_root=self.new)
        x = descriptor([16, 2560])
        w = descriptor([80, 1216, 16, 32], "FRACTAL_NZ")
        w["dtype"] = "DT_INT8"
        w["attr"].extend([attr("format_for_int", {"i": 29}), attr("format_for_int", {"i": 100663325})])
        s = descriptor([20, 19456])
        y = descriptor([16, 19456])
        constant = [attr("value", {"t": {"desc": descriptor([20, 19456])}}), attr("index", {"i": 1})]
        raw_attrs = [
            attr(
                "_kernel_bin_id",
                {
                    "s": "te_dflashgroupquantlinear_9b135bb69ac8f58a0685b3299b1a9a0b6da9ba59c1245d2945be4d8475895cb473e570d30e465b81aa27842f77302b1bc18aaf30a1c2794db46def5ba2a04846"
                },
            ),
            attr("_tbeKernelSize", {"i": 19608}),
            attr("_kernelname", {"s": "bound_symbol"}),
            attr("custom_op_file_path", {"s": str(self.raw / "vendors/customize")}),
            attr("tvm_blockdim", {"i": 8}),
        ]
        self.old = [
            dict(name="arg2_1", type="Data", input_desc=[x]),
            dict(name="arg1_1", type="Const", attr=constant),
            dict(name="DFlashGroupQuantLinear_ac_nz", type="Const", output_desc=[copy.deepcopy(w)]),
            dict(
                name=GATE,
                type=GATE,
                input=["arg2_1:0", "DFlashGroupQuantLinear_ac_nz:0", "arg1_1:0"],
                input_desc=[copy.deepcopy(x), w, s],
                output_desc=[copy.deepcopy(y)],
                attr=raw_attrs,
            ),
            dict(name="NetOutput", type="NetOutput", input=[GATE + ":0"], input_desc=[y]),
        ]
        # Move different keys and identical duplicates. Swap the distinct duplicate
        # flag pair at Gate inputs 0/2 and NetOutput, and the distinct NZ codes at input 1.
        self.candidate = copy.deepcopy(self.old)
        for node in self.candidate:
            for d in node.get("input_desc", []) + node.get("output_desc", []):
                d["attr"].sort(key=lambda r: r["key"])
        for d in [self.candidate[3]["input_desc"][i] for i in (0, 2)] + self.candidate[4]["input_desc"]:
            self.swap_pair(d, "origin_shape_initialized")
        self.swap_pair(self.candidate[3]["input_desc"][1], "format_for_int")
        self.candidate[1]["attr"].reverse()
        self.candidate[1]["attr"][1]["value"]["t"]["desc"]["attr"].sort(key=lambda row: row["key"])
        attrs = self.candidate[3]["attr"]
        attrs[0]["value"]["s"] = (
            "te_dflashgroupquantlinear_9b135bb69ac8f58a0685b3299b1a9a0b6da9ba59c1245d2945be4d8475895cb45e5d700f327aee41b23b4b34128d7360b2c75134358b1310357f8ce7a9d110f9"
        )
        attrs[1]["value"]["i"] = 32304
        attrs[3]["value"]["s"] = str(self.new / "vendors/customize")
        attrs.reverse()

    @staticmethod
    def swap_pair(desc, key):
        i, j = [i for i, r in enumerate(desc["attr"]) if r["key"] == key]
        desc["attr"][i], desc["attr"][j] = desc["attr"][j], desc["attr"][i]

    def compare(self):
        files = []
        for name, nodes in (("raw.json", self.old), ("serial.json", self.candidate)):
            file = self.base / name
            file.write_text(json.dumps({"graph": [{"op": nodes}]}))
            files.append(file)
        return protected_graph(*files, {GATE}, **self.roots)

    def test_reported_five_node_comparison(self):
        self.assertEqual(self.compare(), 4)

    def test_duplicate_multiplicity_missing_value_and_typed_values(self):
        saved = copy.deepcopy(self.candidate)
        for change in (
            "missing",
            "extra",
            "true_as_one",
            "null_versus_missing",
            "NZ_changed",
            "both_NZ_same",
            "extra_member",
        ):
            self.candidate = copy.deepcopy(saved)
            x = self.candidate[3]["input_desc"][0]["attr"]
            flags = [r for r in x if r["key"] == "origin_shape_initialized"]
            nz = [r for r in self.candidate[3]["input_desc"][1]["attr"] if r["key"] == "format_for_int"]
            if change == "missing":
                x.remove(flags[0])
            elif change == "extra":
                x.append(copy.deepcopy(flags[0]))
            elif change == "true_as_one":
                flags[1]["value"]["b"] = 1
            elif change == "null_versus_missing":
                next(r for r in x if r["key"] == "origin_shape_range").pop("value")
            elif change == "NZ_changed":
                nz[0]["value"]["i"] = 100663326
            elif change == "both_NZ_same":
                nz[0]["value"]["i"] = nz[1]["value"]["i"]
            else:
                flags[0]["extra"] = False
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "input_desc"):
                self.compare()

    def test_ordered_fields_stay_ordered(self):
        saved = copy.deepcopy(self.candidate)
        for change in (
            "shape",
            "attr_list",
            "inputs",
            "descriptor_slots",
            "outputs",
            "dtype",
            "layout",
            "protected_launch",
        ):
            self.candidate = copy.deepcopy(saved)
            node = self.candidate[3]
            if change == "shape":
                node["input_desc"][0]["shape"]["dim"].reverse()
            elif change == "attr_list":
                next(r for r in node["input_desc"][0]["attr"] if r["key"] == "origin_shape")["value"]["list"][
                    "i"
                ].reverse()
            elif change == "inputs":
                node["input"].reverse()
            elif change == "descriptor_slots":
                node["input_desc"].reverse()
            elif change == "outputs":
                node["output_desc"].clear()
            elif change == "dtype":
                node["input_desc"][1]["dtype"] = "DT_FLOAT16"
            elif change == "layout":
                node["input_desc"][1]["layout"] = "ND"
            else:
                next(r for r in node["attr"] if r["key"] == "tvm_blockdim")["value"]["i"] = 7
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, GATE):
                self.compare()

    def test_unknown_conflicting_duplicates_and_wrong_nz_slot(self):
        old, new = self.old[3], self.candidate[3]
        pair = [attr("new_field", {"i": 3}), attr("new_field", {"i": 7})]
        old["attr"] += pair
        new["attr"] += copy.deepcopy(pair)
        self.assertEqual(self.compare(), 4)
        new["attr"][-2:] = reversed(new["attr"][-2:])
        with self.assertRaisesRegex(ValueError, "new_field"):
            self.compare()
        new["attr"][-2:] = copy.deepcopy(pair)
        wrong = [attr("format_for_int", {"i": 29}), attr("format_for_int", {"i": 100663325})]
        old["input_desc"][0]["attr"] += wrong
        new["input_desc"][0]["attr"] += wrong[::-1]
        with self.assertRaisesRegex(ValueError, "format_for_int"):
            self.compare()

    def test_opp_roots_independently_bound(self):
        old_prop = next(r for r in self.old[3]["attr"] if r["key"] == "custom_op_file_path")
        new_prop = next(r for r in self.candidate[3]["attr"] if r["key"] == "custom_op_file_path")
        for val in (
            str(self.raw / "vendors/customize"),
            str(self.new / "vendors/other"),
            "vendors/customize",
            "/usr/local/Ascend/cann-9.0.0/opp/vendors/customize",
            str(self.new) + "/vendors/../vendors/customize",
        ):
            new_prop["value"]["s"] = val
            with self.subTest(path=val), self.assertRaisesRegex(ValueError, "custom_op_file_path"):
                self.compare()
        # A literal match in both dumps is still rejected when either private binding is wrong.
        old_prop["value"]["s"] = new_prop["value"]["s"] = str(self.new / "vendors/customize")
        with self.assertRaisesRegex(ValueError, "custom_op_file_path"):
            self.compare()
        old_prop["value"]["s"] = str(self.raw / "vendors/customize")
        self.candidate[3]["attr"].append(copy.deepcopy(new_prop))
        with self.assertRaisesRegex(ValueError, "duplicate custom_op_file_path"):
            self.compare()
        self.candidate[3]["attr"].remove(new_prop)
        vendor = self.new / "vendors/customize"
        vendor.rmdir()
        vendor.symlink_to(self.raw / "vendors/customize", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "redirected"):
            self.compare()

    def test_full_draft_non_target_groupquant_and_head(self):
        old = copy.deepcopy(self.old[3])
        old["name"] = "protected_down"
        new = copy.deepcopy(old)
        next(r for r in new["attr"] if r["key"] == "custom_op_file_path")["value"]["s"] = str(
            self.new / "vendors/customize"
        )
        self.old.append(old)
        self.candidate.append(new)
        self.assertEqual(self.compare(), 5)
        size = next(r for r in new["attr"] if r["key"] == "_tbeKernelSize")
        size["value"]["i"] += 1
        with self.assertRaisesRegex(ValueError, "protected_down.*_tbeKernelSize"):
            self.compare()
        size["value"]["i"] -= 1
        old["type"] = new["type"] = "MatMul"
        with self.assertRaisesRegex(ValueError, "MatMul.*custom_op_file_path"):
            self.compare()

    def test_read_only_recheck_sealed_dumps_and_both_embedded_kernels(self):
        from c_profile_graph import read_om_json
        from gate_plan import embedded, node_binding
        from recheck_graph import recheck
        from test_machine import elf

        def put(path, data):
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(data, bytes):
                path.write_bytes(data)
            else:
                path.write_text(json.dumps(data))
            return c.identity(path)

        root = self.base / "experiment"
        stage = root / "atc"
        case = stage / SERIAL / FIRST
        history = self.base / "history"
        prefix = "te_dflashgroupquantlinear_" + "a" * 64
        symbol = prefix + "__kernel0"

        def artifacts(install, folder, nodes, code):
            blob = elf(code + symbol.encode())
            sha = hashlib.sha256(blob).hexdigest()
            vendor = install / "vendors/customize"
            obj = put(vendor / "op_impl/ai_core/tbe/kernel/ascend310p/fake.o", blob)
            meta = put(Path(obj["path"]).with_suffix(".json"), {"sha256": sha})
            opp = dict(
                install=str(install),
                preflight=dict(
                    vendor=str(vendor),
                    files=[meta],
                    kernels=[
                        dict(binary=obj["path"], sha256=sha, json=meta["path"], json_sha256=meta["sha256"])
                    ],
                ),
            )
            gate = next(n for n in nodes if n["name"] == GATE)
            attrs = {x["key"]: x for x in gate["attr"]}
            for key, val in (
                ("_kernel_bin_id", {"s": prefix + sha}),
                ("_kernelname", {"s": symbol}),
                ("_tbeKernelSize", {"i": len(blob)}),
            ):
                attrs[key]["value"] = val
            gate["attr"] += [
                attr("_ge_attr_op_kernel_lib_name", {"s": "AIcoreEngine"}),
                attr("_op_no_tiling", {"b": False}),
            ]
            dump = put(folder / "om-graph.json", {"graph": [{"op": nodes}]})
            om = put(folder / "compiled/artifacts/om/fixture.om", b"fixture" + blob)
            return opp, dump, om

        raw, old_dump, old_om = artifacts(self.raw, history / "raw", self.old, b"old")
        serial, new_dump, new_om = artifacts(self.new, case, self.candidate, b"serial")
        n = next(n for n in read_om_json(old_dump["path"]) if n["name"] == GATE)
        air = put(history / "frozen.air", b"fixture")
        model = dict(
            opp_root=raw["install"],
            gate_nodes={FIRST: GATE},
            atc_kernels=embedded(c.verify(old_om), [node_binding(n)]),
            om=old_om,
            dump=old_dump,
            air=air,
        )
        old_build = put(history / "build/report.json", dict(variants={"raw": dict(opp=raw)}))
        export = put(
            history / "export/report.json",
            dict(status="GATE8_EXPORT_PASS", models={"raw": {FIRST: model}}, predecessors={"build": old_build}),
        )
        base = dict(root=str(root), error=None, integrity_error=None, sources=[])
        source = put(
            root / "source/report.json",
            dict(
                base,
                status="W16_SOURCE_PASS",
                phase="source",
                predecessors={},
                binding=dict(history=dict(export=export, build=old_build)),
            ),
        )
        build = put(
            root / "build/report.json",
            dict(
                base,
                status="W16_BUILD_PASS",
                phase="build",
                predecessors={"source": source},
                variants={SERIAL: dict(opp=serial)},
            ),
        )
        req = put(case / "request.json", dict(opp=serial, frozen=dict(air=air), output=str(case / "compiled")))
        child = put(
            case / "compiled/report.json",
            dict(status="GATE8_PROJECTION_AIR_COMPILED_NOT_EXECUTED", om=new_om, execute_calls=0),
        )
        failed = put(
            stage / "report.json",
            dict(
                base,
                phase="atc",
                folder=str(stage),
                error="protected compiled node/Head changed: arg2_1",
                status="BLOCKED",
                predecessors=dict(source=source, build=build),
                device_execution="NOT_RUN",
                benchmark="NOT_RUN",
                active_case=[FIRST, SERIAL],
                failure_files=[new_dump, req, child, new_om],
            ),
        )
        result = recheck(root)
        self.assertEqual(result["status"], "ARCHIVED_W16_GRAPH_MATCH")
        self.assertEqual(
            (result["protected_nodes"], result["archived_atc_status"], result["device_execution"]),
            (4, "BLOCKED", "NOT_RUN"),
        )
        self.assertEqual(c.identity(stage / "report.json"), failed)
        Path(new_dump["path"]).write_text("{}")
        with self.assertRaisesRegex(ValueError, "file identity changed"):
            recheck(root)
