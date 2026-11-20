"""CPU contracts for Gate expansion, source dispatch, whole-OM comparison and manual gates."""

import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import run_expansion as driver
import evidence
from evidence import CONTROL, CANDIDATE, FIRST, GATE, GEARS, LAYERS, NODES, SHAPE_KEY, c, full_abi
from dispatch_body import record_scale_compile, select_scale_source
from expansion_models import image_contract
from gate_common import case_contract
from measurement import verify_outputs
from om_constants import weight_partition, compare_constants, inline_constants, KINDS
from pairs import five_decision, groups, layer_statistics
from routing import verify_dispatch, route_rules, wrapper_bytes

RAW = "abc" * 12
GATE_KEY = "te_gate"
OTHER_KEY = "te_down"
TAG = "16x9728x2560"


def put(path, data):
    path.write_bytes(data if isinstance(data, bytes) else json.dumps(data).encode())
    return c.identity(path)


def timing_rows(g, k, raw=2.0, new=1.92):
    return [
        dict(
            gear=g,
            layer=k,
            round=i,
            mode=m,
            folder=f"/{g}/{k}/{i}/{m}",
            bit_mismatches=0,
            profiled=False,
            launch_authorized=True,
            atc_plan=dict(block_dim=8),
            validation={"timing": {"samples_ms": [raw if m == CONTROL else new] * 30}},
        )
        for i, m in evidence.pilot.schedule(True)
    ]


def gear_stats(gear, new=1.92):
    return {
        k: layer_statistics(
            timing_rows(gear, k, new=new),
            f"C{gear}/{k}",
            "FROZEN_PILOT" if (gear, k) == ("16", FIRST) else "FRESH",
        )
        for k in LAYERS
    }


def fake_cases():
    cases = {}
    for g in GEARS:
        cases[g] = {}
        for k in LAYERS:

            def item(n, b, h):
                return dict(path=f"/case/{g}/{k}/{n}", bytes=b, sha256=hashlib.sha256(h.encode()).hexdigest())

            inputs = {
                n: item(n, b, (g if n == "x.bin" else "") + k + n)
                for n, b in (("x.bin", 81920), ("w_nz.bin", 49807360), ("s_gn.bin", 778240))
            }
            cases[g][k] = dict(
                name=k, shape=[16, 2560, 19456], inputs=inputs, golden=item("y", 622592, g + k + "y")
            )
    return cases


def imod(parts):
    contents = [parts.get(i, bytes([i]) * 3) for i in KINDS]
    head = b"IMOD" + struct.pack("<I", 256) + bytes.fromhex("0000002000000000") + bytes(240)
    table = [struct.pack("<Q", len(KINDS))]
    offset = 0
    for kind, data in zip(KINDS, contents):
        table.append(struct.pack("<QQQ", kind, offset, len(data)))
        offset += len(data)
    return head + b"".join(table + contents)


def const_node(value):
    return dict(
        type="Const",
        name="w",
        attr=[
            dict(key="value", value={"t": {"data": value}}),
            dict(key="axes", value={"list": {"i": [2, 3]}}),
        ],
    )


class ExperimentContract(unittest.TestCase):
    def test_freeze_first_only_for_c16(self):
        self.assertEqual(len(groups("16")), 4)
        self.assertNotIn(("16", FIRST), groups("16"))
        self.assertEqual(groups("64"), [("64", k) for k in LAYERS])
        self.assertEqual(groups(None), [("16", "full"), ("64", "full")])
        self.assertEqual(list(NODES.values()), [GATE + "_" + str(i) for i in (4, 9, 14, 19, 24)])
        self.assertEqual(set(evidence.PILOT_END) | set(evidence.PREFIX), set(driver.STEPS))
        self.assertFalse(set(evidence.PILOT_END) & set(evidence.PREFIX))

    def test_gear_own_golden_and_scales(self):
        cases = fake_cases()
        case_contract(cases)
        for field in ("golden", "x.bin", "s_gn.bin", "w_nz.bin"):
            changed = copy.deepcopy(cases)
            if field == "golden":
                changed["64"][FIRST][field] = changed["16"][FIRST][field]
            elif field == "x.bin":
                changed["64"][FIRST]["inputs"][field] = changed["16"][FIRST]["inputs"][field]
            else:
                changed["64"][FIRST]["inputs"][field] = changed["64"][LAYERS[-1]]["inputs"][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                case_contract(changed)

    def test_c16_and_c64_predecessor_boundaries(self):
        self.assertLess(driver.STAGES.index("c16-compare"), driver.STAGES.index("c64-source"))
        self.assertLess(driver.STAGES.index("c64-compare"), driver.STAGES.index("full-draft-export"))
        self.assertEqual(
            driver.STAGES[-4:],
            ("full-draft-export", "full-draft-smoke", "full-draft-benchmark", "final-comparison"),
        )
        self.assertFalse(driver.green("c64-compare", "C64_FIVE_GATE_NO_GO"))
        for stage in driver.STAGES:
            with self.subTest(stage=stage), self.assertRaises(ValueError):
                driver.run(
                    SimpleNamespace(
                        stage=stage, allow_device=stage not in driver.DEVICE, root=Path("/unused")
                    )
                )

    def test_predecessor_sha_and_nonoverwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            with driver.stage_directory(root, "pilot-lock") as fold:
                ref = put(fold / "report.json", {"stub": True})
            with self.assertRaises(ValueError):
                with driver.stage_directory(root, "pilot-lock"):
                    self.fail("overwrote old root")
            with self.assertRaises(ValueError):
                evidence.pinned(ref, "0" * 64)
            stage = "c16-compare"
            old = dict(
                schema=1,
                root=str(root),
                phase=stage,
                folder=str(root / driver.location(stage)),
                predecessors={"atc": ref},
                sources=[ref],
                status="C16_FIVE_GATE_GO",
                error=None,
                integrity_error=None,
                device_execution="NOT_RUN",
                benchmark="NOT_RUN",
            )
            driver.verify_header(root, stage, old, {"atc": ref}, [ref])
            for replace in (
                {"predecessors": {}},
                {"sources": []},
                {"status": "C16_FIVE_GATE_NO_GO"},
                {"device_execution": "COMPLETED"},
            ):
                with self.assertRaises(ValueError):
                    driver.verify_header(root, stage, dict(old, **replace), {"atc": ref}, [ref])

    def test_three_percent_of_latency_not_speedup(self):
        self.assertEqual(five_decision("64", gear_stats("64", 1.94))["decision"], "GO")
        self.assertEqual(five_decision("64", gear_stats("64", 1.941))["decision"], "NO_GO")
        stats = gear_stats("16", 1.85)
        stats[LAYERS[-1]] = layer_statistics(
            timing_rows("16", LAYERS[-1], new=1.99), f"C16/{LAYERS[-1]}", "FRESH"
        )
        self.assertEqual(five_decision("16", stats)["decision"], "GO")
        self.assertEqual(
            stats[LAYERS[-1]]["decision"], "NO_GO"
        )  # aggregate threshold, independent per-layer tails
        stats[FIRST]["round_ratios"][1] = 1.0
        self.assertEqual(five_decision("16", stats)["decision"], "NO_GO")

    def test_one_p95_regression_or_wrong_sample_origin(self):
        rows = timing_rows("64", FIRST)
        for r in rows:
            if r["mode"] == CANDIDATE:
                r["validation"]["timing"]["samples_ms"][-2:] = [3, 3]
        stats = gear_stats("64")
        stats[FIRST] = layer_statistics(rows, f"C64/{FIRST}", "FRESH")
        self.assertEqual(five_decision("64", stats)["layers_with_p95_regression"], [FIRST])
        stats[FIRST]["sample_origin"] = "FROZEN_PILOT"
        with self.assertRaises(ValueError):
            five_decision("64", stats)

    def test_timed_row_count_order_and_device(self):
        for change in ("missing", "reorder", "profile", "short", "seven"):
            rows = timing_rows("16", FIRST)
            if change == "missing":
                rows.pop()
            if change == "reorder":
                rows[:2] = rows[:2][::-1]
            if change == "profile":
                rows[0]["profiled"] = True
            if change == "short":
                rows[0]["validation"]["timing"]["samples_ms"].pop()
            if change == "seven":
                rows[0]["atc_plan"]["block_dim"] = 7
            with self.subTest(change=change), self.assertRaises(ValueError):
                layer_statistics(rows, "C16/" + FIRST, "FRESH")


class OmContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_weights_partition_and_real_header_layout(self):
        # Offsets and extents come directly from the two 310P Raw OM headers.
        for om_size, hex_table in (
            (
                3090429227,
                "050000000000000008000000000000000000000000000000980a0000000000000000000000000000980a0000000000002b4a1100000000000100000000000000c354110000000000003c1bb8000000000300000000000000c3902cb800000000100a0600000000000200000000000000d39a32b800000000d898010000000000",
            ),
            (
                3090516391,
                "050000000000000008000000000000000000000000000000980a0000000000000000000000000000980a000000000000f06911000000000001000000000000008874110000000000003e1bb800000000030000000000000088b22cb800000000c73c07000000000002000000000000004fef33b800000000d898010000000000",
            ),
        ):
            table = bytes.fromhex(hex_table)
            entries = list(struct.iter_unpack("<QQQ", table[8:]))
            self.assertEqual(tuple(row[0] for row in entries), KINDS)
            self.assertEqual(256 + len(table) + sum(row[2] for row in entries), om_size)
        p = self.root / "test.om"
        p.write_bytes(imod({1: b"known weights"}))
        result = weight_partition(p)
        self.assertEqual(
            (result["bytes"], result["sha256"]), (13, hashlib.sha256(b"known weights").hexdigest())
        )
        for index in (4, 256, 264, 272, 280, 312):
            d = bytearray(imod({1: b"abc"}))
            d[index] ^= 1
            p.write_bytes(d)
            with self.subTest(index=index), self.assertRaises(ValueError):
                weight_partition(p)
        p.write_bytes(imod({1: b"abc"}) + b"trailing")
        with self.assertRaises(ValueError):
            weight_partition(p)

    def test_compiled_const_data_and_partition_mutations(self):
        def model(name, data, value):
            node = const_node(value)
            if name == "b":
                node["attr"].reverse()
            return dict(
                om=put(self.root / (name + ".om"), imod({1: data, 3: name.encode()})),
                dump=put(self.root / (name + ".json"), {"graph": [{"op": [node]}]}),
            )

        a = model("a", b"constant", "AQID")
        b = model("b", b"constant", "AQID")
        self.assertEqual(compare_constants(a, b)["raw_partition"]["bytes"], 8)
        with self.assertRaisesRegex(ValueError, "weight bytes"):
            compare_constants(a, model("b", b"ConstanT", "AQID"))
        with self.assertRaisesRegex(ValueError, "inline constant"):
            compare_constants(a, model("b", b"constant", "AQIE"))
        with self.assertRaises(ValueError):
            inline_constants(put(self.root / "empty.json", {"op": []})["path"])

    def test_full_26_selected_bytes(self):
        targets = set(NODES.values())
        left = {
            GATE
            + ("_" + str(i) if i else ""): dict(
                sha256="old", executable_sections=["s"], offset=i, engine="AIcoreEngine", block_dim=8, id="id"
            )
            for i in range(26)
        }
        right = copy.deepcopy(left)
        for n in targets:
            right[n].update(sha256="new", executable_sections=["t"])
        for v in right.values():
            v["offset"] += 200
        self.assertEqual(len(image_contract(left, right, targets, True)), 21)
        for name, field, v in [
            (GATE, "sha256", "other"),
            (GATE, "id", "other"),
            (NODES[FIRST], "block_dim", 7),
            (NODES[FIRST], "executable_sections", ["s"]),
        ]:
            bad = copy.deepcopy(right)
            bad[name][field] = v
            with self.subTest(field=field), self.assertRaises(ValueError):
                image_contract(left, bad, targets, True)

    def test_all_token_and_kv_bytes_including_inactive_tail(self):
        abi = [dict(name="draft_top1", dtype="int64", shape=[1, 15])]
        abi.extend(
            dict(name=f"d{i}_{kind}", dtype="float16", shape=[1, 8, 2112, 128])
            for i in range(5)
            for kind in ("key", "value")
        )
        tokens = put(self.root / "tokens", b"\0" * 120)
        kv = put(self.root / "kv", b"\0" * 4325376)
        spec = dict(
            abi={"outputs": abi}, golden={x["name"]: tokens if i == 0 else kv for i, x in enumerate(abi)}
        )
        full_abi(spec)
        e = dict(
            status="PASS",
            guards=True,
            readonly=True,
            repeat=True,
            cpu_fallback=False,
            golden_match=True,
            poison_verified=True,
            sync_completed=True,
            timed_log_stable=True,
            device=0,
            verification_calls=2,
            row_permutation_checked=False,
            workspace_bytes=10,
            model_weight_bytes=21,
            timing=dict(
                status="NOT_RUN",
                protocol="checked-v1",
                warmup=0,
                repetitions=0,
                correctness_before_calls=2,
                correctness_after_calls=0,
                timed_tail_checked=False,
                per_timed_call_readback=True,
                per_timed_call_poison=True,
                prepare={"samples_ms": []},
                execute_sync={"samples_ms": []},
            ),
        )
        row = dict(
            layer="full",
            execution=put(self.root / "run.json", e),
            outputs={n: [v, v] for n, v in spec["golden"].items()},
        )
        self.assertEqual(verify_outputs(row, spec, False)["status"], "ALL_TOKEN_KV_BITS_AND_SAFETY_MATCH")
        row["outputs"]["d4_value"][-1] = put(self.root / "last", b"\0" * 4325375 + b"\1")
        with self.assertRaisesRegex(ValueError, "d4_value"):
            verify_outputs(row, spec, False)
        spec["abi"]["outputs"][0]["shape"] = [1, 1]
        with self.assertRaises(ValueError):
            full_abi(spec)


class Dispatch(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.raw = put(self.root / "raw.cpp", b"Brcb")
        self.alt = put(self.root / "candidate.cpp", b"Transpose")
        h = put(self.root / "cfg.h", b"raw-on single-W")
        self.package = dict(brcb=self.raw, transpose=self.alt, header_name="cfg.h", header_sha256=h["sha256"])
        self.rules = dict(shapes=[SHAPE_KEY, TAG], keys={GATE_KEY: [SHAPE_KEY], OTHER_KEY: [TAG]})
        self.env = patch.dict(
            os.environ,
            {
                "AI_RUN_DIR": str(self.root),
                "ASCEND_CACHE_PATH": str(self.root / "compiler-cache"),
                "BUILD_KERNEL_SRC": "",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def args(self, shape):
        m, k, n = shape

        def desc(s, dtype="float16", fmt="ND"):
            return dict(shape=s, dtype=dtype, format=fmt)

        return [desc([m, k]), desc([], "int8", "FRACTAL_NZ"), desc([k // 128 if k > 0 else k, n])], [
            desc([m, n])
        ]

    def call(self, shape, key):
        path, receipt = select_scale_source(
            *self.args(shape), key, self.raw["path"], self.rules, self.package
        )
        record_scale_compile(receipt)
        return path

    def test_actual_source_dispatch_and_graph_receipts(self):
        self.assertEqual(self.call((16, 2560, 19456), GATE_KEY + "__kernel0"), self.alt["path"])
        self.assertEqual(self.call((16, 9728, 2560), OTHER_KEY + "__kernel0"), self.raw["path"])
        self.assertEqual(self.call((-1, -1, -1), GATE_KEY), self.alt["path"])
        imgs = {n: dict(symbol=GATE_KEY + "__kernel0") for n in NODES.values()}
        imgs["q"] = {"symbol": OTHER_KEY + "__kernel0"}
        ref = c.identity(self.root / "scale-dispatch.jsonl")
        result = verify_dispatch(ref, dict(routes=self.package, rules=self.rules), imgs, set(NODES.values()))
        self.assertEqual(result["actual_calls"], 3)
        imgs["q"]["symbol"] = GATE_KEY + "__kernel0"
        with self.assertRaises(ValueError):
            verify_dispatch(ref, dict(routes=self.package, rules=self.rules), imgs, set(NODES.values()))

    def test_ambiguity_wrong_source_header_shape_and_override(self):
        for shape, key in [
            ((-1, -1, -1), "unknown"),
            ((16, 2560, 12345), "unknown"),
            ((16, 2560, 19456), OTHER_KEY),
        ]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.call(shape, key)
        with patch.dict(os.environ, {"BUILD_KERNEL_SRC": "override"}), self.assertRaises(ValueError):
            self.call((16, 2560, 19456), GATE_KEY)
        (self.root / "candidate.cpp").write_bytes(b"bad")
        with self.assertRaises(ValueError):
            self.call((16, 2560, 19456), GATE_KEY)
        (self.root / "cfg.h").write_bytes(b"bad")
        with self.assertRaises(ValueError):
            self.call((16, 9728, 2560), OTHER_KEY)

    def test_ambiguous_key_26_node_rules(self):
        nodes = [dict(name=n, ge_type=GATE, k=2560, n=19456) for n in NODES.values()]
        nodes.extend(dict(name="Q" + str(i), ge_type=GATE, k=9728, n=2560) for i in range(21))
        graph = {"runtime_input_abi": {"draft_ac": {"nodes": nodes}}}
        ctl = dict(
            deployment=put(self.root / "dep", dict(graphs=[graph])),
            plans={TAG: {}, SHAPE_KEY: {}},
            atc_kernels={
                x["name"]: {"symbol": ("common" if x["name"] != "Q1" else "other") + "__kernel0"}
                for x in nodes
            },
        )
        rules = route_rules(ctl)
        self.assertEqual(set(rules["keys"]["common"]), {TAG, SHAPE_KEY})
        with self.assertRaises(ValueError):
            select_scale_source(*self.args((-1, -1, -1)), "common", self.raw["path"], rules, self.package)
        nodes[-1].update(k=2560, n=19456)
        ctl["deployment"] = put(self.root / "dep", dict(graphs=[graph]))
        with self.assertRaises(ValueError):
            route_rules(ctl)

    def test_wrapper_compiles_with_frozen_call_site(self):
        text = (
            "def _build_args(x):\n    return x\n\ndef dispatch():\n"
            "    src = get_kernel_source(ascendc_src_file, ascendc_src_dir, ascendc_src_dir_ex)\n"
            "    compile_op(src, origin_func_name, op_info, options, code_channel, '{}', {'valueDepend': {}})\n"
        )
        with patch("routing.WRAPPER", hashlib.sha256(text.encode()).hexdigest()):
            result = wrapper_bytes(text.encode(), self.rules, self.package)
        self.assertIn(inspect.getsource(select_scale_source).encode(), result)
        self.assertEqual(result.count(b"    src, scale_receipt = select_scale_source("), 1)
        self.assertGreater(
            result.index(b"    record_scale_compile(scale_receipt)"), result.index(b"    compile_op(src")
        )
        with self.assertRaises(ValueError):
            wrapper_bytes(b"other", self.rules, self.package)


if __name__ == "__main__":
    unittest.main()
