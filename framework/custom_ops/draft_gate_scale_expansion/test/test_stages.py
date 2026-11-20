"""Exercise stage transitions with synthetic artifacts; no device modules or external server paths."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import evidence
import pairs
import run_expansion as driver
from evidence import CANDIDATE, FIRST, LAYERS, NODES, c
from expansion_models import selected_plans
from routing import wrapper_bytes
from scale_contracts import EXPANDED_SOURCE_SHA, HEADER, RAW_HEADER, WRAPPER
from test_expansion import fake_cases, timing_rows, put


class StageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_runner_matrix_uses_own_gear_and_frozen_full_air(self):
        src = {"cases": fake_cases(), "full": {g: {"io": g} for g in ("16", "64")}}
        chain = {f"c{g}-source": {"binding": {"raw_air": {"payloads": {g: "raw-air"}}}} for g in ("16", "64")}
        variants = {m: m for m in evidence.MODES}

        def simulate(folder, src, build, model, g, k, m, i, label, timed, report, ref, *, constants):
            self.assertIsNone(ref)
            if k == "full":
                self.assertEqual(constants, {g: "raw-air"})
                name, gold = "draft_top1", {"bytes": 120, "sha256": g}
            else:
                self.assertIsNone(constants)
                name, gold = "y", src["cases"][g][k]["golden"]
            report["runs"].append(
                dict(
                    gear=g,
                    layer=k,
                    round=i,
                    mode=m,
                    pair=label,
                    golden={name: gold},
                    outputs={name: [gold, gold]},
                    validation=dict(workspace_bytes=2, model_weight_bytes=3),
                )
            )

        with ExitStack() as stack:
            stack.enter_context(patch.object(pairs, "case_run", side_effect=simulate))
            stack.enter_context(patch.object(pairs, "get_assets", return_value=(variants, variants)))
            verify = stack.enter_context(patch.object(pairs, "verify_runtime_row"))
            stack.enter_context(patch.object(pairs, "full_abi"))
            for gear, expected in (("16", 16), ("64", 20), (None, 8)):
                rec = {}
                pairs.runtime(self.root, {"src": src}, chain, rec, gear, False, launch=True)
                self.assertEqual((len(rec["runs"]), rec["device_execution"]), (expected, "COMPLETED"))
                pairs.runtime(self.root, {"src": src}, chain, rec, gear, False, launch=False)
                rec["runs"].pop()
                with self.assertRaisesRegex(ValueError, "missing"):
                    pairs.runtime(self.root, {"src": src}, chain, rec, gear, False, launch=False)
            self.assertEqual(verify.call_count, 2 * (16 + 20 + 8))

    def test_pilot_loads_external_seven_digests_from_two_exact_roots(self):
        old = self.root / "old"
        fixed = self.root / "fixed"
        raw = self.root / "gate8"
        base = dict(
            schema=1, root=str(fixed), archive=str(raw), sources=[], scope="C16_LAYER0_GATE_SCALE_EXPERIMENT"
        )
        archived = dict(base, root=str(old))
        seen = {}
        prefix = {}
        end = {}
        receipt = {"failed_review": {"sha256": "f"}}
        rows = timing_rows("16", FIRST, raw=2, new=1.6)
        metrics = evidence.pilot.statistics(rows)
        for s in driver.STEPS:
            early = s in evidence.PREFIX
            d = (old if early else fixed) / s
            d.mkdir(parents=True)
            row = dict(
                archived if early else dict(base, reuse=receipt),
                phase=s,
                folder=str(d),
                predecessors=seen.copy(),
                status=evidence.pilot.STATUS[s],
                error=None,
                integrity_error=None,
            )
            if s == "atc":
                row["candidate"] = {"atc_kernels": {evidence.GATE: {"sha256": evidence.PILOT_ELF}}}
            if s == "benchmark":
                row.update(runs=rows, statistics=metrics)
            if s == "compare":
                row["comparison"] = metrics
            item = put(d / "report.json", row)
            seen[s] = item
            if early:
                prefix[s] = item["sha256"]
            else:
                end[s] = (item["bytes"], item["sha256"])
        with ExitStack() as stack:
            stack.enter_context(
                patch.multiple(evidence, OLD=old, PILOT=fixed, GATE8_ROOT=raw, PREFIX=prefix, PILOT_END=end)
            )
            stack.enter_context(patch.object(evidence.pilot, "sources", return_value=[]))
            stack.enter_context(patch.object(evidence, "open_prefix", return_value=(archived, {}, receipt)))
            verified = stack.enter_context(
                patch.object(
                    evidence.pilot, "verify", side_effect=lambda phase, *_: evidence.pilot.STATUS[phase]
                )
            )
            lock, _ = evidence.pilot_chain(({"reports": {}},), replay=True)
            self.assertEqual(lock["reports"], seen)
            self.assertEqual([v.args[0] for v in verified.call_args_list], list(driver.STEPS))
            path = fixed / "compare/report.json"
            bad = json.loads(path.read_bytes())
            bad["predecessors"]["smoke"] = bad["predecessors"]["review"]
            new = put(path, bad)
            with self.assertRaisesRegex(ValueError, "digest"):
                evidence.pilot_chain(({"reports": {}},), replay=False)
            end["compare"] = (new["bytes"], new["sha256"])
            with self.assertRaisesRegex(ValueError, "cross-wired"):
                evidence.pilot_chain(({"reports": {}},), replay=False)

    def test_c64_reuses_models_only_after_gear_specific_binding(self):
        cases = fake_cases()
        ctx = {"pilot": {"atc": {"candidate": "sealed-pilot"}}, "builds": {CANDIDATE: "b"}}
        data = {
            f"c{g}-source": {
                "binding": {
                    "cases": {
                        k: {"case": case, "single_node": NODES[k], "raw_single_om": k}
                        for k, case in cases[g].items()
                    }
                }
            }
            for g in ("16", "64")
        }
        record = {}
        with ExitStack() as stack:
            compiler = stack.enter_context(
                patch.object(driver, "compile_model", side_effect=lambda folder, ctx, b, k, full: k)
            )
            checker = stack.enter_context(patch.object(driver, "check_model", return_value={"checked": True}))
            driver.projection_atc(self.root, ctx, data, record, "16")
            self.assertEqual([args.args[3] for args in compiler.call_args_list], list(LAYERS[1:]))
            self.assertEqual(record["models"][FIRST], "sealed-pilot")
            data["c16-atc"] = record
            another = {}
            driver.projection_atc(self.root, ctx, data, another, "64")
            self.assertEqual(compiler.call_count, 4)
            self.assertEqual(checker.call_count, 10)
            self.assertEqual(another["models"], record["models"])
            data["c64-source"]["binding"]["cases"][FIRST]["case"]["golden"] = cases["16"][FIRST]["golden"]
            with self.assertRaisesRegex(ValueError, "own case"):
                driver.projection_atc(self.root, ctx, data, {}, "64")

    def test_incomplete_or_changed_actual_callbacks_stop(self):
        job = {k: put(self.root / k, b'{"m":16,"k":2560,"n":19456}\n') for k in ("platform", "scheduling")}
        tag = evidence.SHAPE_KEY
        baseline = {tag: {"query": {"host_dso": {"sha256": "x"}}, "contract": "plan"}}
        build = {"opp": {}}
        with ExitStack() as stack:
            stack.enter_context(
                patch("expansion_models.flow.read_query", return_value={"host_dso": {"sha256": "x"}})
            )
            verify = stack.enter_context(patch("expansion_models.same_plan", return_value="plan"))
            selected_plans(self.root, job, build, {}, {"plans": baseline}, write=True)
            self.assertEqual(verify.call_count, 1)
            with self.assertRaisesRegex(ValueError, "inventory"):
                selected_plans(self.root, job, build, {}, {"plans": dict(baseline, other={})}, write=True)
        with patch("expansion_models.flow.read_query", return_value={"host_dso": {"sha256": "changed"}}):
            with self.assertRaisesRegex(ValueError, "tiling DSO"):
                selected_plans(self.root, job, build, {}, {"plans": baseline}, write=False)

    def test_compare_uses_fresh_full_rounds_and_detects_substitution(self):
        runs = [
            r for g in ("16", "64") for r in timing_rows(g, "full", raw=31, new=29 if g == "16" else 30.9)
        ]
        stats = {
            g: driver.layer_statistics([r for r in runs if r["gear"] == g], f"C{g}/full", "FRESH")
            for g in ("16", "64")
        }
        chain = {
            "full-draft-benchmark": {"runs": runs, "statistics": stats},
            "pilot-lock": {"lock": {"reports": {"benchmark": {}}}},
        }
        chain.update({f"c{g}-compare": {"result": {"decision": "GO"}} for g in ("16", "64")})
        result = driver.full_result(chain)
        self.assertEqual(result["decision"], "NO_GO")
        self.assertFalse(result["default_change"])
        runs[0]["validation"]["timing"]["samples_ms"] = [4] * 30
        with self.assertRaisesRegex(ValueError, "statistics"):
            driver.full_result(chain)


@unittest.skipUnless(os.environ.get("SCALE_ROUTER_BUILD"), "requires the SDK-generated private package")
class SDKWrapperTests(unittest.TestCase):
    def test_actual_sdk_entry_and_installed_sources(self):
        report = c.read(os.environ["SCALE_ROUTER_BUILD"])
        cpp = c.verify(report["dynamic_source"])
        self.assertEqual(c.sha(cpp), EXPANDED_SOURCE_SHA)
        py = cpp.with_suffix(".py")
        self.assertEqual(c.sha(py), WRAPPER)
        self.assertEqual(c.sha(cpp.parent / HEADER), RAW_HEADER)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            raw = put(
                root / "src.cpp", (evidence.c.QUANT / "op_kernel/d_flash_group_quant_linear.cpp").read_bytes()
            )
            (root / HEADER).write_bytes((cpp.parent / HEADER).read_bytes())
            package = dict(brcb=raw, transpose=c.identity(cpp), header_name=HEADER, header_sha256=RAW_HEADER)
            tags = [
                "16x2560x4096",
                evidence.SHAPE_KEY,
                "16x4096x2560",
                "16x9728x2560",
                "64x12800x2560",
                "80x2560x2048",
            ]
            result = wrapper_bytes(py.read_bytes(), dict(shapes=tags, keys={}), package)
            # Invoke the selector emitted into the real SDK AST without importing its device compiler.
            import ast

            tree = ast.parse(result)
            selector = next(
                n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "select_scale_source"
            )
            ns = {}
            exec(compile(ast.Module(body=[selector], type_ignores=[]), str(py), "exec"), ns)
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
            self.assertEqual(sum(n.func.id == "select_scale_source" for n in calls), 1)
            self.assertEqual(
                sum(
                    n.func.id == "compile_op" and isinstance(n.args[0], ast.Name) and n.args[0].id == "src"
                    for n in calls
                ),
                1,
            )
            with patch.dict(
                os.environ,
                {
                    "AI_RUN_DIR": str(root),
                    "ASCEND_CACHE_PATH": str(root / "compiler-cache"),
                    "BUILD_KERNEL_SRC": "",
                },
            ):
                for tag in tags:
                    m, k, n = map(int, tag.split("x"))
                    x = dict(dtype="float16", format="ND", shape=[m, k])
                    w = dict(dtype="int8", format="FRACTAL_NZ", shape=[])
                    s = dict(dtype="float16", format="ND", shape=[k // 128, n])
                    y = dict(dtype="float16", format="ND", shape=[m, n])
                    selected, receipt = ns["select_scale_source"](
                        [x, w, s], [y], "sample", raw["path"], dict(shapes=tags, keys={}), package
                    )
                    self.assertEqual(selected, str(cpp) if tag == evidence.SHAPE_KEY else raw["path"])
                    self.assertEqual(receipt[1]["source"]["path"], selected)
