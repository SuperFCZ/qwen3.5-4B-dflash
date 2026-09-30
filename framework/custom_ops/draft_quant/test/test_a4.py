"""CPU-only A4 contracts/metadata fixtures; never device correctness evidence."""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import a4_contract as a4
import reference as ref
from a2_common import case_shape, load_bundle, validate_native_graph, write_json
from a3_launch import PREFIX, launch_evidence, pipeline_plan
from run_a4 import collect_launches
from test_a3 import launch_record
from test_reference import checker_fixture


def bundle(gear):
    result = dict(abi=a4.ABI, status="PASS", context_rows=gear, capture_runtime="native NPU DraftGraph replay",
                  cpu_fallback=False, capture_repeat_equal=True,
                  checkpoint=dict(variant="w8a16", status="PASS", config_sha256="config", model_sha256="model"),
                  source=dict(context_rows=gear, feature_layers=[1, 5, 9]), cases=a4.cases(gear))
    for case in result["cases"]:
        m, k, n = case_shape(case, "a4")
        sizes = {"x.bin": m*k*2, "q_nk.bin": n*k, "w_nz.bin": n*k, "s_gn.bin": n*(k//128)*2,
                 "eager-0.bin": m*n*2, "eager-1.bin": m*n*2}
        case["files"] = {name: dict(path=f"{case['name']}/{name}", bytes=size,
                                    sha256=case["name"] + name.replace("eager-1", "eager-0")) for name, size in sizes.items()}
    return result


class A4Tests(unittest.TestCase):
    def test_projection_inventory_matches_checked_in_model_workloads(self):
        here = Path(__file__).resolve().parents[1]
        requirements = json.loads((here / "workloads.json").read_text())["projections"]
        observed = {}
        for gear in (16, 64):
            cases = a4.cases(gear)
            self.assertEqual(len(cases), 26)
            self.assertEqual(tuple(c["name"] for c in cases), a4.CASE_NAMES)
            for case in cases:
                m, k, n = case_shape(case, "a4")
                observed.setdefault(case["projection"], set()).add((m, k, n))
                for key, value in (("m", 48), ("k", 256), ("n", 64), ("context_rows", 32), ("layer", 8)):
                    with self.subTest(case=case["name"], key=key), self.assertRaises(ValueError):
                        case_shape(dict(case, **{key: value}), "a4")
        self.assertEqual(observed, {p["name"]: {(m, p["k"], p["n"]) for m in p["m_cases"]} for p in requirements})

    def test_capture_targets_are_the_production_packed_modules_in_execution_order(self):
        layers = [SimpleNamespace(**{key: object() for key in
                  ("kv_linear", "q_proj", "o_proj", "gate_up_linear", "down_proj")}) for _ in range(5)]
        graph = SimpleNamespace(fc=object(), layers=layers)
        targets = a4.capture_targets(graph, 64)
        self.assertIs(targets[0][0], graph.fc)
        for layer in range(5):
            for offset, attribute in enumerate(("kv_linear", "q_proj", "o_proj", "gate_up_linear", "down_proj")):
                self.assertIs(targets[1 + layer*5 + offset][0], getattr(layers[layer], attribute))
        self.assertEqual(targets[1][1]["m"], 80)
        self.assertEqual(targets[0][1]["m"], 64)
        self.assertEqual([c["name"] for _, c in targets], list(a4.CASE_NAMES))

    def test_gear_pair_rejects_missing_gears_weight_changes_and_feature_reordering(self):
        left, right = bundle(16), bundle(64)
        a4.paired_gears(left, right)
        for path, value in ((("context_rows",), 16), (("source", "context_rows"), 16),
                            (("source", "feature_layers"), [9, 5, 1]), (("checkpoint", "model_sha256"), "other"),
                            (("cases", 1, "files", "s_gn.bin", "sha256"), "other"), (("cases", 1, "m"), 32)):
            bad = copy.deepcopy(right); parent = bad
            for key in path[:-1]: parent = parent[key]
            parent[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError): a4.paired_gears(left, bad)
        bad = copy.deepcopy(right); bad["cases"].pop()
        with self.assertRaises(ValueError): a4.paired_gears(left, bad)

    def test_real_bundle_requires_a4_abi_full_inventory_and_frozen_gear(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            good = bundle(64)
            write_json(path, good)
            # Only the byte-file reader is mocked; these are metadata fixtures.
            with patch("a2_common.checked_file") as check:
                self.assertEqual(load_bundle(path, "a4")["context_rows"], 64)
                self.assertEqual(check.call_count, 26*6)
                self.assertIn(64*12800*2, [call.args[2] for call in check.call_args_list])
                self.assertIn(80*2560*2, [call.args[2] for call in check.call_args_list])
            with self.assertRaises(ValueError): load_bundle(path)
            for key, value in (("abi", "dflash-group-quant-linear-a2-v1"), ("context_rows", 16),
                               ("cases", good["cases"][1:]), ("capture_runtime", "synthetic")):
                write_json(path, dict(good, **{key: value}))
                with self.subTest(key=key), self.assertRaises(ValueError): load_bundle(path, "a4")

    def test_a4_native_graph_still_requires_exact_immutable_nz_const_and_scope_identity(self):
        case = bundle(64)["cases"][1]  # KV M80
        m, k, n = case_shape(case, "a4")
        graph = {"name": "weight_quant_reference", "input_names": ["x"], "output_names": ["y"],
                 "metadata": {"a4_case": case["name"], "a4_bundle_sha256": "capture",
                     "tensor_abi": {"inputs": [{"name": "x", "dtype": "float16", "shape": [m, k]}]}},
                 "runtime_input_abi": {"weight_quant_layout": {"status": "PASS", "node_count": 1, "inserted_transdata": [],
                    "prepack": {"status": "PASS", "node_count": 1, "constants": [{
                        "logical_sha256": case["files"]["q_nk.bin"]["sha256"],
                        "storage_sha256": case["files"]["w_nz.bin"]["sha256"],
                        "logical_shape": [n, k], "storage_shape": [k//32, n//16, 16, 32], "roundtrip": "BIT_EXACT"}]}}}}
        validate_native_graph(graph, case, "capture", "a4")
        bad = copy.deepcopy(graph); bad["input_names"].append("weight")
        with self.assertRaises(ValueError): validate_native_graph(bad, case, "capture", "a4")
        bad = copy.deepcopy(graph); bad["metadata"]["a4_bundle_sha256"] = "other"
        with self.assertRaises(ValueError): validate_native_graph(bad, case, "capture", "a4")
        bad = copy.deepcopy(graph); bad["runtime_input_abi"]["weight_quant_layout"]["prepack"]["constants"][0]["storage_sha256"] = "other"
        with self.assertRaises(ValueError): validate_native_graph(bad, case, "capture", "a4")

    def test_synthetic_row_boundary_oracle_and_full_m_execution_gate(self):
        self.assertEqual(len(ref.A4_WORKLOADS), 36)
        for m in (32, 64, 80):
            shape = ref.RowWorkload(m=m)
            _, x, q, scales, exact = next(c for c in ref.cases(shape) if c[0] == "m_block_boundaries")
            self.assertTrue(exact)
            values = ref.read_half(x, m*256)
            self.assertEqual(sum(v != 0 for v in values), m)
            self.assertTrue(any(values[(m-1)*256:]))
            s = ref.read_half(scales, 2*64)
            expected = []
            for row in range(m):
                index = (row*31 + row//16*128) % 256
                expected.extend(ref.half((1 + row % 5)/128 * ref.half(q[n*256+index] * s[index//128*64+n])) for n in range(64))
            self.assertEqual(ref.cpu_reference(x, q, scales, shape), ref.half_bytes(expected))
        shape = ref.RowWorkload(m=80)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory) / "fixture"
            checker_fixture(root, shape=shape)
            for case in ref.load_cases(root)["cases"]:
                path = root / case["name"] / "execution.json"
                execution = json.loads(path.read_text())
                execution.update(tile_m=80, weight_reuse_rows=80)
                write_json(path, execution)
            self.assertTrue(ref.check(root))  # CPU checker fixture, never an NPU claim
            path = root / "signed_zero/execution.json"
            execution = json.loads(path.read_text()); execution["tile_m"] = 16
            write_json(path, execution)
            with self.assertRaisesRegex(ValueError, "execution evidence"):
                ref.check(root)

    def test_launch_version_and_m_dimension_cannot_be_inferred_from_old_opp(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "runner.log"
            launch = dict(launch_record(2560, 2048, available=7), version=4, m=80, tile_m=80, weight_reuse_rows=80,
                          dequant_mode="batched", **pipeline_plan(2560, 2048, "serial"), matmul_ub_bytes=65536)
            log.write_text(PREFIX + json.dumps(launch))
            self.assertEqual(launch_evidence(log, (80,2560,2048), 0, 2097152, "batched", "serial", 4)["tile_m"], 80)
            for key, value in (("version", 3), ("m", 16), ("tile_m", 16), ("weight_reuse_rows", 16)):
                log.write_text(PREFIX + json.dumps(dict(launch, **{key: value})))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    launch_evidence(log, (80,2560,2048), 0, 2097152, "batched", "serial", 4)

    def test_aggregate_requires_both_gears_and_every_projection(self):
        config = dict(core_limit=0, dequant_mode="batched", pipeline_mode="serial")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def write_launch(dest, shape):
                m, k, n = shape
                dest.mkdir(parents=True)
                launch = dict(launch_record(k, n, available=7), version=4, m=m, tile_m=m, weight_reuse_rows=m,
                              dequant_mode="batched", **pipeline_plan(k,n,"serial"), matmul_ub_bytes=65536)
                (dest / "runner.log").write_text(PREFIX + json.dumps(launch))
                execution = dict(status="PASS", tile_m=m, weight_reuse_rows=m, workspace_bytes=2097152, timing={})
                write_json(dest / "execution.json", execution)
                return execution
            workloads = []
            for shape in ref.A4_WORKLOADS:
                names = [c[0] for c in ref.cases(shape)]
                for name in names: write_launch(root / "synthetic" / shape.name / name, (shape.m,shape.k,shape.n))
                workloads.append(dict(name=shape.name, mkn=[shape.m,shape.k,shape.n], status="PASS",
                                      custom_cases=[dict(name=name) for name in names]))
            write_json(root / "synthetic/suite.json", dict(status="PASS", workloads=workloads))
            for gear in (16,64):
                cases = []
                for case in a4.cases(gear):
                    execution = write_launch(root / f"c{gear}" / case["name"] / "custom", case_shape(case,"a4"))
                    cases.append(dict(name=case["name"], executions=dict(custom=execution, native_om=execution)))
                write_json(root / f"c{gear}/suite.json", dict(abi=a4.ABI, status="PASS", context_rows=gear, cases=cases))
            evidence = collect_launches(root, config)
            self.assertEqual(len(evidence["synthetic"]), 351)
            self.assertEqual(len(evidence["c16"]), 26)
            self.assertEqual(len(evidence["c64"]), 26)
            path = root / "c64/suite.json"
            summary = json.loads(path.read_text()); summary["cases"].pop(0)
            write_json(path, summary)
            with self.assertRaisesRegex(ValueError, "all 26"): collect_launches(root, config)


if __name__ == "__main__":
    unittest.main()
