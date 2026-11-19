"""Offline negative contracts for exact Golden, expanded stage policy and proof provenance."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import run_w16 as driver
from down_fallback import STAGES as DOWN_STAGES, dma_extents, verify_down
from measurement import verify_outputs
from w16_common import FIRST, LAYERS, c
from w16_runtime import five_layer_decision, paired
from w16_static import proof_index
from test_contracts import rows


class ReportContracts(unittest.TestCase):
    def test_full_graph_protects_head_wiring_and_non_gate_projection(self):
        from w16_export import protected_graph

        def node(name, kind, inputs):
            return dict(
                name=name,
                type=kind,
                inputs=inputs,
                input_desc=[],
                output_desc=[],
                serialized_descriptors=[],
                attributes=[{"key": "_kernel_bin_id", "value": {"s": "before"}}],
            )

        old = [
            node("g", "DFlashGroupQuantLinear", ["x"]),
            node("down", "DFlashGroupQuantLinear", ["z"]),
            node("MatMul", "MatMul", ["hidden", "head_weight"]),
            node("ArgMaxV2", "ArgMaxV2", ["MatMul"]),
        ]
        new = copy.deepcopy(old)
        new[0]["attributes"][0]["value"]["s"] = "after"

        def read(which):
            return old if which == "old" else new

        with patch("w16_export.read_om_json", read):
            self.assertEqual(protected_graph("old", "new", {"g"}), 3)
            for i in (1, 2, 3):
                new[i]["inputs"].append("other")
                with self.assertRaises(ValueError):
                    protected_graph("old", "new", {"g"})
                new[i]["inputs"].pop()

    def test_schema_matches_all_manual_phases(self):
        schema = json.loads((HERE / "report.schema.json").read_text())
        self.assertEqual(schema["properties"]["phase"]["enum"], list(driver.STAGES) + [driver.EXTRA])
        for phase in (*driver.STAGES, driver.EXTRA):
            self.assertIn(driver.passed(phase), schema["properties"]["status"]["enum"])
        down = json.loads((HERE / "down/report.schema.json").read_text())
        self.assertEqual(down["properties"]["phase"]["enum"], list(DOWN_STAGES))
        self.assertEqual(dma_extents()["codes"]["last_exclusive"], 24903680)
        self.assertEqual(dma_extents()["scales"]["last_exclusive"], 389120)

    def test_five_layer_two_orders_and_one_reversal_blocks(self):
        stats = {}
        for layer in LAYERS:
            for pair in ("schedule", "net"):
                data = rows(pair)
                for row in data:
                    row["layer"] = layer
                stats["16/" + layer + "/" + pair] = paired(data, "16", layer, pair)
        good = five_layer_decision(stats, "16")
        self.assertEqual(good["decision"], "GO")
        for pair in ("schedule", "net"):
            self.assertEqual(set(good["orders"][pair]), {"control_first", "candidate_first"})
        bad = copy.deepcopy(stats)
        bad["16/" + LAYERS[-1] + "/net"]["round_ratios"][1] = 0.96
        self.assertEqual(five_layer_decision(bad, "16")["decision"], "NO_GO")

    def test_frozen_bitwise_tail_repeat_and_guard(self):
        for layer, n, scope, validator in (
            (FIRST, 311296, "Gate", verify_outputs),
            ("layer-0-down", 40960, "Down", verify_down),
        ):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                b = bytes(2 * n)
                golden = root / "gold"
                golden.write_bytes(b)
                actual = []
                for i in range(2):
                    p = root / str(i)
                    p.write_bytes(b)
                    actual.append(c.identity(p))
                spec = dict(golden={"y": c.identity(golden)})
                t = dict(
                    protocol="checked-v1",
                    status="NOT_RUN",
                    warmup=0,
                    repetitions=0,
                    correctness_before_calls=2,
                    correctness_after_calls=0,
                    timed_tail_checked=False,
                    per_timed_call_readback=True,
                    per_timed_call_poison=True,
                    prepare={"samples_ms": []},
                    execute_sync={"samples_ms": []},
                )
                e = dict(
                    scope="single " + scope + " frozen Native Golden checked in process",
                    status="PASS",
                    device=0,
                    guards=True,
                    readonly=True,
                    repeat=True,
                    cpu_fallback=False,
                    golden_match=True,
                    poison_verified=True,
                    sync_completed=True,
                    timed_log_stable=True,
                    verification_calls=2,
                    row_permutation_checked=False,
                    timing=t,
                    workspace_bytes=512,
                    model_weight_bytes=32,
                )
                f = root / "exec.json"
                c.write(f, e)
                row = dict(layer=layer, outputs={"y": actual}, execution=c.identity(f))
                validator(row, spec, False)
                for key in (
                    "guards",
                    "readonly",
                    "poison_verified",
                    "sync_completed",
                    "repeat",
                    "cpu_fallback",
                ):
                    c.write(f, dict(e, **{key: not e[key]}))
                    row["execution"] = c.identity(f)
                    with self.subTest(scope=scope, key=key), self.assertRaises(ValueError):
                        validator(row, spec, False)
                c.write(f, e)
                row["execution"] = c.identity(f)
                (root / "1").write_bytes(b[:-2] + b"\1\0")
                actual[1] = c.identity(root / "1")
                with self.assertRaises(ValueError):
                    validator(row, spec, False)

    def test_external_review_index_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proof = root / "per-elf.json"
            proof.write_text("{}")
            f = root / "index.json"
            d = dict(schema=1, kind="W16_EXACT_ATC_MACHINE_EVIDENCE", reports={"a" * 64: c.identity(proof)})
            c.write(f, d)
            h = c.sha(f)
            self.assertEqual(proof_index(f, h)["file"], c.identity(f))
            with self.assertRaises(ValueError):
                proof_index(f, "b" * 64)
            proof.write_text("changed")
            with self.assertRaises(ValueError):
                proof_index(f, h)

    def test_missing_source_and_wrong_stage_do_not_launch(self):
        import argparse

        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            args = argparse.Namespace(
                phase="source",
                root=p / "out",
                gate8_root=p / "absent",
                profile_root=p / "missing",
                objdump=None,
                review_commit=None,
                proof_index=None,
                proof_index_sha256=None,
                allow_device=False,
            )
            with patch.object(driver, "execute") as launch, patch("builtins.print"):
                self.assertNotEqual(driver.main(args), 0)
                args.phase = "smoke"
                args.allow_device = True
                self.assertNotEqual(driver.main(args), 0)
                launch.assert_not_called()
            for phase in ("source", "smoke"):
                report = c.read(args.root / phase / "report.json")
                self.assertEqual(report["status"], "BLOCKED")
                self.assertEqual(report["device_execution"], "NOT_RUN")
                self.assertEqual(report["benchmark"], "NOT_RUN")
