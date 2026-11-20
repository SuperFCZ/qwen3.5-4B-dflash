"""CPU contracts only. The frozen device runner is never invoked by these tests."""

from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import run_scale as run
import scale_contracts as sc
import scale_review as review
from a3_launch import configure_build
from measurement import verify_outputs


class ScaleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()

    def test_full_generated_gate_both_variants(self):
        tmp = self.root
        inc = tmp / "inc"
        shutil.copytree(sc.c.QUANT / "test/cpu_kernel_include", inc)
        header = inc / "kernel_operator.h"
        text = header.read_text()
        if platform.machine() == "aarch64" and sys.platform.startswith("linux"):
            text = sc.replace_once(text, "using half = _Float16;", "using half = __fp16;")
        text = sc.replace_once(
            text,
            "    Tensor operator[]",
            "    uintptr_t GetPhyAddr() const { return reinterpret_cast<uintptr_t>(data); }\n    Tensor operator[]",
        )
        header.write_text(text + (HERE / "test/portable_model.inc").read_text())
        configure_build(
            tmp / sc.HEADER,
            tmp / "config.json",
            0,
            "batched",
            None,
            "prefetch",
            "a-ub",
            "broadcast",
            256,
            "loop",
            "raw",
        )
        self.assertEqual(sc.c.sha(tmp / sc.HEADER), sc.RAW_HEADER)
        raw = (sc.c.QUANT / "op_kernel" / (sc.STEM + ".cpp")).read_bytes()
        for mode, blob in (("brcb", raw), ("transpose", sc.render(raw))):
            (tmp / "variant.cpp").write_bytes(blob)
            exe = tmp / mode
            subprocess.run(
                [
                    "c++",
                    "-std=c++17",
                    "-O2",
                    "-ffp-contract=off",
                    "-I" + str(inc),
                    "-I" + str(tmp),
                    str(HERE / "test/portable_gate.cpp"),
                    "-o",
                    str(exe),
                ],
                check=True,
            )
            self.assertEqual(
                subprocess.check_output([str(exe), mode], text=True).strip(),
                "PORTABLE_GATE_311296_BITS_MATCH",
            )

    def test_scope_only_one_gate_call_and_no_implicit_resources(self):
        raw = (sc.c.QUANT / "op_kernel" / (sc.STEM + ".cpp")).read_bytes()
        changed = sc.render(raw).decode()
        for op in (
            "PipeBarrier<PIPE_ALL>",
            "PipeBarrier<PIPE_V>",
            "AllocEventID",
            "SetFlag<",
            "WaitFlag<",
            "SetTensorB(",
            "mm.Iterate(",
            "GetTensorC(",
            "AscendC::Mul(",
            "pipe.InitBuffer(",
        ):
            self.assertEqual(raw.decode().count(op), changed.count(op), op)
        self.assertIn(
            "if (DFLASH_GROUP_QUANT_GATE_PREFETCH(globalM, globalK, globalN, tileK)) {\n                        ExpandGateScale(",
            changed,
        )
        self.assertEqual(changed.count(sc.CALL), 1)
        with self.assertRaises(ValueError):
            sc.render(raw + b" ")
        (self.root / "kernel").mkdir()
        (self.root / "kernel/scale_expand.inc").write_text(
            (HERE / "kernel/scale_expand.inc").read_text().replace("4, 16, 1", "4, 1, 16")
        )
        with patch.object(sc, "HERE", self.root), self.assertRaisesRegex(ValueError, "reviewed"):
            sc.render(raw)
        self.assertEqual(sc.layout_contract()["additional_tpipe_bytes"], 0)

    def test_each_16bit_pattern_and_all_destination_edges(self):
        table = sc.expansion_map()
        self.assertEqual(table, [v for v in range(64) for _ in range(16)])
        # Nonidentical source words, including both zero signs, denormals, infinities and NaNs.
        for off in range(0, 65536, 64):
            src = [(137 * (off + c) + 0x8000) & 0xFFFF for c in range(64)]
            self.assertEqual([src[i] for i in table], [s for s in src for _ in range(16)])
        self.assertEqual(
            (table[0], table[15], table[16], table[255], table[256], table[-1]), (0, 0, 1, 15, 16, 63)
        )
        self.assertEqual((66048 + 2048, 66048 + 2304 + 185600 + 8192), (68096, 262144))

    def test_sdk_and_report_pin_drift_block(self):
        with self.assertRaises((FileNotFoundError, ValueError)):
            sc.sdk_files(self.root)
        phase = self.root / "source"
        phase.mkdir()
        (phase / "report.json").write_text('{"status":"GATE8_SOURCE_PASS"}')
        with self.assertRaisesRegex(ValueError, "frozen report drift"):
            sc.read_baseline(self.root)

    def test_precise_8_block_50_fields_and_workspace(self):
        contract = {
            "parent_sha256": sc.RAW_PARENT,
            "cube_sha256": sc.CUBE_SHA,
            "cube_fields": {str(i): i for i in range(50)},
            "resources": {"scratch": 185600},
            "block_dim": 8,
        }
        src = {"full": {"16": {"anchor": {"plans": {sc.SHAPE_KEY: {}}}}}}
        with patch.object(sc, "gate_plan", return_value=contract):
            self.assertEqual(sc.same_plan({}, src, contract), contract)
            with self.assertRaises(ValueError):
                sc.same_plan({}, src, dict(contract, block_dim=7))
        for change in ({"parent_sha256": "0" * 64}, {"cube_fields": {}}, {"resources": {"scratch": 185568}}):
            mutated = dict(contract, **change)
            with patch.object(sc, "gate_plan", return_value=mutated), self.assertRaises(ValueError):
                sc.same_plan({}, src, mutated)

    def test_manual_stages_and_predecessors(self):
        for phase in run.DEVICE:
            with patch.object(run, "case_run") as runner, self.assertRaisesRegex(ValueError, "allow-device"):
                run.run(SimpleNamespace(root=self.root, archive=self.root, phase=phase, allow_device=False))
            runner.assert_not_called()
        archive = self.root / "archive"
        archive.mkdir()
        experiment = self.root / "new"
        experiment.mkdir()
        with self.assertRaises(ValueError):
            with run.stage_dir(experiment, "smoke", archive):
                pass
        self.assertFalse((experiment / "smoke").exists())
        with self.assertRaises(ValueError):
            with run.stage_dir(archive / "new", "source", archive):
                pass
        self.assertEqual(
            run.schedule(False), [(0, sc.CONTROL), (0, sc.CANDIDATE), (1, sc.CANDIDATE), (1, sc.CONTROL)]
        )

    def test_replay_rejects_corrupt_final_half_and_cpu_fallback(self):
        a = self.root / "y"
        a.write_bytes(bytes(622592))
        gold = sc.c.identity(a)
        b = self.root / "actual"
        b.write_bytes(bytes(622590) + b"\1\0")
        row = dict(layer=sc.FIRST, execution={}, outputs={"y": [gold, sc.c.identity(b)]})
        e = dict(
            status="PASS",
            scope="single Gate frozen Native Golden checked in process",
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
            workspace_bytes=32,
            model_weight_bytes=64,
            timing=dict(
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
            ),
        )

        def check():
            p = self.root / "ex.json"
            sc.c.write(p, e)
            row["execution"] = sc.c.identity(p)
            return verify_outputs(row, {"golden": {"y": gold}}, False)

        with self.assertRaises(ValueError):
            check()
        row["outputs"]["y"][1] = gold
        self.assertTrue(check())
        for key in ("cpu_fallback", "guards", "readonly", "poison_verified", "sync_completed", "repeat"):
            e[key] = not e[key]
            with self.assertRaises(ValueError):
                check()
            e[key] = not e[key]

    def test_paired_order_p95_all_samples_threshold(self):
        rows = [
            dict(
                round=i,
                mode=m,
                folder=f"{i}-{m}",
                validation={"timing": {"samples_ms": [1.0 if m == sc.CONTROL else 0.97] * 30}},
                profiled=False,
                launch_authorized=True,
                bit_mismatches=0,
                atc_plan={"block_dim": 8},
            )
            for i, m in run.schedule(True)
        ]
        s = run.statistics(rows)
        self.assertEqual(s["decision"], "GO")
        self.assertTrue(all(len(v["samples_ms"]) == 90 for v in s["variants"].values()))
        for m in (rows[::-1], rows[:-1]):
            with self.assertRaises(ValueError):
                run.statistics(m)
        p = copy.deepcopy(rows)
        for x in p:
            if x["mode"] == sc.CANDIDATE:
                x["validation"]["timing"]["samples_ms"][-2:] = [1.1, 1.2]
        self.assertEqual(run.statistics(p)["decision"], "NO_GO")
        p = copy.deepcopy(rows)
        p[1]["validation"]["timing"]["samples_ms"] = [1.0] * 30
        self.assertEqual(run.statistics(p)["decision"], "NO_GO")
        for x in rows:
            if x["mode"] == sc.CANDIDATE:
                x["validation"]["timing"]["samples_ms"] = [0.971] * 30
        self.assertEqual(run.statistics(rows)["decision"], "NO_GO")

    def test_manual_end_to_end_with_cpu_transport(self):
        archive = self.root / "history"
        archive.mkdir()
        base = self.root / "trial"
        y = archive / "golden"
        y.write_bytes(b"fixture")
        gold = sc.c.identity(y)
        ctx = ({"frozen": {"s_gn.bin": gold}}, {"cann": "fixture"}, {}, {})
        fakebuild = {"opp": {"install": str(base / "build/package/opp")}}

        def command(argv, log, env, cwd):
            sc.c.write(cwd / "worker.json", dict(status="SCALE_PRIVATE_BUILD_RECORDED", candidate=fakebuild))

        def launch(folder, src, b, model, g, l, m, i, p, timed, r, prev):
            folder.mkdir(parents=True)
            self.assertEqual(prev is not None, timed)
            x = 1.1 if m == sc.CONTROL else 1.0
            r["runs"].append(
                dict(
                    folder=str(folder),
                    gear=g,
                    layer=l,
                    mode=m,
                    round=i,
                    pair=p,
                    golden={"y": gold},
                    outputs={"y": [gold] * (4 if timed else 2)},
                    profiled=False,
                    launch_authorized=True,
                    atc_plan={"block_dim": 8},
                    bit_mismatches=0,
                    validation=dict(
                        workspace_bytes=32,
                        model_weight_bytes=64,
                        timing={"samples_ms": [x] * 30} if timed else None,
                    ),
                )
            )

        host_ref = self.root / "host"
        sc.c.write(host_ref, {"binding": {"prior": {}}})
        ctx[1]["host_source"] = sc.c.identity(host_ref)

        def fake_compile(folder, *unused):
            folder.mkdir()
            p = folder / "om-graph.json"
            p.write_text("{}")
            return {"dump": sc.c.identity(p)}

        with ExitStack() as st, redirect_stdout(io.StringIO()):
            for name, result in (
                ("read_baseline", ctx),
                ("sdk_files", {}),
                ("validate_package", None),
                ("check_model", {}),
                ("run_cpu", {}),
                ("validate_cpu", None),
                ("verify_runtime_row", None),
            ):
                st.enter_context(patch.object(run, name, return_value=result))
            st.enter_context(patch.object(run, "compile_candidate", side_effect=fake_compile))
            st.enter_context(patch.object(run, "command", side_effect=command))
            st.enter_context(patch.object(run.host.envs, "for_prior", return_value=({}, {})))
            proc = st.enter_context(patch.object(run, "case_run", side_effect=launch))
            for phase in run.STAGES:
                self.assertEqual(
                    run.run(
                        SimpleNamespace(
                            phase=phase, root=base, archive=archive, allow_device=phase in run.DEVICE
                        )
                    ),
                    0,
                    phase,
                )
            self.assertEqual(proc.call_count, 10)
        final = json.loads((base / "compare/report.json").read_text())
        self.assertEqual(final["comparison"]["decision"], "GO")
        self.assertFalse(final["comparison"]["historical_samples_used"])
        self.assertEqual(final["device_execution"], "NOT_RUN")

    @unittest.skipUnless(
        os.environ.get("SCALE_CANN_ROOT"), "actual CANN 9.0 headers/models in ARM64 container"
    )
    def test_real_sdk_files(self):
        self.assertEqual(len(sc.sdk_files(os.environ["SCALE_CANN_ROOT"])), 16)
        self.assertEqual(
            review.expected_line(65536),
            "CPU_SCALE_WORDS 65536 EXPANDED_PER_VARIANT 1048576 PAYLOAD_GUARD_READONLY_ERRORS 0",
        )


if __name__ == "__main__":
    unittest.main()
