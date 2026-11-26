"""Host-only tests of the real scheduler boundaries and profiling joins."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from call_latency_results import samples, task_contrasts, compare, dist
from diagnose_calls import check_same_inputs, workload
from winners import c
from test_context_prefill import fake_plan


def test_actual_chunk_boundary_and_allocation_controls(tmp_path):
    probe = os.getenv("DFLASH_TEST_LATENCY_PROBE")
    runner = os.getenv("QWEN35_CPP_TEST_RUNNER")
    if not probe or not runner:
        pytest.skip("build the host fake ACL runner and experiment first")
    _, plan = fake_plan(tmp_path)
    rmap = {
        "c16": "draft",
        "c64": "draft64",
        "prefill": "target_prefill",
        "verify": "target_verify",
        "context": "context",
    }
    roles = tmp_path / "roles.txt"
    roles.write_text(
        "".join(k + " " + json.dumps(str(tmp_path / (v + ".om"))) + "\n" for k, v in rmap.items())
    )
    tokens = [2] * 33
    real = tmp_path / "real.json"
    proc = subprocess.run(
        [
            runner,
            "--model-kind",
            "chunk",
            "--mode",
            "dflash",
            "--model",
            plan["path"],
            "--model-sha256",
            plan["sha256"],
            "--prompt-token-ids",
            ",".join(map(str, tokens)),
            "--eos-token-ids",
            "63",
            "--pad-token-id",
            "0",
            "--max-new-tokens",
            "48",
            "--max-draft-tokens",
            "15",
            "--warmup",
            "0",
            "--repetitions",
            "1",
            "--trace-rounds",
            "--output",
            real,
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    m = c.read(real)["benchmark"]["measurements"][0]
    _, first, second, *_ = m["rounds"]
    case = tmp_path / "case.txt"
    case.write_text(
        "".join(
            " ".join(map(str, v)) + "\n"
            for v in (
                tokens,
                m["generated_token_ids"][:1],
                first["proposed_token_ids"],
                first["target_token_ids"],
                second["proposed_token_ids"],
            )
        )
    )
    hashes = None
    baseline = None
    controls = {}
    for label, mode, policy in (
        ("clean", "clean", "stock"),
        ("pair", "pair", "stock"),
        ("huge", "pair", "huge-io-work"),
        ("weight", "pair", "draft-weights"),
        ("work", "pair", "draft-work"),
        ("end", "pair", "stock"),
    ):
        d = tmp_path / label
        d.mkdir()
        p = subprocess.run(
            [probe, plan["path"], roles, case, d, mode, policy, "4", "save" if label == "pair" else "check"],
            capture_output=True,
            text=True,
        )
        assert p.returncode == 0 and "HOST_FIXTURE_ONLY" in p.stdout, p.stderr
        s = samples(d / "samples.csv", mode, 4)
        if label == "clean":
            clean = s
            continue
        h = {"hashes": {k: c.identity(d / f"c{k}-hashes.tsv") for k in ("16", "64")}}
        if hashes is None:
            hashes = h
            baseline = s
            for k in ("16", "64"):
                assert len(list((d / ("c" + k) / "inputs").glob("*.bin"))) == 15
                assert len(list((d / ("c" + k) / "golden").glob("*.bin"))) == 11
        check_same_inputs(h, hashes)
        if label != "pair":
            controls["stock-end" if label == "end" else policy] = s
    iso = {k: dist([1, 1]) for k in ("16", "64")}
    result = compare(iso, clean, baseline, iso, None, controls)
    for g in iso:
        e = result[g]
        diff = e["contrasts"]
        reconstructed = (
            diff["predecessor"]["mean"] + diff["hot_io"]["mean"] + diff["resident_minus_isolated_ms"]
        )
        assert reconstructed == pytest.approx(e["paired_wall"]["after_target"]["mean"] - 1)


def test_profile_joins_by_fc_window_and_does_not_guess_missing_rows(tmp_path):
    path = tmp_path / "op_summary_01.csv"
    rows = []
    seq = [(-1, "after_target"), (0, "after_target"), (0, "hot_full"), (0, "hot_bare")]
    for model, gear in (("33", "64"), ("17", "16")):
        for i in range(4):
            start = 1000000000000000 + i * 100
            rows.extend([[model, "FC", "FC" + gear, start, 10], [model, "KV", "KV", start + 40, 20]])
            if not (gear == "16" and i == 2):
                rows.append([model, "Head", "MatMul", start + 70, 30])
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model ID", "Op Name", "OP Type", "Task Start Time(us)", "Task Duration(us)"])
        writer.writerows(reversed(rows))
    specs = {
        g: dict(fc=("FC", "FC" + g), nodes={"FC": "W8", "KV": "W8", "Head": "Head"}) for g in ("16", "64")
    }
    result = task_contrasts(tmp_path, specs, {g: seq for g in specs})["gears"]
    assert result["64"]["status"] == "COMPLETE"
    assert result["64"]["task_sums_ms"]["Head"]["hot_full"]["p50"] == pytest.approx(0.03)
    assert result["16"]["status"] == "PARTIAL"
    assert set(result["16"]["task_sums_ms"]) == {"W8"}
    assert result["16"]["task_sums_ms"]["W8"]["after_target"]["p50"] == pytest.approx(0.03)


def test_archive_excludes_context_preparations(tmp_path):
    plan = {"sha256": "test_plan"}
    cases = []
    reports = {}
    for i in range(20):
        name = "p" + str(i)
        prompt = [9] * (33 if i < 8 else 129)
        m = dict(
            stage_ms=dict(
                draft=[15.7, 15.2], draft_context64=[2.2] * ((len(prompt) - 1) // 64), target_verify=[51, 51]
            ),
            generated_token_ids=[3, 4],
            rounds=[dict(proposed_token_ids=[1], target_token_ids=[2, 5])] * 2,
        )
        z = dict(
            model=plan,
            cpu_fallback=False,
            prompt_token_ids=prompt,
            abi=dict(context_only_prefill=True),
            protocol=dict(repetitions=3),
            dflash=dict(measurements=[m] * 3),
        )
        path = tmp_path / (name + ".json")
        c.write(path, z)
        reports[name] = c.identity(path)
        cases.append(dict(id=name))
    index = tmp_path / "index.json"
    c.write(index, dict(fake_acl=False, model_sha256="test_plan", cases=cases))
    result, prompt = workload(dict(index=c.identity(index), reports=reports), plan)
    assert result["64"]["n"] == 8 * 3
    assert result["16"]["n"] == (8 + 12 * 2) * 3
    assert result["64"]["p50"] == 15.7
    assert prompt["name"] == "p0"


def test_missing_model_id_uses_both_gear_anchors(tmp_path):
    seq = [(0, "after_target"), (0, "hot_full"), (0, "hot_bare")]
    f = tmp_path / "op_summary.csv"
    with f.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Op Name", "Op Type", "Task Start Time(us)", "Task Duration(us)", "Output Shapes"])
        t = 0
        for g in ("64", "16"):
            for _ in seq:
                writer.writerows(
                    [
                        ["FC", "FC" + g, t, 12, ""],
                        ["Head", "MatMul", t + 30, 20, "15,248320"],
                        ["Head", "MatMul", t + 60, 22, "1,248320"],
                    ]
                )
                t += 100
    spec = {
        g: dict(fc=("FC", "FC" + g), nodes=dict(FC="fc", Head="Head"), types=dict(FC="FC" + g, Head="MatMul"))
        for g in ("16", "64")
    }
    result = task_contrasts(tmp_path, spec, {g: seq for g in spec})["gears"]
    assert all(v["status"] == "COMPLETE" and v["model_id"] is None for v in result.values())
    assert result["16"]["task_sums_ms"]["Head"]["after_target"]["p50"] == 0.02
