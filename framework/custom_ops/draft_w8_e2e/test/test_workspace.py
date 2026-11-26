"""Host checks of allocation isolation and the end-to-end latency denominators."""

import copy
import os
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from winners import c
from context_measure import summarize, timings
from run_workspace import archive_inputs, compare, memory
from test_context_prefill import fake_plan


def test_real_entry_point_changes_only_one_device_allocation(tmp_path):
    runner = os.getenv("QWEN35_CPP_TEST_RUNNER")
    if not runner:
        pytest.skip("build the host fake ACL runner first")
    _, plan = fake_plan(tmp_path)
    batch = tmp_path / "batch.txt"
    batch.write_text('QWEN35_PROMPT_BATCH_V1\nlong "' + ",".join(["2"] * 129) + '"\n')
    out = tmp_path / "runner-batch.json"
    allocations = tmp_path / "allocations.txt"
    args = [
        runner,
        "--model-kind",
        "chunk",
        "--mode",
        "dflash",
        "--model",
        plan["path"],
        "--model-sha256",
        plan["sha256"],
        "--prompt-batch",
        batch,
        "--prompt-batch-sha256",
        c.sha(batch),
        "--output",
        out,
        "--low-memory",
        "--trace-rounds",
        "--device-id",
        "0",
        "--eos-token-ids",
        "63",
        "--pad-token-id",
        "0",
        "--max-new-tokens",
        "16",
        "--max-draft-tokens",
        "15",
        "--warmup",
        "1",
        "--repetitions",
        "3",
    ]
    with (tmp_path / "runner.log").open("w") as log:
        subprocess.run(
            args,
            stdout=log,
            stderr=log,
            check=True,
            env=dict(os.environ, QWEN35_FAKE_ALLOC_POLICY_LOG=str(allocations)),
        )
    record = memory(dict(index=c.identity(out)), True)
    requested = [tuple(map(int, l.split())) for l in allocations.read_text().splitlines()]
    assert requested[0] == (record["shared_bytes"], 0)  # ACL_MEM_MALLOC_HUGE_FIRST
    assert len(requested) > 20 and all(policy == 2 for _, policy in requested[1:])  # NORMAL_ONLY
    with pytest.raises(ValueError, match="workspace page request"):
        memory(dict(index=c.identity(out)), False)


def case(prompt, plan, scale):
    n = (len(prompt["tokens"]) - 1) // 64
    m = dict(
        stage_ms=dict(
            draft_context64=[2 * scale] * n,
            target_prefill=[70 * scale] * (n + 1),
            draft=[15 * scale, 11 * scale],
            target_verify=[50 * scale] * 2,
        ),
        rounds=[
            dict(proposed_token_ids=[], target_token_ids=[1]),
            dict(proposed_token_ids=[4], target_token_ids=[4]),
            dict(proposed_token_ids=[9], target_token_ids=[9]),
        ],
        generated_token_ids=[1, 4, 9],
        stop_reason="eos",
        counters=dict(accepted_draft_tokens=2, drafted_tokens=2),
    )
    ms = []
    for i in range(3):
        row = copy.deepcopy(m)
        pre = (70 * (n + 1) + 2 * n) * scale
        dec = (126 + i) * scale
        row["latency_ms"] = dict(prefill=pre, decode=dec, model_total=pre + dec)
        ms.append(row)
    return dict(
        status="PASS",
        cpu_fallback=False,
        device_id=0,
        model=plan,
        prompt_token_ids=prompt["tokens"],
        eos_token_ids=[248044],
        limits=dict(max_new_tokens=128, max_draft_tokens=15),
        protocol=dict(warmup=1, repetitions=3, low_memory=True, round_trace_enabled=True),
        abi=dict(context_only_prefill=True, om_count=5),
        dflash=dict(measurements=ms),
    )


def test_grouped_target_draft_and_request_p95(tmp_path):
    prompts = [dict(id="s", group="short", tokens=[3] * 16), dict(id="l", group="long", tokens=[3] * 128)]
    plan = dict(sha256="plan")
    captures = []
    expected = {}
    for label, scale in (("old", 1), ("new", 0.9)):
        files = {}
        for p in prompts:
            path = tmp_path / (label + p["id"] + ".json")
            v = case(p, plan, scale)
            c.write(path, v)
            files[p["id"]] = c.identity(path)
            if label == "old":
                expected[p["group"]] = timings(v, p, plan, context=True)
        captures.append(dict(reports=files))
    expected["all"] = expected["short"] + expected["long"]
    archived = {k: {"context": summarize(v)} for k, v in expected.items()}
    result = compare(tmp_path, prompts, *captures, plan, archived)
    a, b = result["groups"]["all"].values()
    assert result["token_matches"] == result["round_matches"] == 6
    assert (a["target_prefill"]["count"], a["context"]["count"], a["verify"]["count"]) == (9, 3, 12)
    assert (a["draft_c16"]["count"], a["draft_c64"]["count"]) == (9, 3)
    assert a["draft_per_request"]["mean_ms"] == 27
    assert a["target_per_request"]["mean_ms"] == 205
    assert a["total"]["p95_ms"] == 270 and b["total"]["p95_ms"] == pytest.approx(243)
    assert b["generation_tok_s"] == pytest.approx(18000 / (0.9 * (3 * 197 + 3 * 269)))
    assert result["groups"]["short"]["workspace"]["context"]["p95_ms"] is None
    assert b["p95_request"]["prompt"] == "l"
    path = Path(captures[1]["reports"]["s"]["path"])
    z = c.read(path)
    z["dflash"]["measurements"][2]["rounds"][1]["proposed_token_ids"] = [7]
    c.write(path, z)
    captures[1]["reports"]["s"] = c.identity(path)
    mismatch = compare(tmp_path, prompts, *captures, plan, archived)
    assert mismatch["token_matches"] == 6 and mismatch["round_matches"] == 5


def test_reconstruct_exact_archived_batch_without_launch(tmp_path, monkeypatch):
    import run_workspace as script

    plan = dict(sha256="ut")
    prompts = [dict(id=f"p{i}", group="short" if i < 8 else "long") for i in range(20)]
    monkeypatch.setattr(script.suite, "load_prompts", lambda _: copy.deepcopy(prompts))
    reports = {}
    for p in prompts:
        path = tmp_path / (p["id"] + ".json")
        c.write(path, case(dict(p, tokens=[1, 2]), plan, 1))
        reports[p["id"]] = c.identity(path)
    file = tmp_path / "previous.txt"
    file.write_text("QWEN35_PROMPT_BATCH_V1\n" + "".join(f'{p["id"]} "1,2"\n' for p in prompts))
    sha = c.sha(file)
    monkeypatch.setattr(script, "BATCH_SHA", sha)
    idx = tmp_path / "batch.json"
    c.write(
        idx,
        dict(
            fake_acl=False,
            models_reused_across_prompts=True,
            low_memory=True,
            status="PASS",
            order="DFlash only; ordinary baseline external",
            model_sha256="ut",
            prompt_batch_sha256=sha,
            cases=prompts,
        ),
    )
    capture = dict(index=c.identity(idx), reports=reports)
    assert archive_inputs(tmp_path, capture, plan)["batch"]["sha256"] == sha
    f = Path(reports["p0"]["path"])
    z = c.read(f)
    z["prompt_token_ids"] = [1, 4]
    c.write(f, z)
    reports["p0"] = c.identity(f)
    with pytest.raises(ValueError, match="prompt token batch"):
        archive_inputs(tmp_path, capture, plan)
