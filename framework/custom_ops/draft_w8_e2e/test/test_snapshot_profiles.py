"""The failed resident capture can be recovered from snapshots without running a Target."""

import os
import subprocess
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from winners import c
from diagnose_calls import profile_saved
from experiment import Stages
from full_replay import prepare_case
from snapshot_profiles import POLICIES, ProfileLoadFailure, check_application, replay, workspace_size
from test_context_prefill import fake_plan


def test_zero_profiler_exit_preserves_the_child_load_error():
    log = (
        "phase=before_models free_bytes=4096\nphase=load_failed graph=target_prefill\n"
        + "PROBE FAILED: aclmdlLoadFromFileWithMem failed: 245000 graph=target_prefill\n"
        + "exporting summary\n" * 500
    )
    with pytest.raises(ProfileLoadFailure, match="245000 graph=target_prefill"):
        check_application(0, False, log)
    with pytest.raises(RuntimeError, match="Script params are invalid"):
        check_application(0, False, "Script params are invalid")


def test_use_measured_workspace_without_adding_each_models_request():
    assert workspace_size("workspace policy=shared_serial shared_bytes=100 separate_sum_bytes=300") == 100
    with pytest.raises(ValueError, match="unique measured"):
        workspace_size("workspace policy=per_model")


@pytest.mark.parametrize("gear", [16, 64])
def test_single_model_policies_and_every_golden_byte(tmp_path, gear):
    exe = os.getenv("DFLASH_TEST_GUARDED_REPLAY")
    if not exe:
        pytest.skip("set DFLASH_TEST_GUARDED_REPLAY to the guarded runner linked to host fake ACL")
    fake_plan(tmp_path)
    om = tmp_path / ("draft.om" if gear == 16 else "draft64.om")
    case = dict(
        om=c.identity(om), weight_bytes=2097152, abi=dict(inputs=[], outputs=[]), io={"files": {}}, golden={}
    )
    data = {}
    for line in om.read_text().splitlines()[1:]:
        mark, name, typ, rank, *dims = line.split()
        side = "inputs" if mark == "I" else "outputs"
        shape = list(map(int, dims))
        assert int(rank) == len(shape)
        case["abi"][side].append(dict(name=name, dtype=typ, shape=shape))
        a = np.zeros(shape, dtype=typ)
        if name == "features":
            a.view("<u2")[0, :3, 0] = [0, 1, 2]
        elif name in ("valid_rows", "anchor", "proposal_count"):
            a[:] = dict(valid_rows=3, anchor=3, proposal_count=15)[name]
        elif name == "draft_top1":
            a[0] = np.arange(4, 19)
        elif mark == "I" and name.startswith("d"):
            a.reshape(-1)[1:] = 0.5
            data[name] = a
        elif mark == "O":
            a = data[name].copy()
            a.view("<u2").flat[0] = 3
        f = tmp_path / (side + "-" + name + ".bin")
        a.tofile(f)
        (case["io"]["files"] if mark == "I" else case["golden"])[name] = c.identity(f)
    for label, (work, io) in POLICIES.items():
        run = tmp_path / label
        run.mkdir()
        policy_log = run / "alloc.txt"
        z = replay(
            run,
            c.identity(exe),
            case,
            3145728,
            label,
            dict(os.environ, QWEN35_FAKE_ALLOC_POLICY_LOG=str(policy_log)),
        )
        assert z["allocation"]["golden_outputs"] == 11
        policies = [int(v.split()[1]) for v in policy_log.read_text().splitlines()]
        assert policies == [2 if work == "normal" else 0] + [2 if io == "normal" else 0] * 26
        assert not list((run / "case").glob("*.bin"))
    legacy = tmp_path / "legacy"
    prepare_case(legacy, case["abi"], case["io"], None)
    subprocess.run([exe, "0", str(om), str(legacy), str(legacy / "inputs")], capture_output=True, check=True)
    assert "allocation" not in c.read(legacy / "execution.json")
    assert c.sha(legacy / "draft_top1-postcheck.bin") == case["golden"]["draft_top1"]["sha256"]
    f = Path(case["golden"]["d4_value"]["path"])
    b = bytearray(f.read_bytes())
    b[-1] ^= 1
    f.write_bytes(b)
    case["golden"]["d4_value"] = c.identity(f)
    fail = tmp_path / "different"
    fail.mkdir()
    with pytest.raises(RuntimeError, match="snapshot mismatch: d4_value"):
        replay(fail, c.identity(exe), case, 3145728, "stock", os.environ)


def test_profile_only_admits_completed_receipts(tmp_path, monkeypatch):
    import diagnose_calls as dc
    import panel_profiler

    base, ctx, gap, out = [tmp_path / name for name in ("original", "context", "gap", "new")]
    for root in (base, ctx, gap, out):
        root.mkdir()
    planfile = ctx / "chosen.plan"
    planfile.write_text("frozen")
    plan = c.identity(planfile)
    orig = Stages(base, None)
    old = dict(status="E2E_MEASURED", drafts={})
    for g in ("16", "64"):
        f = base / ("C" + g + ".om")
        f.write_bytes(g.encode())
        model = dict(om=c.identity(f))
        old["drafts"][g] = model
        orig.run("audit-c" + g, {}, lambda p: model)
        orig.run("air-c" + g, {}, lambda p: dict(test=g))
    c.write(base / "result.json", old)
    Stages(ctx, None).run("plan", {}, lambda p: plan)
    c.write(
        ctx / "result.json",
        dict(
            status="MEASURED", token_matches=60, round_matches=60, baseline=c.identity(base / "result.json")
        ),
    )
    f = gap / "case.txt"
    f.write_text("same prompt")
    csv = gap / "samples.csv"
    csv.write_text("paired rows")
    Stages(gap, None).run(
        "paired",
        dict(case=c.sha(f), plan=plan, key=[{}, "pair", "stock", 16, True, None]),
        lambda d: dict(mode="pair", n=16, rows=c.identity(csv)),
    )
    timing = {"16": {}, "64": {}}
    c.write(gap / "timing.json", timing)
    c.write(
        gap / "result.json",
        dict(
            status="FAILED",
            phase="profile-pairs",
            timing=timing,
            actual_generation={},
            baseline=c.identity(ctx / "result.json"),
            plan=plan,
            prompt="known",
        ),
    )
    monkeypatch.setenv("CANN_ROOT", str(tmp_path))
    monkeypatch.setattr(dc, "samples", lambda *a: None)
    monkeypatch.setattr(dc, "receiver_runtime", lambda *a: ({}, {}))
    monkeypatch.setattr(panel_profiler, "bind_msprof", lambda *a: {})
    monkeypatch.setattr(
        dc,
        "snapshot_profiles",
        lambda stage, cann, env, cli, paired, models, graphs: stage(
            "snapshot-test", list(models), lambda d: dict(scope="one OM", complete=True)
        ),
    )
    state = {}
    assert profile_saved(out, gap, None, state) == "MEASURED"
    assert set(c.read(out / "stages.json")) == {"snapshot-test"}
    assert state["timing_source"] == c.identity(gap / "timing.json")
