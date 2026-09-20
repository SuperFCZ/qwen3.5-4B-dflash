"""Seven-OM capture orchestration; synthetic timings are not device evidence."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import profile_om
import profile_om_bundle as bundle


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    root = tmp_path / "bundle with spaces"
    root.mkdir()
    run = tmp_path / "run"
    run.mkdir()
    runner = tmp_path / "runner"
    runner.write_text("#!/bin/sh\nexit 0\n")
    runner.chmod(0o755)
    for name in bundle.OMS:
        (root / (name + ".om")).write_text(name)
    index = dict(artifact_kind="qwen35-draft-variants", status="PASS", bundles={})
    for variant in bundle.VARIANTS:
        index["bundles"][variant] = {}
        for route in bundle.ROUTES:
            graphs = []
            for role, name in (("target_prefill", "prefill"), ("target_decode", "decode"),
                               ("target_verify", "verify_" + route),
                               ("draft", "draft" if variant == "fp16" else "draft_" + variant)):
                graphs.append(dict(name=role, om=dict(path=name + ".om", sha256=name),
                    metadata=dict(incremental_contract=dict(draft_quantization=variant, verify_gdr=route))))
            path = root / (variant + "-" + route + ".json")
            digest = write_json(path, dict(status="PASS", graphs=graphs))
            index["bundles"][variant][route] = dict(status="PASS", manifest=path.name, manifest_sha256=digest)
    write_json(root / "draft-variants.json", index)
    args = profile_om.parser().parse_args(["--bundle-dir", str(root), "--run-dir", str(run),
        "--runner", str(runner), "--prompt-token-ids", "4,4,4", "--profile-audit-draft-inputs"])
    monkeypatch.delenv("ASCEND310P_SIMULATION_ONLY", raising=False)
    monkeypatch.setattr(bundle.shutil, "which", lambda _: "/test/msprof")
    return args, root, index


def synthetic_capture(args, *, output):
    stage = args.profile_stage
    capture = output / "capture"
    capture.mkdir(parents=True)
    count = 2 if stage == "prefill" else 1
    write_json(capture / (stage + "-stage-report.json"), dict(status="PASS_CAPTURE",
        profile_mode=args.profile_mode, profile_stage=stage, profiled_elapsed_ms=10 * count,
        captured_graph_calls={"draft" if stage == "draft" else "target_" + stage: count}))
    for suffix in ("-stage-summary.csv", "-operator-types.csv", "-operator-tasks.csv", "-hotspots.txt"):
        (capture / (stage + suffix)).write_text("SYNTHETIC_HOST_TEST_ONLY\n")
    return output


def test_all_selects_seven_unique_oms_and_two_verify_routes(inputs):
    args, _, _ = inputs
    jobs, _ = bundle.select_jobs(args)
    assert [j["om"] for j in jobs] == list(bundle.OMS)
    assert [j["profile_mode"] for j in jobs] == ["ordinary"] * 2 + ["dflash"] * 5
    assert [j["profile_stage"] for j in jobs] == ["prefill", "decode"] + ["draft"] * 3 + ["verify"] * 2
    assert [j["draft_quantization"] for j in jobs[2:5]] == list(bundle.VARIANTS)
    assert [j["verify_gdr"] for j in jobs[-2:]] == ["chunk", "mtp"]


def test_partial_w8_mtp_bundle_can_profile_just_draft(inputs):
    args, root, index = inputs
    args.profile_om = ["draft_w8a16"]
    index["status"] = "PARTIAL"
    index["bundles"] = {"w8a16": {"mtp": index["bundles"]["w8a16"]["mtp"]}}
    write_json(root / "draft-variants.json", index)
    jobs, _ = bundle.select_jobs(args)
    assert len(jobs) == 1 and jobs[0]["verify_gdr"] == "mtp"
    args.profile_om = ["all"]
    with pytest.raises(ValueError, match="has no compiled bundle"):
        bundle.select_jobs(args)


@pytest.mark.parametrize("failure", ["hash", "contract", "shared", "file"])
def test_bad_bundle_rejected_before_any_capture(inputs, failure):
    args, root, index = inputs
    entry = index["bundles"]["w8a16"]["chunk"]
    path = root / entry["manifest"]
    manifest = json.loads(path.read_text())
    if failure == "hash":
        path.write_text("{}")
    elif failure == "file":
        (root / "draft_w8a16.om").unlink()
    else:
        if failure == "contract":
            manifest["graphs"][-1]["metadata"]["incremental_contract"]["draft_quantization"] = "fp16"
        else:
            manifest["graphs"][0]["om"]["sha256"] = "different-target"
        entry["manifest_sha256"] = write_json(path, manifest)
        write_json(root / "draft-variants.json", index)
    with pytest.raises(ValueError):
        bundle.run_bundle_profile(args, run_single=lambda *a, **kw: pytest.fail("must preflight first"))
    assert not (args.run_dir / "msprof").exists()


@pytest.mark.parametrize("failure", [None, "draft_w4a16", "missing-report"])
def test_serial_capture_failure_retention_and_summary(inputs, failure):
    args, _, _ = inputs
    calls = []
    def run(options, *, output):
        assert output.name not in calls
        assert options.prompt_token_ids == "4,4,4"
        assert options.profile_audit_draft_inputs == (options.profile_stage in {"draft", "verify"})
        calls.append(output.name)
        if output.name == failure:
            raise subprocess.CalledProcessError(7, ["test-only"])
        synthetic_capture(options, output=output)
        if failure == "missing-report" and output.name == "draft_w4a16":
            (output / "capture/draft-stage-report.json").unlink()
    output = bundle.run_bundle_profile(args, run_single=run)
    assert calls == list(bundle.OMS)
    assert args.profile_stage == "all" and args.prompt_report is None
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == ("FAIL_OR_INCOMPLETE" if failure else "PASS_CAPTURE")
    assert report["formal_latency_evidence"] is False
    assert report["oms"][0]["graph_calls"] == 2
    assert report["oms"][0]["profiled_elapsed_ms"] == 20
    assert report["oms"][0]["ms_per_call"] == 10
    assert report["oms"][-1]["status"] == "PASS_CAPTURE"
    if failure:
        assert report["oms"][3]["status"] == "FAIL"
        assert report["oms"][3]["ms_per_call"] is None
    with (output / "summary.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 7


def test_interrupt_preserves_completed_and_unrun_status(inputs):
    args, _, _ = inputs
    def run(options, *, output):
        if output.name == "draft":
            raise KeyboardInterrupt
        synthetic_capture(options, output=output)
    with pytest.raises(KeyboardInterrupt):
        bundle.run_bundle_profile(args, run_single=run)
    report = json.loads(next((args.run_dir / "msprof").glob("oms-*/summary.json")).read_text())
    assert [j["status"] for j in report["oms"]] == ["PASS_CAPTURE"] * 2 + ["INTERRUPTED"] + ["NOT_RUN"] * 4


@pytest.mark.parametrize("selection", [["all", "draft"], ["draft", "draft"]])
def test_duplicate_selection_rejected(inputs, selection):
    args, _, _ = inputs
    args.profile_om = selection
    with pytest.raises(ValueError, match="distinct"):
        bundle.select_jobs(args)


def test_matrix_failure_reaches_cli_exit_status(inputs, monkeypatch):
    args, _, _ = inputs
    monkeypatch.setattr(profile_om, "run_profile", lambda *a, **kw: (_ for _ in ()).throw(ValueError("test failure")))
    assert profile_om.main(["--bundle-dir", str(args.bundle_dir), "--run-dir", str(args.run_dir),
        "--runner", str(args.runner), "--prompt-token-ids", "4", "--profile-om", "draft"]) == 1


# Real C++ runner and msprof controller, with fake ACL and fake collector only.
from test_unified_draft_bundle import unified_export, variant_builder, small_threads, adn_rms_norm_cpu
from test_msprof_stage_script import sandbox


@pytest.mark.usefixtures("small_threads", "adn_rms_norm_cpu")
def test_seven_om_cli_with_real_controller(unified_export, sandbox):
    from qwen35_dflash.ascend310p.compiler import compile_air_bundle

    runner = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if not runner:
        pytest.skip("requires the fake ACL C++ runner")
    export, atc, _ = unified_export
    air = export()
    compiled = compile_air_bundle(air["manifest_path"], atc_bin="/bin/true", soc_version="Ascend310P3",
                                 runner=atc, atc_identity="host-test")
    root = sandbox["tmp"]
    (root / "stubs/torch.py").unlink()
    for module in ("torch_npu", "acl"):
        (root / "stubs" / (module + ".py")).write_text('raise AssertionError("host-only C++ control path")\n')
    events = root / "acl-events.jsonl"
    sandbox["env"]["QWEN35_FAKE_EVENT_LOG"] = str(events)
    result = subprocess.run([sys.executable, "-B", str(profile_om.REPOSITORY / "tools/profile_om.py"),
        "--run-dir", str(root), "--runner", runner,
        "--bundle-dir", str(Path(compiled["manifest_path"]).parent), "--profile-om", "all",
        "--prompt-token-ids", ",".join(["4"] * 65), "--profile-warmup", "0",
        "--max-new-tokens", "16", "--profile-timeout", "5", "--msprof-bin", sandbox["msprof"]],
        env=sandbox["env"], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    output = next((root / "msprof").glob("oms-*"))
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == "PASS_CAPTURE" and len(report["oms"]) == 7
    assert [j["graph_calls"] for j in report["oms"]] == [2, 1, 1, 1, 1, 1, 1]
    captured = [row[0] for row in map(json.loads, events.read_text().splitlines()) if row[1]]
    assert captured == ["target_prefill"] * 2 + ["target_decode"] + ["draft"] * 3 + ["target_verify"] * 2
    for job in report["oms"]:
        stage_report = json.loads(Path(job["artifacts"]["stage_report"]).read_text())
        assert "fake-acl" in stage_report["runner_version"]
        assert Path(job["artifacts"]["operator_types"]).is_file()
