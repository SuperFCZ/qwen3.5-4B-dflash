"""Static OM compilation/dispatch contracts; fake ACL is not target evidence."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from test_draft_variant_bundles import variant_builder
from test_incremental_air_om import small_threads
from weight_quant_test_support import weight_quant_cpu
from qwen35_dflash.ascend310p.compiler import (
    AtcCompileError, compile_air_bundle, recompile_draft_om, _validated_completed_bundle,
)
from qwen35_dflash.ascend310p.draft_gears import STATIC_POLICY, input_shape_arg, om_records
from qwen35_dflash.ascend310p.draft_variants import compose_draft_variant
from qwen35_dflash.ascend310p.incremental_plan import write_incremental_plan
from qwen35_dflash.ascend310p.utils import file_record

pytestmark = pytest.mark.usefixtures("small_threads", "weight_quant_cpu")
EXTRA = dict(draft_quant_matmul="weight_quant", draft_weight_prepack_manifest="explicit-fake-cache")


@pytest.fixture
def static_bundle(variant_builder):
    build, _ = variant_builder
    return build("w8a16", "chunk", extra=EXTRA)


def draft_at(path):
    return next(g for g in json.loads(path.read_text())["graphs"] if g["name"] == "draft")


def test_both_static_oms_bind_the_same_air_math_and_distinct_positive_shapes(static_bundle, tmp_path):
    graph = draft_at(static_bundle)
    records = om_records(graph)
    assert len(records) == 2
    assert graph["metadata"]["draft_compile_policy"] == STATIC_POLICY
    assert graph["metadata"]["dynamic_input_axes"] == {"features": [1]}
    assert graph["metadata"]["tensor_abi"]["inputs"][0]["shape"][1] == 64
    for gear in graph["static_gear_oms"]:
        command = gear["atc_command"]
        assert "--input_shape=" + input_shape_arg(graph, gear["rows"]) in command
        assert "--precision_mode=must_keep_origin_dtype" in command
        assert "--deterministic=0" in command
        assert not any(arg.startswith("--dynamic_") for arg in command)
        assert "," + str(gear["rows"]) + "," in input_shape_arg(graph, gear["rows"])
    plan, _, _ = write_incremental_plan(static_bundle, tmp_path / "plan")
    assert "draft_prefill_policy " + STATIC_POLICY in plan.read_text()
    assert "static_gear64 " in plan.read_text()
    assert "\nC " not in plan.read_text()


def test_static_m64_failure_does_not_publish_a_passing_deployment(static_bundle, variant_builder, tmp_path):
    build, _ = variant_builder
    root = tmp_path / "failed"
    shutil.copytree(static_bundle.parent, root, ignore=shutil.ignore_patterns("om", "deployment-manifest*.json"))
    def fail64(command, cwd):
        if any(arg.endswith("_static64") for arg in command if arg.startswith("--output=")):
            return subprocess.CompletedProcess(command, 255, "static M64 compile failed")
        return build.atc(command, cwd)
    with pytest.raises(AtcCompileError, match="static M64 compile failed"):
        compile_air_bundle(root / "air-manifest.json", atc_bin="/bin/true", soc_version="Ascend310P3",
                           runner=fail64, atc_identity="fake-atc")
    assert (root / "om/draft_w8a16.om").exists()
    assert not (root / "deployment-manifest.json").exists()


@pytest.mark.parametrize("damage", ["missing", "hash", "shape", "precision", "inventory"])
def test_plan_rejects_missing_tampered_or_incompatible_second_gear(static_bundle, tmp_path, damage):
    data = json.loads(static_bundle.read_text())
    graph = next(g for g in data["graphs"] if g["name"] == "draft")
    second = graph["static_gear_oms"][1]
    if damage in ("missing", "hash"):
        path = static_bundle.parent / second["om"]["path"]
        if damage == "missing":
            path.unlink()  # This test's own generated fake OM.
        else:
            path.write_text("tampered fake OM")
    elif damage == "shape":
        second["atc_command"] = ["--input_shape=" + input_shape_arg(graph, 16)
            if arg.startswith("--input_shape=") else arg for arg in second["atc_command"]]
    elif damage == "precision":
        second["atc_command"].append("--precision_mode=allow_fp32_to_fp16")
    else:
        graph["static_gear_oms"].pop()
    static_bundle.write_text(json.dumps(data))
    with pytest.raises((ValueError, FileNotFoundError)):
        write_incremental_plan(static_bundle, tmp_path / "invalid.plan")
    assert not (tmp_path / "invalid.plan").exists()


def test_resume_and_draft_only_recompile_validate_and_build_both_static_gears(static_bundle, variant_builder):
    build, calls = variant_builder
    saved = json.loads(static_bundle.read_text())
    air_path = static_bundle.parent / "air-manifest.json"
    air = json.loads(air_path.read_text())
    assert _validated_completed_bundle(static_bundle, air_path=air_path, graphs=air["graphs"],
        atc_path=Path("/bin/true").resolve(), soc_version="Ascend310P3",
        graph_arguments=saved["compiler"]["graph_extra_args"], identity="fake-atc") == saved
    calls.clear()
    result = recompile_draft_om(static_bundle, output=static_bundle.parent / "det1.json", deterministic=1,
                               atc_bin="/bin/true", runner=build.atc, atc_identity="fake-atc")
    assert calls == [("compile", "draft"), ("compile", "draft")]
    rebuilt = next(g for g in result["graphs"] if g["name"] == "draft")
    om_records(rebuilt)
    assert all("--deterministic=1" in g["atc_command"] for g in rebuilt["static_gear_oms"])
    for original in saved["graphs"]:
        if original["name"] != "draft":
            assert original == next(g for g in result["graphs"] if g["name"] == original["name"])


def test_reuse_and_route_composition_keep_both_gear_payloads(static_bundle, variant_builder, tmp_path):
    build, calls = variant_builder
    calls.clear()
    reused = build("w8a16", "mtp", reuse_common=static_bundle, extra=EXTRA)
    assert ("compile", "draft") not in calls and ("export", "draft") not in calls
    for a, b in zip(om_records(draft_at(static_bundle)), om_records(draft_at(reused))):
        assert os.path.samefile(static_bundle.parent / a["path"], reused.parent / b["path"])
    target = build("fp16", "mtp")
    composed = compose_draft_variant(target_manifest=target, draft_manifest=static_bundle,
                                     bundle_dir=tmp_path / "composed")
    for a, b in zip(om_records(draft_at(static_bundle)), om_records(draft_at(composed))):
        assert os.path.samefile(static_bundle.parent / a["path"], composed.parent / b["path"])


def test_native_runner_checks_static64_physical_descriptor_before_execute(static_bundle, tmp_path, monkeypatch):
    from qwen35_dflash.ascend310p.cpp_runtime import run_cpp_pair
    runner = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if not runner:
        pytest.skip("fake ACL runner required; no target evidence")
    data = json.loads(static_bundle.read_text())
    graph = next(g for g in data["graphs"] if g["name"] == "draft")
    second = graph["static_gear_oms"][1]
    path = static_bundle.parent / second["om"]["path"]
    text = path.read_text()
    assert "I features float16 3 1 64 " in text
    path.write_text(text.replace("I features float16 3 1 64 ", "I features float16 3 1 16 "))
    second["om"] = file_record(path, relative_to=static_bundle.parent)
    static_bundle.write_text(json.dumps(data))
    events = tmp_path / "events"
    monkeypatch.setenv("QWEN35_FAKE_EVENT_LOG", str(events))
    with pytest.raises(RuntimeError, match="OM tensor ABI differs"):
        run_cpp_pair(deployment_manifest=static_bundle, runner=runner,
            runner_options={"device_model": "host-fixture", "cann": "fake", "driver": "fake",
                            "firmware": "fake", "runtime": "fake-acl"},
            prompt_token_ids=[4] * 17, eos_token_ids=[], device_id=0, max_new_tokens=8, max_draft_tokens=7,
            raw_output=tmp_path / "bad-static64.json", log_output=tmp_path / "bad-static64.log")
    assert not events.exists()
