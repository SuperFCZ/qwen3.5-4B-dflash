"""ATC option/recovery tests with fake compilation; no 310P execution claim."""
import json
from pathlib import Path
import subprocess

import pytest

from test_draft_variant_bundles import variant_builder
from test_unified_draft_bundle import unified_export
from test_incremental_air_om import small_threads
from rms_norm_test_support import adn_rms_norm_cpu
from weight_quant_test_support import weight_quant_cpu
from qwen35_dflash.ascend310p.atc_fusion import (
    FUSION_OPTION, WEIGHT_QUANT_TRANSPOSE_PASS as PASS, normalized_atc_options,
)
from qwen35_dflash.ascend310p.compiler import (
    AtcCompileError, compile_air_bundle, _bundle_atc_args, _atc_failure_detail,
)

pytestmark = pytest.mark.usefixtures("small_threads", "adn_rms_norm_cpu", "weight_quant_cpu")


def native_graph(name="draft"):
    return {"name": name, "custom_op_audit": [dict(ge_op_type="WeightQuantBatchMatmulV2",
            status="PASS", ge_node_occurrences=26)]}


def fusion_file(args):
    return Path(next(s.split("=", 1)[1] for s in args if s.startswith(FUSION_OPTION + "=")))


def test_native_draft_no_longer_injects_an_ineffective_fusion_switch():
    flags = ["--deterministic=0", "--precision_mode=must_keep_origin_dtype"]
    _, arguments = _bundle_atc_args([native_graph()], flags, incremental=False, soc_version="Ascend310P3")
    assert arguments["draft"] == flags


@pytest.mark.parametrize("descriptor_length", [100, 5000])
def test_tiling_diagnostic_preserves_attrs_and_constraint_after_tensor_descriptions(descriptor_length):
    # Reproduce the receiver log structure: the actionable shape failure
    # follows seven input slots, outputs and attributes, beyond eight lines.
    reason = "op[WeightQuantBatchMatmulV2], Antiquant shape expect [2, 64], but is [64, 2]"
    output = "\n".join([
        "[Warning]: tiling struct conflict",
        "Inner_Error_Compile_Fail(E90003): Tiling func of op_type WeightQuantBatchMatmulV2 failed",
        "Compile_info: {'_cube_vector_core_type': 'AiCore'}",
        *["Inputs: " + "x" * descriptor_length for _ in range(3)],
        "None", "None", "None", "None", "Outputs: [64,64]",
        "[OP_TILING] Attrs: transpose_weight=True antiquant_group_size=128",
        "TraceBack (most recent call last):", reason,
        "fail to analyze context info", "Op WeightQuantBatchMatmulV2 tiling failed",
    ])
    detail = _atc_failure_detail(output, native_graph())
    assert "E90003" in detail and reason in detail
    assert "[OP_TILING] Attrs: transpose_weight=True antiquant_group_size=128" in detail
    assert "[K/group_size,N]" in detail and "Re-export AIR" in detail
    assert len(detail) < 10500


def test_no_template_diagnostic_does_not_claim_a_shape_error_or_all_group_support_absent():
    output = ("Inner_Error_Compile_Fail(E90003): WeightQuantBatchMatmulV2 tiling failed\n"
              "TraceBack (most recent call last):\nDo op tiling failed, no valid template is found.")
    detail = _atc_failure_detail(output, native_graph())
    assert "No WeightQuant template matched this SoC/layout/group/shape combination" in detail
    assert "--group-size 0 128 --weight-layout nk kn" in detail
    assert "synthetic per-channel control" in detail


def test_probe_failure_does_not_instruct_repeating_the_same_matrix():
    output = ("Inner_Error_Compile_Fail(E90003): WeightQuantBatchMatmulV2 tiling failed\n"
              "TraceBack (most recent call last):\nDo op tiling failed, no valid template is found.")
    graph = dict(native_graph("weight_quant_probe"), metadata={
        "weight_quant_probe": {"group_size": 128, "weight_layout": "nk"}})
    detail = _atc_failure_detail(output, graph)
    assert "compare the other controls" in detail and "--group-size" not in detail


def test_perchannel_shape_failure_is_an_invalid_control_not_unsupported_kernel():
    reason = "per-channel antiquant_scale shape only support [n, 1] or [n,], actual input shape is [1, 64]"
    output = ("Inner_Error_Compile_Fail(E90003): WeightQuantBatchMatmulV2 tiling failed\n"
              "TraceBack (most recent call last):\n" + reason)
    detail = _atc_failure_detail(output, native_graph("weight_quant_probe"))
    assert reason in detail and "before kernel support was tested" in detail
    assert "one-dimensional scale[N]" in detail
    assert "No WeightQuant template matched" not in detail


@pytest.mark.parametrize("explicit", ["on", "off"])
@pytest.mark.parametrize("style", ["json", "text"])
def test_explicit_user_switches_preserved_and_hashed(tmp_path, explicit, style):
    text = (json.dumps({"Switch": {"GraphFusion": {PASS: explicit}}})
            if style == "json" else f"{PASS}:{explicit}\n")
    source = tmp_path / "user.cfg"; source.write_text(text)
    flags = [f"{FUSION_OPTION}={source}"]
    _, arguments = _bundle_atc_args([native_graph()], flags, incremental=False, soc_version="Ascend310P3")
    assert arguments["draft"] == flags and source.read_text() == text
    assert normalized_atc_options(flags)[0].startswith(FUSION_OPTION + "=sha256:")


def interrupted_matrix(unified_export):
    export, atc, calls = unified_export
    air = export(backend="weight_quant")
    root = Path(air["manifest_path"]).parent
    def fail_w4(command, cwd):
        if any(s.endswith("/om/draft_w4a16") for s in command):
            return subprocess.CompletedProcess(command, 1, f"Compilation_Error(E20007): {PASS} failed")
        return atc(command, cwd)
    with pytest.raises(AtcCompileError, match="transpose/NZ graph fusion"):
        compile_air_bundle(air["manifest_path"], atc_bin="/bin/true", soc_version="Ascend310P3",
                           runner=fail_w4, atc_identity="fake-atc")
    assert len(list((root / "om").glob("*.om"))) == 5
    return air, root, atc, calls


def test_resume_reuses_five_completed_oms_and_builds_only_quantized_drafts(unified_export, tmp_path):
    air, root, atc, calls = interrupted_matrix(unified_export)
    before = {p: p.read_bytes() for p in (root / "om").glob("*.om")}
    commands = []
    def resume_atc(command, cwd):
        commands.append(command)
        assert not any(s.startswith(FUSION_OPTION + "=") for s in command)
        return atc(command, cwd)
    result = compile_air_bundle(air["manifest_path"], atc_bin="/bin/true", soc_version="Ascend310P3",
                               runner=resume_atc, atc_identity="fake-atc", resume=True)
    assert len(commands) == 2
    assert {Path(next(s.split("=", 1)[1] for s in c if s.startswith("--output="))).name
            for c in commands} == {"draft_w4a16", "draft_w8a16"}
    assert all(p.read_bytes() == data for p, data in before.items())
    assert len(list((root / "om").glob("*.om"))) == 7
    assert len(result["bundles"]) == 3
    for variant, entries in result["bundles"].items():
        for entry in entries.values():
            data = json.loads((root / entry["manifest"]).read_text())
            for g in data["graphs"]:
                expected = False
                assert bool(g.get("atc_fusion_switch")) == expected
                assert any(s.startswith(FUSION_OPTION + "=") for s in g["atc_command"]) == expected
    count = len(commands)
    compile_air_bundle(air["manifest_path"], atc_bin="/bin/true", soc_version="Ascend310P3",
                       runner=resume_atc, atc_identity="fake-atc", resume=True)
    assert len(commands) == count


@pytest.mark.parametrize("damage", ["om", "compiler", "soc", "options", "unrecorded"])
def test_resume_rejects_unverified_state_before_any_atc(unified_export, damage):
    air, root, atc, calls = interrupted_matrix(unified_export)
    args = dict(atc_bin="/bin/true", soc_version="Ascend310P3", atc_identity="fake-atc", resume=True)
    if damage == "om":
        (root / "om" / "prefill.om").write_text("tampered")
    elif damage == "compiler":
        args["atc_identity"] = "other-atc"
    elif damage == "soc":
        args["soc_version"] = "Ascend310P1"
    elif damage == "options":
        args["extra_args"] = ["--deterministic=1"]
    else:
        (root / "om" / "draft_w4a16.om").write_text("partial")
    attempted = []
    with pytest.raises(ValueError):
        compile_air_bundle(air["manifest_path"], runner=lambda *a: attempted.append(a), **args)
    assert not attempted


def test_resume_rejects_changed_fusion_config(unified_export):
    export, atc, _ = unified_export
    air = export(variants=("w8a16",), routes=("chunk",), backend="weight_quant")
    config = Path(air["manifest_path"]).parent / "user-fusion.cfg"
    config.write_text("OtherPass:off\n")
    flags = [f"{FUSION_OPTION}={config}"]
    result = compile_air_bundle(air["manifest_path"], atc_bin="/bin/true", soc_version="Ascend310P3",
                               extra_args=flags, runner=atc, atc_identity="fake-atc")
    root = Path(result["manifest_path"]).parent
    data = json.loads((root / result["bundles"]["w8a16"]["chunk"]["manifest"]).read_text())
    draft = next(g for g in data["graphs"] if g["name"] == "draft")
    fusion_file(draft["atc_command"]).write_text("{}")
    attempted = []
    with pytest.raises(ValueError, match="integrity|hash"):
        compile_air_bundle(air["manifest_path"], atc_bin="/bin/true", soc_version="Ascend310P3",
                           extra_args=flags, runner=lambda *a: attempted.append(a), atc_identity="fake-atc", resume=True)
    assert not attempted


def test_cli_resume_is_compile_only():
    from qwen35_dflash.ascend310p.cli import build_parser
    args = build_parser().parse_args(["compile-om", "--air-manifest", "air-manifest.json",
                                      "--soc-version", "Ascend310P3", "--resume"])
    assert args.resume
