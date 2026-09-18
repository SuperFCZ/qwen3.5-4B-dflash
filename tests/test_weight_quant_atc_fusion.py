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
    FUSION_OPTION, WEIGHT_QUANT_TRANSPOSE_PASS as PASS, weight_quant_fusion_args,
)
from qwen35_dflash.ascend310p.compiler import AtcCompileError, compile_air_bundle

pytestmark = pytest.mark.usefixtures("small_threads", "adn_rms_norm_cpu", "weight_quant_cpu")


def native_graph(name="draft"):
    return {"name": name, "custom_op_audit": [dict(ge_op_type="WeightQuantBatchMatmulV2",
            status="PASS", ge_node_occurrences=26)]}


def fusion_file(args):
    return Path(next(s.split("=", 1)[1] for s in args if s.startswith(FUSION_OPTION + "=")))


def test_only_native_draft_on_310p_gets_targeted_switch(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    flags = ["--deterministic=0", "--precision_mode=must_keep_origin_dtype"]
    args = weight_quant_fusion_args(native_graph(), flags, soc_version="Ascend310P3")
    path = fusion_file(args)
    assert json.loads(path.read_text()) == {"Switch": {"GraphFusion": {PASS: "off"}}}
    assert args[:-1] == flags
    for graph, soc in ((native_graph("target_verify"), "Ascend310P3"),
                       ({"name": "draft"}, "Ascend310P3"), (native_graph(), "Ascend910B1")):
        assert weight_quant_fusion_args(graph, flags, soc_version=soc) == flags


@pytest.mark.parametrize("explicit", [None, "on", "off"])
@pytest.mark.parametrize("style", ["json", "text"])
def test_user_switches_preserved_and_never_modified(tmp_path, monkeypatch, explicit, style):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    config = {"Switch": {"GraphFusion": {"OtherPass": "off"}, "UBFusion": {"OtherUB": "on"}}}
    if explicit:
        config["Switch"]["GraphFusion"][PASS] = explicit
    original = (json.dumps(config) if style == "json" else "OtherPass:off\nOtherUB:on\n"
                + (f"{PASS}:{explicit}\n" if explicit else ""))
    source = tmp_path / "user.cfg"
    source.write_text(original)
    args = weight_quant_fusion_args(native_graph(), [f"{FUSION_OPTION}={source}"], soc_version="Ascend310P3")
    path = fusion_file(args)
    assert source.read_text() == original and path != source
    if style == "json":
        result = json.loads(path.read_text())
        assert result["Switch"]["GraphFusion"][PASS] == (explicit or "off")
        assert result["Switch"]["GraphFusion"]["OtherPass"] == "off"
        assert result["Switch"]["UBFusion"] == config["Switch"]["UBFusion"]
    else:
        assert f"{PASS}:{explicit or 'off'}" in path.read_text()
        assert "OtherPass:off\nOtherUB:on" in path.read_text()


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
        assert PASS in fusion_file(command).read_text()
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
                expected = variant != "fp16" and g["name"] == "draft"
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
    result = compile_air_bundle(air["manifest_path"], atc_bin="/bin/true", soc_version="Ascend310P3",
                               runner=atc, atc_identity="fake-atc")
    root = Path(result["manifest_path"]).parent
    data = json.loads((root / result["bundles"]["w8a16"]["chunk"]["manifest"]).read_text())
    draft = next(g for g in data["graphs"] if g["name"] == "draft")
    fusion_file(draft["atc_command"]).write_text("{}")
    attempted = []
    with pytest.raises(ValueError, match="integrity|hash"):
        compile_air_bundle(air["manifest_path"], atc_bin="/bin/true", soc_version="Ascend310P3",
                           runner=lambda *a: attempted.append(a), atc_identity="fake-atc", resume=True)
    assert not attempted


def test_cli_resume_is_compile_only():
    from qwen35_dflash.ascend310p.cli import build_parser
    args = build_parser().parse_args(["compile-om", "--air-manifest", "air-manifest.json",
                                      "--soc-version", "Ascend310P3", "--resume"])
    assert args.resume
