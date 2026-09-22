"""Unified command/flat storage checks; fake export/ATC are host evidence only."""
import json
from pathlib import Path
import subprocess

import pytest

from test_draft_variant_bundles import variant_builder, small_threads, adn_rms_norm_cpu
from qwen35_dflash.ascend310p import bundle_matrix, cli
from qwen35_dflash.ascend310p.common_reuse import FACTORY
from qwen35_dflash.ascend310p.compiler import compile_air_bundle
from qwen35_dflash.ascend310p.incremental_plan import write_incremental_plan

pytestmark = pytest.mark.usefixtures("small_threads", "adn_rms_norm_cpu")


@pytest.fixture
def unified_export(variant_builder, tmp_path, monkeypatch):
    build, calls = variant_builder
    monkeypatch.setattr("models.dflash_v1.draft_quantization.require_draft_checkpoint",
                        lambda path, variant: {"variant": variant})
    monkeypatch.setattr("qwen35_dflash.ascend310p.input_manifest.build_quant_input_manifest",
                        lambda **kwargs: kwargs["output"].write_text("{}"))
    def export(variants=("fp16", "w4a16", "w8a16"), routes=("chunk", "mtp"), backend="dequant", **extra):
        return bundle_matrix.export_matrix(FACTORY, dict(target_dir="target", quant_config="quant",
            receiver_models_dir="receiver", draft_quant_matmul=backend, **extra), tmp_path / "bundle",
            variants=variants, routes=routes, draft_dirs={v: v for v in variants}, torchair_module=build.air)
    return export, build.atc, calls


@pytest.mark.parametrize("variants,routes,count", [
    (("fp16", "w4a16", "w8a16"), ("chunk", "mtp"), 7),
    (("w8a16",), ("mtp",), 4),
])
def test_normal_export_compile_flat_directory(unified_export, tmp_path, variants, routes, count):
    export, atc, calls = unified_export
    air = export(variants, routes)
    assert len(calls) == count
    result = compile_air_bundle(air["manifest_path"], atc_bin="/bin/true", soc_version="Ascend310P3",
                               runner=atc, atc_identity="host-test")
    root = Path(result["manifest_path"]).parent
    oms = sorted(root.rglob("*.om"))
    assert len(oms) == count
    assert {p.parent for p in oms} == {root / "om"}
    assert len([c for c in calls if c[0] == "compile"]) == count
    assert {p.name for p in oms} == {"prefill.om", "decode.om",
        *("verify_" + r + ".om" for r in routes),
        *("draft" + ("" if v == "fp16" else "_" + v) + ".om" for v in variants)}
    for variant in variants:
        for route in routes:
            entry = result["bundles"][variant][route]
            assert Path(entry["manifest"]).name == entry["manifest"]
            _, deployment, contract = write_incremental_plan(root / entry["manifest"],
                tmp_path / f"{variant}-{route}.plan", verify_gdr=route)
            assert contract["draft_quantization"] == variant
            assert len(deployment["graphs"]) == 4
            assert next(g for g in deployment["graphs"] if g["name"] == "target_decode")["om"]["path"] == "om/decode.om"
    before = list(calls)
    with pytest.raises(FileExistsError):
        compile_air_bundle(air["manifest_path"], atc_bin="/bin/true", soc_version="Ascend310P3", runner=atc)
    assert calls == before


def test_all_payloads_checked_before_any_compile(unified_export):
    export, atc, calls = unified_export
    air = export()
    root = Path(air["manifest_path"]).parent
    record = json.loads((root / "air-manifest-w8a16-mtp.json").read_text())["graphs"][-1]
    (root / record["air"]["path"]).write_bytes(b"corrupted")
    before = list(calls)
    with pytest.raises(ValueError, match="integrity"):
        compile_air_bundle(air["manifest_path"], atc_bin="/bin/true", soc_version="Ascend310P3", runner=atc)
    assert calls == before


def test_normal_cli_accepts_matrix_and_compile_stays_separate(monkeypatch):
    parser = cli.build_parser()
    args = parser.parse_args(["export-air", "--factory", FACTORY, "--bundle-dir", "/bundle",
                             "--verify-gdr", "both", "--draft-quantizations", "fp16", "w4a16", "w8a16",
                             "--draft-weight-prepack", "nz"])
    seen = {}
    monkeypatch.setattr(bundle_matrix, "export_matrix", lambda *a, **kw: seen.update(kw, config=a[1]) or {"status": "PASS"})
    assert args.handler(args) == 0
    assert seen["variants"] == ["fp16", "w4a16", "w8a16"] and seen["routes"] == ["chunk", "mtp"]
    assert seen["config"]["draft_weight_prepack"] == "nz"
    compile_args = parser.parse_args(["compile-om", "--air-manifest", "/bundle/air-manifest.json",
                                     "--atc", "/atc", "--soc-version", "Ascend310P3"])
    assert compile_args.handler is cli.command_compile


def test_one_matrix_mixes_runtime_and_offline_nz_without_separate_runner(unified_export, tmp_path):
    export, atc, calls = unified_export
    air = export(backend="weight_quant", draft_weight_prepack="nz")
    result = compile_air_bundle(air["manifest_path"], atc_bin="/bin/true", soc_version="Ascend310P3",
                               runner=atc, atc_identity="host-test")
    root = Path(result["manifest_path"]).parent
    assert len(list((root / "om").glob("*.om"))) == 8
    assert len([c for c in calls if c[0] == "compile"]) == 8
    for variant in ("fp16", "w4a16", "w8a16"):
        for route in ("chunk", "mtp"):
            entry = result["bundles"][variant][route]
            _, deployment, contract = write_incremental_plan(root / entry["manifest"],
                tmp_path / f"{variant}-{route}.plan", verify_gdr=route)
            draft = next(g for g in deployment["graphs"] if g["name"] == "draft")
            assert contract["draft_quantization"] == variant
            if variant == "w8a16":
                assert draft["metadata"]["draft_weight_storage"] == "w8-int8-nz-const-v1"
                assert not draft.get("constant_inputs")
                assert (root / "om/draft_w8a16_static64.om").is_file()
            else:
                assert "draft_weight_storage" not in draft["metadata"]
                assert bool(draft.get("constant_inputs")) == (variant == "w4a16")


@pytest.mark.parametrize("variants,backend,extra,message", [
    (("fp16",), "weight_quant", {"draft_weight_prepack": "nz"}, "requires w8a16"),
    (("w8a16",), "dequant", {"draft_weight_prepack": "nz"}, "requires --draft-quant-matmul"),
    (("w8a16",), "weight_quant", {"draft_weight_prepack": "nz", "draft_weight_prepack_manifest": "cache"}, "not both"),
])
def test_invalid_prepack_options_fail_before_model_export(unified_export, tmp_path, variants, backend, extra, message):
    export, _, calls = unified_export
    with pytest.raises(ValueError, match=message):
        export(variants=variants, backend=backend, **extra)
    assert calls == []
    assert not (tmp_path / "bundle").exists()


def test_quant_failure_publishes_fp16_and_continues_other_quant(unified_export, tmp_path):
    export, atc, _ = unified_export
    # FP16 must be ready first even if the export listed another Draft first.
    air = export(variants=("w4a16", "w8a16", "fp16"))
    root = Path(air["manifest_path"]).parent
    attempted = []
    def fail_w4(command, cwd):
        stem = Path(next(s.split("=", 1)[1] for s in command if s.startswith("--output="))).name
        attempted.append(stem)
        if stem == "draft_w4a16":
            index = json.loads((root / "draft-variants.json").read_text())
            assert all(e["status"] == "PASS" for e in index["bundles"]["fp16"].values())
            return subprocess.CompletedProcess(command, 255, "quantized tiling failed")
        return atc(command, cwd)
    result = compile_air_bundle(air["manifest_path"], atc_bin="/bin/true", soc_version="Ascend310P3",
                               runner=fail_w4, atc_identity="host-test")
    assert result["status"] == "PARTIAL" and result["request_status"] == "FAIL"
    assert attempted.count("draft_w4a16") == 1
    assert attempted.count("draft_w8a16") == 1
    assert len(result["unique_oms"]) == 6
    for variant in ("fp16", "w8a16"):
        for route, entry in result["bundles"][variant].items():
            assert entry["status"] == "PASS"
            write_incremental_plan(root / entry["manifest"], tmp_path / f"{variant}-{route}.plan")
    for entry in result["bundles"]["w4a16"].values():
        assert entry["status"] == "FAIL" and "quantized tiling failed" in entry["error"]
        assert "manifest" not in entry
    before = {p: p.read_bytes() for p in (root / "om").glob("*.om")}
    # Recover/use FP16 without trying W4 again; retain the W8 PASS and W4 error.
    selected = compile_air_bundle(air["manifest_path"], atc_bin="/bin/true", soc_version="Ascend310P3",
        runner=lambda *a: pytest.fail("FP16 is already compiled"), atc_identity="host-test",
        resume=True, draft_quantizations=["fp16"])
    assert selected["request_status"] == "PASS" and selected["status"] == "PARTIAL"
    assert selected["bundles"] == result["bundles"]
    assert all(p.read_bytes() == data for p, data in before.items())


def test_fp16_subset_skips_quant_payloads_and_recovers_missing_index(unified_export):
    export, atc, calls = unified_export
    air = export()
    root = Path(air["manifest_path"]).parent
    quant_air = json.loads((root / "air-manifest-w4a16-chunk.json").read_text())
    draft = next(g for g in quant_air["graphs"] if g["name"] == "draft")
    (root / draft["air"]["path"]).write_bytes(b"unused corrupted quantized AIR")
    result = compile_air_bundle(air["manifest_path"], atc_bin="/bin/true", soc_version="Ascend310P3",
        runner=atc, atc_identity="host-test", draft_quantizations=["fp16"])
    assert result["request_status"] == "PASS" and result["status"] == "PARTIAL"
    assert len(result["unique_oms"]) == 5
    assert len([c for c in calls if c[0] == "compile"]) == 5
    assert result["bundles"]["w4a16"]["chunk"]["status"] == "NOT_RUN"
    # Reproduce an older compiler that wrote FP16 manifests but no index.
    (root / "draft-variants.json").unlink()
    leftover = root / "om/draft_w4a16.om"
    leftover.write_bytes(b"failed ATC partial output")
    result = compile_air_bundle(air["manifest_path"], atc_bin="/bin/true", soc_version="Ascend310P3",
        runner=lambda *a: pytest.fail("must reuse FP16"), atc_identity="host-test",
        resume=True, draft_quantizations=["fp16"])
    assert result["request_status"] == "PASS"
    assert result["unverified_unselected_oms"] == ["om/draft_w4a16.om"]
    assert leftover.read_bytes() == b"failed ATC partial output"
    assert leftover.name not in {Path(r["path"]).name for r in result["unique_oms"]}


def test_shared_target_failure_never_publishes_a_passing_member(unified_export):
    export, atc, _ = unified_export
    air = export()
    attempted = []
    def fail_prefill(command, cwd):
        stem = Path(next(s.split("=", 1)[1] for s in command if s.startswith("--output="))).name
        attempted.append(stem)
        if stem == "prefill":
            return subprocess.CompletedProcess(command, 255, "target failed")
        return atc(command, cwd)
    result = compile_air_bundle(air["manifest_path"], atc_bin="/bin/true", soc_version="Ascend310P3",
                               runner=fail_prefill, atc_identity="host-test")
    assert result["status"] == "FAIL" and result["request_status"] == "FAIL"
    assert attempted.count("prefill") == 1
    assert all(e["status"] == "FAIL" for entries in result["bundles"].values() for e in entries.values())
    assert not list(Path(air["manifest_path"]).parent.glob("deployment-manifest*.json"))


def test_interrupt_quant_compile_retains_fp16_index(unified_export):
    export, atc, _ = unified_export
    air = export()
    def interrupt(command, cwd):
        if any(s.endswith("/om/draft_w4a16") for s in command):
            raise KeyboardInterrupt()
        return atc(command, cwd)
    with pytest.raises(KeyboardInterrupt):
        compile_air_bundle(air["manifest_path"], atc_bin="/bin/true", soc_version="Ascend310P3",
                           runner=interrupt, atc_identity="host-test")
    index = json.loads((Path(air["manifest_path"]).parent / "draft-variants.json").read_text())
    assert index["status"] == "PARTIAL" and index["request_status"] == "FAIL"
    assert all(e["status"] == "PASS" for e in index["bundles"]["fp16"].values())
    assert index["bundles"]["w4a16"]["chunk"]["status"] == "INTERRUPTED"


@pytest.mark.parametrize("selection", [[], ["fp16", "fp16"], ["unknown"]])
def test_invalid_compile_subset_rejected_before_atc(unified_export, selection):
    export, _, _ = unified_export
    air = export()
    with pytest.raises(ValueError, match="select distinct"):
        compile_air_bundle(air["manifest_path"], atc_bin="/bin/true", soc_version="Ascend310P3",
            runner=lambda *a: pytest.fail("invalid selection"), draft_quantizations=selection)


@pytest.mark.parametrize("request_status,exit_code", [("PASS", 0), ("FAIL", 1)])
def test_compile_cli_partial_status_is_based_on_requested_drafts(monkeypatch, request_status, exit_code):
    args = cli.build_parser().parse_args(["compile-om", "--air-manifest", "/air-manifest.json",
        "--soc-version", "Ascend310P3", "--draft-quantizations", "fp16", "--resume"])
    def compile_selected(*a, **kw):
        assert kw["draft_quantizations"] == ["fp16"] and kw["resume"]
        return dict(status="PARTIAL", request_status=request_status)
    monkeypatch.setattr(cli, "compile_air_bundle", compile_selected)
    assert args.handler(args) == exit_code
