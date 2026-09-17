"""Unified command/flat storage checks; fake export/ATC are host evidence only."""
import json
from pathlib import Path

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
    def export(variants=("fp16", "w4a16", "w8a16"), routes=("chunk", "mtp")):
        return bundle_matrix.export_matrix(FACTORY, dict(target_dir="target", quant_config="quant",
            receiver_models_dir="receiver"), tmp_path / "bundle",
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
                             "--verify-gdr", "both", "--draft-quantizations", "fp16", "w4a16", "w8a16"])
    seen = {}
    monkeypatch.setattr(bundle_matrix, "export_matrix", lambda *a, **kw: seen.update(kw) or {"status": "PASS"})
    assert args.handler(args) == 0
    assert seen["variants"] == ["fp16", "w4a16", "w8a16"] and seen["routes"] == ["chunk", "mtp"]
    compile_args = parser.parse_args(["compile-om", "--air-manifest", "/bundle/air-manifest.json",
                                     "--atc", "/atc", "--soc-version", "Ascend310P3"])
    assert compile_args.handler is cli.command_compile
