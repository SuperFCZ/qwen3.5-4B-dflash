"""Shared OMs and context; fake export/ATC is not NPU evidence."""
import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess

import pytest
import torch

from test_incremental_air_om import specs, small_threads
from rms_norm_test_support import adn_rms_norm_cpu
from qwen35_dflash.ascend310p.common_reuse import COMMON, FACTORY
from qwen35_dflash.ascend310p.compiler import compile_air_bundle
from qwen35_dflash.ascend310p.exporter import export_air_bundle
from qwen35_dflash.ascend310p.incremental_plan import write_incremental_plan
from qwen35_dflash.ascend310p.utils import sha256_file

pytestmark = pytest.mark.usefixtures("adn_rms_norm_cpu", "small_threads")


@pytest.fixture
def shared_build(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    calls = {"factory": [], "export": [], "atc": []}
    active = {}
    metadata_change = {}
    # Use the production audit property, including tuple-valued counter names.
    # The saved JSON converts these tuples to lists between export processes.
    from test_internal_dflash_bridge_rollback import FakeHIAIModel, FakeWrapper, InternalDFlashTarget
    bridge = InternalDFlashTarget(
        FakeWrapper(FakeHIAIModel()), device=torch.device("cpu"), dtype=torch.float16,
        kv_cache_max_len=64, rollback_enabled=True,
    )

    def factory(config):
        calls["factory"].append(config)
        values = []
        for spec in specs(verify_gdr=config.get("verify_gdr", "chunk"),
                    include_ordinary_decode=config.get("include_ordinary_decode", True)):
            meta = {**spec.metadata, "quant_source_lock": {"sha256": "host-source"},
                    "quant_input_manifest_sha256": "host-inputs",
                    "target_rollback_audit": dict(bridge.dflash_rollback_audit), **metadata_change}
            values.append(replace(spec, metadata=meta))
        active.update({v.name: v for v in values})
        return values

    monkeypatch.setattr("qwen35_dflash.ascend310p.exporter.resolve_callable", lambda ref: factory)

    class FakeTorchAir:
        __version__ = "host-fixture"

        def dynamo_export(self, *args, model, export_path, export_name, **kwargs):
            calls["export"].append(export_name)
            directory = Path(export_path)
            (directory / (export_name + ".air")).write_text(
                json.dumps(active[export_name].metadata["tensor_abi"]))
            # External weight payload is shared too, not merely the small AIR.
            (directory / "weights.bin").write_bytes(b"FAKE_WEIGHT_PAYLOAD")

    def atc(command, cwd):
        prefix = Path(next(x.split("=", 1)[1] for x in command if x.startswith("--output=")))
        name = Path(next(x.split("=", 1)[1] for x in command if x.startswith("--model="))).stem
        calls["atc"].append(name)
        signature = active[name].metadata["tensor_abi"]
        lines = ["FAKE_CHUNK " + name]
        for key, marker in (("inputs", "I"), ("outputs", "O")):
            for t in signature[key]:
                lines.append(" ".join((marker, t["name"], t["dtype"], str(len(t["shape"])),
                                       *(str(d) for d in t["shape"]))))
        if name == "draft" and "--dynamic_dims=16;64" in command:
            lines.append("GEARS 16 64")
        Path(str(prefix) + ".om").write_text("\n".join(lines))
        return subprocess.CompletedProcess(command, 0, "host fake ATC")

    def export(route, name, source=None, config=None):
        return export_air_bundle(
            FACTORY, config or {"verify_gdr": route}, tmp_path / name,
            torchair_module=FakeTorchAir(), reuse_common_from=source,
        )

    def compile(air, **kwargs):
        return compile_air_bundle(
            air["manifest_path"], soc_version=kwargs.pop("soc_version", "Ascend310P3"),
            atc_bin="/bin/true", runner=atc,
            atc_identity=kwargs.pop("atc_identity", "fake-host-test"), **kwargs,
        )

    original = compile(export("chunk", "chunk"))
    return calls, export, compile, original, metadata_change


def test_only_verify_is_exported_and_compiled_with_shared_common_files(shared_build, tmp_path, monkeypatch):
    calls, export, compile, original, _ = shared_build
    source = Path(original["manifest_path"])
    frozen = {p: p.read_bytes() for p in source.parent.rglob("*") if p.is_file()}
    calls["export"].clear()
    calls["atc"].clear()
    air = export("mtp", "mtp", source)
    assert calls["export"] == ["target_verify"]
    result = compile(air)
    assert calls["atc"] == ["target_verify"]
    dest = Path(result["manifest_path"])
    by_name = {g["name"]: g for g in result["graphs"]}
    for graph in original["graphs"]:
        if graph["name"] not in COMMON:
            continue
        current = by_name[graph["name"]]
        assert os.path.samefile(source.parent / graph["om"]["path"], dest.parent / current["om"]["path"])
        assert graph["atc_command"] == current["atc_command"]
        assert current["reused_from"]["method"] == "hardlink"
        for p in next(g for g in air["graphs"] if g["name"] == graph["name"])["payload_files"]:
            assert os.path.samefile(source.parent / p["path"], dest.parent / p["path"])
    all_oms = [*source.parent.glob("om/*.om"), *dest.parent.glob("om/*.om")]
    assert len(all_oms) == 8
    assert len({(p.stat().st_dev, p.stat().st_ino) for p in all_oms}) == 5
    assert all(p.read_bytes() == contents for p, contents in frozen.items())
    plan, _, _ = write_incremental_plan(dest, tmp_path / "shared-plan.txt", verify_gdr="mtp")
    assert plan.read_text().count("\ngraph ") == 4
    # Finished destination is independently loadable even if source paths move.
    moved = source.parent.with_name("moved-source")
    source.parent.rename(moved)
    write_incremental_plan(dest, tmp_path / "standalone-plan.txt", verify_gdr="mtp")
    runner = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if runner:
        from qwen35_dflash.ascend310p.cpp_runtime import run_cpp_pair
        monkeypatch.setenv("QWEN35_FAKE_ACCEPT", "3")
        report = run_cpp_pair(
            deployment_manifest=dest, runner=runner,
            runner_options={"device_model": "host-fixture", "cann": "fake",
                            "driver": "fake", "firmware": "fake", "runtime": "fake-acl"},
            prompt_token_ids=[4] * 65, eos_token_ids=[], device_id=0,
            max_new_tokens=40, max_draft_tokens=15,
            raw_output=tmp_path / "cpp.json", log_output=tmp_path / "cpp.log",
            low_memory=True, trace_rounds=True,
        )
        assert report["ordinary_parity"]["token_id_mismatches"] == 0


def test_single_compact_draft_is_shared_between_routes(shared_build, tmp_path):
    calls, export, compile, _, _ = shared_build
    cfg = {"verify_gdr": "chunk"}
    chunk = compile(export("chunk", "compact-chunk", config=cfg))
    assert {g["name"] for g in chunk["graphs"]} == {*COMMON, "target_verify"}
    for graph in chunk["graphs"]:
        if graph["name"].startswith("draft"):
            assert "--deterministic=0" in graph["atc_command"]
    source = Path(chunk["manifest_path"])
    calls["export"].clear()
    calls["atc"].clear()
    mtp = compile(export("mtp", "compact-mtp", source, {**cfg, "verify_gdr": "mtp"}))
    assert calls["export"] == ["target_verify"]
    assert calls["atc"] == ["target_verify"]
    destination = Path(mtp["manifest_path"])
    for name in COMMON:
        left = next(g for g in chunk["graphs"] if g["name"] == name)
        right = next(g for g in mtp["graphs"] if g["name"] == name)
        assert os.path.samefile(source.parent / left["om"]["path"], destination.parent / right["om"]["path"])
    oms = [*source.parent.glob("om/*.om"), *destination.parent.glob("om/*.om")]
    assert len({(p.stat().st_dev, p.stat().st_ino) for p in oms}) == 5
    plan, _, _ = write_incremental_plan(destination, tmp_path / "compact-plan.txt", verify_gdr="mtp")
    assert plan.read_text().count("\ngraph ") == 4


def test_recompile_determinism_covers_the_single_draft_graph(shared_build, tmp_path):
    from qwen35_dflash.ascend310p.compiler import recompile_draft_om
    _, export, compile, _, _ = shared_build
    original = compile(export("chunk", "compact"))
    source = Path(original["manifest_path"])
    calls = []
    def atc(command, cwd):
        prefix = Path(next(x.split("=", 1)[1] for x in command if x.startswith("--output=")))
        calls.append(prefix.name)
        assert "--deterministic=1" in command
        Path(str(prefix) + ".om").write_bytes(b"host compiled deterministic fixture")
        return subprocess.CompletedProcess(command, 0, "host ATC")
    result = recompile_draft_om(source, output=source.parent / "det1.json", deterministic=1,
                               atc_bin="/bin/true", runner=atc, atc_identity="fake-host-test")
    assert calls == ["draft"]
    assert result["recompilation"]["graphs"] == calls
    for old, new in zip(original["graphs"], result["graphs"]):
        if not old["name"].startswith("draft"):
            assert old == new


@pytest.mark.parametrize("damage", ["weights", "om", "manifest", "config", "source", "tensor_abi"])
def test_reuse_rejects_incompatible_source_before_export(shared_build, tmp_path, damage):
    calls, export, _, original, change = shared_build
    source = Path(original["manifest_path"])
    config = None
    if damage == "weights":
        (source.parent / "air/draft/weights.bin").write_bytes(b"damaged")
    elif damage == "om":
        (source.parent / "om/prefill.om").write_bytes(b"damaged")
    elif damage == "manifest":
        (source.parent / "air-manifest.json").write_text("{}")
    elif damage == "config":
        config = {"verify_gdr": "mtp", "max_sequence_length": 256}
    elif damage == "source":
        change["quant_source_lock"] = {"sha256": "changed-source"}
    else:
        change["tensor_abi"] = {}
    calls["export"].clear()
    with pytest.raises(ValueError):
        export("mtp", "bad", source, config)
    assert not calls["export"] and not (tmp_path / "bad").exists()


@pytest.mark.parametrize("mismatch", ["deterministic", "precision", "soc", "atc", "source_manifest", "local_payload"])
def test_reuse_rejects_compile_mismatch_before_atc(shared_build, tmp_path, mismatch):
    calls, export, compile, original, _ = shared_build
    source = Path(original["manifest_path"])
    air = export("mtp", "mtp", source)
    options = {}
    if mismatch == "deterministic":
        options["extra_args"] = ["--deterministic=0"]
    elif mismatch == "precision":
        options["extra_args"] = ["--precision_mode_v2=origin"]
    elif mismatch == "soc":
        options["soc_version"] = "Ascend310P1"
    elif mismatch == "atc":
        options["atc_identity"] = "different-build"
    elif mismatch == "source_manifest":
        source.write_text(json.dumps({**original, "modified": True}))
    else:
        # Replacing the linked name does not modify the source inode.
        payload = tmp_path / "mtp/air/draft/weights.bin"
        payload.unlink()
        payload.write_bytes(b"damaged")
    calls["atc"].clear()
    with pytest.raises(ValueError):
        compile(air, **options)
    assert not calls["atc"] and not (tmp_path / "mtp/deployment-manifest.json").exists()


def test_compile_failure_preserves_source_and_publishes_no_manifest(shared_build, tmp_path):
    _, export, _, original, _ = shared_build
    source = Path(original["manifest_path"])
    before = sha256_file(source.parent / "om/draft.om")
    air = export("mtp", "mtp", source)
    def fail(command, cwd):
        return subprocess.CompletedProcess(command, 17, "host failure")
    with pytest.raises(RuntimeError, match="host failure"):
        compile_air_bundle(air["manifest_path"], soc_version="Ascend310P3", atc_bin="/bin/true",
                           runner=fail, atc_identity="fake-host-test")
    assert sha256_file(source.parent / "om/draft.om") == before
    assert not (tmp_path / "mtp/deployment-manifest.json").exists()


def test_reuse_reports_real_nested_audit_difference(shared_build, tmp_path):
    calls, export, _, original, change = shared_build
    saved = original["graphs"][0]["metadata"]["target_rollback_audit"]
    assert isinstance(saved["cumulative_counter_fields"], list)
    change["target_rollback_audit"] = {
        **saved, "persistent_gdn_state": "unexpected_fp16_state",
    }
    calls["export"].clear()
    with pytest.raises(ValueError, match=r"target_prefill.metadata.target_rollback_audit.persistent_gdn_state"):
        export("mtp", "changed-audit", Path(original["manifest_path"]))
    assert not calls["export"] and not (tmp_path / "changed-audit").exists()


def test_cli_exposes_common_reuse():
    from qwen35_dflash.ascend310p.cli import build_parser
    for command in ("export-air", "build-om"):
        args = [command, "--factory", FACTORY, "--bundle-dir", "new",
                "--verify-gdr", "mtp", "--reuse-common-from", "chunk/deployment-manifest.json"]
        if command == "build-om":
            args += ["--soc-version", "Ascend310P3"]
        parsed = build_parser().parse_args(args)
        assert parsed.reuse_common_from == Path("chunk/deployment-manifest.json")


def test_reuse_requires_ordinary_decode_for_the_paired_benchmark(shared_build, tmp_path):
    calls, export, compile, _, _ = shared_build
    config = {"verify_gdr": "chunk", "include_ordinary_decode": False}
    source = compile(export("chunk", "no-decode", config=config))
    calls["factory"].clear()
    with pytest.raises(ValueError, match="include_ordinary_decode"):
        export("mtp", "new", source["manifest_path"],
               config={**config, "verify_gdr": "mtp"})
    assert not calls["factory"] and not (tmp_path / "new").exists()


def test_one_directory_has_exactly_five_oms_and_two_loadable_routes(shared_build, tmp_path, monkeypatch):
    calls, export, compile, original, _ = shared_build
    source = Path(original["manifest_path"])
    before = {p: p.read_bytes() for p in source.parent.rglob("*") if p.is_file()}
    calls["export"].clear()
    calls["atc"].clear()
    air = export("mtp", "chunk", source)
    assert air["manifest_path"] == str(source.parent / "air-manifest-mtp.json")
    result = compile(air)
    destination = Path(result["manifest_path"])
    assert destination.name == "deployment-manifest-mtp.json"
    assert calls["export"] == calls["atc"] == ["target_verify"]
    assert result["common_reuse"]["method"] == "same_directory"
    assert sorted(p.name for p in source.parent.glob("om/*.om")) == [
        "decode.om", "draft.om", "prefill.om", "verify_chunk.om", "verify_mtp.om",
    ]
    assert all(p.read_bytes() == data for p, data in before.items())
    old_verify = next(g for g in original["graphs"] if g["name"] == "target_verify")
    new_verify = next(g for g in result["graphs"] if g["name"] == "target_verify")
    assert old_verify["atc_log"] != new_verify["atc_log"]
    for route, manifest in (("chunk", source), ("mtp", destination)):
        write_incremental_plan(manifest, tmp_path / (route + "-five-om-plan.txt"), verify_gdr=route)
        runner = os.environ.get("QWEN35_CPP_TEST_RUNNER")
        if runner:
            from qwen35_dflash.ascend310p.cpp_runtime import run_cpp_pair
            monkeypatch.setenv("QWEN35_FAKE_ACCEPT", "3")
            report = run_cpp_pair(
                deployment_manifest=manifest, runner=runner,
                runner_options={"device_model": "host-fixture", "cann": "fake",
                                "driver": "fake", "firmware": "fake", "runtime": "fake-acl"},
                prompt_token_ids=[4] * 65, eos_token_ids=[], device_id=0,
                max_new_tokens=40, max_draft_tokens=15,
                raw_output=tmp_path / (route + "-cpp.json"),
                log_output=tmp_path / (route + "-cpp.log"), low_memory=True, trace_rounds=True,
            )
            assert report["ordinary_parity"]["token_id_mismatches"] == 0
    with pytest.raises(FileExistsError):
        export("mtp", "chunk", source)
    with pytest.raises(FileExistsError):
        compile(air)
    assert all(p.read_bytes() == data for p, data in before.items())
