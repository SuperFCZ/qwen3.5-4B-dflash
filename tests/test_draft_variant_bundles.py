"""Fake AIR/ACL contract tests, not checkpoint accuracy or device evidence."""
import copy
import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess

import pytest
import torch

from test_incremental_air_om import TinyTarget, gdr, mtp_gdr, attention_op, rotary, small_threads
from test_draft_quantization import config
from rms_norm_test_support import adn_rms_norm_cpu
from models.dflash_v1.modeling_dflash import DFlashDraftModel, WeightOnlyLinear
from models.dflash_v1.draft_quantization import GroupQuantLinear, pack_device_weight
from qwen35_dflash.ascend310p.incremental import incremental_graph_specs
from qwen35_dflash.ascend310p.quant_factory import AirDFlashOps
from qwen35_dflash.ascend310p.common_reuse import FACTORY
from qwen35_dflash.ascend310p.exporter import export_air_bundle
from qwen35_dflash.ascend310p.compiler import compile_air_bundle
from qwen35_dflash.ascend310p.draft_variants import compose_draft_variant
from qwen35_dflash.ascend310p.incremental_plan import write_incremental_plan
from qwen35_dflash.ascend310p.utils import sha256_file

pytestmark = pytest.mark.usefixtures("adn_rms_norm_cpu", "small_threads")


@pytest.fixture
def variant_builder(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    active, calls = {}, []
    def factory(cfg):
        layers = cfg.get("draft_layers", 2)
        draft = DFlashDraftModel(replace(config(), vocab_size=64, mask_token_id=63, block_size=16,
            num_hidden_layers=layers, layer_types=("full_attention",) * layers,
            target_layer_ids=(0, 1), num_target_layers=2), ops=AirDFlashOps(), dtype=torch.float16).eval()
        variant = cfg["draft_quantization"]
        if variant != "fp16":
            bits = 4 if variant == "w4a16" else 8
            for path, module in list(draft.named_modules()):
                if not isinstance(module, WeightOnlyLinear):
                    continue
                quant = GroupQuantLinear(pack_device_weight(torch.ones_like(module.weight, dtype=torch.int8), bits),
                    torch.full((module.out_features, module.in_features // 128), .5, dtype=torch.float16),
                    bits=bits, in_features=module.in_features, ops=draft.ops,
                    matmul_backend=cfg.get("draft_quant_matmul", "dequant"))
                parent, attr = path.rsplit('.', 1) if '.' in path else ('', path)
                setattr(draft.get_submodule(parent), attr, quant)
            draft.draft_quantization = variant
        # This fixture serializes ABIs; its Target math is never executed.
        specs = incremental_graph_specs(TinyTarget(), draft, capacity=128,
            metadata={"quant_source_lock": {"sha256": "fixture-source"}, "quant_input_manifest_sha256": variant,
                      "target_input_identity": {"target": cfg.get("target_identity", "fixed-target")},
                      "draft_dir": variant,
                      **({"draft_weight_prepack_manifest": cfg["draft_weight_prepack_manifest"]}
                         if cfg.get("draft_weight_prepack_manifest") and variant == "w8a16" else {})},
            gdr=gdr, attention=attention_op, rotary=rotary,
            verify_gdr=cfg["verify_gdr"], gdr_mtp=mtp_gdr, target_feature_layers=(0, 1))
        active.update({s.name: s for s in specs})
        return specs
    monkeypatch.setattr("qwen35_dflash.ascend310p.exporter.resolve_callable", lambda _: factory)
    class Air:
        __version__ = "host-fixture"
        def dynamo_export(self, *args, model, export_path, export_name, **kw):
            calls.append(("export", export_name))
            (Path(export_path) / (export_name + ".air")).write_text(json.dumps(active[export_name].metadata["tensor_abi"]))
            ops = active[export_name].custom_ops
            if ops:
                (Path(export_path) / "dynamo.pbtxt").write_text("".join(
                    f'op {{ op: "{op.ge_op_type}" }}\n' * op.minimum_occurrences for op in ops))
    def atc(command, cwd):
        name = Path(next(c.split("=", 1)[1] for c in command if c.startswith("--model="))).stem
        path = Path(next(c.split("=", 1)[1] + ".om" for c in command if c.startswith("--output=")))
        calls.append(("compile", name))
        lines = ["FAKE_CHUNK " + name]
        air_path = Path(next(c.split("=", 1)[1] for c in command if c.startswith("--model=")))
        signature = json.loads(air_path.read_text())
        shape_arg = next((c.split("=", 1)[1] for c in command if c.startswith("--input_shape=")), "")
        shapes = {field.split(":")[0]: [int(d) for d in field.split(":")[1].split(",")]
                  for field in shape_arg.split(";") if field}
        for direction, tag in (("inputs", "I"), ("outputs", "O")):
            for t in signature[direction]:
                if tag == "I" and t["name"] in shapes and -1 not in shapes[t["name"]]:
                    t = dict(t, shape=shapes[t["name"]])
                lines.append(" ".join(map(str, (tag, t["name"], t["dtype"], len(t["shape"]), *t["shape"])) ))
        if name == "draft" and "--dynamic_dims=16;64" in command:
            lines.append("GEARS 16 64")
        path.write_text("\n".join(lines))
        return subprocess.CompletedProcess(command, 0, "fake ATC")
    def build(variant, route, *, reuse_target=None, reuse_common=None, extra=None):
        cfg = dict(draft_quantization=variant, draft_dir=variant, input_manifest=variant,
                   shared_draft_features=True, verify_gdr=route, **(extra or {}))
        air = export_air_bundle(FACTORY, cfg, tmp_path / variant / route,
            torchair_module=Air(), reuse_target_from=reuse_target, reuse_common_from=reuse_common)
        result = compile_air_bundle(air["manifest_path"], soc_version="Ascend310P3", atc_bin="/bin/true",
                                   runner=atc, atc_identity="fake-atc")
        return Path(result["manifest_path"])
    build.air, build.atc = Air(), atc
    return build, calls


def test_seven_unique_oms_for_three_drafts_and_two_routes(variant_builder, tmp_path):
    build, calls = variant_builder
    fp16 = build("fp16", "chunk")
    mtp = build("fp16", "mtp", reuse_common=fp16)
    for variant in ("w4a16", "w8a16"):
        calls.clear()
        chunk = build(variant, "chunk", reuse_target=fp16)
        assert calls == [("export", "draft"), ("compile", "draft")]
        combined = compose_draft_variant(target_manifest=mtp, draft_manifest=chunk, bundle_dir=tmp_path / variant / "mtp")
        _, manifest, contract = write_incremental_plan(combined, tmp_path / (variant + ".plan"), verify_gdr="mtp")
        assert contract["draft_quantization"] == variant
        graph = next(g for g in manifest["graphs"] if g["name"] == "draft")
        assert len(graph["constant_inputs"]) == 22
        assert graph["om"]["path"] == "om/draft_" + variant + ".om"
        assert len((tmp_path / (variant + ".plan")).read_text().split('\nC ')) == 23
    oms = list(tmp_path.glob("*/*/om/*.om"))
    assert len(oms) == 24
    assert len({(p.stat().st_dev, p.stat().st_ino) for p in oms}) == 7


@pytest.mark.parametrize("variant", ["w4a16", "w8a16"])
def test_constants_survive_reset_upload_once_and_tamper_rejected(variant_builder, tmp_path, monkeypatch, variant):
    from qwen35_dflash.ascend310p.cpp_runtime import run_cpp_pair
    build, _ = variant_builder
    manifest = build(variant, "chunk")
    runner = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if not runner:
        pytest.skip("requires fake ACL test runner")
    log = tmp_path / "uploads.txt"
    monkeypatch.setenv("QWEN35_FAKE_CONSTANT_UPLOAD_LOG", str(log))
    monkeypatch.setenv("QWEN35_FAKE_CONSTANT_EXPECT", "w4a16" if variant == "w4a16" else "w8a16")
    report = run_cpp_pair(deployment_manifest=manifest, runner=runner,
        runner_options={"device_model": "host-fixture", "cann": "fake", "driver": "fake", "firmware": "fake", "runtime": "fake-acl"},
        prompt_token_ids=[4] * 65, eos_token_ids=[], device_id=0, max_new_tokens=40, max_draft_tokens=15,
        raw_output=tmp_path / "result.json", log_output=tmp_path / "result.log", low_memory=True)
    assert report["ordinary_parity"]["token_id_mismatches"] == 0
    assert len(log.read_text().splitlines()) == 22
    graph = next(g for g in json.loads(manifest.read_text())["graphs"] if g["name"] == "draft")
    payload = manifest.parent / graph["constant_inputs"][0]["path"]
    payload.write_bytes(b'\0' * payload.stat().st_size)
    with pytest.raises(ValueError, match="integrity"):
        write_incremental_plan(manifest, tmp_path / "tampered.plan")


def test_reuse_rejects_different_target_input_identity(variant_builder):
    build, calls = variant_builder
    source = build("fp16", "chunk")
    calls.clear()
    with pytest.raises(ValueError, match="configuration differs|target_input_identity"):
        build("w4a16", "chunk", reuse_target=source, extra={"target_identity": "changed"})
    assert calls == []


@pytest.mark.parametrize("variant", ["w4a16", "w8a16"])
def test_five_layer_quantized_draft_fits_acl_gears(variant_builder, tmp_path, variant):
    from copy import deepcopy
    from qwen35_dflash.ascend310p.incremental_plan import validate_incremental_bundle
    from qwen35_dflash.ascend310p.cpp_runtime import run_cpp_pair

    build, _ = variant_builder
    manifest = build(variant, "chunk", extra={"draft_layers": 5})
    data = json.loads(manifest.read_text())
    graph = next(g for g in data["graphs"] if g["name"] == "draft")
    constants = graph["constant_inputs"]
    assert len(constants) == 52
    assert all(len(t["shape"]) == 1 for t in constants)
    assert sum(len(t["shape"]) for t in graph["metadata"]["tensor_abi"]["inputs"]) == 99
    # Production layer count, not a two-layer fixture: the old 2-D ABI needs
    # 151 slots. Reject it before compiling/loading another unusable OM.
    old = deepcopy(data["graphs"])
    for g in old:
        for descriptor, payload in zip(g["metadata"]["incremental_contract"]["draft_constants"], constants):
            descriptor["shape"] = payload["logical_shape"]
    with pytest.raises(ValueError, match="151 dimensions.*capacity 128"):
        validate_incremental_bundle(old)
    runner = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if not runner:
        pytest.skip("requires fake ACL test runner")
    result = run_cpp_pair(deployment_manifest=manifest, runner=runner,
        runner_options={"device_model": "host-fixture", "cann": "fake", "driver": "fake",
                        "firmware": "fake", "runtime": "fake-acl"},
        prompt_token_ids=[4] * 65, eos_token_ids=[], device_id=0,
        max_new_tokens=40, max_draft_tokens=15,
        raw_output=tmp_path / "five-layer.json", log_output=tmp_path / "five-layer.log", low_memory=True)
    assert result["ordinary_parity"]["token_id_mismatches"] == 0
