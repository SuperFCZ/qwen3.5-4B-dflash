"""Offline carrier and actual AIR-edge checks; no target latency claims."""
import copy
import hashlib
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch

from qwen35_dflash.ascend310p.runtime_input_export import canonical_runtime_input_abi
from qwen35_dflash.ascend310p.weight_prepack import (
    CONST_DESC_POLICY, CONST_SHAPE_LOCK, PREPACK_POLICY, _const_value_desc, validate_prepacked_graph,
    pack_int8_nz, load_prepacked_weights, prepack_weight_quant_constants,
)
from qwen35_dflash.ascend310p.weight_quant_layout import normalize_weight_quant_layout, validate_weight_quant_layout
from test_weight_quant_layout import fixture_graph, evaluate_weight_quant_graph
from test_draft_variant_bundles import variant_builder
from test_incremental_air_om import small_threads
from rms_norm_test_support import adn_rms_norm_cpu
from weight_quant_test_support import weight_quant_cpu
from tools.pack_draft_weights_nz import convert

pytestmark = pytest.mark.usefixtures("small_threads")


def cached_weight(tmp_path, matrix):
    tmp_path.mkdir(parents=True, exist_ok=True)
    packed = pack_int8_nz(matrix)
    data = packed.numpy().tobytes()
    (tmp_path / "weight.bin").write_bytes(data)
    record = {"logical_shape": list(matrix.shape), "storage_shape": list(packed.shape),
        "dtype": "int8", "format": "FRACTAL_NZ", "logical_sha256": hashlib.sha256(matrix.numpy().tobytes()).hexdigest(),
        "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "path": "weight.bin"}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "policy": PREPACK_POLICY,
                                    "status": "PASS", "weight_count": 1, "weights": [record]}))
    return manifest


def const(node, value):
    desc = copy.deepcopy(node.output_desc[0])
    name = node.name
    node.Clear(); node.name, node.type = name, "Const"
    node.output_desc.add().CopyFrom(desc)
    node.attr["value"].t.desc.CopyFrom(desc)
    node.attr["value"].t.data = value.numpy().tobytes()


def check_const_value_shape(constant, x_shape):
    """Check serialized storage and logical K, not CANN inference success."""
    desc = constant.attr["value"].t.desc
    shape = list(desc.shape.dim)
    origin = list(desc.attr["origin_shape"].list.i)
    if desc.layout != "FRACTAL_NZ" or len(shape) != 4 or len(origin) != 2:
        raise ValueError("Const.value must describe physical NZ with a logical matrix origin")
    if x_shape[-1] != origin[-1]:
        raise ValueError(f"Ka[{x_shape[-1]}] != Kb[{origin[-1]}]")
    n, k = origin
    assert shape == [(k + 31) // 32, (n + 15) // 16, 16, 32]
    return [x_shape[0], n]


@pytest.mark.parametrize("n,k", [(64, 256), (19456, 2560), (2560, 9728), (2560, 12800)])
@pytest.mark.parametrize("m", [16, 64])
def test_const_value_encoding_keeps_logical_k_and_physical_storage(n, k, m):
    # Shape-only counterpart of the tiny/full projection probes. No large
    # weight allocation or target kernel simulation is needed for this gate.
    graph = fixture_graph()
    node = next(op for op in graph.op if op.name == "w")
    desc = node.output_desc[0]
    desc.layout = "FRACTAL_NZ"
    desc.shape.dim[:] = [(k + 31) // 32, (n + 15) // 16, 16, 32]
    desc.attr["format_for_int"].i = 29
    desc.attr["origin_format_for_int"].i = 2
    desc.attr["origin_shape"].list.i[:] = [n, k]
    desc.attr["origin_shape_initialized"].b = True
    desc.attr["origin_format_is_set"].b = True
    # The former public-Tensor encoding gave value ND[N,K] and output NZ4D.
    # It passed a logical-K-only check but cannot recreate the physical edge.
    node.attr["value"].t.desc.CopyFrom(desc)
    node.attr["value"].t.desc.shape.dim[:] = [n, k]
    node.attr["value"].t.desc.layout = "ND"
    with pytest.raises(ValueError, match="physical NZ"):
        check_const_value_shape(node, [m, k])
    node.attr["value"].t.desc.CopyFrom(_const_value_desc(desc))
    saved = type(graph).FromString(graph.SerializeToString())
    saved_weight = next(op for op in saved.op if op.name == "w")
    assert check_const_value_shape(saved_weight, [m, k]) == [m, n]
    # Inference metadata must not turn the outgoing carrier into ND.
    assert saved_weight.output_desc[0] == desc
    assert saved_weight.attr["value"].t.desc == desc


@pytest.mark.parametrize("n,k", [(1, 1), (17, 130), (67, 256), (19456, 2560), (2560, 9728)])
def test_all_signed_codes_and_nz_addressing_including_tails(n, k):
    q = (torch.arange(n * k).reshape(n, k) % 256 - 128).to(torch.int8)
    packed = pack_int8_nz(q)
    # Independent indexed NZ formula; sample lane and tile boundaries, then
    # compare the entire logical matrix, including all signed INT8 codes.
    rows = torch.arange(n)
    for column in sorted({0, k - 1, min(k - 1, 31), min(k - 1, 32), min(k - 1, 127), min(k - 1, 128)}):
        assert torch.equal(packed[column // 32, rows // 16, rows % 16, column % 32], q[:, column])
    columns = torch.arange(k)
    # Bound test memory at production gate/up sizes.
    for first in range(0, n, 128):
        r = rows[first:first + 128].reshape(-1, 1)
        assert torch.equal(packed[columns // 32, r // 16, r % 16, columns % 32], q[first:first + 128])
    restored = packed.permute(1, 2, 0, 3).reshape(packed.shape[1] * 16, packed.shape[0] * 32)
    assert not torch.count_nonzero(restored[n:, :])
    assert not torch.count_nonzero(restored[:n, k:])


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("externalized", [False, True])
def test_air_nz_constants_equal_original_graph_for_both_gears(tmp_path, shared, externalized):
    g = fixture_graph(shared=shared)
    original = copy.deepcopy(g)
    q = (torch.arange(64 * 256).reshape(64, 256) % 256 - 128).to(torch.int8)
    s = (1 + torch.arange(128).reshape(64, 2) % 4).half() / 32
    w = next(n for n in g.op if n.name == "w")
    const(w, q)
    if externalized:
        w.type = "FileConstant"; del w.attr["value"]
    cache = load_prepacked_weights(cached_weight(tmp_path, q))
    audit = normalize_weight_quant_layout(g)
    audit = prepack_weight_quant_constants(g, audit, {"w:0": q}, cache)
    assert not any(n.type == "TransData" for n in g.op)
    assert ("w" in {n.name for n in g.op}) is shared
    nz = next(n for n in g.op if n.name == "quant_weight_nz")
    assert nz.type == "Const" and not nz.input
    assert nz.attr["_out_shape_locked"].b is True
    assert CONST_SHAPE_LOCK == "_out_shape_locked"  # GE's literal has a leading underscore.
    assert [n.name for n in g.op if CONST_SHAPE_LOCK in n.attr and n.attr[CONST_SHAPE_LOCK].b] == [nz.name]
    tensor_desc = nz.attr["value"].t.desc
    assert list(tensor_desc.shape.dim) == [8, 4, 16, 32]
    assert tensor_desc.layout == "FRACTAL_NZ" and tensor_desc.attr["format_for_int"].i == 29
    assert tensor_desc.attr["origin_format_is_set"].b
    assert tensor_desc.attr["origin_format_for_int"].i == 2
    assert list(tensor_desc.attr["origin_shape"].list.i) == [64, 256]
    assert "storage_shape" not in tensor_desc.attr and "storage_format" not in tensor_desc.attr
    assert tensor_desc == nz.output_desc[0]
    assert list(nz.output_desc[0].shape.dim) == [8, 4, 16, 32]
    assert nz.output_desc[0].layout == "FRACTAL_NZ"
    assert list(nz.output_desc[0].attr["origin_shape"].list.i) == [64, 256]
    # Exercise the serialized physical value, including origin metadata.
    g = type(g).FromString(g.SerializeToString())
    nz = next(node for node in g.op if node.name == "quant_weight_nz")
    for rows in (16, 64):
        assert check_const_value_shape(nz, [rows, 256]) == [rows, 64]
        x = (torch.arange(rows * 256).reshape(rows, 256) % 7 - 3).half() / 16
        args = dict(x=x, w=q, s=s)
        assert torch.equal(evaluate_weight_quant_graph(g, args), evaluate_weight_quant_graph(original, args))
    entry = {"custom_op_audit": [{"ge_op_type": "WeightQuantBatchMatmulV2", "ge_node_occurrences": 1}],
             "runtime_input_abi": {"weight_quant_layout": audit},
             "metadata": {"draft_quantization": "w8a16", "draft_weight_storage": PREPACK_POLICY}}
    validate_weight_quant_layout(entry)
    broken = copy.deepcopy(entry)
    broken["runtime_input_abi"]["weight_quant_layout"]["nodes"][0]["prepacked_constant"]["storage_sha256"] = "bad"
    with pytest.raises(ValueError): validate_weight_quant_layout(broken)
    broken = copy.deepcopy(entry); broken["metadata"]["draft_quantization"] = "w4a16"
    with pytest.raises(ValueError): validate_weight_quant_layout(broken)
    for field, bad in (("descriptor_policy", None), ("value_shape", [64, 256]),
                       ("value_format", "ND"), ("value_origin_shape", [8, 4, 16, 32]),
                       ("value_origin_format", "FRACTAL_NZ"), ("output_shape_locked", False)):
        broken = copy.deepcopy(entry)
        bad_audit = broken["runtime_input_abi"]["weight_quant_layout"]
        # Keep duplicate audit records internally consistent; validate the
        # actual descriptor contract rather than just their equality.
        bad_audit["nodes"][0]["prepacked_constant"][field] = bad
        bad_audit["prepack"]["constants"][0][field] = bad
        with pytest.raises(ValueError): validate_weight_quant_layout(broken)
    legacy = copy.deepcopy(entry)
    del legacy["runtime_input_abi"]["weight_quant_layout"]["prepack"]["descriptor_policy"]
    with pytest.raises(ValueError, match="re-export AIR"):
        validate_weight_quant_layout(legacy)


@pytest.mark.parametrize("damage", ["physical_origin", "double_nz", "consumer", "payload", "unlocked", "nd_value",
                                    "public_input_lock", "weightquant_lock"])
def test_shape_lock_cannot_hide_bad_carrier_or_lock_runtime_computation(tmp_path, damage):
    g = fixture_graph()
    q = (torch.arange(64 * 256).reshape(64, 256) % 256 - 128).to(torch.int8)
    const(next(n for n in g.op if n.name == "w"), q)
    audit = prepack_weight_quant_constants(g, normalize_weight_quant_layout(g), {"w:0": q},
                                          load_prepacked_weights(cached_weight(tmp_path, q)))
    nz = next(n for n in g.op if n.name == "quant_weight_nz")
    quant = next(n for n in g.op if n.name == "quant")
    if damage == "physical_origin":
        nz.output_desc[0].attr["origin_shape"].list.i[:] = [8, 4, 16, 32]
    elif damage == "double_nz":
        nz.output_desc[0].shape.dim[:] = [8, 4, 1, 1, 16, 32]
    elif damage == "consumer":
        quant.input_desc[1].attr["origin_shape"].list.i[:] = [8, 4, 16, 32]
    elif damage == "payload":
        nz.attr["value"].t.data = q.numpy().tobytes()  # same length, wrong ordering
    elif damage == "unlocked":
        del nz.attr[CONST_SHAPE_LOCK]
    elif damage == "nd_value":
        # Same packed bytes and logical origin as the bad receiver artifact;
        # correct input/output descriptors alone must no longer admit this.
        desc = nz.attr["value"].t.desc
        desc.shape.dim[:] = [64, 256]
        desc.layout = "ND"
        desc.attr["format_for_int"].i = 2
        desc.attr["storage_shape"].list.i[:] = [8, 4, 16, 32]
        desc.attr["storage_format"].i = 29
    elif damage == "public_input_lock":
        next(n for n in g.op if n.name == "x").attr[CONST_SHAPE_LOCK].b = True
    else:
        quant.attr[CONST_SHAPE_LOCK].b = True
    with pytest.raises(ValueError):
        validate_prepacked_graph(g, audit)


@pytest.mark.parametrize("rows", [16, 64])
def test_value_alone_recreates_the_same_nz_edge_and_exact_output(tmp_path, rows):
    """Const recreation uses the raw GeTensor descriptor, not storage_* attrs.

    Model GE CreateConstOp's value-to-output boundary after serialization.
    This is a host regression for the AIR contract, not a GE execution mock.
    """
    graph = fixture_graph()
    original = copy.deepcopy(graph)
    # Vary both axes so a row/tile permutation cannot pass by coincidence.
    q = ((torch.arange(64).reshape(-1, 1) * 17 + torch.arange(256) * 13) % 256 - 128).to(torch.int8)
    const(next(n for n in graph.op if n.name == "w"), q)
    prepack_weight_quant_constants(graph, normalize_weight_quant_layout(graph), {"w:0": q},
                                  load_prepacked_weights(cached_weight(tmp_path, q)))
    graph = type(graph).FromString(graph.SerializeToString())
    weight = next(n for n in graph.op if n.name == "quant_weight_nz")
    tensor = copy.deepcopy(weight.attr["value"].t)
    # Discard the old output entirely, as a newly created Const would do.
    del weight.output_desc[:]
    weight.output_desc.add().CopyFrom(tensor.desc)
    physical = list(tensor.desc.shape.dim)
    logical = list(tensor.desc.attr["origin_shape"].list.i)
    assert tensor.desc.layout == "FRACTAL_NZ" and len(physical) == len(logical) + 2
    assert physical == [8, 4, 16, 32] and logical == [64, 256]
    consumer = copy.deepcopy(next(n for n in graph.op if n.name == "quant").input_desc[1])
    consumer.name = tensor.desc.name  # port labels need not match across an edge
    assert tensor.desc == consumer
    # Interpret bytes directly by value.shape; no storage-attribute override.
    nz = torch.frombuffer(bytearray(tensor.data), dtype=torch.int8).reshape(physical)
    for k in (0, 31, 32, 127, 128, 255):
        for n in (0, 15, 16, 63):
            assert nz[k // 32, n // 16, n % 16, k % 32] == q[n, k]
    x = (torch.arange(rows * 256).reshape(rows, 256) % 7 - 3).half() / 16
    s = (1 + (torch.arange(64).reshape(-1, 1) * 3 + torch.arange(2) * 5) % 13).half() / 64
    assert torch.equal(evaluate_weight_quant_graph(graph, dict(x=x, w=q, s=s)),
                       evaluate_weight_quant_graph(original, dict(x=x, w=q, s=s)))


@pytest.mark.parametrize("damage", ["runtime_input", "weight_changed", "payload_changed", "rehashed_wrong_layout"])
def test_cannot_use_runtime_or_corrupt_offline_weights(tmp_path, damage):
    g = fixture_graph()
    q = (torch.arange(64 * 256).reshape(64, 256) % 256 - 128).to(torch.int8)
    manifest = cached_weight(tmp_path, q)
    if damage != "runtime_input": const(next(n for n in g.op if n.name == "w"), q)
    audit = normalize_weight_quant_layout(g)
    if damage == "weight_changed": q = q.clone(); q[0, 0] += 1
    if damage in {"payload_changed", "rehashed_wrong_layout"}:
        wrong = q.numpy().tobytes()
        (tmp_path / "weight.bin").write_bytes(wrong)
        if damage == "rehashed_wrong_layout":
            data = json.loads(manifest.read_text()); data["weights"][0]["sha256"] = hashlib.sha256(wrong).hexdigest()
            manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        prepack_weight_quant_constants(g, audit, {"w:0": q}, load_prepacked_weights(manifest))


def test_actual_air_save_hook_binds_immutable_values_before_conversion(monkeypatch, tmp_path):
    g = fixture_graph(); g.op[0].output_desc[0].shape.dim[0] = 16
    x = torch.zeros(16, 256).half()
    q = (torch.arange(64 * 256).reshape(64, 256) % 256 - 128).to(torch.int8)
    s = torch.ones(64, 2).half() / 32
    manifest = cached_weight(tmp_path / "cache", q)
    torchair = ModuleType("torchair")
    def original(inputs, graph, path, weight_name):
        for i, value in enumerate(inputs):
            if id(value) in weight_name: const(graph.op[i], value)
        return False, len(weight_name)
    module = SimpleNamespace(_convert_data_to_const=original)
    monkeypatch.setitem(sys.modules, "torchair", torchair)
    monkeypatch.setitem(sys.modules, "torchair._utils.export_utils", module)
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    with canonical_runtime_input_abi(torchair, public_inputs=[x], public_names=["x"],
                                    weight_prepack_manifest=str(manifest)) as audit:
        module._convert_data_to_const([x, q, s], g, str(tmp_path), {id(q): "qweight", id(s): "scales"})
    assert audit["bindings"][0]["logical_name"] == "x"
    assert [n.name for n in g.op if n.type == "Data"] == ["x"]
    assert audit["weight_quant_layout"]["prepack"]["node_count"] == 1
    assert audit["weight_quant_layout"]["prepack"]["descriptor_policy"] == CONST_DESC_POLICY
    nz = next(op for op in g.op if op.name == "quant_weight_nz")
    assert check_const_value_shape(nz, [16, 256]) == [16, 64]
    assert not any(n.type == "TransData" for n in g.op)
    assert module._convert_data_to_const is original


@pytest.mark.usefixtures("weight_quant_cpu", "adn_rms_norm_cpu")
def test_script_export_and_compile_contracts_preserve_w4_and_target(tmp_path, variant_builder):
    build, calls = variant_builder
    original = build("w8a16", "chunk", extra={"draft_layers": 5, "draft_quant_matmul": "weight_quant"})
    manifest = convert(original, tmp_path / "offline")
    cache = load_prepacked_weights(manifest)
    assert len(json.loads(manifest.read_text())["weights"]) == 26
    assert cache["weights"]
    fp16 = build("fp16", "chunk")
    calls.clear()
    # Fresh output path/route; fake AIR/ATC validates ABI and reuse only. Real
    # constant data and serialization are covered by the hook test above.
    path = build("w8a16", "mtp", extra={"draft_layers": 5, "draft_quant_matmul": "weight_quant",
                                       "draft_weight_prepack_manifest": str(manifest)})
    data = json.loads(path.read_text())
    graph = next(g for g in data["graphs"] if g["name"] == "draft")
    assert not graph.get("constant_inputs")
    assert graph["metadata"]["draft_weight_storage"] == PREPACK_POLICY
    assert sum(len(t["shape"]) for t in graph["metadata"]["tensor_abi"]["inputs"]) == 47
    from qwen35_dflash.ascend310p.incremental_plan import write_incremental_plan
    plan, _, _ = write_incremental_plan(path, tmp_path / "prepacked.plan", verify_gdr="mtp")
    assert "\nC " not in plan.read_text()
    calls.clear()
    build("w4a16", "chunk", reuse_target=fp16,
          extra={"draft_quant_matmul": "weight_quant", "draft_weight_prepack_manifest": str(manifest)})
    assert calls == [("export", "draft"), ("compile", "draft")]
    with pytest.raises(FileExistsError): convert(original, tmp_path / "offline")


@pytest.mark.usefixtures("weight_quant_cpu", "adn_rms_norm_cpu")
@pytest.mark.parametrize("prefix", [1, 16, 17, 64, 65, 80, 81])
@pytest.mark.parametrize("accepted", [0, 3, 15])
@pytest.mark.parametrize("low_memory", [True, False])
@pytest.mark.parametrize("route", ["chunk", "mtp"])
def test_embedded_w8_static_oms_share_state_across_gears(tmp_path, monkeypatch, variant_builder, prefix, accepted, low_memory, route):
    runner = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if not runner:
        pytest.skip("requires fake ACL test runner; not device evidence")
    from qwen35_dflash.ascend310p.cpp_runtime import run_cpp_pair
    build, _ = variant_builder
    # The explicit fake exporter simulates an OM with embedded weights; this
    # test exercises the real C++ loading/gear/loop code through fake ACL.
    manifest = build("w8a16", route, extra={"draft_layers": 5, "draft_quant_matmul": "weight_quant",
                                            "draft_weight_prepack_manifest": "explicit-fake-cache"})
    uploads = tmp_path / "uploads.txt"
    gears = tmp_path / "gears.txt"
    monkeypatch.setenv("QWEN35_FAKE_GEAR_LOG", str(gears))
    monkeypatch.setenv("QWEN35_FAKE_ACCEPT", str(accepted))
    monkeypatch.setenv("QWEN35_FAKE_CONSTANT_UPLOAD_LOG", str(uploads))
    report = run_cpp_pair(deployment_manifest=manifest, runner=runner,
        runner_options={"device_model": "host-fixture", "cann": "fake", "driver": "fake",
                        "firmware": "fake", "runtime": "fake-acl"},
        prompt_token_ids=[4] * prefix, eos_token_ids=[], device_id=0, max_new_tokens=40, max_draft_tokens=15,
        raw_output=tmp_path / "result.json", log_output=tmp_path / "result.log", low_memory=low_memory)
    assert report["ordinary_parity"]["token_id_mismatches"] == 0
    assert report["abi"]["draft_prefill_policy"] == "static_draft16_64_oms"
    assert report["abi"]["om_count"] == 5
    assert report["protocol"]["max_resident_models"] == (4 if low_memory else 5)
    selected = [tuple(map(int, line.split())) for line in gears.read_text().splitlines()]
    assert all(rows == (64 if valid > 16 else 16) for _, valid, rows in selected)
    assert any(rows == 16 for _, _, rows in selected)
    assert any(rows == 64 for _, _, rows in selected) == (prefix > 16)
    assert not uploads.exists() or not uploads.read_text()
