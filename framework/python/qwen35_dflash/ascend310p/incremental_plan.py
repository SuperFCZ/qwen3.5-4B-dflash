"""Complete chunk ABI validation and hash-locked native C++ loading plans."""

from __future__ import annotations

import json
from pathlib import Path

from .utils import contained_path, load_json_object, require_run_output, sha256_file

LEGACY_CHUNK_ABI = "qwen35-dflash-chunk-v3"
ABI = "qwen35-dflash-chunk-v4"
LEGACY_MTP_ABI = "qwen35-dflash-mtp-v1"
MTP_ABI = "qwen35-dflash-mtp-v2"
VERIFY_GDR_ROUTES = ("chunk", "mtp")
ATTENTION_EXPORT_POLICY = "receiver_adn_all_seq_lengths_q_static_capacity_causal_mask"
DRAFT_LENGTH_POLICY = "anchor_plus_runtime_K_masked_in_every_attention_layer"
DRAFT_PREFILL_POLICY = "single_draft16_64_gears"
VERIFY_STATE_OUTPUT_POLICY = "raw_fp32_device_only_discard_first_pass_commit_second_pass"
MTP_STATE_OUTPUT_POLICY = "internal_fp32_bank_gather_accepted_slot"
ROLES = ("target_prefill", "target_decode", "target_verify", "draft")
DTYPES = {"int64": 8, "int16": 2, "float16": 2, "float32": 4, "int8": 1, "uint8": 1}


def verify_gdr_route(contract):
    route = {LEGACY_CHUNK_ABI: "chunk", ABI: "chunk",
             LEGACY_MTP_ABI: "mtp", MTP_ABI: "mtp"}.get(contract.get("abi"))
    if route is None or contract.get("verify_gdr", route) != route:
        raise ValueError("unsupported or inconsistent verification route/ABI")
    return route


def require_verify_gdr(contract, requested=None):
    route = verify_gdr_route(contract)
    if requested is not None and requested != route:
        raise ValueError(
            f"requested verify_gdr={requested}, but deployment contains {route}; "
            "select the matching manifest or export/compile a new bundle"
        )
    return route


def descriptor(name, dtype, shape):
    return {"name": name, "dtype": dtype, "shape": shape}


def verify_discard_descriptors(c):
    """Raw first-pass GDR outputs, ordered by linear layer, never cache state."""
    if verify_gdr_route(c) == "mtp":
        return []
    states = {s["name"]: s for s in c["target_states"]}
    return [
        descriptor("verify_discard_" + name, "float32", states[name]["shape"])
        for name in c["gdn_states"][1::2]
    ]


def _validate_tensor(tensor):
    if not isinstance(tensor, dict) or set(tensor) != {"name", "dtype", "shape"}:
        raise ValueError("invalid tensor ABI descriptor")
    if not isinstance(tensor["name"], str) or not tensor["name"].isidentifier():
        raise ValueError("invalid tensor name")
    if tensor["dtype"] not in DTYPES:
        raise ValueError("unsupported tensor dtype")
    shape = tensor["shape"]
    if (
        not isinstance(shape, list)
        or not shape
        or len(shape) > 8
        or any(type(d) is not int or d <= 0 for d in shape)
    ):
        raise ValueError("tensor ABI requires positive static dimensions")
    size = DTYPES[tensor["dtype"]]
    for dim in shape:
        size *= dim
    if size > 2**40:
        raise ValueError("tensor ABI exceeds the size limit")


def expected_signatures(c):
    states, drafts = c["target_states"], c["draft_states"]
    start, valid = (
        descriptor("start_position", "int64", [1]),
        descriptor("valid_rows", "int16", [1]),
    )
    feature = lambda rows: descriptor(
        "features", "float16", [1, rows, c["feature_width"]]
    )
    result = {}
    for name, rows, verify in (
        ("target_prefill", 64, False),
        ("target_decode", 1, False),
        ("target_verify", 16, True),
    ):
        outputs = [descriptor("target_top1", "int64", [1, rows if verify else 1])]
        if verify:
            outputs.append(descriptor("accepted_count", "int64", [1]))
        if rows != 1:
            outputs.append(feature(64))
        outputs += states
        if verify:
            outputs += c["verify_discard_states"]
        result[name] = {
            "inputs": [
                descriptor("input_ids", "int64", [1, rows]),
                start,
                valid,
                *states,
            ],
            "outputs": outputs,
        }
    result["draft"] = {
        "inputs": [
            feature(64),
            start,
            valid,
            descriptor("anchor", "int64", [1]),
            descriptor("proposal_count", "int16", [1]),
            *drafts,
            *c.get("draft_constants", []),
        ],
        "outputs": [descriptor("draft_top1", "int64", [1, 15]), *drafts],
    }
    return result


def validate_incremental_bundle(graphs):
    candidates = [
        g for g in graphs if g.get("metadata", {}).get("incremental_contract")
    ]
    if not candidates:
        return None
    names = {g["name"] for g in graphs}
    c = candidates[0]["metadata"]["incremental_contract"]
    if type(c.get("draft_context_rows")) is not int or c["draft_context_rows"] != 16:
        raise ValueError("incremental Draft requires draft_context_rows=16; regenerate AIR/OM in a new bundle directory")
    if (c.get("draft_prefill_policy") != DRAFT_PREFILL_POLICY
            or c.get("draft_context_gears") != [16, 64]):
        raise ValueError("Draft needs single_draft16_64_gears with [16, 64]; regenerate AIR/OM in a new bundle directory")
    roles = set(ROLES)
    required = roles - {"target_decode"}
    if (
        len(candidates) != len(graphs)
        or len(names) != len(graphs)
        or names not in (roles, required)
    ):
        raise ValueError(
            "incremental bundle needs prefill, verify, one draft and optional ordinary decode"
        )
    c = candidates[0]["metadata"]["incremental_contract"]
    route = verify_gdr_route(c)
    if c["abi"] in (ABI, MTP_ABI) and c.get("recurrent_state_dtype") != "float32":
        raise ValueError("Current incremental ABI requires FP32 recurrent state")
    if c.get("block_size") != 16 or c.get("prefill_rows") != 64:
        raise ValueError("unsupported incremental ABI; regenerate AIR/OM and rebuild the C++ runner")
    if c.get("draft_length_policy") != DRAFT_LENGTH_POLICY:
        raise ValueError("incremental Draft requires runtime proposal_count in every layer")
    if c.get("attention_export") != ATTENTION_EXPORT_POLICY:
        raise ValueError(
            "unsupported attention export ABI: regenerate AIR with "
            "all_seq_lengths_q and no integer pse_shift in a new bundle directory"
        )
    capacity = c.get("capacity")
    if (
        type(capacity) is not int
        or capacity < 64
        or capacity % 64
        or capacity > 32704
        or c.get("cache_capacity") != capacity + 64
    ):
        raise ValueError("invalid logical/scratch cache capacity")
    for key in ("vocab_size", "feature_width"):
        if type(c.get(key)) is not int or c[key] <= 0:
            raise ValueError(f"invalid {key}")
    for key in ("target_states", "draft_states", "capsules"):
        tensors = c.get(key)
        if not isinstance(tensors, list) or not tensors:
            raise ValueError(f"missing {key}")
        for tensor in tensors:
            _validate_tensor(tensor)
        if len({t["name"] for t in tensors}) != len(tensors):
            raise ValueError(f"duplicate tensor in {key}")
    names = {s["name"] for s in c["target_states"]}
    gdn, kv = c["gdn_states"], c["kv_states"]
    if (
        not gdn
        or not kv
        or len(gdn) % 2
        or len(kv) % 2
        or len(gdn) + len(kv) != len(names)
        or set(gdn) & set(kv)
        or set(gdn) | set(kv) != names
    ):
        raise ValueError("state partition is inconsistent")
    if len(c["capsules"]) != len(gdn) // 2 * (7 if route == "chunk" else 2):
        raise ValueError("GDR capsule count differs from linear layer count")
    if route == "mtp" or c["abi"] == ABI:
        states = {t["name"]: t for t in c["target_states"]}
        if any(states[n]["dtype"] != "float32" for n in gdn[1::2]):
            raise ValueError("Current Chunk/MTP ABI requires committed recurrent state in FP32")
    output_policy = VERIFY_STATE_OUTPUT_POLICY if route == "chunk" else MTP_STATE_OUTPUT_POLICY
    if (
        c.get("verify_state_output_policy") != output_policy
        or c.get("verify_discard_states") != verify_discard_descriptors(c)
    ):
        raise ValueError(
            "verify requires separate raw FP32 first-pass discard outputs" if route == "chunk"
            else "MTP verify requires internal FP32 bank selection without discard outputs"
        )
    expected = expected_signatures(c)
    gear_rank = sum(len(t["shape"]) for t in expected["draft"]["inputs"])
    if gear_rank > 128:
        raise ValueError(f"Draft dynamic gear requires {gear_rank} dimensions, exceeding "
                         "aclmdlIODims capacity 128; re-export and recompile the quantized "
                         "Draft with flat weight inputs")
    constants = c.get("draft_constants", [])
    variant = c.get("draft_quantization", "fp16")
    if variant not in ("fp16", "w4a16", "w8a16"):
        raise ValueError("unknown Draft quantization")
    from .weight_prepack import PREPACK_POLICY
    storage = c.get("draft_weight_storage")
    if storage is not None and (storage != PREPACK_POLICY or variant != "w8a16"):
        raise ValueError("invalid offline NZ Draft storage policy")
    count = 0 if variant == "fp16" or storage else 2 + 5 * len(c["draft_states"])
    if len(constants) != count:
        raise ValueError("Draft constant count differs from packed projection contract")
    for i, tensor in enumerate(constants):
        _validate_tensor(tensor)
        if tensor["name"] != f"draft_weight_{i:03d}" or tensor["dtype"] != (
                "float16" if i % 2 else "uint8" if variant == "w4a16" else "int8"):
            raise ValueError("Draft compressed constant ordering/dtype differs")
    for graph in graphs:
        if graph["metadata"]["incremental_contract"] != c:
            raise ValueError("incremental graph contracts differ")
        if graph["name"] == "draft" and graph["metadata"].get("draft_weight_storage") != storage:
            raise ValueError("Draft weight storage differs from bundle contract")
        signature = graph["metadata"].get("tensor_abi")
        if signature != expected[graph["name"]]:
            raise ValueError(f"incremental tensor ABI differs: {graph['name']}")
        axes = graph["metadata"].get("dynamic_input_axes", {})
        if axes != ({"features": [1]} if graph["name"] == "draft" else {}):
            raise ValueError("only Draft features axis 1 may be dynamic")
        if "dynamic" in graph and graph["dynamic"] is not (graph["name"] == "draft"):
            raise ValueError("Draft requires dynamic AIR; Target graphs must stay static")
        if graph.get("role") != graph["name"].replace("_", "-"):
            raise ValueError("incremental graph role differs")
        for direction in ("input", "output"):
            tensors = signature[direction + "s"]
            for tensor in tensors:
                _validate_tensor(tensor)
            if list(graph[direction + "_names"]) != [t["name"] for t in tensors]:
                raise ValueError("ordered graph tensor names differ from ABI")
    return c


def write_incremental_plan(deployment_manifest, output, *, mode="paired", verify_gdr=None):
    if mode not in {"paired", "ordinary", "dflash"}:
        raise ValueError("mode must be paired, ordinary or dflash")
    path = Path(deployment_manifest).resolve()
    manifest = load_json_object(path)
    if (
        manifest.get("status") != "PASS"
        or manifest.get("artifact_kind") != "qwen35-dflash-ascend310p-om-bundle"
    ):
        raise ValueError("incremental runner requires a passing OM bundle")
    graphs = manifest.get("graphs", [])
    c = validate_incremental_bundle(graphs)
    if c is None:
        raise ValueError("deployment is not an incremental chunk bundle")
    require_verify_gdr(c, verify_gdr)
    if mode != "dflash" and not any(g["name"] == "target_decode" for g in graphs):
        raise ValueError("ordinary/paired mode requires an exported target_decode OM")
    record = manifest["air_manifest"]
    air = contained_path(path.parent, record["path"])
    if sha256_file(air) != record["sha256"]:
        raise ValueError("AIR manifest hash differs")
    if validate_incremental_bundle(load_json_object(air)["graphs"]) != c:
        raise ValueError("deployment contract differs from AIR")
    output = require_run_output(output)
    if output.exists():
        raise FileExistsError(output)
    lines = [c["abi"], f"capacity {c['capacity']} {c['vocab_size']}",
             f"draft_prefill_policy {c['draft_prefill_policy']}"]
    for name in ROLES:
        if name == "target_decode" and mode == "dflash":
            continue
        graph = next(g for g in graphs if g["name"] == name)
        om = contained_path(path.parent, graph["om"]["path"])
        if (
            om.stat().st_size != graph["om"]["bytes"]
            or sha256_file(om) != graph["om"]["sha256"]
        ):
            raise ValueError(f"OM integrity check failed: {name}")
        if any(ch in str(om) for ch in "\r\n\t"):
            raise ValueError("OM path contains a control character")
        lines.append(
            f"graph {name} {json.dumps(str(om), ensure_ascii=False)} {graph['om']['sha256']}"
        )
        for direction, marker in (("inputs", "I"), ("outputs", "O")):
            for tensor in graph["metadata"]["tensor_abi"][direction]:
                lines.append(
                    " ".join(
                        (
                            marker,
                            tensor["name"],
                            tensor["dtype"],
                            str(len(tensor["shape"])),
                            *(str(d) for d in tensor["shape"]),
                        )
                    )
                )
        if graph.get("constant_inputs") or (graph["name"] == "draft" and c.get("draft_quantization", "fp16") != "fp16"):
            from .draft_constants import verify_constant_inputs
            verify_constant_inputs(graph, path.parent)
        for record in graph.get("constant_inputs", []):
            payload = contained_path(path.parent, record["path"])
            lines.append(f'C {record["name"]} {json.dumps(str(payload))} {record["sha256"]} {record["bytes"]}')
        lines.append("end")
    lines.append("done")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output, manifest, c
