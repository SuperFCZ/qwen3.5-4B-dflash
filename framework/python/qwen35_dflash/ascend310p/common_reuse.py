"""Share immutable ordinary and Draft artifacts between verification routes."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from .incremental_plan import ABI, MTP_ABI, validate_incremental_bundle, verify_gdr_route
from .utils import contained_path, file_record, load_json_object, sha256_file

FACTORY = "qwen35_dflash.ascend310p.quant_factory:create_quant_incremental_graphs"
COMMON = ("target_prefill", "target_decode", "draft")
_ROUTE_CONTRACT = {
    "abi", "verify_gdr", "target_states", "capsules", "state_policy",
    "verify_state_output_policy", "verify_discard_states",
}


def artifact_stem(graph):
    """Keep runtime roles stable while giving the five current OMs distinct names."""
    contract = graph.get("metadata", {}).get("incremental_contract", {})
    if contract.get("abi") not in (ABI, MTP_ABI):
        return graph["name"]
    name = graph["name"]
    if name == "target_verify":
        return "verify_" + verify_gdr_route(contract)
    return {"target_prefill": "prefill", "target_decode": "decode"}.get(name, name)


def _verified_file(root, record):
    path = contained_path(root, record["path"])
    if (not path.is_file() or path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]):
        raise ValueError(f"Common reuse artifact integrity check failed: {path}")
    return path


def _metadata(value):
    # Compare the representation that actually survives an AIR-manifest write.
    # The live bridge audit contains tuples (e.g. cumulative_counter_fields);
    # JSON reloads them as lists, without changing the graph or its contract.
    result = json.loads(json.dumps(value, allow_nan=False))
    for key in ("factory_id", "verify_gdr"):
        result.pop(key, None)
    for key in _ROUTE_CONTRACT:
        result["incremental_contract"].pop(key, None)
    return result


def _difference(old, new, path):
    """Identify the first real mismatch without dumping the whole manifest."""
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(old.keys() | new.keys()):
            child = f"{path}.{key}"
            if key not in old or key not in new:
                return child + (" (missing from saved manifest)" if key not in old else " (missing from current export)")
            if old[key] != new[key]:
                return _difference(old[key], new[key], child)
    if isinstance(old, list) and isinstance(new, list):
        for i, (left, right) in enumerate(zip(old, new)):
            if left != right:
                return _difference(left, right, f"{path}[{i}]")
        if len(old) != len(new):
            return f"{path}.length (saved={len(old)}, current={len(new)})"
    return f"{path} (saved={repr(old)[:160]}, current={repr(new)[:160]})"


def _config(config):
    return {k: v for k, v in config.items() if k != "verify_gdr"}


def load_common_source(path, *, factory, config, destination, expected_sha256=None):
    """Validate provenance/configuration before constructing any model graphs."""
    path = Path(path).expanduser().resolve()
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Common reuse deployment manifest hash differs")
    deployment = load_json_object(path)
    if (deployment.get("status") != "PASS"
            or deployment.get("artifact_kind") != "qwen35-dflash-ascend310p-om-bundle"):
        raise ValueError("Common reuse requires a passing deployment manifest")
    reference = deployment["air_manifest"]
    air_path = contained_path(path.parent, reference["path"])
    if sha256_file(air_path) != reference["sha256"]:
        raise ValueError("Common reuse AIR manifest hash differs")
    air = load_json_object(air_path)
    if (air.get("status") != "PASS"
            or air.get("artifact_kind") != "qwen35-dflash-torchair-bundle"):
        raise ValueError("Common reuse requires a passing AIR manifest")
    if factory != FACTORY or air.get("factory") != FACTORY:
        raise ValueError("Common reuse supports only the quant incremental factory")
    if _config(config) != _config(air["factory_config"]):
        raise ValueError("Common reuse factory configuration differs (only verify_gdr may change)")
    if validate_incremental_bundle(air["graphs"]) != validate_incremental_bundle(deployment["graphs"]):
        raise ValueError("Common reuse AIR/deployment contracts differ")
    by_name = {g["name"]: g for g in deployment["graphs"]}
    graphs, paths = {}, []
    for graph in air["graphs"]:
        name = graph["name"]
        if name not in COMMON:
            continue
        compiled = by_name[name]
        for key in ("metadata", "role", "input_names", "output_names", "air",
                    "runtime_input_abi", "custom_op_audit", "standard_op_overrides"):
            if graph.get(key) != compiled.get(key):
                raise ValueError(f"Common reuse AIR/deployment differ: {name}.{key}")
        payload = graph["payload_files"]
        if not payload or graph["air"] not in payload:
            raise ValueError(f"Common reuse has no complete AIR payload manifest: {name}")
        if len({p["path"] for p in payload}) != len(payload):
            raise ValueError("Common reuse payload paths are duplicated")
        for item in payload:
            if Path(item["path"]).parts[:2] != ("air", name):
                raise ValueError("Common reuse payload escapes its graph directory")
            paths.append(_verified_file(path.parent, item))
        om = _verified_file(path.parent, compiled["om"])
        paths.append(om)
        graphs[name] = {"air": graph, "compiled": compiled, "om": om}
    if set(graphs) != set(COMMON):
        raise ValueError("Common reuse requires prefill, decode and draft; set include_ordinary_decode=true")
    ancestor = Path(destination)
    while not ancestor.exists():
        ancestor = ancestor.parent
    if any(p.stat().st_dev != ancestor.stat().st_dev for p in paths):
        raise ValueError("Common reuse needs bundles on the same filesystem for hard links")
    return {"path": path, "sha256": digest, "air": air, "destination": Path(destination),
            "deployment": deployment, "graphs": graphs}


def validate_common_export(source, headers, environment):
    if set(headers) != set(source["graphs"]):
        raise ValueError("Common reuse graph set differs")
    for name, header in headers.items():
        original = source["graphs"][name]["air"]
        for key, value in header.items():
            old, new = ((_metadata(original[key]), _metadata(value))
                        if key == "metadata" else (original.get(key), value))
            if key == "metadata" and (
                    not new.get("quant_source_lock") or not new.get("quant_input_manifest_sha256")):
                raise ValueError("Common reuse requires locked source and input identities")
            if old != new:
                raise ValueError("Common reuse export differs: " + _difference(old, new, f"{name}.{key}")
                                 + "; use matching source/configuration")
    if environment != source["air"]["environment"]:
        raise ValueError("Common reuse export environment differs")


def link_common_air(source, header, root):
    original = source["graphs"][header["name"]]["air"]
    for record in original["payload_files"]:
        target = contained_path(root, record["path"])
        if root == source["path"].parent:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(contained_path(source["path"].parent, record["path"]), target)
    return {**copy.deepcopy(original), **header}


def reuse_record(source):
    return {
        "deployment_manifest": {"path": str(source["path"]), "sha256": source["sha256"]},
        "graphs": sorted(source["graphs"]),
        "method": ("same_directory" if source["destination"] == source["path"].parent
                   else "hardlink"),
    }


def validate_common_compile(air, root, *, atc_path, soc_version, arguments, identity):
    record = air["common_reuse"]
    reference = record["deployment_manifest"]
    source = load_common_source(
        reference["path"], factory=air["factory"], config=air["factory_config"],
        destination=root, expected_sha256=reference["sha256"],
    )
    if record != reuse_record(source):
        raise ValueError("Unsupported common reuse record")
    by_name = {g["name"]: g for g in air["graphs"]}
    for name, item in source["graphs"].items():
        graph, original = by_name[name], item["air"]
        if set(graph) != set(original):
            raise ValueError(f"Common reuse AIR graph fields differ: {name}")
        for key, value in graph.items():
            old, new = ((_metadata(original[key]), _metadata(value))
                        if key == "metadata" else (original[key], value))
            if old != new:
                raise ValueError("Common reuse AIR graph differs: " + _difference(old, new, f"{name}.{key}"))
        for payload in graph["payload_files"]:
            _verified_file(root, payload)
        expected = [str(atc_path), "--mode=0", "--framework=1",
                    f"--soc_version={soc_version}", *arguments[name]]
        actual = [s for s in item["compiled"]["atc_command"]
                  if not s.startswith(("--model=", "--output="))]
        if actual != expected:
            raise ValueError(f"Common reuse ATC options differ: {name}, including deterministic/precision")
    if source["deployment"]["compiler"]["identity"] != identity:
        raise ValueError("Common reuse ATC identity differs")
    if source["deployment"]["target"]["soc_version"] != soc_version:
        raise ValueError("Common reuse SoC differs")
    if air["environment"] != source["air"]["environment"]:
        raise ValueError("Common reuse export environment differs")
    return source


def link_common_om(source, graph, root, om_root):
    item = source["graphs"][graph["name"]]
    path = (item["om"] if root == source["path"].parent
            else om_root / (artifact_stem(graph) + ".om"))
    if path != item["om"]:
        os.link(item["om"], path)
    # Retain the command/log of the actual compilation; never invent an ATC run.
    return {
        **copy.deepcopy(item["compiled"]),
        "metadata": copy.deepcopy(graph["metadata"]),
        "om": file_record(path, relative_to=root),
        "reused_from": reuse_record(source),
    }
