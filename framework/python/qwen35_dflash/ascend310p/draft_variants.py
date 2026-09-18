"""Compose hash-locked Target and Draft artifacts without re-export or ATC."""
from __future__ import annotations

import copy
import os
from pathlib import Path

from .common_reuse import FACTORY, TARGETS, _metadata, _verified_file, artifact_stem
from .incremental_plan import validate_incremental_bundle, write_incremental_plan
from .utils import atomic_write_json, contained_path, file_record, load_json_object, require_run_output, sha256_file


def _load(path):
    path = Path(path).resolve()
    deployment = load_json_object(path)
    if deployment.get("status") != "PASS" or deployment.get("artifact_kind") != "qwen35-dflash-ascend310p-om-bundle":
        raise ValueError("composition needs a passing deployment bundle")
    air_path = contained_path(path.parent, deployment["air_manifest"]["path"])
    if sha256_file(air_path) != deployment["air_manifest"]["sha256"]:
        raise ValueError("composition AIR manifest hash differs")
    air = load_json_object(air_path)
    if air.get("factory") != FACTORY or not air["factory_config"].get("shared_draft_features"):
        raise ValueError("composition requires shared_draft_features incremental bundles")
    contract = validate_incremental_bundle(deployment["graphs"])
    if contract != validate_incremental_bundle(air["graphs"]):
        raise ValueError("composition AIR/deployment contracts differ")
    by_name = {g["name"]: g for g in air["graphs"]}
    for graph in deployment["graphs"]:
        original = by_name[graph["name"]]
        for key in ("metadata", "input_names", "output_names", "air", "runtime_input_abi", "constant_inputs", "constant_inputs_table"):
            if graph.get(key) != original.get(key):
                raise ValueError(f"composition AIR/deployment differ: {graph['name']}.{key}")
        _verified_file(path.parent, graph["om"])
        for payload in original["payload_files"]:
            _verified_file(path.parent, payload)
    return path, deployment, air, contract


def compose_draft_variant(*, target_manifest, draft_manifest, bundle_dir):
    """Select an existing route's Target and another route's identical Draft.

    Both source bundles must use the shared feature ABI and identical Target
    inputs, source, export environment and compiler identity. Only Draft and
    verification-route metadata may differ; Tensor ABIs are validated again.
    """
    root = require_run_output(bundle_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"composition destination must be empty: {root}")
    tp, target, ta, tc = _load(target_manifest)
    dp, draft, da, dc = _load(draft_manifest)
    if target["target"] != draft["target"] or ta["environment"] != da["environment"]:
        raise ValueError("composition target/export environment differs")
    for key in ("identity", "framework", "extra_args", "precision_policy"):
        if target["compiler"].get(key) != draft["compiler"].get(key):
            raise ValueError(f"composition compiler {key} differs")
    ignored = {"draft_dir", "draft_quantization", "draft_quant_matmul", "input_manifest", "verify_gdr"}
    if ({k: v for k, v in ta["factory_config"].items() if k not in ignored} !=
            {k: v for k, v in da["factory_config"].items() if k not in ignored}):
        raise ValueError("composition Target factory configuration differs")
    for name in TARGETS[:2]:
        if target["compiler"]["graph_extra_args"][name] != draft["compiler"]["graph_extra_args"][name]:
            raise ValueError(f"composition Target compiler options differ: {name}")
        tm = next(g["metadata"] for g in ta["graphs"] if g["name"] == name)
        dm = next(g["metadata"] for g in da["graphs"] if g["name"] == name)
        if _metadata(tm, targets=True) != _metadata(dm, targets=True):
            raise ValueError(f"composition Target metadata differs: {name}")
    contract = copy.deepcopy(tc)
    contract.update({k: copy.deepcopy(v) for k, v in dc.items() if k.startswith("draft_")})
    draft_meta = next(g["metadata"] for g in da["graphs"] if g["name"] == "draft")
    graphs, air_graphs, links = [], [], []
    root.mkdir(parents=True, exist_ok=True)
    for name in ("target_prefill", "target_decode", "target_verify", "draft"):
        path, deployment, air = (dp, draft, da) if name == "draft" else (tp, target, ta)
        compiled = copy.deepcopy(next(g for g in deployment["graphs"] if g["name"] == name))
        exported = copy.deepcopy(next(g for g in air["graphs"] if g["name"] == name))
        meta = exported["metadata"]
        meta["incremental_contract"] = copy.deepcopy(contract)
        meta["verify_gdr"] = tc["verify_gdr"]
        for key, value in draft_meta.items():
            if key.startswith("draft_") or key == "quant_input_manifest_sha256":
                meta[key] = copy.deepcopy(value)
        compiled["metadata"] = copy.deepcopy(meta)
        for payload in exported["payload_files"]:
            links.append((contained_path(path.parent, payload["path"]), contained_path(root, payload["path"])))
        om = root / "om" / (artifact_stem(compiled) + ".om")
        links.append((contained_path(path.parent, compiled["om"]["path"]), om))
        compiled["om"]["path"] = om.relative_to(root).as_posix()
        compiled["reused_from"] = {"deployment_manifest": str(path), "sha256": sha256_file(path), "method": "hardlink"}
        air_graphs.append(exported)
        graphs.append(compiled)
    validate_incremental_bundle(graphs)
    if len({b for _, b in links}) != len(links):
        raise ValueError("composition artifact paths collide")
    if any(a.stat().st_dev != root.stat().st_dev for a, _ in links):
        raise ValueError("composition requires the same filesystem for hard links")
    for source, destination in links:
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, destination)
    provenance = {"target": {"path": str(tp), "sha256": sha256_file(tp)},
                  "draft": {"path": str(dp), "sha256": sha256_file(dp)}, "method": "hardlink"}
    air = {k: copy.deepcopy(v) for k, v in da.items() if k not in ("graphs", "common_reuse", "manifest_path")}
    air["factory_config"]["verify_gdr"] = tc["verify_gdr"]
    air.update(graphs=air_graphs, artifact_composition=provenance)
    air_path = atomic_write_json(root / "air-manifest.json", air)
    result = {k: copy.deepcopy(v) for k, v in target.items() if k not in ("graphs", "common_reuse", "manifest_path")}
    result.update(graphs=graphs, air_manifest=file_record(air_path, relative_to=root), artifact_composition=provenance)
    result["compiler"]["graph_extra_args"]["draft"] = copy.deepcopy(draft["compiler"]["graph_extra_args"]["draft"])
    output = atomic_write_json(root / "deployment-manifest.json", result)
    write_incremental_plan(output, root / "runner-plan.txt", verify_gdr=tc["verify_gdr"])
    return output
