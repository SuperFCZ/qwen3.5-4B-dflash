"""One AIR/OM directory for shared Target graphs and selectable Drafts."""
from __future__ import annotations

import gc
from pathlib import Path

from .common_reuse import FACTORY, _metadata, _verified_file, artifact_stem
from .incremental_plan import validate_incremental_bundle
from .utils import atomic_write_json, contained_path, file_record, load_json_object, require_run_output

VARIANTS = ("fp16", "w4a16", "w8a16")
AIR_KIND = "qwen35-dflash-torchair-matrix"


def deployment_name(variant, route):
    if variant not in VARIANTS or route not in ("chunk", "mtp"):
        raise ValueError("invalid Draft type or verification route")
    suffix = ("" if variant == "fp16" else "-" + variant) + ("" if route == "chunk" else "-mtp")
    return "deployment-manifest" + suffix + ".json"


def validate_shared_graph(original, current):
    """Only checkpoint-specific and route-specific descriptive metadata may vary."""
    for key in ("name", "role", "input_names", "output_names", "model_class", "dynamic",
                "example_args", "example_kwargs", "metadata", "runtime_input_abi",
                "air", "payload_files", "constant_inputs", "constant_inputs_table",
                "custom_op_audit", "standard_op_overrides"):
        if key not in current:
            continue
        old, new = original.get(key), current[key]
        if key == "metadata":
            targets = current["name"] != "draft"
            old, new = _metadata(old, targets=targets), _metadata(new, targets=targets)
            if not new.get("quant_source_lock"):
                raise ValueError("shared graphs require locked source identity")
        if old != new:
            raise ValueError(f"shared graph differs: {artifact_stem(current)}.{key}")


def export_matrix(factory, config, bundle_dir, *, variants, routes, draft_dirs,
                  torchair_module=None):
    """Export seven unique AIR graphs at most, in separate model lifetimes."""
    from models.dflash_v1.draft_quantization import require_draft_checkpoint
    from .input_manifest import build_quant_input_manifest, verify_quant_input_manifest
    from .exporter import export_air_bundle

    if factory != FACTORY:
        raise ValueError("Draft/route matrix requires the quant incremental factory")
    if not variants or len(set(variants)) != len(variants) or any(v not in VARIANTS for v in variants):
        raise ValueError("select distinct Draft types: fp16, w4a16, w8a16")
    if not routes or len(set(routes)) != len(routes) or any(r not in ("chunk", "mtp") for r in routes):
        raise ValueError("select distinct verification routes: chunk, mtp")
    root = require_run_output(bundle_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"matrix export needs an empty bundle directory: {root}")
    if config.get("input_manifest"):
        verify_quant_input_manifest(config["input_manifest"])
    directories, audits = {}, {}
    # Validate all selected checkpoints before starting any model export.
    for variant in variants:
        directory = draft_dirs.get(variant) or (config.get("draft_dir") if variant == "fp16" else None)
        if not directory:
            raise ValueError(f"set DRAFT_{variant.upper()}_DIR or --{variant}-draft-dir")
        directories[variant] = str(Path(directory).expanduser().resolve())
        audits[variant] = require_draft_checkpoint(directories[variant], variant)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config").mkdir()
    cached, members = {}, []
    for variant in variants:
        inputs = root / "config" / (variant + "-inputs.json")
        build_quant_input_manifest(target_dir=config["target_dir"], draft_dir=directories[variant],
            quant_config=config["quant_config"], receiver_models_dir=config["receiver_models_dir"], output=inputs)
        for route in routes:
            current = dict(config, draft_dir=directories[variant], draft_quantization=variant,
                           verify_gdr=route, input_manifest=str(inputs),
                           shared_draft_features=True, include_ordinary_decode=True)
            name = f"air-manifest-{variant}-{route}.json"
            print(f"[export-air] {variant}/{route} START", flush=True)
            exported = export_air_bundle(factory, current, root, torchair_module=torchair_module,
                                        _shared_air=cached, _manifest_name=name)
            members.append(dict(draft_quantization=variant, verify_gdr=route,
                                checkpoint=audits[variant],
                                air_manifest=file_record(Path(exported["manifest_path"]), relative_to=root),
                                deployment_manifest=deployment_name(variant, route)))
            # TorchDynamo caches must not retain three loaded Drafts/Targets.
            import torch
            torch._dynamo.reset()
            gc.collect()
            if hasattr(torch, "npu"):
                torch.npu.empty_cache()
            print(f"[export-air] {variant}/{route} DONE", flush=True)
    result = dict(schema_version=1, artifact_kind=AIR_KIND, status="PASS",
                  draft_quantizations=list(variants), routes=list(routes), members=members,
                  unique_graphs=sorted(cached))
    path = atomic_write_json(root / "air-manifest.json", result)
    return dict(result, manifest_path=str(path))


def compile_matrix(path, *, soc_version, atc_bin=None, extra_args=(), runner=None, atc_identity=None,
                   resume=False):
    """Compile each unique graph once; every deployment refers to the same om/."""
    from .compiler import (compile_air_bundle, _validated_custom_op_audit,
                           _validated_standard_op_overrides, _validated_completed_bundle,
                           _bundle_atc_args, _atc_identity, resolve_atc_executable, validate_soc_version)
    from .runtime_input_export import validated_runtime_input_abi
    from .draft_constants import verify_constant_inputs
    from .utils import sha256_file

    path = Path(path).expanduser().resolve()
    root = require_run_output(path.parent)
    catalog = load_json_object(path)
    if catalog.get("artifact_kind") != AIR_KIND or catalog.get("status") != "PASS":
        raise ValueError("expected a passing AIR matrix")
    variants, routes = catalog["draft_quantizations"], catalog["routes"]
    expected = {(v, r) for v in variants for r in routes}
    members = catalog["members"]
    if not expected or len(members) != len(expected) or {
            (m["draft_quantization"], m["verify_gdr"]) for m in members} != expected:
        raise ValueError("AIR matrix selections and members differ")
    if not resume and ((root / "draft-variants.json").exists() or any((root / "om").glob("*.om"))):
        raise FileExistsError("matrix output already contains compiled artifacts; use a new bundle directory")
    seen, environment, air_members = {}, None, {}
    # Check every member and payload before the first ATC invocation.
    for member in members:
        variant, route = member["draft_quantization"], member["verify_gdr"]
        if variant not in VARIANTS or route not in ("chunk", "mtp"):
            raise ValueError("invalid Draft/route member")
        if member["deployment_manifest"] != deployment_name(variant, route):
            raise ValueError("unexpected matrix deployment filename")
        if not resume and (root / member["deployment_manifest"]).exists():
            raise FileExistsError(root / member["deployment_manifest"])
        air = load_json_object(_verified_file(root, member["air_manifest"]))
        air_members[(variant, route)] = air
        if air.get("status") != "PASS" or air.get("factory") != FACTORY:
            raise ValueError("matrix member is not a passing incremental AIR bundle")
        if air.get("artifact_kind") != "qwen35-dflash-torchair-bundle" or "common_reuse" in air:
            raise ValueError("unexpected matrix member artifact/reuse policy")
        if not air["factory_config"].get("shared_draft_features"):
            raise ValueError("matrix member requires shared Target feature ABI")
        if environment is not None and environment != air["environment"]:
            raise ValueError("matrix export environments differ")
        environment = air["environment"]
        contract = validate_incremental_bundle(air["graphs"])
        if contract is None or contract.get("draft_quantization", "fp16") != variant or contract["verify_gdr"] != route:
            raise ValueError("matrix member labels differ from the graph contract")
        for graph in air["graphs"]:
            _validated_custom_op_audit(graph)
            _validated_standard_op_overrides(graph)
            validated_runtime_input_abi(graph, required=True, allow_test_double=runner is not None)
            for payload in graph["payload_files"]:
                _verified_file(root, payload)
            _verified_file(root, graph["air"])
            if graph.get("constant_inputs") or (graph["name"] == "draft" and variant != "fp16"):
                verify_constant_inputs(graph, root)
            key = artifact_stem(graph)
            if key in seen:
                validate_shared_graph(seen[key], graph)
            else:
                seen[key] = graph
    if sorted(seen) != catalog.get("unique_graphs"):
        raise ValueError("AIR matrix unique graph inventory differs")
    cache, bundles, completed = {}, {}, {}
    if resume:
        atc_path = resolve_atc_executable(atc_bin)
        soc_version = validate_soc_version(soc_version)
        atc_identity = atc_identity or _atc_identity(atc_path)
        # Admit every existing member before invoking any new ATC process.
        for member in members:
            deployment_path = root / member["deployment_manifest"]
            if not deployment_path.exists():
                continue
            key = (member["draft_quantization"], member["verify_gdr"])
            graphs = air_members[key]["graphs"]
            _, options = _bundle_atc_args(graphs, extra_args, incremental=True, soc_version=soc_version)
            saved = _validated_completed_bundle(deployment_path,
                air_path=root / member["air_manifest"]["path"], graphs=graphs, atc_path=atc_path,
                soc_version=soc_version, graph_arguments=options, identity=atc_identity)
            completed[key] = dict(saved, manifest_path=str(deployment_path))
            for graph in saved["graphs"]:
                stem = artifact_stem(graph)
                if stem in cache:
                    validate_shared_graph(cache[stem][0], graph)
                    if cache[stem][0]["om"] != graph["om"] or cache[stem][1] != options[graph["name"]]:
                        raise ValueError(f"resume shared OM/options differ: {stem}")
                else:
                    cache[stem] = (graph, options[graph["name"]])
        admitted = {root / graph["om"]["path"] for graph, _ in cache.values()}
        unknown = set((root / "om").glob("*.om")) - admitted
        if unknown:
            raise ValueError("resume found OM files without a verified passing deployment; "
                             "retain/move these files aside before retrying: "
                             + ", ".join(str(p) for p in sorted(unknown)))
    for member in members:
        variant, route = member["draft_quantization"], member["verify_gdr"]
        if (variant, route) in completed:
            compiled = completed[(variant, route)]
            print(f"[compile-om] {variant}/{route} REUSE completed bundle", flush=True)
        else:
            compiled = compile_air_bundle(contained_path(root, member["air_manifest"]["path"]),
                soc_version=soc_version, atc_bin=atc_bin, extra_args=extra_args, runner=runner,
                atc_identity=atc_identity, _shared_compiled=cache,
                _deployment_name=member["deployment_manifest"])
        manifest = Path(compiled["manifest_path"])
        bundles.setdefault(variant, {})[route] = dict(
            manifest=manifest.name, manifest_sha256=sha256_file(manifest),
            checkpoint=member["checkpoint"], status="PASS")
    result = dict(schema_version=2, artifact_kind="qwen35-draft-variants", status="PASS",
                  draft_quantizations=variants, routes=routes, bundles=bundles,
                  unique_oms=[file_record(root / "om" / (name + ".om"), relative_to=root) for name in sorted(cache)],
                  air_manifest=file_record(path, relative_to=root), layout="shared_directory")
    output = atomic_write_json(root / "draft-variants.json", result)
    return dict(result, manifest_path=str(output))
