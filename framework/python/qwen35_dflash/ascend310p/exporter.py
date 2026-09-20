"""Export a factory-provided DFlash graph suite to standard TorchAir AIR."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import copy
import importlib
import os
from pathlib import Path
import platform
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from .contracts import AirGraphSpec
from .custom_op_export import audit_custom_op_export, prepare_custom_op_export
from .standard_op_export import prepare_aten_softplus_export, audit_aten_softplus_export
from .runtime_input_export import canonical_runtime_input_abi
from .utils import atomic_write_json, file_record, require_run_output, resolve_callable


def _mark_input_shapes(spec):
    """Keep caches/controls/weights static; expose only the declared feature axis."""
    axes = spec.metadata.get("dynamic_input_axes", {})
    if not axes:
        return
    for name, tensor in zip(spec.input_names, spec.example_args):
        dynamic = axes.get(name, ())
        for axis in range(tensor.ndim):
            if axis in dynamic:
                torch._dynamo.mark_dynamic(tensor, axis, min=16, max=64)
            else:
                torch._dynamo.mark_static(tensor, axis)
    # dynamic=True must not generalize coincident parameter/cache dimensions.
    for tensor in (*spec.model.parameters(), *spec.model.buffers()):
        torch._dynamo.mark_static(tensor)


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _tensor_record(value: Any) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
            "device": str(value.device),
            "requires_grad": bool(value.requires_grad),
        }
    return {"kind": type(value).__name__}


def _module_version(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    value = getattr(module, "__version__", None)
    return None if value is None else str(value)


def _spec_header(spec):
    return {
        "name": spec.name, "role": spec.role, "dynamic": bool(spec.dynamic),
        "model_class": f"{type(spec.model).__module__}.{type(spec.model).__qualname__}",
        "input_names": list(spec.input_names), "output_names": list(spec.output_names),
        "example_args": [_tensor_record(item) for item in spec.example_args],
        "example_kwargs": {n: _tensor_record(v) for n, v in spec.example_kwargs.items()},
        "metadata": dict(spec.metadata),
    }


def _normalize_specs(value: Any) -> tuple[AirGraphSpec, ...]:
    if isinstance(value, AirGraphSpec):
        specs = (value,)
    elif isinstance(value, Iterable):
        specs = tuple(value)
    else:
        raise TypeError("AIR factory must return AirGraphSpec or an iterable of them")
    if not specs:
        raise ValueError("AIR factory returned no graphs")
    if not all(isinstance(item, AirGraphSpec) for item in specs):
        raise TypeError("AIR factory returned a non-AirGraphSpec item")
    names = [item.name for item in specs]
    if len(set(names)) != len(names):
        raise ValueError("AIR graph names must be unique")
    return specs


def export_air_bundle(
    factory: str | Callable[[Mapping[str, Any]], Sequence[AirGraphSpec]],
    factory_config: Mapping[str, Any],
    bundle_dir: str | Path,
    *,
    torchair_module: Any | None = None,
    reuse_common_from: str | Path | None = None,
    reuse_target_from: str | Path | None = None,
    _shared_air: dict | None = None,
    _manifest_name: str | None = None,
) -> dict[str, Any]:
    """Export every graph from ``factory`` and retain a hash-complete manifest."""

    root = require_run_output(bundle_dir)
    matrix = _shared_air is not None
    if matrix and (reuse_common_from or reuse_target_from):
        raise ValueError("matrix export manages shared graphs internally")
    from .common_reuse import (
        COMMON, artifact_stem, load_common_source, validate_common_export,
        link_common_air, reuse_record,
    )
    if reuse_common_from and reuse_target_from:
        raise ValueError("select either common-route reuse or target-only reuse")
    reuse_path = reuse_common_from or reuse_target_from
    reused = (load_common_source(reuse_path, factory=factory,
                              config=factory_config, destination=root,
                              targets=reuse_target_from is not None)
              if reuse_path is not None else None)
    shared_root = reused is not None and root == reused["path"].parent
    if shared_root and reuse_target_from:
        raise ValueError("use a separate bundle directory for each Draft quantization")
    if not matrix and not shared_root and root.exists() and any(root.iterdir()):
        raise FileExistsError(f"AIR bundle directory is not empty: {root}")
    suffix = "-" + str(factory_config.get("verify_gdr", "chunk")) if shared_root else ""
    manifest_path = root / (_manifest_name or ("air-manifest" + suffix + ".json"))
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    torchair = torchair_module
    if torchair is None:
        try:
            torchair = importlib.import_module("torchair")
        except ImportError as error:
            raise RuntimeError(
                "TorchAir is required for AIR export; activate the declared CANN/TorchAir environment"
            ) from error

    # Import TorchAir before invoking the factory. The production factory loads
    # both 4B checkpoints, so a missing export runtime must fail before that
    # expensive and memory-heavy operation starts.
    factory_callable = resolve_callable(factory)
    prepare = getattr(factory_callable, "prepare_export", None)
    preflight = prepare(dict(factory_config), torchair) if callable(prepare) else None
    specs = _normalize_specs(factory_callable(dict(factory_config)))
    for spec in specs:
        probe = spec.metadata.get("weight_quant_probe")
        if probe is not None:
            from .weight_quant_layout import validate_probe_config
            validate_probe_config(probe)
            if spec.name != "weight_quant_probe" or spec.role != "diagnostic":
                raise ValueError("WeightQuant probe controls cannot be applied to model graphs")
    from .incremental_plan import validate_incremental_bundle
    validate_incremental_bundle([
        {"name": spec.name, "role": spec.role, "input_names": list(spec.input_names),
         "output_names": list(spec.output_names), "metadata": dict(spec.metadata)}
        for spec in specs
    ])
    environment = {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "torch_npu": _module_version("torch_npu"),
        "torchair": str(getattr(torchair, "__version__", "unknown")),
    }
    if reused is not None:
        validate_common_export(
            reused, {s.name: _spec_header(s) for s in specs if s.name in reused["graphs"]}, environment,
        )
    if matrix:
        from .bundle_matrix import validate_shared_graph
        for spec in specs:
            header = _spec_header(spec)
            key = artifact_stem(header)
            if key in _shared_air:
                original, previous_environment = _shared_air[key]
                validate_shared_graph(original, header)
                if environment != previous_environment:
                    raise ValueError("shared AIR export environments differ")
    for spec in specs:
        if matrix and artifact_stem(_spec_header(spec)) in _shared_air:
            continue
        if reused is None or spec.name not in reused["graphs"]:
            graph_dir = root / "air" / (artifact_stem(_spec_header(spec))
                                       if matrix or spec.name == "target_verify" else spec.name)
            if graph_dir.exists():
                raise FileExistsError(f"AIR graph output already exists: {graph_dir}")
    root.mkdir(parents=True, exist_ok=True)
    air_root = root / "air"
    air_root.mkdir(exist_ok=shared_root or matrix)

    graphs: list[dict[str, Any]] = []
    for spec in specs:
        key = artifact_stem(_spec_header(spec))
        if matrix and key in _shared_air:
            graphs.append({**copy.deepcopy(_shared_air[key][0]), **_spec_header(spec)})
            continue
        if reused is not None and spec.name in reused["graphs"]:
            graphs.append(link_common_air(reused, _spec_header(spec), root))
            continue
        # Common graph directories retain their original names for payload validation.
        graph_dir = air_root / (artifact_stem(_spec_header(spec))
                                if matrix or spec.name == "target_verify" else spec.name)
        graph_dir.mkdir()
        custom_op_sessions = [
            prepare_custom_op_export(item, torchair) for item in spec.custom_ops
        ]
        softplus_session = (
            prepare_aten_softplus_export(torchair)
            if spec.metadata.get("standard_op_export_contracts") else None
        )
        call_kwargs = {
            "model": spec.model.eval(),
            "export_path": str(graph_dir),
            "export_name": spec.name,
            "dynamic": bool(spec.dynamic),
        }
        if spec.compiler_config is not None:
            call_kwargs["config"] = spec.compiler_config
        call_kwargs.update(dict(spec.example_kwargs))
        input_abi_context = (
            canonical_runtime_input_abi(
                torchair, public_inputs=spec.example_args,
                public_names=spec.input_names,
                explicit_test_double=torchair_module is not None,
                require_static_shapes=True,
                dynamic_input_axes=spec.metadata.get("dynamic_input_axes"),
                capture_weight_quant_shapes=any(
                    op.ge_op_type == "WeightQuantBatchMatmulV2" for op in spec.custom_ops
                ),
                weight_quant_probe=spec.metadata.get("weight_quant_probe"),
                weight_prepack_manifest=(spec.metadata.get("draft_weight_prepack_manifest")
                                         if spec.metadata.get("draft_weight_storage") else None),
                public_output_names=spec.output_names,
                verify_discard_output_names=(
                    [s["name"] for s in spec.metadata["incremental_contract"]["verify_discard_states"]]
                    if spec.name == "target_verify" else ()
                ),
            )
            if (spec.metadata.get("incremental_contract") or any(
                op.ge_op_type == "WeightQuantBatchMatmulV2" for op in spec.custom_ops
            )) else nullcontext(None)
        )
        with (
            torch.inference_mode(), _working_directory(graph_dir),
            input_abi_context as runtime_input_abi,
        ):
            _mark_input_shapes(spec)
            torchair.dynamo_export(*spec.example_args, **call_kwargs)

        custom_op_audit = audit_custom_op_export(
            custom_op_sessions,
            graph_dir,
            relative_to=root,
        )

        standard_op_audit = [] if softplus_session is None else [
            audit_aten_softplus_export(softplus_session, graph_dir,
                                      calls_before=0, relative_to=root)
        ]

        air_files = sorted(graph_dir.glob("*.air"))
        if len(air_files) != 1:
            raise RuntimeError(
                f"TorchAir export for {spec.name!r} produced {len(air_files)} AIR files"
            )
        from .draft_constants import write_constant_inputs
        constant_records = write_constant_inputs(spec, graph_dir, root)
        payload_files = sorted(path for path in graph_dir.rglob("*") if path.is_file())
        records = [file_record(path, relative_to=root) for path in payload_files]
        air_record = next(
            item for item in records if item["path"] == air_files[0].relative_to(root).as_posix()
        )
        graphs.append(
            {
                **_spec_header(spec),
                **({"runtime_input_abi": runtime_input_abi}
                   if runtime_input_abi is not None else {}),
                "custom_op_audit": custom_op_audit,
                "standard_op_overrides": standard_op_audit,
                "air": air_record,
                "payload_files": records,
                **constant_records,
            }
        )
        if matrix:
            _shared_air[key] = (copy.deepcopy(graphs[-1]), environment)

    factory_name = (
        factory
        if isinstance(factory, str)
        else f"{factory_callable.__module__}:{factory_callable.__qualname__}"
    )
    manifest = {
        "schema_version": 2,
        "artifact_kind": "qwen35-dflash-torchair-bundle",
        "status": "PASS",
        "factory": factory_name,
        "factory_config": dict(factory_config),
        "environment": environment,
        **({"common_reuse": reuse_record(reused)} if reused is not None else {}),
        "operator_preflight": preflight,
        "graphs": graphs,
    }
    manifest_path = atomic_write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest
