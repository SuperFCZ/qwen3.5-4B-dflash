"""Compile a hash-locked TorchAir bundle into Ascend 310P OM artifacts."""

from __future__ import annotations

from pathlib import Path
import json
import os
import re
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence

from .runtime_input_export import validated_runtime_input_abi
from .atc_fusion import (WEIGHT_QUANT_TRANSPOSE_PASS, fusion_switch_record,
                         normalized_atc_options)

from .utils import (
    atomic_write_json,
    contained_path,
    file_record,
    load_json_object,
    require_run_output,
    sha256_file,
)


_FORBIDDEN_ATC_PREFIXES = (
    "--framework",
    "--model",
    "--output",
    "--soc_version",
    "--mode",
)


class AtcCompileError(RuntimeError):
    """ATC failed or did not produce the promised OM artifact."""


def _atc_failure_detail(stdout: str, graph: Mapping[str, Any]) -> str:
    """Surface ATC's diagnostic instead of just the Python wrapper traceback."""
    lines = stdout.splitlines()
    # Prefer the structured ATC summary over earlier engine-placement errors
    # and later shutdown messages. Retain the complete output in the log file.
    start = next((i for i, line in enumerate(lines)
                  if re.search(r"\b\w+\(E[A-Z0-9]+\):", line)), None)
    if start is None:
        start = next((i for i, line in enumerate(lines) if "[ERROR]" in line),
                     max(0, len(lines) - 8))
    # Tiling dumps list inputs/optional slots before the actual constraint.
    # Keep the header bounded, and reserve room for attributes and traceback
    # independently so tensor descriptions cannot hide the reason for failure.
    excerpt = "\n".join(lines[start:start + 8])[:3500].strip()
    attrs = next((line for line in lines[start:] if "[OP_TILING] Attrs:" in line), None)
    if attrs and attrs not in excerpt:
        excerpt += "\n" + attrs[:2000]
    trace = next((i for i in range(start, len(lines))
                  if "TraceBack (most recent call last)" in lines[i]
                  or "Traceback (most recent call last)" in lines[i]), None)
    if trace is not None:
        context = "\n".join(lines[trace:trace + 8])[:4000]
        if context:
            excerpt += "\n" + context
    detail = f"\nATC diagnostic:\n{excerpt}" if excerpt else ""
    if WEIGHT_QUANT_TRANSPOSE_PASS in stdout:
        detail += (
            "\nWeight-quant transpose/NZ graph fusion failed before OM execution. "
            "An off switch has not prevented this pass on the receiver. "
            "Re-export with the NK weight / GN scale normalization and run "
            "tools/probe_draft_matmul_atc.py before rebuilding a full Draft. "
            "Retain weight-quant-layout.json and the complete ATC log if it still fails; "
            "group-128 kernel support requires a separate target check."
        )
    if "WeightQuantBatchMatmulV2" in stdout and "Antiquant shape expect" in stdout:
        detail += (
            "\nWeight-quant group scale must be [K/group_size,N], independently of "
            "transpose_weight. Re-export AIR to retain the scale transpose; "
            "do not reshape scales or change the quantization group size. "
            "This shape error alone does not establish unsupported kernel functionality."
        )
    if "WeightQuantBatchMatmulV2" in stdout and "no valid template is found" in stdout:
        detail += (
            "\nNo WeightQuant template matched this SoC/layout/group/shape combination. "
            "This is distinct from an invalid scale shape and from the transpose fusion pass. "
            "Run probe_draft_matmul_atc.py with --bits 8 --projection tiny "
            "--group-size 0 128 --weight-layout nk kn to isolate support. "
            "Group 0 is a synthetic per-channel control, not a replacement for checkpoint group-128 scales."
        )
    if ("ChunkGatedDeltaRule" in stdout and
            re.search(r"DT_FLOAT of output\s*\[core_attn\]", stdout)):
        detail += (
            "\nGDR output contract: core_attn=FP16, last_recurrent_state=FP32. "
            "The reported core_attn dtype mismatch does not establish a missing kernel."
        )
        runtime_abi = graph.get("runtime_input_abi")
        audit = runtime_abi.get("gdr_output_dtypes") if isinstance(runtime_abi, Mapping) else None
        if (isinstance(audit, Mapping) and audit.get("status") == "PASS"
                and audit.get("node_count", 0) > 0):
            detail += (
                " The Python GE descriptors passed the pre-save dtype check; "
                "compare the saved AIR and CANN InferShape/InferDataType results "
                "with gdr-output-dtypes.json before changing Fake/Meta or precision."
            )
        else:
            detail += (
                " This AIR has no pre-save GDR dtype audit; inspect its GE output "
                "descriptors or re-export to obtain gdr-output-dtypes.json."
            )
    return detail


def validate_soc_version(soc_version: str) -> str:
    """Require an ATC SoC variant instead of the generic 310P family name."""

    value = str(soc_version).strip()
    if not value:
        raise ValueError("soc_version must be the exact ATC target identity")
    normalized = re.sub(r"[^a-z0-9]", "", value.lower())
    if normalized in {"310p", "ascend310p", "atlas310p"}:
        raise ValueError(
            "soc_version must identify the concrete 310P ATC variant, "
            "for example Ascend310P3; a generic Ascend310P value is not sufficient"
        )
    return value


def resolve_atc_executable(atc_bin: str | Path | None = None) -> Path:
    """Resolve ATC from the declared profile and fail before graph construction."""

    configured_atc = atc_bin or os.environ.get("ASCEND310P_ATC_BIN")
    if configured_atc is None:
        raise RuntimeError(
            "ATC is unavailable in the declared target profile; provide --atc from a CANN profile"
        )
    atc_path = Path(configured_atc).expanduser().resolve()
    if not atc_path.is_file() or not os.access(atc_path, os.X_OK):
        raise RuntimeError(f"ATC is not an executable file: {atc_path}")
    return atc_path


def _validate_extra_args(arguments: Sequence[str]) -> list[str]:
    result = []
    for argument in arguments:
        value = str(argument)
        if not value.startswith("--"):
            raise ValueError(f"ATC extra argument must start with '--': {value!r}")
        if value.startswith(_FORBIDDEN_ATC_PREFIXES):
            raise ValueError(f"ATC core option cannot be overridden: {value!r}")
        result.append(value)
    return result


def _chunk_precision_args(arguments: Sequence[str], *, incremental: bool) -> list[str]:
    """Preserve the FP32 islands specified by the native chunk/Draft graphs.

    ATC's performance default can lower FP32 RMSNorm, Softmax, RoPE and
    attention matmuls even though the Python graph contains explicit casts.
    The FP16 checkpoint dtype is not permission to downcast these operations.
    """
    result = list(arguments)
    if not incremental:
        return result
    precision = [value for value in result
                 if value.split("=", 1)[0] in {"--precision_mode", "--precision_mode_v2"}]
    allowed = {"--precision_mode=must_keep_origin_dtype", "--precision_mode_v2=origin"}
    if len(precision) > 1 or (precision and precision[0] not in allowed):
        raise ValueError(
            "chunk AIR requires original graph precision: use "
            "--precision_mode=must_keep_origin_dtype or --precision_mode_v2=origin"
        )
    if not precision:
        result.append("--precision_mode=must_keep_origin_dtype")
    return result


def _graph_atc_args(arguments: Sequence[str], *, name: str, incremental: bool) -> list[str]:
    """Default Draft to deterministic=0; retain explicit flags and Target settings."""
    result = list(arguments)
    settings = [s for s in result if s.split("=", 1)[0] == "--deterministic"]
    if len(settings) > 1 or (settings and settings[0] not in {"--deterministic=0", "--deterministic=1"}):
        raise ValueError("use exactly one --deterministic=0 or --deterministic=1")
    if incremental and name == "draft" and not settings:
        result.append("--deterministic=0")
    return result


def _default_runner(command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _atc_identity(atc_bin: Path) -> str:
    result = subprocess.run(
        [str(atc_bin), "--version"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    text = result.stdout.strip()
    return text if text else f"exit={result.returncode} (no version text)"


def _validated_custom_op_audit(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = graph.get("custom_op_audit", [])
    if not isinstance(raw, list):
        raise TypeError("AIR custom_op_audit must be a list")
    metadata = graph.get("metadata", {})
    contract_items: list[Mapping[str, Any]] = []
    if isinstance(metadata, Mapping):
        plural = metadata.get("custom_op_export_contracts")
        singular = metadata.get("custom_op_export_contract")
        if plural is not None:
            if not isinstance(plural, list) or not all(
                isinstance(item, Mapping) for item in plural
            ):
                raise TypeError(
                    "AIR custom_op_export_contracts must be a list of objects"
                )
            contract_items = list(plural)
        elif singular is not None:
            if not isinstance(singular, Mapping):
                raise TypeError("AIR custom_op_export_contract must be an object")
            contract_items = [singular]
    if contract_items and not raw:
        raise ValueError("AIR graph requires a passing custom-operator audit")
    contract_by_target: dict[str, Mapping[str, Any]] = {}
    for item in contract_items:
        target = str(item.get("torch_target", ""))
        if not target or target in contract_by_target:
            raise ValueError(
                "AIR custom-operator contracts contain a missing or duplicate target"
            )
        contract_by_target[target] = item

    result: list[dict[str, Any]] = []
    audit_targets: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping) or item.get("status") != "PASS":
            raise ValueError("AIR graph contains a non-passing custom-operator audit")
        target = str(item.get("torch_target", ""))
        if not target or target in audit_targets:
            raise ValueError(
                "AIR custom-operator audits contain a missing or duplicate target"
            )
        audit_targets.add(target)
        minimum = int(item.get("minimum_occurrences", 0))
        ge_nodes = int(item.get("ge_node_occurrences", 0))
        converter_policy = str(
            item.get("converter_policy", "framework-registered-ge-ir")
        )
        if converter_policy not in {
            "framework-registered-ge-ir",
            "torchair-builtin",
        }:
            raise ValueError("AIR custom-operator converter policy is invalid")
        if minimum < 0 or ge_nodes < minimum:
            raise ValueError("AIR custom-operator preservation counts are invalid")
        raw_converter_calls = item.get("converter_calls")
        if converter_policy == "torchair-builtin":
            if raw_converter_calls is not None:
                raise ValueError(
                    "TorchAir builtin converter audit must use a null call count"
                )
        else:
            converter_calls = int(
                0 if raw_converter_calls is None else raw_converter_calls
            )
            if converter_calls < minimum or ge_nodes < converter_calls:
                raise ValueError(
                    "AIR custom-operator preservation counts are invalid"
                )
        result.append(dict(item))
    if contract_by_target and audit_targets != set(contract_by_target):
        raise ValueError(
            "AIR custom-operator audits do not cover every declared contract"
        )
    for item in result:
        contract = contract_by_target.get(str(item["torch_target"]))
        if contract is None:
            continue
        if (
            str(item.get("ge_op_type", ""))
            != str(contract.get("ge_op_type", ""))
            or int(item.get("minimum_occurrences", 0))
            != int(contract.get("minimum_occurrences", 1))
        ):
            raise ValueError(
                "AIR custom-operator audit differs from its declared contract"
            )
    from .weight_quant_layout import validate_weight_quant_layout
    validate_weight_quant_layout(graph)
    return result


def _validated_standard_op_overrides(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Keep exact SoftplusV2 evidence mandatory through AIR -> OM."""
    contracts = graph.get("metadata", {}).get("standard_op_export_contracts", [])
    raw = graph.get("standard_op_overrides", [])
    if not isinstance(contracts, list) or not isinstance(raw, list):
        raise TypeError("AIR standard-op contracts and audits must be lists")
    if not contracts and not raw:
        return []
    expected = [{"torch_target": "aten.softplus.default", "ge_op_type": "SoftplusV2",
                 "minimum_occurrences": 1}]
    if contracts != expected or len(raw) != 1 or not isinstance(raw[0], Mapping):
        raise ValueError("AIR standard-op audits must cover the declared SoftplusV2 contract")
    item = raw[0]
    if (item.get("status") != "PASS"
        or any(item.get(key) != value for key, value in expected[0].items())
        or item.get("converter_policy") not in {
            "framework-registered-ge-ir", "torchair-existing-ge-ir"}):
        raise ValueError("AIR SoftplusV2 audit differs from the declared contract")
    calls, nodes = int(item.get("converter_calls", -1)), int(item.get("ge_node_occurrences", 0))
    if calls < 0 or nodes < max(1, calls):
        raise ValueError("AIR SoftplusV2 preservation counts are invalid")
    return [dict(item)]


def _dynamic_atc_args(graph, arguments):
    axes = graph.get("metadata", {}).get("dynamic_input_axes", {})
    if not axes:
        return list(arguments)
    abi = graph["runtime_input_abi"]
    tensors = graph["metadata"]["tensor_abi"]["inputs"]
    bindings = abi.get("bindings", [])
    names = ([t["name"] for t in tensors]
             if abi["status"] == "NOT_APPLICABLE_EXPLICIT_TEST_DOUBLE"
             else [b["data_node_name"] for b in bindings])
    shapes = []
    for tensor, name in zip(tensors, names):
        if any(char in name for char in ";:\n\r"):
            raise ValueError("AIR Data node name is not safe for ATC input_shape")
        shape = list(tensor["shape"])
        for axis in axes.get(tensor["name"], ()):
            shape[axis] = -1
        shapes.append(name + ":" + ",".join(map(str, shape)))
    required = {"--input_format": "ND", "--input_shape": ";".join(shapes),
                "--dynamic_dims": "16;64"}
    output = []
    for arg in arguments:
        key, _, value = arg.partition("=")
        if key in ("--dynamic_batch_size", "--dynamic_image_size", "--input_shape_range"):
            raise ValueError("Draft 16/64 gears cannot be overridden with another shape policy")
        if key in required:
            if value != required[key]:
                raise ValueError(f"Draft gear contract requires {key}={required[key]}")
            continue
        output.append(arg)
    return output + [key + "=" + value for key, value in required.items()]


def _bundle_atc_args(graphs, extra_args, *, incremental, soc_version):
    arguments = _chunk_precision_args(_validate_extra_args(extra_args), incremental=incremental)
    graph_arguments = {
        graph["name"]: _dynamic_atc_args(graph, _graph_atc_args(
            arguments, name=graph["name"], incremental=incremental))
        for graph in graphs
    }
    return arguments, graph_arguments


def _compile_air_graph(
    graph: Mapping[str, Any], *, root: Path, om_root: Path, log_root: Path,
    atc_path: Path, exact_soc_version: str, arguments: Sequence[str],
    execute: Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    """Shared audited graph compile for complete builds and Draft-only rebuilds."""
    run_dir = Path(os.environ["AI_RUN_DIR"]).expanduser().resolve()
    if not isinstance(graph, Mapping):
        raise TypeError("AIR graph manifest entry must be an object")
    name = str(graph["name"])
    custom_op_audit = _validated_custom_op_audit(graph)
    air_record = graph["air"]
    payload_records = graph.get("payload_files")
    if not isinstance(payload_records, list) or not payload_records:
        raise ValueError(f"AIR graph has no payload manifest: {name}")
    for record in payload_records:
        payload_path = contained_path(root, str(record["path"]))
        if not payload_path.is_file():
            raise FileNotFoundError(f"AIR payload is missing: {payload_path}")
        if payload_path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"AIR payload size mismatch before ATC: {record['path']}")
        if sha256_file(payload_path) != record["sha256"]:
            raise ValueError(f"AIR payload hash mismatch before ATC: {record['path']}")
    air_path = contained_path(root, str(air_record["path"]))
    if not air_path.is_file():
        raise FileNotFoundError(f"AIR graph is missing: {air_path}")
    actual_hash = sha256_file(air_path)
    if actual_hash != air_record["sha256"]:
        raise ValueError(f"AIR graph hash mismatch before ATC: {name}")

    from .common_reuse import artifact_stem
    output_prefix = om_root / artifact_stem(graph)
    command = [
        str(atc_path),
        "--mode=0",
        "--framework=1",
        f"--model={air_path}",
        f"--output={output_prefix}",
        f"--soc_version={exact_soc_version}",
        *arguments,
    ]
    fusion = fusion_switch_record(arguments)
    result = execute(command, air_path.parent)
    log_path = log_root / f"{name}.log"
    log_path.write_text(result.stdout or "", encoding="utf-8")
    om_path = Path(str(output_prefix) + ".om")
    if result.returncode != 0:
        raise AtcCompileError(
            f"ATC failed for {name!r} with exit {result.returncode}; log={log_path}"
            + _atc_failure_detail(result.stdout or "", graph)
        )
    if fusion != fusion_switch_record(arguments):
        raise AtcCompileError("ATC fusion configuration changed during compilation")
    if not om_path.is_file() or om_path.stat().st_size == 0:
        raise AtcCompileError(
            f"ATC returned success but produced no non-empty OM for {name!r}; log={log_path}"
        )
    return {
        "name": name,
        "role": graph["role"],
        "metadata": dict(graph.get("metadata", {})),
        "input_names": list(graph.get("input_names", [])),
        "output_names": list(graph.get("output_names", [])),
        "custom_op_audit": custom_op_audit,
        "standard_op_overrides": _validated_standard_op_overrides(graph),
        **({"runtime_input_abi": graph["runtime_input_abi"]}
           if "runtime_input_abi" in graph else {}),
        "air": dict(air_record),
        "om": file_record(om_path, relative_to=root),
        "atc_command": command,
        "atc_log": str(log_path.relative_to(run_dir)),
        **({"atc_fusion_switch": fusion} if fusion is not None else {}),
        **{key: graph[key] for key in ("constant_inputs", "constant_inputs_table") if key in graph},
    }


def _validated_completed_bundle(path, *, air_path, graphs, atc_path, soc_version,
                                graph_arguments, identity):
    """Admit a completed matrix member only with matching inputs and build identity."""
    from .common_reuse import _verified_file, artifact_stem

    root = path.parent
    saved = load_json_object(path)
    if (saved.get("status") != "PASS"
            or saved.get("artifact_kind") != "qwen35-dflash-ascend310p-om-bundle"
            or saved.get("air_manifest") != {"path": air_path.name, "sha256": sha256_file(air_path)}):
        raise ValueError(f"resume requires a matching passing deployment/AIR manifest: {path}")
    if (saved.get("target", {}).get("soc_version") != soc_version
            or saved.get("compiler", {}).get("identity") != identity
            or saved.get("compiler", {}).get("path") != str(atc_path)):
        raise ValueError(f"resume compiler/SoC identity differs: {path}")
    compiled = saved.get("graphs", [])
    by_name = {graph["name"]: graph for graph in compiled}
    if len(by_name) != len(compiled) or set(by_name) != {graph["name"] for graph in graphs}:
        raise ValueError(f"resume graph inventory differs: {path}")
    for graph in graphs:
        original = by_name[graph["name"]]
        for key in ("name", "role", "metadata", "air", "input_names", "output_names",
                    "runtime_input_abi", "custom_op_audit", "standard_op_overrides",
                    "constant_inputs", "constant_inputs_table"):
            if original.get(key) != graph.get(key):
                raise ValueError(f"resume graph differs: {artifact_stem(graph)}.{key}")
        if original["om"]["path"] != f"om/{artifact_stem(graph)}.om":
            raise ValueError("resume requires the matrix's shared om/ paths")
        _verified_file(root, original["om"])
        command = original.get("atc_command")
        if not isinstance(command, list) or not command or not all(isinstance(s, str) for s in command):
            raise ValueError("resume requires the original ATC command")
        if original.get("atc_fusion_switch") != fusion_switch_record(command):
            raise ValueError("resume fusion switch provenance/hash differs")
        actual = [s for s in command if not s.startswith(("--model=", "--output="))]
        expected = [str(atc_path), "--mode=0", "--framework=1", f"--soc_version={soc_version}",
                    *graph_arguments[graph["name"]]]
        if normalized_atc_options(actual) != normalized_atc_options(expected):
            raise ValueError(f"resume ATC options differ: {artifact_stem(graph)}")
    return saved


def compile_air_bundle(
    air_manifest_path: str | Path,
    *,
    soc_version: str,
    atc_bin: str | Path | None = None,
    extra_args: Sequence[str] = (),
    runner: Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]] | None = None,
    atc_identity: str | None = None,
    resume: bool = False,
    _shared_compiled: dict | None = None,
    _deployment_name: str | None = None,
) -> dict[str, Any]:
    """Compile all AIR graphs with ``framework=1`` into the same run bundle."""

    manifest_path = Path(air_manifest_path).expanduser().resolve()
    root = require_run_output(manifest_path.parent)
    air_manifest = load_json_object(manifest_path)
    from .bundle_matrix import AIR_KIND, compile_matrix, validate_shared_graph
    if air_manifest.get("artifact_kind") == AIR_KIND:
        return compile_matrix(manifest_path, soc_version=soc_version, atc_bin=atc_bin,
                              extra_args=extra_args, runner=runner, atc_identity=atc_identity,
                              resume=resume)
    if resume:
        raise ValueError("--resume requires the unified AIR matrix air-manifest.json")
    if air_manifest.get("status") != "PASS":
        raise ValueError("AIR manifest is not passing")
    if air_manifest.get("artifact_kind") != "qwen35-dflash-torchair-bundle":
        raise ValueError("unexpected AIR artifact kind")
    exact_soc_version = validate_soc_version(soc_version)
    atc_path = resolve_atc_executable(atc_bin)
    graphs = air_manifest.get("graphs")
    if not isinstance(graphs, list) or not graphs:
        raise ValueError("AIR manifest requires a non-empty graph list")
    names = [graph.get("name") if isinstance(graph, Mapping) else None for graph in graphs]
    if any(not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None for name in names):
        raise ValueError("AIR graph has an invalid name")
    if len(set(names)) != len(names):
        raise ValueError("AIR graph names must be unique")
    from .incremental_plan import validate_incremental_bundle
    incremental = validate_incremental_bundle(graphs)

    # Validate the entire suite before starting any ATC process.
    for graph in graphs:
        if graph.get("constant_inputs") or (graph["name"] == "draft" and
                graph.get("metadata", {}).get("draft_quantization", "fp16") != "fp16"):
            from .draft_constants import verify_constant_inputs
            verify_constant_inputs(graph, root)
        _validated_custom_op_audit(graph)
        _validated_standard_op_overrides(graph)
        validated_runtime_input_abi(
            graph, required=bool(graph.get("metadata", {}).get("incremental_contract")),
            allow_test_double=runner is not None,
        )

    arguments, graph_arguments = _bundle_atc_args(
        graphs, extra_args, incremental=incremental is not None, soc_version=exact_soc_version)
    from .common_reuse import artifact_stem, validate_common_compile, link_common_om
    reused = None
    if "common_reuse" in air_manifest:
        atc_identity = atc_identity or _atc_identity(atc_path)
        reused = validate_common_compile(
            air_manifest, root, atc_path=atc_path, soc_version=exact_soc_version,
            arguments=graph_arguments, identity=atc_identity,
        )
    execute = runner or _default_runner
    om_root = root / "om"
    shared_root = reused is not None and root == reused["path"].parent
    suffix = "-" + incremental["verify_gdr"] if shared_root else ""
    deployment_path = root / (_deployment_name or ("deployment-manifest" + suffix + ".json"))
    if deployment_path.exists():
        raise FileExistsError(deployment_path)
    if _shared_compiled is None and not shared_root and om_root.exists() and any(om_root.iterdir()):
        raise FileExistsError(f"OM output directory is not empty: {om_root}")
    for graph in graphs:
        key = artifact_stem(graph)
        if _shared_compiled is not None and key in _shared_compiled:
            original, options = _shared_compiled[key]
            validate_shared_graph(original, {k: v for k, v in graph.items() if k in original})
            if options != graph_arguments[graph["name"]]:
                raise ValueError(f"shared graph ATC options differ: {key}")
            continue
        if reused is None or graph["name"] not in reused["graphs"]:
            output_path = om_root / (artifact_stem(graph) + ".om")
            if output_path.exists():
                raise FileExistsError(output_path)
    om_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(os.environ["AI_RUN_DIR"]).expanduser().resolve()
    log_root = run_dir / "log" / "dflash-atc"
    log_root.mkdir(parents=True, exist_ok=True)
    log_root = Path(tempfile.mkdtemp(prefix=root.name + "-", dir=log_root))

    compiled = []
    for graph in graphs:
        key = artifact_stem(graph)
        if _shared_compiled is not None and key in _shared_compiled:
            import copy
            current = copy.deepcopy(_shared_compiled[key][0])
            current["metadata"] = copy.deepcopy(graph["metadata"])
        elif reused is not None and graph["name"] in reused["graphs"]:
            current = link_common_om(reused, graph, root, om_root)
        else:
            print(f"[compile-om] {key} START", flush=True)
            current = _compile_air_graph(
                graph, root=root, om_root=om_root, log_root=log_root, atc_path=atc_path,
                exact_soc_version=exact_soc_version, arguments=graph_arguments[graph["name"]],
                execute=execute,
            )
            print(f"[compile-om] {key} DONE", flush=True)
        compiled.append(current)
        if _shared_compiled is not None and key not in _shared_compiled:
            _shared_compiled[key] = (current, graph_arguments[graph["name"]])

    deployment = {
        "schema_version": 1,
        "artifact_kind": "qwen35-dflash-ascend310p-om-bundle",
        "status": "PASS",
        "target": {"target_id": "ascend310p", "soc_version": exact_soc_version},
        "air_manifest": {
            "path": manifest_path.relative_to(root).as_posix(),
            "sha256": sha256_file(manifest_path),
        },
        "compiler": {
            "path": str(atc_path),
            "identity": atc_identity or _atc_identity(atc_path),
            "framework": 1,
            "extra_args": arguments,
            "graph_extra_args": graph_arguments,
            "precision_policy": "preserve_graph_dtypes" if incremental else "explicit_args_or_atc_default",
        },
        "graphs": compiled,
        **({"common_reuse": air_manifest["common_reuse"]} if reused is not None else {}),
    }
    output = atomic_write_json(deployment_path, deployment)
    deployment["manifest_path"] = str(output)
    return deployment


def recompile_draft_om(
    deployment_manifest_path: str | Path, *, output: str | Path,
    atc_bin: str | Path | None = None,
    deterministic: int = 0,
    runner: Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]] | None = None,
    atc_identity: str | None = None,
) -> dict[str, Any]:
    """Recompile the single Draft OM, retaining the Target OMs.

    The new manifest shares its parent's bundle root so existing hash-locked
    AIR payloads and Target OMs need neither copies nor path/ABI changes.
    The old deployment remains usable for rollback and paired comparisons.
    """
    from .incremental_plan import validate_incremental_bundle

    if type(deterministic) is not int or deterministic not in (0, 1):
        raise ValueError("deterministic must be 0 or 1")
    source = Path(deployment_manifest_path).expanduser().resolve()
    root = require_run_output(source.parent)
    destination = require_run_output(output)
    if destination.parent != root:
        raise ValueError("new deployment manifest must be beside the existing manifest")
    if destination.exists():
        raise FileExistsError(destination)
    parent_record = file_record(source, relative_to=root)
    deployment = load_json_object(source)
    if (deployment.get("status") != "PASS"
            or deployment.get("artifact_kind") != "qwen35-dflash-ascend310p-om-bundle"):
        raise ValueError("Draft recompilation requires a passing deployment bundle")
    graphs = deployment["graphs"]
    contract = validate_incremental_bundle(graphs)
    if contract is None:
        raise ValueError("Draft recompilation requires an incremental chunk bundle")
    soc = validate_soc_version(deployment["target"]["soc_version"])
    air_record = deployment["air_manifest"]
    air_path = contained_path(root, air_record["path"])
    if air_path.parent != root or sha256_file(air_path) != air_record["sha256"]:
        raise ValueError("AIR manifest path/hash differs from deployment")
    air = load_json_object(air_path)
    if (air.get("status") != "PASS" or air.get("artifact_kind") != "qwen35-dflash-torchair-bundle"
            or validate_incremental_bundle(air["graphs"]) != contract):
        raise ValueError("AIR bundle contract differs from deployment")
    air_by_name = {graph["name"]: graph for graph in air["graphs"]}
    if set(air_by_name) != {graph["name"] for graph in graphs}:
        raise ValueError("AIR and deployment graph sets differ")
    for graph in graphs:
        exported = air_by_name[graph["name"]]
        for key in ("role", "metadata", "input_names", "output_names", "air", "runtime_input_abi"):
            if graph.get(key) != exported.get(key):
                raise ValueError(f"AIR/deployment {key} differs: {graph['name']}")
        if graph.get("custom_op_audit", []) != _validated_custom_op_audit(exported):
            raise ValueError("AIR/deployment custom-operator audit differs")
        if graph.get("standard_op_overrides", []) != _validated_standard_op_overrides(exported):
            raise ValueError("AIR/deployment standard-operator audit differs")
        validated_runtime_input_abi(exported, required=True, allow_test_double=runner is not None)
        old_om = contained_path(root, graph["om"]["path"])
        if file_record(old_om, relative_to=root) != graph["om"]:
            raise ValueError(f"OM integrity check failed: {graph['name']}")

    selected = [g for g in graphs if g["name"] == "draft"]
    graph_arguments = {}
    for graph in selected:
        name, old_command = graph["name"], graph["atc_command"]
        if (not isinstance(old_command, list) or len(old_command) < 2
                or not all(isinstance(value, str) for value in old_command)):
            raise ValueError(f"missing original Draft ATC command: {name}")
        # Preserve each graph's actual flags, not only the common bundle flags.
        core = {"--mode", "--framework", "--model", "--output", "--soc_version"}
        inherited = _validate_extra_args([
            value for value in old_command[1:] if value.split("=", 1)[0] not in core
        ])
        _graph_atc_args(inherited, name=name, incremental=True)
        inherited = [value for value in inherited if value.split("=", 1)[0] != "--deterministic"]
        inherited.append(f"--deterministic={deterministic}")
        graph_arguments[name] = _graph_atc_args(
            _chunk_precision_args(inherited, incremental=True), name=name, incremental=True)
        graph_arguments[name] = _dynamic_atc_args(air_by_name[name], graph_arguments[name])
    atc_path = resolve_atc_executable(
        atc_bin or os.environ.get("ASCEND310P_ATC_BIN") or selected[0]["atc_command"][0])
    stage = Path(tempfile.mkdtemp(prefix=f"draft-det{deterministic}-", dir=root))
    log_root = stage / "log"
    log_root.mkdir()
    compiled = {
        name: _compile_air_graph(
            air_by_name[name], root=root, om_root=stage, log_root=log_root,
            atc_path=atc_path, exact_soc_version=soc, arguments=arguments,
            execute=runner or _default_runner,
        ) for name, arguments in graph_arguments.items()
    }
    # Publish only after ATC succeeded and the reused authority still matches.
    if file_record(source, relative_to=root) != parent_record or sha256_file(air_path) != air_record["sha256"]:
        raise ValueError("source manifests changed during Draft compilation")
    for graph in graphs:
        if file_record(contained_path(root, graph["om"]["path"]), relative_to=root) != graph["om"]:
            raise ValueError(f"original OM changed during Draft compilation: {graph['name']}")
    deployment["graphs"] = [compiled.get(graph["name"], graph) for graph in graphs]
    deployment["recompilation"] = {
        "parent_manifest": parent_record,
        "graphs": list(compiled),
        "reason": "Draft-only deterministic setting change; device validation pending",
        "deterministic": deterministic,
        "compiler": {"path": str(atc_path), "identity": atc_identity or _atc_identity(atc_path),
                     "extra_args": graph_arguments["draft"],
                     "graph_extra_args": graph_arguments},
        "ordinary_parity": "NOT_RUN", "formal_latency_evidence": False,
    }
    # Common compiler metadata describes the original build. Graph commands
    # and this override record describe the changed artifact precisely.
    deployment["compiler"].setdefault("graph_extra_args", {}).update(graph_arguments)
    deployment.pop("manifest_path", None)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(deployment, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    deployment["manifest_path"] = str(destination)
    return deployment
