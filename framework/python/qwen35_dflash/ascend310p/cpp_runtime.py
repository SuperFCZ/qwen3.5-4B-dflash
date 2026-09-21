"""Control plane for the low-overhead AscendCL C++ paired OM runner."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

from .utils import (
    atomic_write_json,
    contained_path,
    file_record,
    load_json_object,
    require_run_output,
    sha256_file,
)


CPP_RUNNER_ID = "qwen35-dflash-ascendcl-cpp-v1"
_GENERIC_DEVICE_NAMES = {"310p", "ascend310p", "atlas310p"}


def resolve_cpp_runner(path: str | Path) -> Path:
    executable = Path(path).expanduser().resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError(f"C++ ACL runner is not executable: {executable}")
    return executable


def preflight_cpp_runner(path: str | Path) -> Path:
    """Prove that the target binary and its dynamic AscendCL deps can start."""

    executable = resolve_cpp_runner(path)
    result = subprocess.run(
        [str(executable), "--help"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0 or "qwen35_dflash_acl_runner" not in result.stdout:
        raise RuntimeError(
            "C++ ACL runner cannot start in the activated target environment: "
            f"exit={result.returncode}, output={result.stdout!r}"
        )
    return executable


def _runtime_identity(options: Mapping[str, Any], device_id: int) -> dict[str, Any]:
    required = ("device_model", "cann", "driver", "firmware", "runtime")
    missing = [name for name in required if not str(options.get(name, "")).strip()]
    if missing:
        raise ValueError(f"C++ runner config is missing identities: {missing}")
    model = str(options["device_model"]).strip()
    normalized = re.sub(r"[^a-z0-9]", "", model.lower())
    if normalized in _GENERIC_DEVICE_NAMES:
        raise ValueError("C++ runner config must name the concrete 310P product")
    graph_name = str(options.get("graph_name", "quant_dflash_recompute"))
    pad_token_id = int(options.get("pad_token_id", 0))
    if pad_token_id < 0:
        raise ValueError("C++ runner pad_token_id must be non-negative")
    return {
        "cpu_fallback": False,
        "device": {
            "target_id": "ascend310p",
            "model": model,
            "device_id": int(device_id),
        },
        "cann": str(options["cann"]),
        "driver": str(options["driver"]),
        "firmware": str(options["firmware"]),
        "runtime": str(options["runtime"]),
        "graph_name": graph_name,
        "pad_token_id": pad_token_id,
    }


def validate_cpp_runner_options(
    options: Mapping[str, Any], device_id: int
) -> dict[str, Any]:
    return _runtime_identity(options, device_id)


def build_cpp_runner(
    *,
    build_dir: str | Path,
    output: str | Path,
    cmake: str | Path = "cmake",
    ascendcl_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build the production ACL binary without writing into the model repo."""

    explicit_source = os.environ.get("QWEN35_DFLASH_CPP_SOURCE")
    candidates: list[Path] = []
    if explicit_source:
        candidates.append(Path(explicit_source).expanduser().resolve())
    candidates.append(Path(__file__).resolve().parents[3] / "runtime" / "cpp")
    model_root_value = os.environ.get("AI_MODEL_ROOT")
    if model_root_value:
        candidates.append(
            Path(model_root_value).expanduser().resolve()
            / "targets"
            / "ascend310p"
            / "runtime"
            / "cpp"
        )
    source = next(
        (item for item in candidates if (item / "CMakeLists.txt").is_file()),
        candidates[0],
    )
    if not (source / "CMakeLists.txt").is_file():
        raise FileNotFoundError(
            "C++ runner source is missing; searched: "
            + ", ".join(str(item) for item in candidates)
        )
    build = require_run_output(build_dir)
    if build.exists() and any(build.iterdir()):
        raise FileExistsError(f"C++ runner build directory is not empty: {build}")
    report_path = require_run_output(output)
    if report_path.exists():
        raise FileExistsError(f"C++ runner build report already exists: {report_path}")
    configured = str(cmake)
    cmake_path = Path(configured).expanduser()
    if cmake_path.parent == Path("."):
        resolved = shutil.which(configured)
        if resolved is None:
            raise RuntimeError(f"CMake executable is unavailable: {configured}")
        cmake_path = Path(resolved)
    cmake_path = cmake_path.resolve()
    if not cmake_path.is_file() or not os.access(cmake_path, os.X_OK):
        raise RuntimeError(f"CMake executable is invalid: {cmake_path}")
    build.mkdir(parents=True, exist_ok=True)
    run_root = Path(os.environ["AI_RUN_DIR"]).expanduser().resolve()
    log_root = run_root / "log" / "dflash-cpp-build"
    log_root.mkdir(parents=True, exist_ok=True)
    configure_command = [
        str(cmake_path),
        "-S",
        str(source),
        "-B",
        str(build),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DQWEN35_DFLASH_BUILD_ACL_RUNNER=ON",
        "-DQWEN35_DFLASH_BUILD_TESTS=ON",
    ]
    if ascendcl_root is not None:
        configure_command.append(
            f"-DASCENDCL_ROOT={Path(ascendcl_root).expanduser().resolve()}"
        )
    build_command = [
        str(cmake_path),
        "--build",
        str(build),
        "--config",
        "Release",
        "--parallel",
    ]
    test_command = [
        "ctest",
        "--test-dir",
        str(build),
        "--build-config",
        "Release",
        "--output-on-failure",
    ]
    commands = (
        ("configure", configure_command),
        ("build", build_command),
        ("host-tests", test_command),
    )
    logs: dict[str, dict[str, Any]] = {}
    for name, command in commands:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_path = log_root / f"{name}.log"
        log_path.write_text(result.stdout or "", encoding="utf-8")
        logs[name] = file_record(log_path, relative_to=run_root)
        if result.returncode != 0:
            raise RuntimeError(
                f"C++ runner {name} failed with exit {result.returncode}; log={log_path}"
            )
    candidates = (
        build / "qwen35_dflash_acl_runner",
        build / "Release" / "qwen35_dflash_acl_runner",
    )
    runner = next((item for item in candidates if item.is_file()), None)
    if runner is None or not os.access(runner, os.X_OK):
        raise RuntimeError("C++ build succeeded but produced no executable ACL runner")
    preflight_cpp_runner(runner)
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "artifact_kind": "qwen35-dflash-ascendcl-cpp-runner",
        "source": str(source),
        "cmake": str(cmake_path),
        "ascendcl_root": (
            None
            if ascendcl_root is None
            else str(Path(ascendcl_root).expanduser().resolve())
        ),
        "runner": file_record(runner, relative_to=run_root),
        "logs": logs,
        "claim_boundary": (
            "Host scheduler and fake-ACL integration tests passed; physical-device "
            "latency is established only by infer-cpp/run-e2e-cpp target reports."
        ),
    }
    atomic_write_json(report_path, payload)
    payload["report_path"] = str(report_path)
    payload["runner_path"] = str(runner)
    return payload


def _resolve_integrated_om(
    deployment_manifest: str | Path,
    *,
    graph_name: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    manifest_path = Path(deployment_manifest).expanduser().resolve()
    manifest = load_json_object(manifest_path)
    if manifest.get("artifact_kind") != "qwen35-dflash-ascend310p-om-bundle":
        raise ValueError("C++ runner requires a DFlash Ascend 310P OM bundle")
    if manifest.get("status") != "PASS":
        raise ValueError("deployment manifest is not passing")
    matches = [
        graph
        for graph in manifest.get("graphs", [])
        if isinstance(graph, Mapping) and str(graph.get("name")) == graph_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"deployment manifest needs one {graph_name!r} graph, got {len(matches)}"
        )
    graph = dict(matches[0])
    if graph.get("role") != "generation-recompute":
        raise ValueError("C++ runner graph role must be generation-recompute")
    if list(graph.get("input_names", [])) != ["input_ids", "attention_mask"]:
        raise ValueError("C++ runner OM input order differs from the locked ABI")
    if list(graph.get("output_names", [])) != ["target_top1", "draft_top1"]:
        raise ValueError("C++ runner OM output order differs from the locked ABI")
    record = graph.get("om")
    if not isinstance(record, Mapping):
        raise ValueError("deployment graph has no OM record")
    om_path = contained_path(manifest_path.parent, str(record["path"]))
    if not om_path.is_file() or sha256_file(om_path) != str(record["sha256"]):
        raise ValueError("OM artifact integrity check failed before C++ runner launch")
    return om_path, manifest, graph


def _token_csv(values: Sequence[int]) -> str:
    result = [int(item) for item in values]
    if any(item < 0 for item in result):
        raise ValueError("token IDs must be non-negative")
    return ",".join(str(item) for item in result)


def _validate_mode_report(
    name: str,
    report: Mapping[str, Any],
    *,
    generation_mode: str,
    warmup: int = 3,
    repetitions: int = 10,
    require_repeatability: bool = False,
    max_new_tokens: int | None = None,
    eos_token_ids: Sequence[int] = (),
) -> None:
    from .repeatability import repeatability_observation, representative_output

    if report.get("status") not in ("PASS", "PASS_WITH_OBSERVATIONS"):
        raise RuntimeError(f"C++ {name} report is not passing")
    if report.get("generation_mode") != generation_mode:
        raise RuntimeError(f"C++ {name} generation mode differs")
    if (type(report.get("warmup")) is not int or type(report.get("repetitions")) is not int
            or report["warmup"] != warmup or report["repetitions"] != repetitions):
        raise RuntimeError(f"C++ {name} report differs from requested {warmup}+{repetitions} measurements")
    measurements = report.get("measurements")
    if not isinstance(measurements, list) or len(measurements) != repetitions:
        raise RuntimeError(f"C++ {name} report must retain {repetitions} raw measurements")
    for measurement in measurements:
        tokens, stop = measurement.get("generated_token_ids"), measurement.get("stop_reason")
        if (not isinstance(tokens, list) or not tokens
                or any(type(t) is not int or t < 0 for t in tokens)
                or stop not in ("eos", "max_new_tokens")
                or (max_new_tokens is not None and len(tokens) > max_new_tokens)
                or any(t in eos_token_ids for t in tokens[:-1])
                or (stop == "eos" and tokens[-1] not in eos_token_ids)
                or (stop == "max_new_tokens" and
                    (len(tokens) != max_new_tokens or tokens[-1] in eos_token_ids))):
            raise RuntimeError(f"C++ {name} measurement has invalid output length/EOS termination")
    observation = repeatability_observation(report)
    expected, stop = representative_output(report)
    if "repeatability" not in report and (
            report.get("stable_generated_token_ids") != expected
            or report.get("stable_stop_reason") != stop):
        raise RuntimeError(f"C++ {name} saved reference differs from measurement 0")
    if "representative_generated_token_ids" in report and (
            report["representative_generated_token_ids"] != expected
            or report.get("representative_stop_reason") != stop
            or report.get("representative_repetition") != 0):
        raise RuntimeError(f"C++ {name} representative output differs from measurement 0")
    if "repeatability" in report:
        drift = observation["status"] == "DRIFT_OBSERVED"
        if (report["repeatability"] != observation
                or report["status"] != ("PASS_WITH_OBSERVATIONS" if drift else "PASS")
                or report.get("stable_generated_token_ids") != (None if drift else expected)
                or report.get("stable_stop_reason") != (None if drift else stop)):
            raise RuntimeError(f"C++ {name} repeatability metadata disagrees with measurements")
    if require_repeatability and observation["differences"]:
        raise RuntimeError(f"C++ {name} repetitions are not token-stable/EOS-stable")


def validate_cpp_runner_report(
    report: Mapping[str, Any],
    *,
    prompt_token_ids: Sequence[int],
    om_sha256: str,
    device_id: int,
    max_new_tokens: int,
    max_draft_tokens: int,
    chunk_abi: bool = False,
    verify_gdr: str | None = None,
    low_memory: bool = False,
    draft_context_rows: int | None = None,
    draft_prefill_policy: str | None = None,
    allow_output_differences: bool = False,
    warmup: int = 3,
    repetitions: int = 10,
    require_repeatability: bool = False,
) -> None:
    if type(warmup) is not int or warmup < 0 or type(repetitions) is not int or repetitions <= 0:
        raise ValueError("warmup must be non-negative and repetitions must be positive integers")
    allowed_parity_failure = (
        allow_output_differences
        and report.get("status") == "FAIL"
        and report.get("failure_stage") == "ordinary_dflash_parity"
    )
    if (report.get("status") not in ("PASS", "PASS_WITH_OBSERVATIONS") and not allowed_parity_failure) or report.get("runner_id") != CPP_RUNNER_ID:
        raise RuntimeError("C++ ACL runner did not produce a passing known report")
    if report.get("cpu_fallback") is not False:
        raise RuntimeError("C++ target report indicates CPU fallback")
    if int(report.get("device_id", -1)) != int(device_id):
        raise RuntimeError("C++ runner used a different device ID")
    model = report.get("model")
    if not isinstance(model, Mapping) or model.get("sha256") != om_sha256:
        raise RuntimeError("C++ runner OM hash differs from the deployment manifest")
    if list(report.get("prompt_token_ids", [])) != [int(item) for item in prompt_token_ids]:
        raise RuntimeError("C++ runner prompt token IDs differ")
    limits = report.get("limits", {})
    if int(limits.get("max_new_tokens", -1)) != int(max_new_tokens):
        raise RuntimeError("C++ runner max_new_tokens differs")
    if int(limits.get("max_draft_tokens", -1)) != int(max_draft_tokens):
        raise RuntimeError("C++ runner max_draft_tokens differs")
    protocol = report.get("protocol", {})
    if (type(protocol.get("warmup")) is not int or type(protocol.get("repetitions")) is not int
            or protocol["warmup"] != warmup or protocol["repetitions"] != repetitions):
        raise RuntimeError(f"C++ runner protocol differs from requested {warmup}+{repetitions}")
    if protocol.get("low_memory", False) is not low_memory:
        raise RuntimeError("C++ runner low-memory mode differs from the request")
    abi = report.get("abi", {})
    # Preserve analysis of saved reports; live bundles require a single Draft.
    # Live deployments are checked by validate_incremental_bundle before execution.
    reported_rows = abi.get("draft_context_rows", 64)
    if chunk_abi and (type(reported_rows) is not int or reported_rows not in (16, 64)
                      or draft_context_rows is not None and reported_rows != draft_context_rows):
        raise RuntimeError("C++ runner Draft execution gear differs")
    reported_policy = abi.get("draft_prefill_policy")
    if chunk_abi and (reported_policy not in (None, "single_draft16_subchunks", "single_draft16_64_gears", "static_draft16_64_oms")
                      or draft_prefill_policy is not None and reported_policy != draft_prefill_policy):
        raise RuntimeError("C++ runner Draft prefill policy differs")
    if chunk_abi and reported_policy in ("single_draft16_64_gears", "static_draft16_64_oms") and abi.get("draft_context_gears") != [16, 64]:
        raise RuntimeError("C++ runner Draft dynamic gears differ")
    extra_context_graph = chunk_abi and reported_rows == 16 and reported_policy is None
    if low_memory and (
        not chunk_abi
        or protocol.get("order") not in (
            "ordinary then DFlash with model unload between modes", "saved ordinary baseline then DFlash")
        or protocol.get("max_resident_models") != 3 + extra_context_graph + (reported_policy == "static_draft16_64_oms")
    ):
        raise RuntimeError("C++ runner low-memory protocol differs")
    abi = report.get("abi", {})
    if chunk_abi:
        from .incremental_plan import require_verify_gdr
        try:
            require_verify_gdr({"abi": abi.get("id")}, verify_gdr)
        except ValueError as error:
            raise RuntimeError("C++ runner incremental ABI differs: " + str(error)) from error
        if abi.get("graph_count") != 4 + extra_context_graph:
            raise RuntimeError("C++ runner incremental graph count differs")
    if not chunk_abi and abi.get("input_names") != ["input_ids", "attention_mask"]:
        raise RuntimeError("C++ runner input ABI differs")
    if not chunk_abi and abi.get("output_names") != ["target_top1", "draft_top1"]:
        raise RuntimeError("C++ runner output ABI differs")
    if not chunk_abi and str(abi.get("dtype", "")).lower() != "int64":
        raise RuntimeError("C++ runner ABI dtype differs")
    ordinary = report.get("ordinary")
    dflash = report.get("dflash")
    if not isinstance(ordinary, Mapping) or not isinstance(dflash, Mapping):
        raise RuntimeError("C++ runner omitted paired mode reports")
    _validate_mode_report(
        "ordinary", ordinary, generation_mode="ordinary-greedy", warmup=warmup, repetitions=repetitions,
        require_repeatability=require_repeatability, max_new_tokens=max_new_tokens,
        eos_token_ids=report.get("eos_token_ids", [])
    )
    _validate_mode_report(
        "DFlash", dflash, generation_mode="dflash-strict-greedy", warmup=warmup, repetitions=repetitions,
        require_repeatability=require_repeatability, max_new_tokens=max_new_tokens,
        eos_token_ids=report.get("eos_token_ids", [])
    )
    from .repeatability import representative_output, repeatability_observation
    expected, expected_stop = representative_output(ordinary)
    actual, actual_stop = representative_output(dflash)
    drift = any(repeatability_observation(mode)["differences"] for mode in (ordinary, dflash))
    if report.get("status") == "PASS_WITH_OBSERVATIONS" and not drift:
        raise RuntimeError("C++ runner observation status has no observed drift")
    if allow_output_differences:
        mismatches = sum(
            i >= len(expected) or i >= len(actual) or expected[i] != actual[i]
            for i in range(max(len(expected), len(actual)))
        )
        eos_mismatches = int(expected_stop != actual_stop)
        parity_status = "FAIL" if mismatches or eos_mismatches else "PASS"
        parity = report.get("ordinary_parity", {})
        expected_status = "PASS_WITH_OBSERVATIONS" if drift and parity_status == "PASS" else parity_status
        if (report.get("status") not in (parity_status, expected_status)
                or parity.get("status") != parity_status
                or parity.get("token_id_mismatches") != mismatches
                or parity.get("eos_mismatches") != eos_mismatches):
            raise RuntimeError("C++ runner parity metadata disagrees with saved outputs")
        return
    if expected != actual:
        raise RuntimeError("C++ DFlash tokens differ from ordinary authority")
    if expected_stop != actual_stop:
        raise RuntimeError("C++ DFlash EOS/stop reason differs from ordinary authority")
    parity = report.get("ordinary_parity", {})
    if (
        parity.get("status") != "PASS"
        or parity.get("token_id_mismatches") != 0
        or parity.get("eos_mismatches") != 0
    ):
        raise RuntimeError("C++ runner ordinary parity gate failed")


def run_cpp_pair(
    *,
    deployment_manifest: str | Path,
    runner: str | Path,
    runner_options: Mapping[str, Any],
    prompt_token_ids: Sequence[int],
    eos_token_ids: Sequence[int],
    device_id: int,
    max_new_tokens: int,
    max_draft_tokens: int,
    raw_output: str | Path,
    log_output: str | Path,
    trace_rounds: bool = False,
    low_memory: bool = False,
    verify_gdr: str | None = None,
    execute: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Run paired ordinary/DFlash generation entirely inside one C++ process."""

    if max_new_tokens <= 0 or max_draft_tokens <= 0:
        raise ValueError("C++ generation limits must be positive")
    tokens = [int(item) for item in prompt_token_ids]
    if not tokens:
        raise ValueError("C++ runner prompt tokens must not be empty")
    identity = _runtime_identity(runner_options, device_id)
    deployment = load_json_object(Path(deployment_manifest).expanduser().resolve())
    chunk = any(g.get("metadata", {}).get("incremental_contract") for g in deployment.get("graphs", []))
    if trace_rounds and not chunk:
        raise ValueError("round tracing requires an incremental chunk OM bundle")
    if low_memory and not chunk:
        raise ValueError("low-memory mode requires an incremental chunk OM bundle")
    if chunk:
        from .incremental_plan import write_incremental_plan
        om_path, deployment, contract = write_incremental_plan(
            deployment_manifest, Path(raw_output).with_suffix(".chunk-plan.txt"), verify_gdr=verify_gdr)
        from .incremental_plan import verify_gdr_route
        verify_gdr = verify_gdr_route(contract)
        graph = {"name": contract["abi"], "om": file_record(om_path, relative_to=om_path.parent)}
    else:
        if verify_gdr is not None:
            raise ValueError("--verify-gdr requires an incremental OM bundle")
        om_path, deployment, graph = _resolve_integrated_om(
            deployment_manifest, graph_name=identity["graph_name"])
    om_record = dict(graph["om"])
    executable = resolve_cpp_runner(runner)
    raw_path = require_run_output(raw_output)
    log_path = require_run_output(log_output)
    if raw_path.exists() or log_path.exists():
        raise FileExistsError("C++ runner output/log already exists in this run")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(executable),
        "--model",
        str(om_path),
        "--model-sha256",
        str(om_record["sha256"]),
        "--output",
        str(raw_path),
        "--prompt-token-ids",
        _token_csv(tokens),
        "--eos-token-ids",
        _token_csv(eos_token_ids),
        "--pad-token-id",
        str(identity["pad_token_id"]),
        "--max-new-tokens",
        str(int(max_new_tokens)),
        "--max-draft-tokens",
        str(int(max_draft_tokens)),
        "--warmup",
        "3",
        "--repetitions",
        "10",
        "--device-id",
        str(int(device_id)),
    ]
    if chunk:
        command.extend(("--model-kind", "chunk"))
    if trace_rounds:
        command.append("--trace-rounds")
    if low_memory:
        command.append("--low-memory")
    start_ns = time.perf_counter_ns()
    result = execute(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    end_ns = time.perf_counter_ns()
    log_path.write_text(result.stdout or "", encoding="utf-8")
    if result.returncode != 0:
        detail = next((line for line in reversed((result.stdout or "").splitlines())
                       if line.startswith("qwen35_dflash_acl_runner:")), "")
        raise RuntimeError(
            f"C++ ACL runner failed with exit {result.returncode}; log={log_path}"
            + (f"\n{detail}" if detail else "")
        )
    if not raw_path.is_file():
        raise RuntimeError("C++ ACL runner returned success without a JSON report")
    report = load_json_object(raw_path)
    validate_cpp_runner_report(
        report,
        prompt_token_ids=tokens,
        om_sha256=str(om_record["sha256"]),
        device_id=device_id,
        max_new_tokens=max_new_tokens,
        max_draft_tokens=max_draft_tokens,
        chunk_abi=chunk,
        verify_gdr=verify_gdr,
        low_memory=low_memory,
        draft_context_rows=contract["draft_context_rows"] if chunk else None,
        draft_prefill_policy=contract["draft_prefill_policy"] if chunk else None,
    )
    run_root = Path(os.environ["AI_RUN_DIR"]).expanduser().resolve()
    air_record = deployment.get("air_manifest")
    if not isinstance(air_record, Mapping):
        raise ValueError("deployment manifest has no AIR manifest record")
    air_manifest_path = contained_path(
        Path(deployment_manifest).expanduser().resolve().parent,
        str(air_record.get("path", "")),
    )
    if (
        not air_manifest_path.is_file()
        or sha256_file(air_manifest_path) != str(air_record.get("sha256", ""))
    ):
        raise ValueError("AIR manifest integrity check failed after C++ execution")
    report["backend_metadata"] = {
        **identity,
        "graph_name": str(graph["name"]),
        "artifacts": ({g["name"]: g["om"]["sha256"] for g in deployment["graphs"]} if chunk else {str(graph["name"]): str(om_record["sha256"])}),
        "state_policy": contract["state_policy"] if chunk else "recompute committed prefixes",
        "host_hot_path": "AscendCL C++",
    }
    report["control_plane"] = {
        "process_wall_ms": (end_ns - start_ns) / 1_000_000.0,
        "runner": {
            "path": str(executable),
            "bytes": executable.stat().st_size,
            "sha256": sha256_file(executable),
        },
        "deployment_manifest": file_record(
            Path(deployment_manifest).expanduser().resolve(), relative_to=run_root
        ),
        "air_manifest": file_record(air_manifest_path, relative_to=run_root),
        "runner_raw_report": file_record(raw_path, relative_to=run_root),
        "runner_log": file_record(log_path, relative_to=run_root),
        "compiler": dict(deployment.get("compiler", {})),
        "target": dict(deployment.get("target", {})),
    }
    return report


def write_cpp_prompt_report(
    *,
    payload: Mapping[str, Any],
    output: str | Path,
    prompt: str,
    chat: bool,
    tokenizer_source: Mapping[str, Any],
    tokenize_ms: float,
    detokenize_ms: float,
    generated_text: str,
) -> dict[str, Any]:
    from .repeatability import representative_output

    result = dict(payload)
    result["report_kind"] = "cpp-ascendcl-paired-target"
    result["prompt"] = prompt
    result["chat"] = bool(chat)
    result["tokenizer_source"] = dict(tokenizer_source)
    result["output"] = {
        "token_ids": list(representative_output(result["dflash"])[0]),
        "text": generated_text,
        "stop_reason": str(representative_output(result["dflash"])[1]),
        "measurement_repetition": 0,
    }
    result["host_text_stage_ms"] = {
        "tokenize": float(tokenize_ms),
        "detokenize": float(detokenize_ms),
        "note": (
            "single host measurements outside the 3+10 C++ OM model-loop "
            "distribution; do not combine them into a claimed service latency"
        ),
    }
    result["claim_boundary"] = (
        "C++ removes Python from the OM generation hot path and reports paired "
        "synchronized model-loop latency. Comparable closed-runtime latency still "
        "requires same-device A/B evidence. Preserve protocol.order when comparing "
        "grouped low-memory measurements with alternating measurements."
    )
    target = require_run_output(output)
    atomic_write_json(target, result)
    return result
