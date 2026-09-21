"""Native AscendCL transport for the synthetic offline MatMul validator."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np

from qwen35_dflash.ascend310p.cpp_runtime import preflight_cpp_runner
from qwen35_dflash.ascend310p.utils import contained_path, sha256_file


def preflight(runner):
    path = preflight_cpp_runner(runner)
    help_result = subprocess.run([str(path), "--help"], capture_output=True, text=True)
    if help_result.returncode or "--matmul-probe" not in help_result.stdout:
        raise RuntimeError("C++ runner lacks --matmul-probe; rebuild it or use --build-cpp-runner")
    return path


def execute(runner, job, rows, samples, *, device_id, repetitions, root, result):
    """One native process loads one OM and reuses it for all vectors/repetitions."""
    k, n = samples[0][1].shape[1], samples[0][2].shape[1]
    gear = job["graph"]["static_gear_oms"][int(rows == 64)]["om"]
    model = contained_path(job["deployment"].parent, gear["path"])
    case_dir = root / f"{job['projection']}-m{rows}"
    case_dir.mkdir()
    inputs = case_dir / "inputs.bin"
    inputs.write_bytes(b"".join(x.tobytes() for _, x, _ in samples))
    input_hash = sha256_file(inputs)
    output_dir = case_dir / "native"
    command = [
        str(runner), "--matmul-probe", "--model", str(model),
        "--model-sha256", gear["sha256"], "--input", str(inputs),
        "--input-sha256", input_hash, "--rows", str(rows),
        "--k", str(k), "--n", str(n), "--repetitions", str(repetitions),
        "--device-id", str(device_id), "--output-dir", str(output_dir),
    ]
    result["native_command"] = command
    completed = subprocess.run(command, capture_output=True, text=True)
    log = case_dir / "runner.log"
    log.write_text(completed.stdout + completed.stderr)
    result["native_log"] = str(log)
    report_path = output_dir / "execution.json"
    native = json.loads(report_path.read_text()) if report_path.is_file() else {}
    result["native_execution"] = native
    result["native_report"] = str(report_path)
    if native.get("phase") in ("load", "execute", "cleanup", "complete"):
        result["phase"] = native["phase"]
    if completed.returncode != 0 or native.get("status") != "PASS":
        if native.get("phase") in ("execute", "cleanup"):
            result["execution_status"] = "FAIL"
        raise RuntimeError(f"C++ probe failed: {native.get('error', 'missing execution report')}; log={log}")
    # A successful host test double is not physical device evidence.
    required = dict(
        schema_version=1, runtime="AscendCL C++ matmul probe", fake_acl=False,
        cpu_fallback=False, status="PASS", phase="complete", device_id=device_id,
        model_sha256=gear["sha256"], input_sha256=input_hash, rows=rows,
        k=k, n=n, vectors=4, repetitions=repetitions, calls=4 * repetitions,
        io_validated=True,
    )
    for key, expected in required.items():
        if key not in native or native[key] != expected or type(native[key]) is not type(expected):
            raise ValueError(f"C++ probe execution identity differs: {key}")
    output = output_dir / "outputs.bin"
    size = 4 * repetitions * rows * n * 2
    if (output.stat().st_size != size or sha256_file(output) != native.get("output_sha256")
            or sha256_file(inputs) != input_hash or sha256_file(model) != gear["sha256"]):
        raise ValueError("C++ probe source/output size or hash differs")
    result["om_sha256"] = gear["sha256"]
    result["execution_status"] = "PASS"
    result["phase"] = "compare"
    return np.fromfile(output, dtype=np.float16).reshape(4, repetitions, rows, n)
