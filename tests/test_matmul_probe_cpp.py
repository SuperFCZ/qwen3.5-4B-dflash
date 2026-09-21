"""C++ probe transport tests use fake ACL, never device correctness evidence."""
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from tools import matmul_probe_cpp
from tools import validate_draft_matmul_om as validator
from qwen35_dflash.ascend310p.utils import sha256_file
from test_draft_matmul_om_validation import saved_probe, options  # noqa: F401


@pytest.fixture
def runner():
    path = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if not path:
        pytest.skip("set QWEN35_CPP_TEST_RUNNER to the fake ACL runner")
    return Path(path)


def native_case(tmp_path, rows, defect=""):
    model = tmp_path / "fixture.om"
    dtype = "float32" if defect == "dtype" else "float16"
    actual_rows = 64 if defect == "shape" else rows
    model.write_text(f"FAKE_CHUNK matmul_probe\nI x {dtype} 2 {actual_rows} 256\nO y float16 2 {rows} 256\n")
    inputs = tmp_path / "inputs.bin"
    values = np.arange(4 * rows * 256, dtype=np.int64).reshape(4, rows, 256) % 31
    values.astype(np.float16).tofile(inputs)
    output = tmp_path / "native"
    command = ["--matmul-probe", "--model", str(model), "--model-sha256", sha256_file(model),
               "--input", str(inputs), "--input-sha256", sha256_file(inputs),
               "--rows", str(rows), "--k", "256", "--n", "256", "--repetitions", "3",
               "--device-id", "0", "--output-dir", str(output)]
    return command, output, values.astype(np.float16)


@pytest.mark.parametrize("rows", [16, 64])
def test_native_reuses_model_and_roundtrips_all_vectors(runner, tmp_path, rows):
    command, output, values = native_case(tmp_path, rows)
    cleanup = tmp_path / "cleanup.json"
    env = dict(os.environ, QWEN35_FAKE_CLEANUP_LOG=str(cleanup))
    completed = subprocess.run([str(runner)] + command, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert all(v == 0 for v in json.loads(cleanup.read_text()).values())
    report = json.loads((output / "execution.json").read_text())
    assert report["fake_acl"] is True and report["status"] == "PASS"
    assert report["io_validated"] is True and report["calls"] == 12
    assert report["output_sha256"] == sha256_file(output / "outputs.bin")
    outputs = np.fromfile(output / "outputs.bin", dtype=np.float16).reshape(4, 3, rows, 256)
    assert np.array_equal(outputs, np.repeat(values[:, None], 3, axis=1))
    saved = (output / "outputs.bin").read_bytes()
    repeated = subprocess.run([str(runner)] + command, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert (output / "outputs.bin").read_bytes() == saved


@pytest.mark.parametrize("defect,phase", [
    ("shape", "load"), ("dtype", "load"), ("load", "load"),
    ("execute", "execute"), ("cleanup", "cleanup"), ("hash", None),
])
def test_native_rejects_invalid_artifacts_abi_and_acl_failure(runner, tmp_path, defect, phase):
    command, output, _ = native_case(tmp_path, 16, defect)
    cleanup = tmp_path / "cleanup.json"
    env = dict(os.environ, QWEN35_FAKE_CLEANUP_LOG=str(cleanup))
    if defect == "load":
        env["QWEN35_FAKE_FAIL_LOAD_GRAPH"] = "matmul_probe"
    elif defect == "execute":
        env["QWEN35_FAKE_FAIL_GRAPH"] = "matmul_probe"
    elif defect == "cleanup":
        env["QWEN35_FAKE_CLEANUP_FAIL"] = "aclrtResetDevice"
    elif defect == "hash":
        command[command.index("--model-sha256") + 1] = "0" * 64
    completed = subprocess.run([str(runner)] + command, env=env, capture_output=True, text=True)
    assert completed.returncode != 0
    if phase is not None:
        assert all(v == 0 for v in json.loads(cleanup.read_text()).values())
    if phase is not None:
        report = json.loads((output / "execution.json").read_text())
        assert report["status"] == "FAIL" and report["phase"] == phase
        if phase == "load":
            assert not report["io_validated"] and report["calls"] == 0
    else:
        assert not output.exists()


@pytest.mark.parametrize("defect", ["", "fake", "device", "hash", "truncated", "runtime", "numeric"])
def test_python_checks_native_identity_and_output(saved_probe, tmp_path, monkeypatch, defect):
    _, jobs = validator.prepare(saved_probe)
    job = jobs[0]
    samples = list(validator.vectors(validator.weight_units(job), 16))
    runner = tmp_path / "runner"
    runner.write_bytes(b"explicit host mock")
    root = tmp_path / "out"
    root.mkdir()
    def native(command, **kwargs):
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir()
        values = np.stack([np.repeat(y[None], 3, axis=0) for _, _, y in samples])
        if defect == "numeric":
            values[0, 0, 0, 0] += 1
        values.tofile(output / "outputs.bin")
        report = dict(schema_version=1, runtime="AscendCL C++ matmul probe", status="PASS", phase="complete",
                      fake_acl=False, cpu_fallback=False, device_id=0, rows=16, k=256, n=64,
                      vectors=4, repetitions=3, calls=12, io_validated=True,
                      model_sha256=job["graph"]["static_gear_oms"][0]["om"]["sha256"],
                      input_sha256=sha256_file(Path(command[command.index("--input") + 1])),
                      output_sha256=sha256_file(output / "outputs.bin"))
        if defect == "fake": report["fake_acl"] = True
        if defect == "device": report["device_id"] = 1
        if defect == "runtime": report["runtime"] = "CPU simulation"
        if defect == "hash": report["output_sha256"] = "0" * 64
        if defect == "truncated": (output / "outputs.bin").write_bytes(b"")
        (output / "execution.json").write_text(json.dumps(report))
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(matmul_probe_cpp.subprocess, "run", native)
    result = dict(phase="load", execution_status="NOT_RUN", vectors=[])
    def execute():
        return matmul_probe_cpp.execute(runner, job, 16, samples, device_id=0,
                                       repetitions=3, root=root, result=result)
    if defect in ("fake", "device", "hash", "truncated", "runtime"):
        with pytest.raises(ValueError): execute()
    else:
        output = execute()
        validator.evaluate_vectors(result, samples, 3, lambda i, r, x: output[i, r])
        assert result["status"] == ("FAIL" if defect == "numeric" else "PASS")


def test_cpp_selection_never_constructs_python_acl(saved_probe, tmp_path, monkeypatch):
    args = options(tmp_path, saved_probe)
    runner = tmp_path / "runner"
    runner.write_bytes(b"explicit host mock")
    args.runner = runner
    monkeypatch.setattr(matmul_probe_cpp, "preflight", lambda p: p)
    monkeypatch.setattr(validator, "AclOmRuntime", lambda *a, **kw: pytest.fail("Python acl must not be used"))
    def execute(runner, job, rows, samples, **kwargs):
        kwargs["result"].update(execution_status="PASS", phase="compare")
        return np.stack([np.repeat(y[None], args.repetitions, axis=0) for _, _, y in samples])
    monkeypatch.setattr(matmul_probe_cpp, "execute", execute)
    assert validator.run(args) == 0


def test_build_option_builds_runner_once_and_uses_existing_oms(saved_probe, tmp_path, monkeypatch):
    args = options(tmp_path, saved_probe)
    args.build_cpp_runner = True
    args.ascendcl_root = tmp_path / "declared-cann"
    builds, calls = [], []
    def build(**kwargs):
        builds.append(kwargs)
        kwargs["build_dir"].mkdir()
        path = kwargs["build_dir"] / "qwen35_dflash_acl_runner"
        path.write_bytes(b"explicit host mock")
        return {"runner_path": str(path)}
    monkeypatch.setattr(validator, "build_cpp_runner", build)
    monkeypatch.setattr(matmul_probe_cpp, "preflight", Path)
    monkeypatch.setattr(validator, "AclOmRuntime", lambda *a, **kw: pytest.fail("no Python acl"))
    def execute(runner, job, rows, samples, **kwargs):
        calls.append(rows)
        return np.stack([np.repeat(y[None], args.repetitions, axis=0) for _, _, y in samples])
    monkeypatch.setattr(matmul_probe_cpp, "execute", execute)
    before = {p: p.read_bytes() for p in saved_probe.parent.rglob("*") if p.is_file()}
    assert validator.run(args) == 0
    assert len(builds) == 1 and calls == [16, 64]
    assert builds[0] == dict(build_dir=args.output_dir / "cpp-build",
                             output=args.output_dir / "cpp-build.json", ascendcl_root=args.ascendcl_root)
    assert all(p.read_bytes() == value for p, value in before.items())


def test_old_runner_rejected_with_build_command(tmp_path):
    runner = tmp_path / "old-runner"
    runner.write_text("#!/bin/sh\nprintf 'Usage: qwen35_dflash_acl_runner\\n'\n")
    runner.chmod(0o755)
    with pytest.raises(RuntimeError, match="--build-cpp-runner"):
        matmul_probe_cpp.preflight(runner)


def test_fake_native_report_cannot_pass_device_gate(saved_probe, tmp_path, monkeypatch):
    _, jobs = validator.prepare(saved_probe)
    # The transport identity gate above rejects fake_acl=True. This higher-level
    # failure must also remain FAIL and must not attempt the Python backend.
    args = options(tmp_path, saved_probe)
    args.runner = tmp_path / "fake"
    args.runner.write_bytes(b"explicit host mock")
    monkeypatch.setattr(matmul_probe_cpp, "preflight", lambda p: p)
    def reject(*a, **kw):
        raise ValueError("C++ probe execution identity differs: fake_acl")
    monkeypatch.setattr(matmul_probe_cpp, "execute", reject)
    monkeypatch.setattr(validator, "AclOmRuntime", lambda *a, **kw: pytest.fail("no fallback"))
    assert validator.run(args) == 1
    report = json.loads((args.output_dir / "summary.json").read_text())
    assert report["status"] == "FAIL"
    assert all(c["status"] == "FAIL" for c in report["cases"])
