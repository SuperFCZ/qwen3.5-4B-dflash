"""Host-only contract tests: fake execution never establishes 310P correctness."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools import validate_draft_matmul_om as validator
from tools.probe_draft_matmul_atc import make_spec
from qwen35_dflash.ascend310p import acl_runtime
from qwen35_dflash.ascend310p.draft_gears import STATIC_POLICY
from qwen35_dflash.ascend310p.utils import file_record, sha256_file
from test_incremental_air_om import small_threads  # noqa: F401

pytestmark = pytest.mark.usefixtures("small_threads")


@pytest.fixture
def saved_probe(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    root = tmp_path / "compiled"
    root.mkdir()
    spec = make_spec(8, "tiny", "cpu", prepack_dir=root / "offline", static_om_gears=True)
    offline = root / "offline/manifest.json"
    abi = {"status": "NOT_APPLICABLE_EXPLICIT_TEST_DOUBLE", "weight_quant_layout": {
        "prepack": {"status": "PASS", "offline_manifest_sha256": sha256_file(offline)}}}
    graph = dict(name=validator.GRAPH, input_names=["x"], output_names=["y"],
                 metadata=spec.metadata, runtime_input_abi=abi, static_gear_oms=[])
    for rows in (16, 64):
        om = root / f"probe-{rows}.om"
        om.write_bytes(f"host fixture only {rows}".encode())
        graph["static_gear_oms"].append(dict(rows=rows, om=file_record(om, relative_to=root),
            atc_command=["atc", "--input_format=ND", f"--input_shape=x:{rows},256",
                         "--precision_mode=must_keep_origin_dtype"],
            atc_log=f"static{rows}.log"))
    graph.update({key: graph["static_gear_oms"][0][key] for key in ("om", "atc_command", "atc_log")})
    deployment = root / "deployment.json"
    deployment.write_text(json.dumps(dict(status="PASS", artifact_kind="qwen35-dflash-ascend310p-om-bundle",
                                         graphs=[graph])))
    summary = root / "summary.json"
    summary.write_text(json.dumps(dict(status="PASS", soc_version="Ascend310P3",
        cases=[dict(projection="tiny", bits=8, group_size=128, weight_layout="nk", weight_format="nz",
            offline_weight_prepack=True, control="prepacked-static-gears", status="PASS", phase="complete",
            deployment_manifest=str(deployment), static_gear_oms=graph["static_gear_oms"])])))
    return summary


def options(tmp_path, source):
    return SimpleNamespace(probe_summary=source, output_dir=tmp_path / "validation", device_id=0, repetitions=2)


def fake_runtime(monkeypatch, job, defect=None):
    units = validator.weight_units(job).numpy()
    loaded, closed = [], []
    class Runtime:
        def __init__(self, path, device_id, static_gear_rows):
            assert Path(path) == job["deployment"] and device_id == 0
            self.rows = static_gear_rows
            self.graph_names = (validator.GRAPH,)
            self.acl = SimpleNamespace(__file__="HOST_TEST_DOUBLE")
            self.calls = 0
            loaded.append(self.rows)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            closed.append(self.rows)
        def artifact_hashes(self):
            return {validator.GRAPH: job["graph"]["static_gear_oms"][int(self.rows == 64)]["om"]["sha256"]}
        def graph_inputs(self, _):
            return (dict(name="x", shape=[16 if defect == "abi" else self.rows, 256], dtype="float16"),)
        def graph_outputs(self, _):
            return (dict(name="y", shape=[self.rows, 64], dtype="float16"),)
        def run_graph(self, _, inputs):
            self.calls += 1
            result = (inputs["x"].astype(np.float32) @ units.T / 32).astype(np.float16)
            if defect == "nan":
                result[0, 0] = np.nan
            elif defect == "drift" and self.calls % 2 == 0:
                result[0, 0] += 1
            elif defect == "value":
                result[0, 0] += 1
            return {"y": result}
        def synchronize(self):
            pass
    monkeypatch.setattr(validator, "AclOmRuntime", Runtime)
    return loaded, closed


def test_executes_both_gears_and_repetitions_without_mutating_probe(saved_probe, tmp_path, monkeypatch):
    _, jobs = validator.prepare(saved_probe)
    loaded, closed = fake_runtime(monkeypatch, jobs[0])
    before = {p: p.read_bytes() for p in saved_probe.parent.rglob("*") if p.is_file()}
    assert validator.run(options(tmp_path, saved_probe)) == 0
    report = json.loads((tmp_path / "validation/summary.json").read_text())
    assert loaded == closed == [16, 64]
    assert len({case["om_sha256"] for case in report["cases"]}) == 2
    assert report["full_draft_validation"] == report["latency_status"] == "NOT_RUN"
    for case in report["cases"]:
        assert case["status"] == case["execution_status"] == "PASS"
        assert len(case["vectors"]) == 4
        for vector in case["vectors"]:
            assert vector["status"] == "PASS" and not vector["repeat_drift"]
            assert len(vector["measurements"]) == 2
            assert all(m["mismatched_values"] == 0 for m in vector["measurements"])
    assert all(p.read_bytes() == value for p, value in before.items())


@pytest.mark.parametrize("defect", ["abi", "value", "nan", "drift"])
def test_invalid_abi_nonfinite_incorrect_or_unstable_output_is_failure(saved_probe, tmp_path, monkeypatch, defect):
    _, jobs = validator.prepare(saved_probe)
    loaded, closed = fake_runtime(monkeypatch, jobs[0], defect)
    assert validator.run(options(tmp_path, saved_probe)) == 1
    report = json.loads((tmp_path / "validation/summary.json").read_text())
    assert report["status"] == "FAIL" and loaded == closed == [16, 64]
    if defect == "abi":
        assert report["cases"][1]["execution_status"] == "NOT_RUN"
        assert "ABI differs" in report["cases"][1]["error"]
    else:
        assert report["cases"][0]["vectors"][0]["status"] == "FAIL"


@pytest.mark.parametrize("corruption", ["om64", "gear", "control", "cache", "duplicate"])
def test_hashes_inventory_and_probe_kind_fail_before_execution(saved_probe, tmp_path, monkeypatch, corruption):
    summary = json.loads(saved_probe.read_text())
    deployment = Path(summary["cases"][0]["deployment_manifest"])
    manifest = json.loads(deployment.read_text())
    graph = manifest["graphs"][0]
    if corruption == "om64":
        (deployment.parent / graph["static_gear_oms"][1]["om"]["path"]).write_bytes(b"changed")
    elif corruption == "gear":
        graph["static_gear_oms"][1]["atc_command"][2] = "--input_shape=x:16,256"
        deployment.write_text(json.dumps(manifest))
    elif corruption == "cache":
        Path(graph["metadata"]["draft_weight_prepack_manifest"]).write_text(
            Path(graph["metadata"]["draft_weight_prepack_manifest"]).read_text() + " ")
    elif corruption == "duplicate":
        summary["cases"].append(copy.deepcopy(summary["cases"][0]))
        saved_probe.write_text(json.dumps(summary))
    else:
        summary["cases"][0]["control"] = "runtime-dynamic"
        saved_probe.write_text(json.dumps(summary))
    monkeypatch.setattr(validator, "AclOmRuntime", lambda *a, **kw: pytest.fail("must reject before load"))
    assert validator.run(options(tmp_path, saved_probe)) == 1


def test_exact_oracle_matches_integer_dot_products_on_both_gears(saved_probe):
    _, jobs = validator.prepare(saved_probe)
    units = validator.weight_units(jobs[0])
    for rows in (16, 64):
        for (_, x, expected), (_, x_units) in zip(validator.vectors(units, rows), validator.patterns(rows, 256)):
            integer = x_units.to(torch.int64) @ units.to(torch.int64).t()
            reference = (integer.double() / 4096).half().numpy()
            assert np.array_equal(expected, reference)
            assert np.array_equal(x, (x_units / 128).half().numpy())


@pytest.mark.parametrize("rows", [16, 64])
def test_acl_runtime_loads_selected_om_and_reports_its_hash(saved_probe, monkeypatch, rows):
    _, jobs = validator.prepare(saved_probe)
    job = jobs[0]
    loaded, closed = [], []
    class Model:
        def __init__(self, acl, name, path, **kwargs):
            loaded.append(path)
        def close(self):
            closed.append(True)
    monkeypatch.setattr(acl_runtime, "_AclModel", Model)
    acl = SimpleNamespace(init=lambda: 0, finalize=lambda: 0,
                          rt=SimpleNamespace(set_device=lambda n: 0, reset_device=lambda n: 0))
    with acl_runtime.AclOmRuntime(job["deployment"], acl_module=acl, static_gear_rows=rows) as runtime:
        selected = job["graph"]["static_gear_oms"][int(rows == 64)]["om"]
        assert loaded == [job["deployment"].parent / selected["path"]]
        assert runtime.artifact_hashes() == {validator.GRAPH: selected["sha256"]}
    assert closed == [True]


def test_rejects_missing_pyacl_instead_of_falling_back(saved_probe, tmp_path, monkeypatch):
    def missing(*args, **kwargs):
        raise ModuleNotFoundError("No module named acl")
    monkeypatch.setattr(validator, "AclOmRuntime", missing)
    assert validator.run(options(tmp_path, saved_probe)) == 1
    report = json.loads((tmp_path / "validation/summary.json").read_text())
    assert report["status"] == "FAIL" and report["cpu_fallback"] is False
    assert all(c["execution_status"] == "NOT_RUN" for c in report["cases"])


def test_existing_output_and_bad_cli_counts_preserve_evidence(saved_probe, tmp_path):
    args = options(tmp_path, saved_probe)
    args.output_dir.mkdir()
    with pytest.raises(FileExistsError):
        validator.run(args)
    args.output_dir = tmp_path / "new"
    args.repetitions = 0
    with pytest.raises(ValueError, match="repetitions"):
        validator.run(args)
    assert not args.output_dir.exists()


def test_plain_static_om_remains_compatible_without_gear_selection(saved_probe, monkeypatch):
    _, jobs = validator.prepare(saved_probe)
    path = jobs[0]["deployment"]
    manifest = json.loads(path.read_text())
    graph = manifest["graphs"][0]
    del graph["static_gear_oms"], graph["metadata"]["draft_compile_policy"]
    path.write_text(json.dumps(manifest))
    loaded = []
    class Model:
        def __init__(self, acl, name, path, **kwargs):
            loaded.append(path)
        def close(self):
            pass
    monkeypatch.setattr(acl_runtime, "_AclModel", Model)
    acl = SimpleNamespace(init=lambda: 0, finalize=lambda: 0,
                          rt=SimpleNamespace(set_device=lambda n: 0, reset_device=lambda n: 0))
    with acl_runtime.AclOmRuntime(path, acl_module=acl) as runtime:
        assert runtime.artifact_hashes() == {validator.GRAPH: graph["om"]["sha256"]}
    assert loaded == [path.parent / graph["om"]["path"]]
