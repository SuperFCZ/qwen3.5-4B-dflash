"""Hash-bound A2 evidence and shape ABI. Importable without torch/CANN."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[4]
ABI = "dflash-group-quant-linear-a2-v1"
SHAPES = {"gate_up": (16, 2560, 19456), "down": (16, 9728, 2560)}
CASE_NAMES = tuple(f"layer-{layer}-{kind}" for layer in range(5) for kind in SHAPES)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def record(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)}


def checked_file(root, item, size=None):
    root = Path(root).resolve()
    relative = Path(item["path"])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"artifact escapes bundle or is missing: {relative}")
    if (type(item.get("bytes")) is not int or item["bytes"] != path.stat().st_size or
            (size is not None and item["bytes"] != size) or sha256(path) != item.get("sha256")):
        raise ValueError(f"artifact bytes/hash differ: {relative}")
    return path


def case_shape(case):
    shape = SHAPES.get(case.get("projection"))
    if (shape is None or type(case.get("layer")) is not int or not 0 <= case["layer"] < 5 or
            case.get("name") != f"layer-{case['layer']}-{case['projection']}" or
            [case.get(d) for d in ("m", "k", "n")] != list(shape) or
            case.get("group_size") != 128 or case.get("layout") != "nz_int8_v1"):
        raise ValueError("invalid A2 case identity/shape/layout")
    return shape


def load_bundle(path):
    path = Path(path).resolve()
    bundle = json.loads(path.read_text())
    if (bundle.get("abi") != ABI or bundle.get("status") != "PASS" or
            bundle.get("capture_runtime") != "native NPU DraftGraph replay" or
            bundle.get("cpu_fallback") is not False or bundle.get("capture_repeat_equal") is not True or
            bundle.get("checkpoint", {}).get("variant") != "w8a16" or
            bundle.get("checkpoint", {}).get("status") != "PASS" or
            tuple(c.get("name") for c in bundle.get("cases", [])) != CASE_NAMES):
        raise ValueError("A2 requires a completed real W8 replay with all five gate/up and down pairs")
    for case in bundle["cases"]:
        m, k, n = case_shape(case)
        sizes = {"x.bin": m * k * 2, "q_nk.bin": n * k, "w_nz.bin": n * k,
                 "s_gn.bin": n * (k // 128) * 2, "eager-0.bin": m * n * 2, "eager-1.bin": m * n * 2}
        if set(case["files"]) != set(sizes):
            raise ValueError("A2 file inventory differs")
        for name, size in sizes.items():
            checked_file(path.parent, case["files"][name], size)
        if case["files"]["eager-0.bin"]["sha256"] != case["files"]["eager-1.bin"]["sha256"]:
            raise ValueError("capture eager output drift")
    return bundle


def validate_layout(root, case):
    """Use array views instead of millions of Python integers for real weights."""
    import numpy as np
    m, k, n = case_shape(case)
    def array(name, dtype, shape):
        return np.memmap(checked_file(root, case["files"][name]), mode="r", dtype=dtype, shape=shape)
    q = array("q_nk.bin", "i1", (n, k))
    nz = array("w_nz.bin", "i1", (k // 32, n // 16, 16, 32))
    # Compare one K32 plane at a time, without another full-size copy.
    for plane in range(k // 32):
        if not np.array_equal(nz[plane].reshape(n, 32), q[:, plane * 32:(plane + 1) * 32]):
            raise ValueError("NZ carrier does not restore the exact checkpoint codes")
    x = array("x.bin", "<f2", (m, k))
    scale = array("s_gn.bin", "<f2", (k // 128, n))
    if not np.isfinite(x).all() or not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("nonfinite X or nonpositive/nonfinite scale")


def snapshot_contract(text):
    """Read the existing C++ frozen-input contract, with no guessed dimensions."""
    lines = text.splitlines()
    if len(lines) < 5 or lines[0] != "qwen35-draft-replay-inputs-v1":
        raise ValueError("unsupported Draft replay contract")
    index = 3 + int(lines[3].startswith("static64 "))
    specs = {}
    widths = {"float16": 2, "int16": 2, "int64": 8, "int8": 1, "float32": 4}
    for line in lines[index + 1:]:
        parts = line.split()
        if len(parts) < 4 or not re.fullmatch(r"[A-Za-z0-9_]+", parts[0]) or parts[1] not in widths:
            raise ValueError("invalid snapshot tensor descriptor")
        name, dtype = parts[:2]
        size, *shape = map(int, parts[2:])
        if name in specs or min(shape) <= 0 or math.prod(shape) * widths[dtype] != size:
            raise ValueError("snapshot tensor shape/bytes differ")
        specs[name] = {"dtype": dtype, "shape": shape, "bytes": size}
    return lines[1], specs


def replay_context_rows(valid, storage_rows, declared_rows):
    # C++ snapshots retain the maximum feature allocation, but execute the
    # selected 16/64 gear. Replaying all 64 rows for a 16-row call changes KV
    # writes and potentially the native tiler, even with masked feature rows.
    rows = 16 if valid <= 16 else 64
    if (type(valid) is not int or not 1 <= valid <= 64 or storage_rows not in (16, 64) or
            storage_rows < rows or declared_rows != rows):
        raise ValueError("frozen snapshot context gear/valid rows differ")
    return rows


def validate_native_graph(graph, case, bundle_hash):
    """Bind the exporter's actual NZ Const audit to the captured weight bytes."""
    m, k, n = case_shape(case)
    metadata = graph.get("metadata", {})
    layout = graph.get("runtime_input_abi", {}).get("weight_quant_layout", {})
    prepack = layout.get("prepack", {})
    if (graph.get("name") != "weight_quant_reference" or graph.get("input_names") != ["x"] or
            graph.get("output_names") != ["y"] or metadata.get("a2_case") != case["name"] or
            metadata.get("a2_bundle_sha256") != bundle_hash or
            metadata.get("tensor_abi", {}).get("inputs") != [{"name": "x", "dtype": "float16", "shape": [m, k]}] or
            layout.get("status") != "PASS" or layout.get("node_count") != 1 or
            prepack.get("status") != "PASS" or prepack.get("node_count") != 1 or
            layout.get("inserted_transdata") != [] or len(prepack.get("constants", [])) != 1):
        raise ValueError("native AIR is not the requested single offline-NZ WeightQuant projection")
    weight = prepack["constants"][0]
    if (weight.get("logical_sha256") != case["files"]["q_nk.bin"]["sha256"] or
            weight.get("storage_sha256") != case["files"]["w_nz.bin"]["sha256"] or
            weight.get("logical_shape") != [n, k] or weight.get("storage_shape") != [k // 32, n // 16, 16, 32] or
            weight.get("roundtrip") != "BIT_EXACT"):
        raise ValueError("exported native NZ Const differs from the captured checkpoint codes")


def validate_execution(report, shape, device, runtime, warmup, repetitions):
    if (report.get("status") != "PASS" or report.get("runtime") != runtime or
            report.get("cpu_fallback") is not False or report.get("input_readonly") is not True or
            report.get("guards_intact") is not True or report.get("repetitions") != 2 or
            [report.get(d) for d in ("m", "k", "n")] != list(shape) or report.get("device_id") != device):
        raise ValueError("execution report does not prove the requested device/shape/guards")
    if runtime == "AscendCL native OM" and report.get("io_validated") is not True:
        raise ValueError("native OM IO was not validated")
    if runtime == "AscendCL ACLNN" and (report.get("op") != "DFlashGroupQuantLinear" or
                                        report.get("tile_n") != 64 or report.get("tile_k") != 128):
        raise ValueError("custom execution contract differs")
    timing = report.get("timing", {})
    samples = timing.get("execute_sync", {}).get("samples_ms", [])
    if (timing.get("status") != "MEASURED" or timing.get("warmup") != warmup or
            timing.get("repetitions") != repetitions or len(samples) != repetitions or
            any(not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0 for v in samples)):
        raise ValueError("incomplete timing evidence")
    if runtime == "AscendCL ACLNN":
        prepare = timing.get("prepare", {}).get("samples_ms", [])
        if len(prepare) != repetitions or any(not math.isfinite(v) or v < 0 for v in prepare):
            raise ValueError("incomplete workspace-query timing evidence")
    for key in ("workspace_bytes", "tracked_device_allocation_bytes"):
        if type(report.get(key)) is not int or report[key] < 0:
            raise ValueError("missing device allocation accounting")
