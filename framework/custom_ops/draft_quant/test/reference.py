#!/usr/bin/env python3
"""A1 N/K tiling data/oracle tools; default arguments preserve tiny fixtures.

The exact cases have dyadic products whose partial integer sums fit in FP32.
The rounding probe's CPU result is diagnostic; native WeightQuant is its gate.
Neither CPU nor eager NPU comparison establishes native OM equivalence.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import sys

M, K, N, GROUP = 16, 256, 64, 128
NZ_SHAPE = (8, 4, 16, 32)
REPETITIONS = 2
ABI = "dflash-group-quant-linear-tiny-v1"
A1_ABI = "dflash-group-quant-linear-a1-v1"
A4_ABI = "dflash-group-quant-linear-a4-synthetic-v1"
NATIVE_POLICY = "production-ng-backed-scale-view-v1"
REPO = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class Workload:
    m: int = M
    k: int = K
    n: int = N

    def __post_init__(self):
        if (any(type(v) is not int for v in (self.m, self.k, self.n)) or
                self.m != 16 or self.k not in (256, 512, 1024) or self.n not in (64, 128, 256)):
            raise ValueError("A1 requires M=16, K=256/512/1024, N=64/128/256")

    @property
    def groups(self):
        return self.k // GROUP

    @property
    def nz_shape(self):
        return (self.k // 32, self.n // 16, 16, 32)

    @property
    def tile_k(self):
        return 256 if self.k == 256 else 128

    @property
    def name(self):
        return f"m{self.m}-k{self.k}-n{self.n}"


TINY = Workload()
A1_WORKLOADS = tuple(Workload(k=k, n=n) for k in (256, 512, 1024) for n in (64, 128, 256))


@dataclass(frozen=True)
class RowWorkload(Workload):
    m: int = 32

    def __post_init__(self):
        if (any(type(v) is not int for v in (self.m, self.k, self.n)) or
                self.m not in (32, 64, 80) or self.k not in (256, 512, 1024) or self.n not in (64, 128, 256)):
            raise ValueError("A4 synthetic requires M=32/64/80, K=256/512/1024, N=64/128/256")


A4_WORKLOADS = A1_WORKLOADS + tuple(RowWorkload(m=m, k=k, n=n) for m in (32, 64, 80)
                                  for k in (256, 512, 1024) for n in (64, 128, 256))


def workload_abi(shape):
    return A4_ABI if shape.m != 16 else (ABI if shape == TINY else A1_ABI)


def half(value):
    return struct.unpack("<e", struct.pack("<e", value))[0]


def half_bytes(values):
    return struct.pack(f"<{len(values)}e", *values)


def read_half(raw, count):
    if len(raw) != count * 2:
        raise ValueError(f"expected {count * 2} FP16 bytes, got {len(raw)}")
    return struct.unpack(f"<{count}e", raw)


def nz_offset(n, k, shape=TINY):
    return ((k // 32 * (shape.n // 16) + n // 16) * 16 + n % 16) * 32 + k % 32


def pack_nz(q, shape=TINY):
    if len(q) != shape.n * shape.k or any(type(v) is not int or not -128 <= v <= 127 for v in q):
        raise ValueError(f"expected signed INT8 [{shape.n},{shape.k}]")
    packed = bytearray(shape.n * shape.k)
    for n in range(shape.n):
        for k in range(shape.k):
            packed[nz_offset(n, k, shape)] = q[n * shape.k + k] & 255
    return bytes(packed)


def unpack_nz(raw, shape=TINY):
    if len(raw) != shape.n * shape.k:
        raise ValueError(f"expected INT8 NZ {shape.nz_shape}")
    signed = struct.unpack(f"<{shape.n * shape.k}b", raw)
    return [signed[nz_offset(n, k, shape)] for n in range(shape.n) for k in range(shape.k)]


def scales_to_gn(scales_ng, shape=TINY):
    """Transpose storage without floating-point conversions or requantization."""
    if len(scales_ng) != shape.n * shape.groups * 2:
        raise ValueError(f"expected FP16 [{shape.n},{shape.groups}] scale bytes")
    return b"".join(scales_ng[2 * (n * shape.groups + g):2 * (n * shape.groups + g + 1)]
                    for g in range(shape.groups) for n in range(shape.n))


def scales_to_ng(scales_gn, shape=TINY):
    """Restore production [N,G] storage, preserving every original FP16 bit.

    The custom kernel consumes contiguous GN. Native WeightQuant with q.t()
    instead receives the production GN *view* of contiguous NG storage. CANN
    9.0's TensorContiguousProcess uses the weight transpose flag for scale too.
    """
    if len(scales_gn) != shape.n * shape.groups * 2:
        raise ValueError(f"expected FP16 [{shape.groups},{shape.n}] scale bytes")
    return b"".join(scales_gn[2 * (g * shape.n + n):2 * (g * shape.n + n + 1)]
                    for n in range(shape.n) for g in range(shape.groups))


def validate_inputs(x_raw, q, s_raw, shape=TINY):
    x = read_half(x_raw, shape.m * shape.k)
    scales = read_half(s_raw, shape.groups * shape.n)
    if any(not math.isfinite(v) for v in x):
        raise ValueError("A1 fixtures require finite X")
    if any(not math.isfinite(v) or v <= 0 for v in scales):
        raise ValueError("scales must be positive finite FP16")
    if len(q) != shape.n * shape.k or any(type(v) is not int or not -128 <= v <= 127 for v in q):
        raise ValueError("invalid signed INT8 codes")
    return x, scales


def cpu_reference(x_raw, q, s_raw, shape=TINY):
    x, scales = validate_inputs(x_raw, q, s_raw, shape)
    weight = [[half(q[n * shape.k + k] * scales[(k // GROUP) * shape.n + n])
               for k in range(shape.k)] for n in range(shape.n)]
    # Python/FP64 accumulation is exact for the bounded dyadic fixtures. For
    # arbitrary values this is only an indexing/FP16-dequantization diagnostic,
    # not a claim about the receiver's Cube reduction order or denormal mode.
    # Omitting zero products only accelerates sparse CPU fixture preparation;
    # the independent exact-integer tests verify all resulting FP16 bits.
    rows = [[(k, x[m * shape.k + k]) for k in range(shape.k) if x[m * shape.k + k] != 0]
            for m in range(shape.m)]
    return half_bytes([sum(value * weight[n][k] for k, value in row)
                       for row in rows for n in range(shape.n)])


def cases(shape=TINY):
    q = [((n * 53 + k * 29 + (n // 16) * 7 + (k // 32) * 11) % 256) - 128
         for n in range(shape.n) for k in range(shape.k)]
    scales_ng = half_bytes([(1 + (n * 3 + g * 2) % 4) / 32 if shape == TINY else
                           (1 + (n * 3 + g * 5 + (n // 64) * 7) % 13) / 64
                           for n in range(shape.n) for g in range(shape.groups)])
    scales = scales_to_gn(scales_ng, shape)
    zero = [-0.0 if index % 2 else 0.0 for index in range(shape.m * shape.k)]
    yield "signed_zero", half_bytes(zero), q, scales, True

    positions = (0, 15, 16, 31, 32, 63, 64, 95, 96, 127, 128, 129, 159, 160, 224, 255)
    basis = [0.0] * (shape.m * shape.k)
    for m, k in enumerate(positions):
        basis[m * shape.k + k] = (-1 if m % 2 else 1) / 128
    yield "group_nz_boundaries", half_bytes(basis), q, scales, True
    if shape.m != 16:
        # Every row is nonzero, including the second/fifth M16 block and the
        # last row. Distinguish row stride, stale tails and an accidental M16 cap.
        rows = [0.0] * (shape.m * shape.k)
        for row in range(shape.m):
            rows[row * shape.k + (row * 31 + row // 16 * 128) % shape.k] = (1 + row % 5) / 128
        yield "m_block_boundaries", half_bytes(rows), q, scales, True

    for seed in (391, 817):
        rng = random.Random(seed)
        x = half_bytes([rng.choice((-1, 1)) / 128 for _ in range(shape.m * shape.k)])
        yield f"dense_signed_{seed}", x, q, scales, True

    x = half_bytes([(-1 if (m + k) % 3 == 0 else 1) / 128 if GROUP <= k < 2 * GROUP else 0.0
                    for m in range(shape.m) for k in range(shape.k)])
    yield "second_group_only", x, q, scales, True

    rng = random.Random(20260928)
    x = half_bytes([rng.uniform(-1, 1) for _ in range(shape.m * shape.k)])
    rounded_scales = half_bytes([rng.uniform(0.002, 0.01) for _ in range(shape.groups * shape.n)])
    yield "rounding_probe", x, q, rounded_scales, False

    if shape != TINY:
        # Exercise both sides of EVERY K-group boundary, including the last
        # group. The first 16-row basis fixture alone stops at K=255.
        positions = sorted({0, 31, 32, shape.k - 1} |
                           {g * GROUP + delta for g in range(1, shape.groups) for delta in (-1, 0)})
        for start in range(0, len(positions), shape.m):
            basis = [0.0] * (shape.m * shape.k)
            for m, k in enumerate(positions[start:start + shape.m]):
                basis[m * shape.k + k] = (-1 if m % 2 else 1) / 128
            yield f"all_group_boundaries_{start // shape.m}", half_bytes(basis), q, scales, True
        x = half_bytes([(-1 if (m + k) % 3 == 0 else 1) / 128
                        if k >= shape.k - GROUP else 0.0
                        for m in range(shape.m) for k in range(shape.k)])
        yield "last_group_only", x, q, scales, True

        # Each first group produces >65504, then group 1 cancels it. All exact
        # partial sums in units of 1/64 are bounded by 16646151 < 2**24. FP16
        # per-group results overflow, while an FP32 accumulator retains residue.
        cancellation_q = [0] * (shape.n * shape.k)
        for n in range(shape.n):
            cancellation_q[n * shape.k] = 127
            cancellation_q[n * shape.k + 1] = n % 7 + 1
            cancellation_q[n * shape.k + GROUP] = -127
        cancellation_x = [0.0] * (shape.m * shape.k)
        for m in range(shape.m):
            sign = -1 if m % 2 else 1
            cancellation_x[m * shape.k] = sign * 1024
            cancellation_x[m * shape.k + 1] = sign / 64
            cancellation_x[m * shape.k + GROUP] = sign * 1024
        yield "fp32_accumulator_cancellation", half_bytes(cancellation_x), cancellation_q, \
            half_bytes([1.0] * (shape.groups * shape.n)), True


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare(root, shape=TINY):
    root.mkdir(parents=True, exist_ok=False)
    manifest = {"abi": workload_abi(shape),
                "m": shape.m, "k": shape.k, "n": shape.n, "group_size": GROUP,
                "layout": "nz_int8_v1", "w_origin": [shape.n, shape.k], "w_storage": list(shape.nz_shape),
                "tile": [shape.m, 64, shape.tile_k], "n_tiles": shape.n // 64,
                "k_tiles": shape.k // shape.tile_k, "accumulation": "cube-l0c-fp32",
                "scale_layout": "GN", "repetitions": REPETITIONS, "cases": []}
    for name, x, q, scales, exact in cases(shape):
        folder = root / name
        folder.mkdir()
        nz = pack_nz(q, shape)
        if unpack_nz(nz, shape) != q:
            raise ValueError("NZ byte roundtrip failed")
        files = {"x.bin": x, "q_nk.bin": struct.pack(f"<{len(q)}b", *q),
                 "w_nz.bin": nz, "s_gn.bin": scales, "cpu.bin": cpu_reference(x, q, scales, shape)}
        for filename, raw in files.items():
            (folder / filename).write_bytes(raw)
        manifest["cases"].append({"name": name, "cpu_exact": exact,
                                  "files": {filename: digest(raw) for filename, raw in files.items()}})
    write_json(root / "manifest.json", manifest)
    (root / "cases.txt").write_text("".join(c["name"] + "\n" for c in manifest["cases"]))
    print(f"Prepared {shape.name}: {len(manifest['cases'])} cases; no NPU execution performed", flush=True)


def workload_from_manifest(manifest):
    cls = RowWorkload if manifest.get("abi") == A4_ABI else Workload
    shape = cls(manifest.get("m"), manifest.get("k"), manifest.get("n"))
    if manifest.get("abi") != workload_abi(shape):
        raise ValueError("unsupported workload ABI")
    if shape != TINY and (manifest.get("tile") != [shape.m, 64, shape.tile_k] or
                          manifest.get("n_tiles") != shape.n // 64 or
                          manifest.get("k_tiles") != shape.k // shape.tile_k or
                          manifest.get("accumulation") != "cube-l0c-fp32"):
        raise ValueError("A1 tile/accumulation contract differs")
    return shape


def load_cases(root):
    manifest = json.loads((root / "manifest.json").read_text())
    shape = workload_from_manifest(manifest)
    if (manifest.get("repetitions") != REPETITIONS or manifest.get("group_size") != GROUP or
            manifest.get("layout") != "nz_int8_v1" or manifest.get("w_origin") != [shape.n, shape.k] or
            manifest.get("w_storage") != list(shape.nz_shape) or
            manifest.get("scale_layout") != "GN"):
        raise ValueError("unsupported workload manifest")
    expected_cases = {name: exact for name, _, _, _, exact in cases(shape)}
    if [c["name"] for c in manifest["cases"]] != list(expected_cases):
        raise ValueError("workload case inventory differs")
    for case in manifest["cases"]:
        if case["cpu_exact"] is not expected_cases[case["name"]]:
            raise ValueError("CPU gate classification differs")
        folder = root / case["name"]
        if set(case["files"]) != {"x.bin", "q_nk.bin", "w_nz.bin", "s_gn.bin", "cpu.bin"}:
            raise ValueError("workload input file inventory differs")
        for filename, expected in case["files"].items():
            if digest((folder / filename).read_bytes()) != expected:
                raise ValueError(f"fixture changed: {case['name']}/{filename}")
        q = list(struct.unpack(f"<{shape.n * shape.k}b", (folder / "q_nk.bin").read_bytes()))
        if unpack_nz((folder / "w_nz.bin").read_bytes(), shape) != q:
            raise ValueError("NZ and logical weight differ")
        validate_inputs((folder / "x.bin").read_bytes(), q, (folder / "s_gn.bin").read_bytes(), shape)
    return manifest


def ordered_half(bits):
    # Adjacent finite FP16 numbers have adjacent ranks, with -0 and +0 adjacent.
    # The separate bit comparison makes zero sign differences fail as required.
    return 0xFFFF - bits if bits & 0x8000 else 0x8000 + bits


def compare(expected, actual, shape=TINY):
    ev = read_half(expected, shape.m * shape.n)
    av = read_half(actual, shape.m * shape.n)
    eb = struct.unpack(f"<{shape.m * shape.n}H", expected)
    ab = struct.unpack(f"<{shape.m * shape.n}H", actual)
    mismatch = [i for i, (a, e) in enumerate(zip(ab, eb)) if a != e]
    finite = all(math.isfinite(v) for v in (*ev, *av))
    first = mismatch[0] if mismatch else None
    return {"bitwise_equal": not mismatch, "bit_mismatches": len(mismatch), "finite": finite,
            "max_abs_error": max(abs(a - e) for a, e in zip(av, ev)) if finite else None,
            "max_ulp": max(abs(ordered_half(a) - ordered_half(e))
                           for a, e in zip(ab, eb)) if finite else None,
            "first_difference": {"index": [first // shape.n, first % shape.n],
                                 "expected_bits": f"0x{eb[first]:04x}",
                                 "actual_bits": f"0x{ab[first]:04x}"} if first is not None else None}


def cpu_gate(difference, exact):
    if not difference["finite"]:
        return "FAIL"
    if not exact:
        return "DIAGNOSTIC_ONLY"
    return "PASS" if difference["bitwise_equal"] else "FAIL"


def native(root, device_id):
    manifest = load_cases(root)
    shape = workload_from_manifest(manifest)
    report = {"status": "RUNNING", "backend": "torch_npu native WeightQuant (eager)",
              "inner_precise": 0, "group_size": GROUP, "cpu_fallback": False,
              "reference_policy": NATIVE_POLICY,
              "entrypoint": "models.dflash_v1.weight_quant_matmul.weight_quant_linear",
              "mkn": [shape.m, shape.k, shape.n],
              "scale_storage_layout": "NG", "scale_view_shape": [shape.groups, shape.n],
              "scale_view_stride": [1, shape.groups],
              "manifest_sha256": digest((root / "manifest.json").read_bytes()), "cases": []}
    path = root / "native-eager.json"
    try:
        import torch
        import torch_npu
        # Reuse exactly the same weight/scale view construction as the model.
        # This changes only the standalone native oracle, never the model path.
        sys.path.insert(0, str(REPO))
        from models.dflash_v1.weight_quant_matmul import weight_quant_linear
        report["entrypoint_sha256"] = digest(
            (REPO / "models/dflash_v1/weight_quant_matmul.py").read_bytes())
        torch.npu.set_device(device_id)
        device = f"npu:{device_id}"
        report["environment"] = {"torch": str(torch.__version__), "torch_npu": str(torch_npu.__version__),
                                  "device_id": device_id, "device_name": torch.npu.get_device_name(device_id)}
        with torch.inference_mode():
            for case in manifest["cases"]:
                folder = root / case["name"]
                def tensor(filename, dtype, dims):
                    return torch.frombuffer(bytearray((folder / filename).read_bytes()),
                                            dtype=dtype).reshape(dims).to(device)
                x = tensor("x.bin", torch.float16, (shape.m, shape.k))
                q = tensor("q_nk.bin", torch.int8, (shape.n, shape.k))
                # Reorder bytes on CPU, then transfer contiguous [N,G]. The
                # production helper passes scales_ng.t(): logical GN, stride
                # [1,G], backed by NG. Do NOT make that transposed view contiguous.
                ng_raw = scales_to_ng((folder / "s_gn.bin").read_bytes(), shape)
                scales_ng = torch.frombuffer(bytearray(ng_raw), dtype=torch.float16).reshape(
                    shape.n, shape.groups).to(device)
                scale_view = scales_ng.t()
                if (tuple(scale_view.shape) != (shape.groups, shape.n) or
                        tuple(scale_view.stride()) != (1, shape.groups)):
                    raise ValueError("native scale must be the production GN view of NG storage")
                outputs, cpu_comparisons = [], []
                for repeat in range(REPETITIONS):
                    y = weight_quant_linear(x, q, scales_ng)
                    torch.npu.synchronize()
                    if y.dtype != torch.float16 or tuple(y.shape) != (shape.m, shape.n):
                        raise ValueError("native WeightQuant output ABI differs")
                    bits = y.cpu().contiguous().view(torch.int16).flatten().tolist()
                    raw = struct.pack(f"<{len(bits)}h", *bits)
                    (folder / f"native-{repeat}.bin").write_bytes(raw)
                    outputs.append(digest(raw))
                    difference = compare((folder / "cpu.bin").read_bytes(), raw, shape)
                    cpu_comparisons.append({"repeat": repeat, "difference": difference,
                                            "status": cpu_gate(difference, case["cpu_exact"])})
                repeat_drift = outputs[0] != outputs[1]
                valid = not repeat_drift and all(c["status"] != "FAIL" for c in cpu_comparisons)
                report["cases"].append({"name": case["name"], "output_sha256": outputs,
                                        "repeat_drift": repeat_drift, "cpu_exact": case["cpu_exact"],
                                        "cpu_comparisons": cpu_comparisons,
                                        "status": "PASS" if valid else "FAIL"})
                print(f"native reference {case['name']}: {'PASS' if valid else 'FAIL'} "
                      f"native_vs_cpu bits={cpu_comparisons[0]['difference']['bit_mismatches']} "
                      f"gate={cpu_comparisons[0]['status']}", flush=True)
        report["status"] = "PASS" if all(c["status"] == "PASS" for c in report["cases"]) else "FAIL"
    except Exception as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(path, report)
    return report["status"] == "PASS"


def check(root):
    manifest = load_cases(root)
    shape = workload_from_manifest(manifest)
    native_report = json.loads((root / "native-eager.json").read_text())
    if native_report.get("reference_policy") != NATIVE_POLICY:
        raise ValueError("stale native reference: rerun native with the production NG-backed scale view")
    if (native_report.get("status") != "PASS" or native_report.get("cpu_fallback") is not False or
            native_report.get("group_size") != GROUP or native_report.get("inner_precise") != 0 or
            native_report.get("manifest_sha256") != digest((root / "manifest.json").read_bytes()) or
            [c["name"] for c in native_report["cases"]] != [c["name"] for c in manifest["cases"]]):
        raise ValueError("native reference execution/CPU self-check failed, or input evidence differs; "
                         "inspect native-eager.json before diagnosing the custom kernel")
    if shape != TINY and native_report.get("mkn") != [shape.m, shape.k, shape.n]:
        raise ValueError("native workload identity differs")
    report = {"abi": manifest["abi"], "status": "PASS", "scope": "synthetic ACLNN correctness only",
              "mkn": [shape.m, shape.k, shape.n], "tile": [shape.m, 64, shape.tile_k],
              "reference_policy": NATIVE_POLICY,
              "native_om_parity": "NOT_RUN", "full_draft_validation": "NOT_RUN", "performance": "NOT_RUN",
              "numerical_gate": "bitwise FP16; atol=0 rtol=0; nonfinite fails", "cases": []}
    for case, native_case in zip(manifest["cases"], native_report["cases"]):
        folder = root / case["name"]
        execution = json.loads((folder / "execution.json").read_text())
        required = {"status": "PASS", "runtime": "AscendCL ACLNN", "op": "DFlashGroupQuantLinear",
                    "cpu_fallback": False, "input_readonly": True, "guards_intact": True,
                    "repetitions": REPETITIONS, "device_id": native_report["environment"]["device_id"]}
        if shape != TINY:
            required.update(m=shape.m, k=shape.k, n=shape.n, tile_n=64, tile_k=shape.tile_k)
        if shape.m != 16:
            required.update(tile_m=shape.m, weight_reuse_rows=shape.m)
        if any(execution.get(k) != v for k, v in required.items()):
            raise ValueError(f"invalid candidate execution evidence: {case['name']}")
        outputs = []
        item = {"name": case["name"], "cpu_exact": case["cpu_exact"], "comparisons": []}
        for repeat in range(REPETITIONS):
            actual = (folder / f"actual-{repeat}.bin").read_bytes()
            golden = (folder / f"native-{repeat}.bin").read_bytes()
            if digest(golden) != native_case["output_sha256"][repeat]:
                raise ValueError("native output changed after execution")
            cpu_raw = (folder / "cpu.bin").read_bytes()
            cpu_diff = compare(cpu_raw, actual, shape)
            native_cpu_diff = compare(cpu_raw, golden, shape)
            native_diff = compare(golden, actual, shape)
            custom_cpu_gate = cpu_gate(cpu_diff, case["cpu_exact"])
            native_cpu_gate = cpu_gate(native_cpu_diff, case["cpu_exact"])
            passed = (native_diff["bitwise_equal"] and native_diff["finite"] and
                      custom_cpu_gate != "FAIL" and native_cpu_gate != "FAIL")
            failures = []
            if custom_cpu_gate == "FAIL": failures.append("custom_vs_cpu")
            if native_cpu_gate == "FAIL": failures.append("native_reference_vs_cpu")
            if not native_diff["finite"] or not native_diff["bitwise_equal"]:
                failures.append("custom_vs_native")
            item["comparisons"].append({"repeat": repeat, "status": "PASS" if passed else "FAIL",
                                         "cpu": cpu_diff, "native_eager": native_diff,
                                         "native_vs_cpu": native_cpu_diff,
                                         "custom_cpu_gate": custom_cpu_gate,
                                         "native_cpu_gate": native_cpu_gate, "failed_checks": failures})
            outputs.append(digest(actual))
        item["repeat_drift"] = outputs[0] != outputs[1]
        item["status"] = "PASS" if not item["repeat_drift"] and all(
            c["status"] == "PASS" for c in item["comparisons"]) else "FAIL"
        report["cases"].append(item)
        if item["status"] != "PASS": report["status"] = "FAIL"
        first = item["comparisons"][0]
        print(f"{case['name']}: {item['status']} "
              f"custom_vs_cpu bits={first['cpu']['bit_mismatches']} "
              f"native_vs_cpu bits={first['native_vs_cpu']['bit_mismatches']} "
              f"custom_vs_native bits={first['native_eager']['bit_mismatches']} "
              f"ULP={first['native_eager']['max_ulp']}", flush=True)
    write_json(root / "comparison.json", report)
    print(f"{report['status']}: {shape.name}; native OM / full Draft / performance NOT_RUN")
    return report["status"] == "PASS"


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("prepare", "native", "check"))
    cli.add_argument("data_dir", type=Path)
    cli.add_argument("--device-id", type=int, default=0)
    cli.add_argument("--k", type=int, choices=(256, 512, 1024), default=256,
                     help="prepare only; native/check read dimensions from the manifest")
    cli.add_argument("--n", type=int, choices=(64, 128, 256), default=64,
                     help="prepare only; native/check read dimensions from the manifest")
    args = cli.parse_args()
    if args.mode != "prepare" and (args.k != 256 or args.n != 64):
        cli.error("native/check infer dimensions from the manifest; omit --k/--n")
    if args.mode == "prepare": prepare(args.data_dir, Workload(k=args.k, n=args.n))
    elif args.mode == "native":
        if not native(args.data_dir, args.device_id): return 1
    elif not check(args.data_dir): return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
