#!/usr/bin/env python3
"""Tiny data/oracle tools. CPU preparation needs only Python's standard library.

The exact cases have dyadic products whose partial integer sums fit in FP32.
The rounding probe's CPU result is diagnostic; native WeightQuant is its gate.
Neither CPU nor eager NPU comparison establishes native OM equivalence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import struct

M, K, N, GROUP = 16, 256, 64, 128
NZ_SHAPE = (8, 4, 16, 32)
REPETITIONS = 2
ABI = "dflash-group-quant-linear-tiny-v1"


def half(value):
    return struct.unpack("<e", struct.pack("<e", value))[0]


def half_bytes(values):
    return struct.pack(f"<{len(values)}e", *values)


def read_half(raw, count):
    if len(raw) != count * 2:
        raise ValueError(f"expected {count * 2} FP16 bytes, got {len(raw)}")
    return struct.unpack(f"<{count}e", raw)


def nz_offset(n, k):
    return ((k // 32 * (N // 16) + n // 16) * 16 + n % 16) * 32 + k % 32


def pack_nz(q):
    if len(q) != N * K or any(type(v) is not int or not -128 <= v <= 127 for v in q):
        raise ValueError("expected signed INT8 [64,256]")
    packed = bytearray(N * K)
    for n in range(N):
        for k in range(K):
            packed[nz_offset(n, k)] = q[n * K + k] & 255
    return bytes(packed)


def unpack_nz(raw):
    if len(raw) != N * K:
        raise ValueError("expected INT8 NZ [8,4,16,32]")
    signed = struct.unpack(f"<{N * K}b", raw)
    return [signed[nz_offset(n, k)] for n in range(N) for k in range(K)]


def scales_to_gn(scales_ng):
    """Transpose storage without floating-point conversions or requantization."""
    if len(scales_ng) != N * (K // GROUP) * 2:
        raise ValueError("expected FP16 [64,2] scale bytes")
    return b"".join(scales_ng[2 * (n * 2 + g):2 * (n * 2 + g + 1)]
                    for g in range(2) for n in range(N))


def validate_inputs(x_raw, q, s_raw):
    x = read_half(x_raw, M * K)
    scales = read_half(s_raw, (K // GROUP) * N)
    if any(not math.isfinite(v) for v in x):
        raise ValueError("tiny fixtures require finite X")
    if any(not math.isfinite(v) or v <= 0 for v in scales):
        raise ValueError("scales must be positive finite FP16")
    pack_nz(q)  # validate signed codes and length
    return x, scales


def cpu_reference(x_raw, q, s_raw):
    x, scales = validate_inputs(x_raw, q, s_raw)
    weight = [[half(q[n * K + k] * scales[(k // GROUP) * N + n])
               for k in range(K)] for n in range(N)]
    # Python/FP64 accumulation is exact for the bounded dyadic fixtures. For
    # arbitrary values this is only an indexing/FP16-dequantization diagnostic,
    # not a claim about the receiver's Cube reduction order or denormal mode.
    return half_bytes([sum(x[m * K + k] * weight[n][k] for k in range(K))
                       for m in range(M) for n in range(N)])


def cases():
    q = [((n * 53 + k * 29 + (n // 16) * 7 + (k // 32) * 11) % 256) - 128
         for n in range(N) for k in range(K)]
    scales_ng = half_bytes([(1 + (n * 3 + g * 2) % 4) / 32
                            for n in range(N) for g in range(2)])
    scales = scales_to_gn(scales_ng)
    zero = [-0.0 if index % 2 else 0.0 for index in range(M * K)]
    yield "signed_zero", half_bytes(zero), q, scales, True

    positions = (0, 15, 16, 31, 32, 63, 64, 95, 96, 127, 128, 129, 159, 160, 224, 255)
    basis = [0.0] * (M * K)
    for m, k in enumerate(positions):
        basis[m * K + k] = (-1 if m % 2 else 1) / 128
    yield "group_nz_boundaries", half_bytes(basis), q, scales, True

    for seed in (391, 817):
        rng = random.Random(seed)
        x = half_bytes([rng.choice((-1, 1)) / 128 for _ in range(M * K)])
        yield f"dense_signed_{seed}", x, q, scales, True

    x = half_bytes([0.0 if k < GROUP else (-1 if (m + k) % 3 == 0 else 1) / 128
                    for m in range(M) for k in range(K)])
    yield "second_group_only", x, q, scales, True

    rng = random.Random(20260928)
    x = half_bytes([rng.uniform(-1, 1) for _ in range(M * K)])
    rounded_scales = half_bytes([rng.uniform(0.002, 0.01) for _ in range(2 * N)])
    yield "rounding_probe", x, q, rounded_scales, False


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    manifest = {"abi": ABI, "m": M, "k": K, "n": N, "group_size": GROUP,
                "layout": "nz_int8_v1", "w_origin": [N, K], "w_storage": list(NZ_SHAPE),
                "scale_layout": "GN", "repetitions": REPETITIONS, "cases": []}
    for name, x, q, scales, exact in cases():
        folder = root / name
        folder.mkdir()
        nz = pack_nz(q)
        if unpack_nz(nz) != q:
            raise ValueError("NZ byte roundtrip failed")
        files = {"x.bin": x, "q_nk.bin": struct.pack(f"<{len(q)}b", *q),
                 "w_nz.bin": nz, "s_gn.bin": scales, "cpu.bin": cpu_reference(x, q, scales)}
        for filename, raw in files.items():
            (folder / filename).write_bytes(raw)
        manifest["cases"].append({"name": name, "cpu_exact": exact,
                                  "files": {filename: digest(raw) for filename, raw in files.items()}})
    write_json(root / "manifest.json", manifest)
    (root / "cases.txt").write_text("".join(c["name"] + "\n" for c in manifest["cases"]))
    print(f"Prepared {len(manifest['cases'])} tiny cases; no NPU execution performed", flush=True)


def load_cases(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("abi") != ABI or manifest.get("repetitions") != REPETITIONS or
            (manifest.get("m"), manifest.get("k"), manifest.get("n"), manifest.get("group_size")) !=
            (M, K, N, GROUP) or manifest.get("layout") != "nz_int8_v1" or
            manifest.get("w_origin") != [N, K] or manifest.get("w_storage") != list(NZ_SHAPE) or
            manifest.get("scale_layout") != "GN"):
        raise ValueError("unsupported tiny manifest")
    expected_cases = {name: exact for name, _, _, _, exact in cases()}
    if [c["name"] for c in manifest["cases"]] != list(expected_cases):
        raise ValueError("tiny case inventory differs")
    for case in manifest["cases"]:
        if case["cpu_exact"] is not expected_cases[case["name"]]:
            raise ValueError("CPU gate classification differs")
        folder = root / case["name"]
        if set(case["files"]) != {"x.bin", "q_nk.bin", "w_nz.bin", "s_gn.bin", "cpu.bin"}:
            raise ValueError("tiny input file inventory differs")
        for filename, expected in case["files"].items():
            if digest((folder / filename).read_bytes()) != expected:
                raise ValueError(f"fixture changed: {case['name']}/{filename}")
        q = list(struct.unpack(f"<{N * K}b", (folder / "q_nk.bin").read_bytes()))
        if unpack_nz((folder / "w_nz.bin").read_bytes()) != q:
            raise ValueError("NZ and logical weight differ")
        validate_inputs((folder / "x.bin").read_bytes(), q, (folder / "s_gn.bin").read_bytes())
    return manifest


def ordered_half(bits):
    # Adjacent finite FP16 numbers have adjacent ranks, with -0 and +0 adjacent.
    # The separate bit comparison makes zero sign differences fail as required.
    return 0xFFFF - bits if bits & 0x8000 else 0x8000 + bits


def compare(expected, actual):
    ev = read_half(expected, M * N)
    av = read_half(actual, M * N)
    eb = struct.unpack(f"<{M * N}H", expected)
    ab = struct.unpack(f"<{M * N}H", actual)
    mismatch = [i for i, (a, e) in enumerate(zip(ab, eb)) if a != e]
    finite = all(math.isfinite(v) for v in (*ev, *av))
    first = mismatch[0] if mismatch else None
    return {"bitwise_equal": not mismatch, "bit_mismatches": len(mismatch), "finite": finite,
            "max_abs_error": max(abs(a - e) for a, e in zip(av, ev)) if finite else None,
            "max_ulp": max(abs(ordered_half(a) - ordered_half(e))
                           for a, e in zip(ab, eb)) if finite else None,
            "first_difference": {"index": [first // N, first % N],
                                 "expected_bits": f"0x{eb[first]:04x}",
                                 "actual_bits": f"0x{ab[first]:04x}"} if first is not None else None}


def native(root, device_id):
    manifest = load_cases(root)
    report = {"status": "RUNNING", "backend": "torch_npu native WeightQuant (eager)",
              "inner_precise": 0, "group_size": GROUP, "cpu_fallback": False,
              "manifest_sha256": digest((root / "manifest.json").read_bytes()), "cases": []}
    path = root / "native-eager.json"
    try:
        import torch
        import torch_npu
        torch.npu.set_device(device_id)
        device = f"npu:{device_id}"
        report["environment"] = {"torch": str(torch.__version__), "torch_npu": str(torch_npu.__version__),
                                  "device_id": device_id, "device_name": torch.npu.get_device_name(device_id)}
        with torch.inference_mode():
            for case in manifest["cases"]:
                folder = root / case["name"]
                def tensor(filename, dtype, shape):
                    return torch.frombuffer(bytearray((folder / filename).read_bytes()),
                                            dtype=dtype).reshape(shape).to(device)
                x = tensor("x.bin", torch.float16, (M, K))
                q = tensor("q_nk.bin", torch.int8, (N, K))
                s = tensor("s_gn.bin", torch.float16, (K // GROUP, N))
                outputs = []
                for repeat in range(REPETITIONS):
                    y = torch.ops.npu.npu_weight_quant_batchmatmul.default(
                        x, q.t(), s, antiquant_group_size=GROUP, inner_precise=0)
                    torch.npu.synchronize()
                    if y.dtype != torch.float16 or tuple(y.shape) != (M, N):
                        raise ValueError("native WeightQuant output ABI differs")
                    bits = y.cpu().contiguous().view(torch.int16).flatten().tolist()
                    raw = struct.pack(f"<{len(bits)}h", *bits)
                    (folder / f"native-{repeat}.bin").write_bytes(raw)
                    outputs.append(digest(raw))
                report["cases"].append({"name": case["name"], "output_sha256": outputs,
                                        "repeat_drift": outputs[0] != outputs[1]})
        report["status"] = "PASS" if all(not c["repeat_drift"] for c in report["cases"]) else "FAIL"
    except Exception as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(path, report)


def check(root):
    manifest = load_cases(root)
    native_report = json.loads((root / "native-eager.json").read_text())
    if (native_report.get("status") != "PASS" or native_report.get("cpu_fallback") is not False or
            native_report.get("group_size") != GROUP or native_report.get("inner_precise") != 0 or
            native_report.get("manifest_sha256") != digest((root / "manifest.json").read_bytes()) or
            [c["name"] for c in native_report["cases"]] != [c["name"] for c in manifest["cases"]]):
        raise ValueError("native eager evidence missing, failed, or from different inputs")
    report = {"abi": ABI, "status": "PASS", "scope": "tiny ACLNN correctness only",
              "native_om_parity": "NOT_RUN", "full_draft_validation": "NOT_RUN", "performance": "NOT_RUN",
              "numerical_gate": "bitwise FP16; atol=0 rtol=0; nonfinite fails", "cases": []}
    for case, native_case in zip(manifest["cases"], native_report["cases"]):
        folder = root / case["name"]
        execution = json.loads((folder / "execution.json").read_text())
        required = {"status": "PASS", "runtime": "AscendCL ACLNN", "op": "DFlashGroupQuantLinear",
                    "cpu_fallback": False, "input_readonly": True, "guards_intact": True,
                    "repetitions": REPETITIONS, "device_id": native_report["environment"]["device_id"]}
        if any(execution.get(k) != v for k, v in required.items()):
            raise ValueError(f"invalid candidate execution evidence: {case['name']}")
        outputs = []
        item = {"name": case["name"], "cpu_exact": case["cpu_exact"], "comparisons": []}
        for repeat in range(REPETITIONS):
            actual = (folder / f"actual-{repeat}.bin").read_bytes()
            golden = (folder / f"native-{repeat}.bin").read_bytes()
            if digest(golden) != native_case["output_sha256"][repeat]:
                raise ValueError("native output changed after execution")
            cpu_diff = compare((folder / "cpu.bin").read_bytes(), actual)
            native_diff = compare(golden, actual)
            passed = (native_diff["bitwise_equal"] and native_diff["finite"] and
                      (not case["cpu_exact"] or (cpu_diff["bitwise_equal"] and cpu_diff["finite"])))
            item["comparisons"].append({"repeat": repeat, "status": "PASS" if passed else "FAIL",
                                         "cpu": cpu_diff, "native_eager": native_diff})
            outputs.append(digest(actual))
        item["repeat_drift"] = outputs[0] != outputs[1]
        item["status"] = "PASS" if not item["repeat_drift"] and all(
            c["status"] == "PASS" for c in item["comparisons"]) else "FAIL"
        report["cases"].append(item)
        if item["status"] != "PASS": report["status"] = "FAIL"
        print(f"{case['name']}: {item['status']} "
              f"native bits={item['comparisons'][0]['native_eager']['bit_mismatches']} "
              f"ULP={item['comparisons'][0]['native_eager']['max_ulp']}", flush=True)
    write_json(root / "comparison.json", report)
    print(f"{report['status']}: tiny only; native OM / full Draft / performance NOT_RUN")
    return report["status"] == "PASS"


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("prepare", "native", "check"))
    cli.add_argument("data_dir", type=Path)
    cli.add_argument("--device-id", type=int, default=0)
    args = cli.parse_args()
    if args.mode == "prepare": prepare(args.data_dir)
    elif args.mode == "native": native(args.data_dir, args.device_id)
    elif not check(args.data_dir): return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
