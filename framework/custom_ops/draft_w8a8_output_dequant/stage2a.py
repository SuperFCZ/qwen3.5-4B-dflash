#!/usr/bin/env python3
"""Stage 2A only: read-only CANN evidence index and *hypothetical* CPU reference.

Neither this script nor its tests establish Ascend 310P3 SetQuantVector support.
No kernel, OPP installation, device calls or performance timing are performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SHAPE = (16, 128, 64)
# Targeted files ONLY. Missing locations are unresolved, not evidence of no support.
SDK_FILES = (
    "include/adv_api/matmul/matmul_config.h",
    "impl/adv_api/detail/matmul/matmul_impl.h",
    "impl/adv_api/detail/matmul/matmul_impl_base.h",
    "impl/adv_api/detail/matmul/feature_trait/matmul_chip_cap.h",
    "impl/adv_api/detail/matmul/utils/matmul_utils.h",
    "impl/adv_api/detail/matmul/stage/copy_cube_out/copy_cube_out_datacopy.h",
    "impl/adv_api/detail/matmul/stage/copy_cube_out/copy_cube_out_datacopy_wrapper.h",
    "impl/adv_api/detail/matmul/stage/copy_cube_out/quant/quant_processor_intf.h",
    "impl/adv_api/detail/matmul/stage/copy_cube_out/quant/quant_processor_datacopy.h",
    "impl/adv_api/detail/matmul/stage/copy_cube_out/quant/quant_processor_using_ub.h",
    "impl/adv_api/detail/matmul/stage/copy_cube_out/quant/quant_processor_utils.h",
    "include/adv_api/matmul/matmul_tiling.h",
)
TOKENS = (
    "SetQuantVector", "SetDequantType", "IsNeedUB", "IsQuantSenario",
    "MatmulQuantProcessor", "SetCType", "IsPerChannelSenario", "CopyQuantTensor",
    "CopyLocal2GMNZ2NDOnTheFly", "DEQF16", "VDEQF16", "NoQuant",
    "__NPU_ARCH__", "2002", "uint64_t", "0xFFFFE000", "PIPE_V",
)
CLAIMS = (
    "310P3/dav_2002 output specialization actually selected",
    "INT8/INT8/half Matmul specialization supported",
    "Host SetCType(DT_FLOAT16) accepted by actual 310P3 tiler",
    "SetDequantType(TENSOR) and SetQuantVector(GM uint64[64]) consumed",
    "per-N-channel mapping and actual uint64 coefficient encoding",
    "rounding, overflow, subnormals and NaN/Inf policy",
    "UB/DataCopy workspace, lifetimes and alignment",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sdk_index(cann_root: Path, extra: tuple[str, ...] = ()) -> dict:
    root = cann_root.resolve(strict=True)
    asc = (root / "aarch64-linux" / "asc").resolve(strict=True)
    paths = tuple(dict.fromkeys((*SDK_FILES, *extra)))
    files = []
    for relative in paths:
        p = Path(relative)
        if not relative or p.is_absolute() or ".." in p.parts:
            raise ValueError("SDK relative paths only; no traversal")
        source = asc / p
        entry: dict = {"relative":relative, "state":"MISSING_UNRESOLVED"}
        if source.is_file():
            real = source.resolve(strict=True)
            if real != asc and asc not in real.parents:
                raise ValueError("SDK symlink escapes selected root")
            if real.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("SDK source >16 MiB")
            data = real.read_bytes()
            lines = data.decode("utf-8", errors="replace").splitlines()
            hits = {term:[i for i,line in enumerate(lines, 1) if term in line][:24]
                    for term in TOKENS}
            entry = {"relative":relative, "state":"INDEXED_NOT_REVIEWED",
                     "bytes":len(data), "sha256":digest(data),
                     "hits":{k:v for k,v in hits.items() if v}}
        files.append(entry)
    versions = []
    for relative in ("version.info", "version.cfg", "aarch64-linux/asc/version.info"):
        p = root / relative
        if p.is_file() and p.resolve().is_relative_to(root):
            data = p.read_bytes()
            versions.append({"relative":relative,"sha256":digest(data),
                             "content":data.decode("utf-8",errors="replace")[:256]})
    return {"stage":"2A", "status":"UNRESOLVED_SOURCE_REVIEW_REQUIRED",
            "chip_claim":"Ascend310P3 / __NPU_ARCH__ == 2002 (must verify locally)",
            "cann_claim":"9.0.0 (verify version info)",
            "scope":"read-only fixed SDK files; hit lines are locators, never support proofs",
            "source_root":str(asc),"version_candidates":versions,
            "files":files,"claims":[{"name":x,"status":"UNRESOLVED"} for x in CLAIMS],
            "cann_compile":"NOT_RUN","npu":"NOT_RUN","performance":"NOT_RUN"}


def dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != (16,128) or b.shape != (128,64) or a.dtype != np.int8 or b.dtype != np.int8:
        raise ValueError("requires signed INT8 ND A[16,128], B[128,64]")
    return a.astype(np.int64) @ b.astype(np.int64)


def encode_doc_candidate(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PUBLIC CANN 9.0 DOC HYPOTHESIS ONLY: lower 13 bits cleared + bit 46.

    The 310P3/CANN9.0 implementation may use another codec; never auto-deploy.
    """
    s = np.asarray(scales, dtype="<f4")
    if s.shape != (64,) or not np.isfinite(s).all() or np.any(s < 0):
        raise ValueError("requires 64 finite nonnegative float32 scales")
    bits = s.view("<u4").copy()
    truncated = bits & np.uint32(0xFFFFE000)
    words = truncated.astype("<u8") | np.uint64(1 << 46)
    return words, truncated.view("<f4").copy()


def convert(acc: np.ndarray, scales: np.ndarray) -> np.ndarray:
    if acc.shape != (16,64) or acc.dtype != np.int64:
        raise ValueError("requires exact INT64 accumulator [16,64]")
    if np.any(np.abs(acc) > 2097152):
        raise ValueError("outside K128 signed INT8 bound")
    s = np.asarray(scales, dtype="<f4")
    if s.shape != (64,):
        raise ValueError("requires one scale per output column")
    with np.errstate(over="ignore",under="ignore",invalid="ignore"):
        return (acc.astype("<f4") * s[None,:]).astype("<f2")


def write_fresh(path: Path, value: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(value)


def make_cases(out: Path) -> dict:
    """CPU-only fixtures: not a device reference until local codec is reviewed."""
    out.mkdir(parents=True,exist_ok=False)
    i = np.arange(2048)
    a = ((i*17)%127-63).astype(np.int8).reshape(16,128)
    b = ((np.arange(8192)*71)%125-62).astype(np.int8).reshape(128,64)
    ss = np.ones(64,dtype="<f4")
    specs = {
        "scale_one":ss,
        "scale_half":np.full(64,0.5,dtype="<f4"),
        "per_channel_powers":np.array([2.0**(-(j%8)) for j in range(64)],dtype="<f4"),
        "zeros_and_ones":np.array([0.0 if j%2 else 1.0 for j in range(64)],dtype="<f4"),
        "mantissa_probe":np.full(64,1.0009,dtype="<f4"),
    }
    acc = dot(a,b)
    manifest = {"status":"CPU_MODEL_ONLY_NOT_NPU_VERIFIED", "shape":list(SHAPE),
                "encoding":"CANN9 public-doc candidate; local 310P3 codec unresolved",
                "rounding":"candidate fp32 multiplication then fp16 RNE, not NPU claim", "cases":[]}
    for name,scale in specs.items():
        folder = out/name;folder.mkdir()
        words,effective = encode_doc_candidate(scale)
        items = {"a_i8.bin":a,"b_i8.bin":b,"z_i32.bin":acc.astype("<i4"),
                 "scale_f32.bin":scale,"quant_u64_doc_candidate.bin":words,
                 "scale_effective_f32.bin":effective,
                 "ideal_fp16.bin":convert(acc,scale),
                 "doc_candidate_fp16.bin":convert(acc,effective)}
        for fn,array in items.items():
            write_fresh(folder/fn,array.tobytes())
        manifest["cases"].append({"name":name,"files":{
            fn:{"bytes":arr.nbytes,"sha256":digest(arr.tobytes())} for fn,arr in items.items()}})
    write_fresh(out/"manifest.json",(json.dumps(manifest,indent=2)+"\n").encode())
    return manifest


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("sdk-index","cpu-cases"))
    parser.add_argument("--cann-root",type=Path,default=Path("/usr/local/Ascend/cann-9.0.0"))
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--sdk-file",action="append",default=[],help="additional exact asc-relative SDK file")
    args=parser.parse_args()
    if args.command=="sdk-index":
        result=sdk_index(args.cann_root,tuple(args.sdk_file))
        args.out.parent.mkdir(parents=True,exist_ok=True)
        write_fresh(args.out,(json.dumps(result,ensure_ascii=False,indent=2)+"\n").encode())
        print("UNRESOLVED_SOURCE_REVIEW_REQUIRED",args.out)
        print("Indexed:",sum(x["state"]=="INDEXED_NOT_REVIEWED" for x in result["files"]),
              "missing:",sum(x["state"]=="MISSING_UNRESOLVED" for x in result["files"]))
    else:
        result=make_cases(args.out)
        print(result["status"],args.out, "cases:",len(result["cases"]))


if __name__=="__main__":
    main()
