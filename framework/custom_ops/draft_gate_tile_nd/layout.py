"""Physical Gate/Up layout and scoped generation from the accepted single-W B16 sources."""

import hashlib
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
OP = "DFlashGateTileNd"
STEM = "d_flash_gate_tile_nd"
OLD_OP = "DFlashGroupQuantLinear"
OLD_STEM = "d_flash_group_quant_linear"
OLD_CPP = "ed11b76a2a602558e0f24bd393f33d7fe4246988326e13fd57205d22284f348b"
HEADER = OLD_STEM + "_build_config.h"
HEADER_SHA = "26017103ea631e701088f9d7e44e8645737228b4ba869b062c5726b47bb543e6"
M, K, N = 16, 2560, 19456
TILE = (N // 64, K // 256, 64, 256)
NZ = (K // 32, N // 16, 16, 32)


def require(test, message):
    if not test:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def once(text, old, new):
    require(text.count(old) == 1, "changed or ambiguous source anchor: " + old[:80])
    return text.replace(old, new, 1)


def pack(nz, n=N, k=K):
    import numpy as np

    require(
        n > 0 and k > 0 and n % 64 == 0 and k % 256 == 0 and len(nz) == n * k,
        "invalid N64/K256 weight extent",
    )
    src = np.frombuffer(nz, dtype=np.int8).reshape(k // 256, 8, n // 64, 64, 32)
    start = time.perf_counter_ns()
    data = src.transpose(2, 0, 3, 1, 4).tobytes(order="C")
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    inverse = (
        np.frombuffer(data, dtype=np.int8).reshape(n // 64, k // 256, 64, 8, 32).transpose(1, 3, 0, 2, 4)
    )
    require(np.array_equal(src, inverse), "offline NZ/tile-ND inverse differs")
    return data, elapsed_ms


def kernel_source(accepted):
    require(sha(accepted) == OLD_CPP, "expected the accepted Gate B16 device source")
    text = accepted.decode()
    start = text.index("#if DFLASH_GROUP_QUANT_GATE_COPY_MODE == 1", text.index("inline void CopyRawTile("))
    stop = text.index("    for (uint32_t group = 0;", start)
    require("qLocal[k1 * kTileN * kNzK0]" in text[start:stop], "missing NZ plane copy")
    text = text[:start] + "    CopyGateTileNd(qLocal, qGm, nBegin, kBegin);\n" + text[stop:]
    start = text.index("            // INT8 NZ tile:")
    stop = text.index("            AscendC::PipeBarrier<PIPE_V>();", start)
    require("AscendC::Cast(wLocal[k1 * kNzK0]" in text[start:stop], "missing eight plane Casts")
    text = text[:start] + "            CastGateTileNd(wLocal, qLocal);\n" + text[stop:]
    text = once(
        text,
        "__aicore__ inline void CopyRawTile(",
        (HERE / "tile_nd.inc").read_text() + "\n__aicore__ inline void CopyRawTile(",
    )
    return once(text, "void " + OLD_STEM + "(", "void " + STEM + "(").replace("w_nz", "w_tiles").encode()


def host_source(original):
    text = original.replace(OLD_OP, OP).replace(OLD_STEM + "_tiling.h", STEM + "_tiling.h")
    start = text.index("bool ValidWeightInput(")
    end = text.index("bool ReadDimensions(", start)
    require("IsNzWeightDescriptor" in text[start:end], "missing original Host NZ contract")
    text = (
        text[:start] + """bool ValidWeightInput(gert::TilingContext *context, int64_t n, int64_t k, int64_t m)
{
    return m == 16 && k == 2560 && n == 19456 &&
        ValidInput(context, 1, ge::DT_INT8, ge::FORMAT_ND, {304,10,64,256}, {304,10,64,256});
}

""" + text[end:]
    )
    text = once(
        text,
        "draft_quant_contract::IsSupportedShape(m, k, n) && s->GetDim(0) == k / kGroup",
        "m == 16 && k == 2560 && n == 19456 && s->GetDim(0) == k / kGroup",
    )
    text = once(
        text,
        "(!IsShape(w, {n, k}) && !IsShape(w, {k / 32, n / 16, 16, 32}))",
        "!IsShape(w, {304, 10, 64, 256})",
    )
    text = once(
        text,
        'Input("w_nz").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_FRACTAL_NZ})',
        'Input("w_tiles").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND})',
    )
    begin = text.index("        std::fprintf(stderr,", text.index("!ValidWeightInput(context"))
    end = text.index("        for (size_t i", begin)
    return (
        text[:begin]
        + '        std::fprintf(stderr, "DFlashGateTileNd expects FP16 x[16,2560], INT8 ND w_tiles[304,10,64,256], FP16 s[20,19456]\\n");\n'
        + text[end:]
    )
