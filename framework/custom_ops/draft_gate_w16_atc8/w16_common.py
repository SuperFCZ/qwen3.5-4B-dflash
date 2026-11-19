"""Shared identities for the independent, static-OM experiment."""

import hashlib
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
OPS = HERE.parent
GATE8 = OPS / "draft_gate_raw_atc8"
REVIEW = OPS / "draft_w8a16_overlap_review"
sys.path.insert(0, str(GATE8))
import run_gate8 as historical
from gate_common import (
    API,
    GATE,
    HEADER,
    KERNEL_SOURCE,
    LAYERS,
    RAW_HEADER,
    SHAPE,
    SHAPE_KEY,
    STEM,
    c,
    command,
    flow,
    gate_kernel,
    host,
    one,
    require,
    resume,
    stats,
    tree,
)

__all__ = [
    "API",
    "GATE",
    "HEADER",
    "KERNEL_SOURCE",
    "LAYERS",
    "RAW_HEADER",
    "SHAPE",
    "SHAPE_KEY",
    "STEM",
    "c",
    "command",
    "flow",
    "gate_kernel",
    "historical",
    "host",
    "one",
    "require",
    "resume",
    "stats",
    "tree",
    "DOUBLE",
    "FIRST",
    "GATE8",
    "GATE8_ROOT",
    "HERE",
    "OVERLAP",
    "PROFILE_ROOT",
    "REVIEW",
    "SERIAL",
    "SINGLE",
    "VARIANTS",
    "CUBE_SHA",
    "sources",
    "sdk_files",
    "replace_once",
    "render_kernel",
    "render_host",
    "patch_project",
]

SINGLE = "single-w-raw"
SERIAL = "double-w-serial"
OVERLAP = "double-w-overlap"
DOUBLE = (SERIAL, OVERLAP)
VARIANTS = (SINGLE, *DOUBLE)
FIRST = LAYERS[0]
CUBE_SHA = "745b846a7e3b31de10827caa5e493026d3984e282dd698177d2da380b826359a"
GATE8_ROOT = Path("/home/w00949577/qwen35-dflash-run/gate-raw-atc8-Np20yu")
PROFILE_ROOT = GATE8_ROOT.parent / "w8a16-deep-profile-acJHJJ"


def sources():
    # Machine dossiers are immutable Git *data* bound at the static stage.
    # Adding a dossier later must not invalidate the already-built executable.
    own = [
        c.identity(p)
        for p in sorted(HERE.rglob("*"))
        if p.is_file()
        and "__pycache__" not in p.parts
        and not (p.parent == HERE / "machine_reviews" and p.suffix == ".json")
    ]
    review = [c.identity(p) for p in sorted(REVIEW.glob("*.md"))]
    return own + review + historical.sources()


def sdk_files(cann):
    review = (REVIEW / "w16-event-lifecycle-revised.md").read_text()
    rows = re.findall(r"^\| `((?:include|impl)/[^`]+)` \| [^|]+ \| `([0-9a-f]{64})` \|", review, re.M)
    require(len(rows) == len(dict(rows)) == 23, "SDK audit file inventory changed")
    base = Path(cann).resolve(strict=True)
    result = {}
    for name, digest in rows:
        item = c.identity(base / "aarch64-linux/asc" / name)
        require(base in Path(item["path"]).parents and item["sha256"] == digest, "unreviewed SDK: " + name)
        result[name] = item
    return result


def replace_once(text, original, replacement):
    require(text.count(original) == 1, "frozen source anchor changed: " + original)
    return text.replace(original, replacement)


def render_kernel(data, variant):
    if variant == "down-2d":
        require(hashlib.sha256(data).hexdigest() == KERNEL_SOURCE, "wrong Down control source")
        text = replace_once(
            data.decode(), "bool gateCopy = false)", "bool gateCopy = false, bool down2d = false)"
        )
        text = replace_once(
            text,
            "#if DFLASH_GROUP_QUANT_GATE_COPY_MODE == 1",
            (HERE / "down/dma.inc").read_text() + "\n#if DFLASH_GROUP_QUANT_GATE_COPY_MODE == 1",
        )
        condition = "false, globalM == 16 && globalK == 9728 && globalN == 2560 && tileK == 256"
        text = replace_once(text, "tileK, nBegin, 0);", "tileK, nBegin, 0, " + condition + ");")
        return replace_once(
            text, "tileK, nBegin, kBegin + tileK);", "tileK, nBegin, kBegin + tileK, " + condition + ");"
        ).encode()
    require(variant in DOUBLE and hashlib.sha256(data).hexdigest() == KERNEL_SOURCE, "wrong Raw math source")
    prefix = (HERE / "kernel/tile_schedule.h").read_text() + "\n" + (HERE / "kernel/double_w.inc").read_text()
    main = 'extern "C" __global__ __aicore__ void d_flash_group_quant_linear('
    source = replace_once(data.decode(), main, prefix + "\n" + main)
    anchor = "    GET_TILING_DATA(tilingData, tiling);"
    mode = "true" if variant == OVERLAP else "false"
    return replace_once(
        source,
        anchor,
        anchor
        + """
    if (tilingData.globalM == 16 && tilingData.globalK == 2560 && tilingData.globalN == 19456) {
        w16::Run<"""
        + mode
        + """>(x, w_nz, s, y, tilingData);
        return;
    }""",
    ).encode()


def render_host(data):
    frozen = OPS / "draft_a_core_scheduler_flow/op_host" / (STEM + ".cpp")
    require(data == frozen.read_bytes(), "wrong Scheduler A Host source")
    source = replace_once(
        data.decode(),
        "    const uint64_t weightBytes = kTileN * tileK * 2;",
        (HERE / "host/w16_budget.inc").read_text().rstrip(),
    )
    anchor = "    tiling.set_globalM(m);"
    source = replace_once(source, anchor, (HERE / "host/w16_admission.inc").read_text() + anchor)
    source = replace_once(
        source,
        "    if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) {",
        "    const auto sdkRc = cubeTiling.GetTiling(tiling.cubeTilingData);\n"
        "    if (doubleW ? sdkRc != 0 : sdkRc == -1) {",
    )
    return replace_once(
        source,
        "+ aStageBytes + broadcastBytes + broadcastReservedBytes;",
        "+ aStageBytes + broadcastBytes + broadcastReservedBytes + extraWeightBytes;",
    ).encode()


def patch_project(project, variant):
    for side in ("op_kernel", "op_host"):
        require(c.sha(project / side / HEADER) == RAW_HEADER, "expected accepted Raw generated header")
    require(
        (project / "op_host/scheduler_variant.h").read_text()
        == "#pragma once\n#define DFLASH_SCHEDULER_POLICY 0\n",
        "wrong scheduler",
    )
    kernel = project / "op_kernel" / (STEM + ".cpp")
    host_cpp = project / "op_host" / (STEM + ".cpp")
    kernel.write_bytes(render_kernel(kernel.read_bytes(), variant))
    if variant in DOUBLE:
        host_cpp.write_bytes(render_host(host_cpp.read_bytes()))
