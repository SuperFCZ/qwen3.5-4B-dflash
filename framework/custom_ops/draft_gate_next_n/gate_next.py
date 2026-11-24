"""Port the measured next-N issue order to the deployed full-K Gate/Up geometry."""

import mmap
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OPS = HERE.parent
sys.path.insert(0, str(OPS / "draft_all_w8"))
from selection import c, gate, q, require, OLD_OP
from bench import compile_image, embedded, SHAPE_KEY, ABI
from layout import once

CONTROL = "panel-k2560-w2"
NEXT = "panel-k2560-w2-next-n"
EARLY = "panel-k2560-w2-next-n-early-dma"
CANDIDATES = (NEXT, EARLY)
ROLE = "layer0.gate_up"
SHAPE = (16, 2560, 19456)
SCOPE = "measured Layer0 Gate/Up checked against captured raw projection"


def device_source(mode):
    require(mode in CANDIDATES, "unknown Gate schedule")
    # The two projections share M16/K2560, N64, single raw and dual W16. Keep
    # the Gate's *original* A1 gather; the Q implementation uses a different A1 copy.
    src = (q.HERE / "op_kernel/next_n.cpp").read_text()
    src = once(src, '"q_spec.h"', '"panel_spec.h"')
    src = once(src, "using namespace q_panel;", "using namespace gate_panel;")
    src = once(src, q.STEM + "(", gate.STEM + "(")
    control = (gate.HERE / "op_kernel/full_l1.cpp").read_text()
    a = control.index("constexpr uint32_t Stripe =")
    b = control.index("} // namespace", a)
    start = src.index("__aicore__ inline void CacheActivation(")
    end = src.index("__aicore__ inline void ScaleStripe(", start)
    src = src[:start] + control[a:b] + "\n" + src[end:]
    if mode == EARLY:
        # Issue next INT8/scale copies while the current last UB->B1 store can
        # still be in flight. The existing all-pipe barrier covers both transfers.
        start = src.index("        PipeBarrier<PIPE_ALL>(); // All B1 stores")
        end = src.index("        // Cube consumes", start)
        src = src[:start] + """        const uint32_t next = nt + blocks;
        if (next < Tiles) {
            Pair<HardEvent::V_MTE2>(rawConsumed); // retire the last current raw and scale reads
            DataCopy(scales, sgm[next * Groups * Columns], Groups * Columns);
            DataCopy(raw, qgm[next * Columns * K], Stripe);
        }
        PipeBarrier<PIPE_ALL>(); // current B1 and next raw/scale are ready
""" + src[end:]
    return src


def compile_single(out, pkg, env, atc, src, mode, control=None):
    require(mode in (CONTROL, *CANDIDATES) and (mode == CONTROL) == (control is None), "invalid Gate pair")
    # compile_image reads its input from the working directory. Use exactly the same AIR bytes.
    air = out / "gate.air"
    air.symlink_to(c.verify(src["single_air"]))
    argv, om, dump, query, node, desc = compile_image(
        out, pkg, env, atc, gate.OP, shape=SHAPE, model_name="gate"
    )
    full = src["deployed_query"]
    proof = gate.validate_plan(query, {"plans": {SHAPE_KEY: {"query": full}}}, 2560, w16_banks=2)
    require(
        query["schedule"] == full["schedule"] and query["tiling_sha256"] == full["tiling_sha256"],
        "Gate scheduling, complete Cube tiling or buffers differ from deployed C16",
    )
    selected = embedded(om, [desc])[node["name"]]
    require(
        (selected["executable_sections"] == src["deployed"]["executable_sections"]) == (mode == CONTROL),
        "wrong Gate machine text relative to the deployed OM: " + mode,
    )
    if control is not None:
        require(
            control["air"] == src["single_air"]
            and proof == control["contract"]
            and query["schedule"]["tiling_hex"] == control["query"]["schedule"]["tiling_hex"],
            "different AIR, Cube or resources in same-layout comparison",
        )
    offsets = []
    with om.open("rb") as fp, mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ) as mem:
        for d, dims, item in zip(
            node["input_desc"][1:], src["packing"]["physical_shapes"], src["packing"]["constants"]
        ):
            require(
                d["layout"] == "ND" and tuple(map(int, d["shape"]["dim"])) == tuple(dims),
                "wrong Gate constant layout",
            )
            offset = mem.find(c.verify(item).read_bytes())
            require(offset >= 0, "missing Gate constant in OM")
            offsets.append(offset)
    return dict(
        op=gate.OP,
        mode=mode,
        abi=ABI,
        air=src["single_air"],
        om=c.identity(om),
        dump=c.identity(dump),
        query=query,
        contract=proof,
        selected=selected,
        constant_offsets=offsets,
        argv=argv,
    )
