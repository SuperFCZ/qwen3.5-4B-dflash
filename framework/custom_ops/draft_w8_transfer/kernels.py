"""Specialize the measured kernels, retaining their local B1 consumers and full fences."""

import shutil
from recipes import OPS, c, gate, down, fc, q, kv, require, header
from layout import once
from package import generate, install_project, write_preset


def loader(src, r, first, last):
    a = src.index(first)
    b = src.index(last, a)
    if first.startswith("template"):
        length = "LocalK"
        offset = "k0"
        x = "x"
        dst = "a"
    else:
        length = "K"
        offset = "0"
        x = "src"
        dst = "dst"
    body = src[a:b]
    if r.activation == "row":
        return src
    begin = body.index("{") + 1
    if r.activation == "gather":
        copy = f"""
    const DataCopyParams gather{{M, 1, K/16-1, 0}};
    for (uint32_t kg=0; kg<{length}/16; ++kg)
        DataCopy({dst}[kg*M*16], {x}[{offset}+kg*16], gather);
"""
    else:
        copy = f"\n    DataCopy({dst}, {x}[{offset}*M], M*{length});\n"
    return src[:a] + body[:begin] + copy + "}\n\n" + src[b:]


def packet_converter(src, r, elem, piece, first, next_address):
    """In a paired packet both Cast readers retire before MTE2 may refill it."""
    if r.packet == 1:
        return src
    src = once(src, first, "if ((" + piece + " % RawPieces) == 0) " + first.strip())
    src = once(
        src, f"CastCodes<{elem}>(w, raw);", f"CastCodes<{elem}>(w, raw[({piece} % RawPieces)*{elem}]);"
    )
    src = once(src, f"if ({piece} + 1 < Pieces) {{", f"if (({piece}+1)%RawPieces==0 && {piece}+1<Pieces) {{")
    return once(src, next_address, next_address.replace(", " + elem + ");", ", RawPieces*" + elem + ");"))


def full_kernel(r):
    src = (q.HERE / "op_kernel/next_n.cpp").read_text()
    src = once(src, '"q_spec.h"', '"transfer_spec.h"')
    src = once(src, "namespace q_panel", "namespace transfer_panel")
    src = once(src, q.STEM + "(", r.spec.stem + "(")
    # Use the same A1 copy as the incumbent except in the explicit row-loader trials.
    src = loader(src, r, "__aicore__ inline void CacheActivation(", "__aicore__ inline void ScaleStripe(")
    src = once(src, "g < Groups; ++g", "g < GroupsPerPart; ++g")
    src = once(
        src,
        "scales[g * Columns + stripe * 16]",
        "scales[((stripe % Parts)*GroupsPerPart+g)*Columns+(stripe/Parts)*16]",
    )
    src = once(src, "stripe + 1 < 4", "stripe + 1 < Pieces")
    src = once(src, "stripe < 4; ++stripe", "stripe < Pieces; ++stripe")
    src = once(src, "stripe < 3) Pair", "stripe + 1 < Pieces) Pair")
    src = packet_converter(
        src,
        r,
        "Stripe",
        "stripe",
        "Pair<HardEvent::MTE2_V>(inputReady);\n    CastCodes",
        "DataCopy(raw, codes[tile * Columns * K + (stripe + 1) * Stripe], Stripe);",
    )
    if r.packet == 2:
        src = once(src, "codes.Get<int8_t>(Stripe)", "codes.Get<int8_t>(CodeBytes)")
        src = once(src, "codes[tile * Columns * K], Stripe", "codes[tile * Columns * K], CodeBytes")
        src = once(src, "qgm[next * Columns * K], Stripe", "qgm[next * Columns * K], CodeBytes")
        src = once(
            src,
            "        if (next < Tiles) DataCopy(raw, qgm[next * Columns * K + Stripe], Stripe);",
            "        // The second piece is still in the prefetched raw packet.",
        )
    return src


def multi_kernel(r):
    src = (fc.HERE / "op_kernel" / (fc.STEM + ".cpp")).read_text()
    src = once(src, '"fc_spec.h"', '"transfer_spec.h"')
    src = once(src, "namespace fc_panel", "namespace transfer_panel")
    src = once(src, fc.STEM + "(", r.spec.stem + "(")
    src = loader(src, r, "template<uint32_t M, uint32_t LocalK>", "__aicore__ inline void ScalePiece(")
    src = once(
        src,
        "if (td.globalM == 16) Project<16>(x, w_tiles, s, y, td);\n    else if (td.globalM == 64) Project<64>",
        "if (td.globalM == RowsCount) Project<RowsCount>",
    )
    src = once(src, "qbuf.Get<int8_t>(Elements)", "qbuf.Get<int8_t>(CodeBytes)")
    src = once(src, "wbuf.Get<half>(2 * Elements)", "wbuf.Get<half>(W16Banks * Elements)")
    src = once(src, "weights[(piece % 2) * Elements]", "weights[(piece % W16Banks) * Elements]")
    if r.banks == 1:
        src = once(
            src,
            "                ConvertPiece(w, raw, scales, codes, start, piece, ready, consumed);",
            "                Pair<HardEvent::MTE3_V>(stored); // last UB->B1 reader of W0\n"
            "                ConvertPiece(w, raw, scales, codes, start, piece, ready, consumed);",
        )
        src = once(src, "                if (piece + 1 < Pieces) Pair<HardEvent::MTE3_V>(stored);", "")
    if r.role == "down":
        # Down already fences after every Iterate and *after each final GetTensorC*.
        # An interior next-piece DMA is consumed by the next explicit MTE2->V pair.
        src = once(
            src,
            "            PipeBarrier<PIPE_ALL>();\n        }\n    }\n    mm.End();",
            "            if (lastK) PipeBarrier<PIPE_ALL>();\n        }\n    }\n    mm.End();",
        )
    src = packet_converter(
        src,
        r,
        "Elements",
        "piece",
        "Pair<HardEvent::MTE2_V>(ready);\n    CastCodes",
        "DataCopy(raw, codes[start + (piece + 1) * Elements], Elements);",
    )
    if r.packet == 2:
        src = once(src, "DataCopy(raw, codes[start], Elements);", "DataCopy(raw, codes[start], CodeBytes);")
        src = once(
            src, "codes[next * Columns * PanelK], Elements", "codes[next * Columns * PanelK], CodeBytes"
        )
        src = once(
            src,
            "            if (hasNext) DataCopy(raw, codes[next * Columns * PanelK + Elements], Elements);",
            "            // Both halves of the first raw packet are already present.",
        )
    return src


def down_serial_pair(r):
    require(r.role == "down" and r.packet == 2 and r.banks == 1 and r.keep_a, "unexpected serial candidate")
    src = (down.HERE / "op_kernel" / (down.STEM + ".cpp")).read_text()
    src = once(src, '"down_spec.h"', '"transfer_spec.h"')
    src = once(src, "namespace down_panel", "namespace transfer_panel")
    src = once(src, "qbuf.Get<int8_t>(Elements)", "qbuf.Get<int8_t>(CodeBytes)")
    src = once(src, "DataCopy(raw, qgm[start], Elements);", "DataCopy(raw, qgm[start], CodeBytes);")
    src = once(
        src,
        "Pair<HardEvent::MTE2_V>(inputReady);",
        "if (piece%RawPieces==0) Pair<HardEvent::MTE2_V>(inputReady);",
    )
    src = once(
        src, "CastCodes<Elements>(w, raw);", "CastCodes<Elements>(w, raw[(piece%RawPieces)*Elements]);"
    )
    src = once(src, "if (piece + 1 < Pieces) {", "if ((piece+1)%RawPieces==0 && piece+1<Pieces) {")
    return once(
        src, "qgm[start + (piece + 1) * Elements], Elements", "qgm[start + (piece + 1) * Elements], CodeBytes"
    )


def device(r):
    if not r.ahead:
        return once(
            down_serial_pair(r),
            "using namespace transfer_panel;",
            "using namespace transfer_panel;\nconstexpr uint32_t M = RowsCount;",
        )
    if r.panel == r.shape[1]:
        # next_n.cpp uses M in its non-template functions.
        return once(
            full_kernel(r),
            "using namespace transfer_panel;",
            "using namespace transfer_panel;\nconstexpr uint32_t M = RowsCount;",
        )
    return multi_kernel(r)


def host(r):
    src = (kv.HERE / "d_flash_kv_panel_host.cpp").read_text()
    src = src.replace('"kv_spec.h"', '"transfer_spec.h"').replace(
        "namespace kv_panel", "namespace transfer_panel"
    )
    src = src.replace(kv.STEM, r.spec.stem).replace(kv.OP, r.spec.op)
    src = once(src, "m != 32 && m != 80", "m != RowsCount")
    return once(src, "draft_quant_contract::LaunchBlockCount(N,cores,0)", "std::min(cores,Tiles)")


def build(out, cann, env, r):
    name = r.spec.op
    stem = r.spec.stem
    rows = c.read(OPS / "draft_gate_tile_nd/DFlashGateTileNd.json")
    rows[0]["op"] = name
    spec = out / (name + ".json")
    c.write(spec, rows)
    proj = generate(out, spec, env)
    h = proj / "op_host"
    d = proj / "op_kernel"
    data = header(r)
    for folder in (h, d):
        (folder / "transfer_spec.h").write_bytes(data)
    (h / (stem + ".cpp")).write_text(host(r))
    (d / (stem + ".cpp")).write_text(device(r))
    (h / (stem + "_tiling.h")).write_text(
        (c.QUANT / "op_host" / (q.OLD_STEM + "_tiling.h")).read_text().replace(q.OLD_OP, name)
    )
    shutil.copyfile(c.QUANT / "op_host" / (q.OLD_STEM + "_contract.h"), h / (q.OLD_STEM + "_contract.h"))
    diag = OPS / "draft_a_core_scheduler_flow/op_host"
    shutil.copyfile(diag / "platform_diag.h", h / "platform_diag.h")
    (h / "scheduler_evidence.h").write_text(
        once(
            (diag / "scheduler_evidence.h").read_text(),
            r",\"tile_n\":64,\"n_tiles\":",
            r',\"tile_n\":"<<transfer_panel::Columns<<",\"n_tiles\":',
        )
    )
    helper = gate.HERE / "panel_vector.h"
    shutil.copyfile(helper, d / helper.name)
    preset = proj / "CMakePresets.json"
    write_preset(preset, preset, cann)
    return install_project(
        out,
        proj,
        cann,
        env,
        name,
        stem,
        {"transfer_spec.h": q.sha(data), helper.name: c.sha(helper)},
        "W8_TRANSFER_" + r.name,
    )
