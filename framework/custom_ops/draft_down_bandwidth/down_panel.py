"""Build and bind isolated C16 Down projection OM variants."""

import mmap
from pathlib import Path
import shutil
import struct
import sys

HERE = Path(__file__).resolve().parent
OPS = HERE.parent
sys.path.insert(0, str(OPS / "draft_gate_bandwidth"))
from gate_panel import make_projection_air
from gate_plan import schema
from package import c, generate, install_project, make_package, write_preset
from bench import compile_image, embedded, require
from layout import OLD_OP, OLD_STEM, once, sha
from scale_contracts import EXPANDED_SOURCE_SHA, REPORTS
from tile_gate_graph import model_proto
from fresh_inputs import synthetic_operands

M, K, N = 16, 9728, 2560
SHAPE = (M, K, N)
KEY = "layer-0-down"
TAG = "16x9728x2560"
PANEL = 4864
CONTROL = "raw-nz-brcb"
SINGLE, DOUBLE = "l1-k4864-w1", "l1-k4864-w2"
OP = "DFlashDownPanel"
STEM = "d_flash_down_panel"
GATE = OPS / "draft_gate_bandwidth"
RAW_PARENT = "39dd1427cafee3d85bd8736498e19b5f167f746399bd08912695ab0c5b9570d6"
ABI = {
    "inputs": [dict(name="x", dtype="float16", shape=[M, K])],
    "outputs": [dict(name="y", dtype="float16", shape=[M, N])],
}
EXTENTS = {"x.bin": M * K * 2, "w_nz.bin": K * N, "s_gn.bin": K // 128 * N * 2, "golden.bin": M * N * 2}
FLOPS = 2 * M * K * N
LOGICAL_BYTES = sum(EXTENTS.values())


def inputs(out, gate8_source=None):
    if gate8_source is None:
        directory = out / "inputs"
        directory.mkdir()
        for name, data in synthetic_operands(m=M, k=K, n=N, seed=310).items():
            (directory / name).write_bytes(data)
        files = {name: c.identity(directory / name) for name in EXTENTS}
        label = "SYNTHETIC_SEEDED_C16_DOWN"
        provenance = {}
    else:
        pin = c.identity(gate8_source)
        require((pin["bytes"], pin["sha256"]) == REPORTS["source"], "unexpected Gate8 Source report")
        source = c.read(c.verify(pin))
        require(source["status"] == "GATE8_SOURCE_PASS", "Gate8 source did not pass")
        flow = source["binding"]["history"]["source"]
        old = c.read(c.verify(flow))
        require(old["status"] == "FLOW_SOURCE_PASS", "wrong predecessor source")
        case = old["binding"]["cases"]["16/" + KEY]
        require(case["shape"] == list(SHAPE), "wrong frozen Down case")
        files = {n: case["inputs"][n] for n in EXTENTS if n != "golden.bin"}
        files["golden.bin"] = case["golden"]
        provenance = dict(gate8_source=pin, flow_source=flow)
        label = "FROZEN_NATIVE_C16_LAYER0_DOWN"
    require(all(row["bytes"] == EXTENTS[n] for n, row in files.items()), "Down tensor byte extent differs")
    c.verify_tree(files)
    return files, label, provenance


def header(banks):
    require(banks in (1, 2), "Down needs one or two W16 banks")
    return f"#define DFLASH_DOWN_W16_BANKS {banks}U\n".encode() + (HERE / "down_spec.h").read_bytes()


def host_source():
    # Carry the original shape/format checks, scheduler, capacity checks and joined sidecars.
    # Only this private project receives the specialization. All operands are explicitly in A1/B1.
    s = (
        (GATE / "op_host/d_flash_gate_panel.cpp")
        .read_text()
        .replace("DFlashGatePanel", OP)
        .replace("gate_panel", "down_panel")
    )
    s = once(s, '#include "panel_spec.h"', '#include "down_spec.h"')
    s = once(
        s,
        "FullL1 ? TPosition::A1 : TPosition::GM, FullL1 ? CubeFormat::NZ : CubeFormat::ND",
        "TPosition::A1, CubeFormat::NZ",
    )
    s = once(s, "FullL1 ? TPosition::B1 : TPosition::VECOUT", "TPosition::B1")
    s = once(
        s, "FullL1 ? static_cast<int>(l1Capacity-ResidentL1) : -1", "static_cast<int>(l1Capacity-ResidentL1)"
    )
    s = once(s, "mm.SetOrgShape(M, N, K, PanelK);", "mm.SetOrgShape(M, N, PanelK, PanelK);")
    s = once(
        s,
        "cube.get_stepKa() != PanelK/256 || cube.get_stepKb() != PanelK/256",
        "cube.get_stepKa() != 1 || cube.get_stepKb() != 1",
    )
    return s.replace("Gate panel SDK plan rejected", "Down panel SDK plan rejected")


def build(out, cann, env, banks):
    out.mkdir()
    rows = c.read(OPS / "draft_gate_tile_nd/DFlashGateTileNd.json")
    rows[0]["op"] = OP
    spec = out / (OP + ".json")
    c.write(spec, rows)
    project = generate(out, spec, env)
    host = project / "op_host"
    dev = project / "op_kernel"
    hdr = header(banks)
    (host / "down_spec.h").write_bytes(hdr)
    (dev / "down_spec.h").write_bytes(hdr)
    (host / (STEM + ".cpp")).write_text(host_source())
    copy = HERE / "op_kernel" / (STEM + ".cpp")
    shutil.copyfile(copy, dev / copy.name)
    vector = GATE / "panel_vector.h"
    shutil.copyfile(vector, dev / vector.name)
    (host / (STEM + "_tiling.h")).write_text(
        (c.QUANT / "op_host" / (OLD_STEM + "_tiling.h")).read_text().replace(OLD_OP, OP)
    )
    base = OLD_STEM + "_contract.h"
    shutil.copyfile(c.QUANT / "op_host" / base, host / base)
    for name in ("scheduler_evidence.h", "platform_diag.h"):
        shutil.copyfile(OPS / "draft_a_core_scheduler_flow/op_host" / name, host / name)
    preset = project / "CMakePresets.json"
    write_preset(preset, preset, cann)
    return install_project(
        out,
        project,
        cann,
        env,
        OP,
        STEM,
        {"down_spec.h": sha(hdr), vector.name: c.sha(vector)},
        "PRIVATE_DOWN_L1_PANEL",
    )


def baseline_package(out, cann, env):
    out.mkdir()
    pkg = make_package(out, None, cann, env, tiled=False)
    config = c.read(out / "gate-template/build-config.json")
    expected = dict(
        k_tile=256,
        scale_mode="broadcast",
        pipeline_mode="prefetch",
        dequant_mode="batched",
        gate_copy_mode="loop",
        gate_prefetch_mode="raw",
        core_limit=0,
    )
    require(
        all(config[k] == v for k, v in expected.items()) and pkg["source"]["sha256"] == EXPANDED_SOURCE_SHA,
        "Down control did not bind the Raw/loop/K256 configuration with Gate-only B16",
    )
    return pkg


def prepare_air(out, types, files, control):
    if not control:
        return make_projection_air(
            out, types, files, PANEL, shape=SHAPE, op_type=OP, model_name="down", graph_name="c16_down"
        )
    out.mkdir()
    codes = c.verify(files["w_nz.bin"]).read_bytes()
    gn = c.verify(files["s_gn.bin"]).read_bytes()
    msg = model_proto(types, codes, gn, projection_shape=SHAPE, name="c16_down")
    (out / "down.air").write_bytes(msg.SerializeToString())
    return dict(
        constants=[files["w_nz.bin"], files["s_gn.bin"]],
        physical_shapes=((K // 32, N // 16, 16, 32), (K // 128, N)),
    )


def plan(q, mode, control=None):
    p, s = q["platform"], q["schedule"]
    require(
        mode in (CONTROL, SINGLE, DOUBLE)
        and p["blockDim"] == 8
        and p["availableCores"] == 8
        and p["coreLimit"] == 0
        and p["workspace"] == 2097152
        and p["capacity"]["UB"] == 262144
        and s["baseline_assignments"] == [list(range(i, N // 64, 8)) for i in range(8)],
        "Down requires the actual 8-block cyclic N64 plan",
    )
    data = bytes.fromhex(s["tiling_hex"])
    require(len(data) == 224 and sha(data) == q["tiling_sha256"], "invalid parent serialization")
    cube = {f["name"]: struct.unpack_from("<I", data, 24 + f["offset"])[0] for f in schema()}
    if mode == CONTROL:
        require(q["tiling_sha256"] == RAW_PARENT, "unexpected baseline Down ATC tiling")
        pk, raw_banks, codes, scales, weight, user, resident = 256, 2, 32768, 512, 32768, 76544, 0
    else:
        pk, raw_banks, codes, scales, resident = PANEL, 1, 38912, 4864, 933888
        weight = 77824 * (2 if mode == DOUBLE else 1)
        user = codes + scales + weight + 8192
    scratch = p["capacity"]["UB"] - user
    require(struct.unpack_from("<4IQ", data) == (M, K, N, pk, scratch), "wrong Down dimensions or scratch")
    expected = dict(
        singleCoreM=M,
        singleCoreN=64,
        singleCoreK=pk,
        baseM=M,
        baseN=64,
        baseK=256,
        stepKa=1,
        stepKb=1,
        dbL0A=2,
        dbL0B=2,
        dbL0C=1,
        isBias=0,
    )
    require(
        all(cube[k] == v for k, v in expected.items())
        and 0 < cube["transLength"] <= scratch
        and min(cube["depthA1"], cube["depthB1"]) > 0,
        "invalid Down Cube plan",
    )
    resources = dict(
        codes_bytes=codes,
        scale_bytes=scales,
        weight_bytes=weight,
        raw_banks=raw_banks,
        tile_k=pk,
        user_ub=user,
        scratch=scratch,
        workspace=p["workspace"],
        l1_used=resident + 8192 * cube["depthA1"] + 32768 * cube["depthB1"],
        l0a_used=8192 * cube["dbL0A"],
        l0b_used=32768 * cube["dbL0B"],
        l0c_used=4096 * cube["dbL0C"],
    )
    require(
        s["resources"] == resources
        and all(
            resources[key] <= p["capacity"][cap]
            for key, cap in (("l1_used", "L1"), ("l0a_used", "L0A"), ("l0b_used", "L0B"), ("l0c_used", "L0C"))
        ),
        "Down resource or capacity mismatch",
    )
    if control is not None:
        common = control["query"]["platform"]
        require(
            {k: v for k, v in p.items() if k != "host_dso"}
            == {k: v for k, v in common.items() if k != "host_dso"},
            "Down A/B platform views differ",
        )
    return dict(
        cube=cube,
        cube_sha256=sha(data[24:]),
        block_dim=p["blockDim"],
        tiling_sha256=q["tiling_sha256"],
        outer_iterates_per_n64=K // pk,
        resident_l1_bytes=resident,
        resources=resources,
    )


def compile_model(out, pkg, env, atc, mode, packing, base=None):
    op = OLD_OP if mode == CONTROL else OP
    argv, om, dump, q, node, image = compile_image(out, pkg, env, atc, op, shape=SHAPE, model_name="down")
    proof = plan(q, mode, base["plans"][TAG] if base else None)
    layouts = ("FRACTAL_NZ", "ND") if mode == CONTROL else ("ND", "ND")
    offsets = []
    with om.open("rb") as fd, mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ) as view:
        for desc, layout, shape, bound in zip(
            node["input_desc"][1:], layouts, packing["physical_shapes"], packing["constants"]
        ):
            require(
                desc["layout"] == layout and tuple(map(int, desc["shape"]["dim"])) == tuple(shape),
                "Down constant shape/layout changed in the OM",
            )
            offset = view.find(c.verify(bound).read_bytes())
            require(offset >= 0, "compiled Down OM lacks the original packed bytes")
            offsets.append(offset)
    selected = embedded(om, [image])
    receipt = out / "compile.json"
    c.write(
        receipt,
        dict(
            argv=argv, query=q, packing=packing, constant_offsets=offsets, selected=selected, contract=proof
        ),
    )
    return dict(
        mode=mode,
        key=KEY,
        abi=ABI,
        scope="single Down",
        air=c.identity(out / "down.air"),
        om=c.identity(om),
        dump=c.identity(dump),
        deployment=c.identity(receipt),
        atc_kernels=selected,
        plans={TAG: dict(query=q, contract=proof)},
        opp_root=pkg["opp"]["install"],
        private_kernel=pkg["kernel"],
    )
