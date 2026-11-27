"""Package the new Cube operator and bind its external weight panels to GE."""

import os
import shutil

import numpy as np

from pair_graph import HERE, OP, K, N, c, require, tensor_node
from pair_reference import matrix_view
from package import generate, write_preset, build_install


VARIANTS = {
    "nd128": (128, False, False),
    "pack128": (128, True, False),
    "pack256": (256, True, False),
    "pack128-prefetch": (128, True, True),
    "pack128-vector": (128, True, False),
}


def build_operator(folder, cann, env):
    project = generate(folder, HERE / (OP + ".json"), env)
    sources = {}
    for name in ("op_host", "op_kernel"):
        for f in sorted((HERE / name).iterdir()):
            shutil.copyfile(f, project / name / f.name)
            sources[f.name] = c.sha(f)
    preset = project / "CMakePresets.json"
    write_preset(preset, preset, cann)
    install = build_install(folder, project, cann, env)
    root = install / "vendors/customize"
    sdk = root / "op_impl/ai_core/tbe"
    objects = sorted((sdk / "kernel/ascend310p/target_gate_up_panel").glob("*.o"))
    require(
        objects and all(c.read(p.with_suffix(".json"))["sha256"] == c.sha(p) for p in objects),
        "new 310P kernel not present in isolated package",
    )
    (impl,) = sdk.glob("*/dynamic/target_gate_up_panel.cpp")
    require(
        c.sha(impl) == c.sha(HERE / "op_kernel/target_gate_up_panel.cpp"),
        "different installed custom implementation",
    )
    return dict(
        vendor=str(root),
        binaries=[c.identity(p) for p in objects],
        implementation=c.identity(impl),
        sources=sources,
    )


def use_operator(base, package):
    c.verify(package["implementation"])
    vendor = package["vendor"]
    paths = base.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
    return dict(base, ASCEND_CUSTOM_OPP_PATH=os.pathsep.join(dict.fromkeys([vendor, *filter(None, paths)])))


def pack_matrix(w):
    k, n = w.shape
    return w.reshape(k // 16, 16, n // 32, 32).transpose(2, 0, 1, 3).copy()


def pack(folder, weights, layouts):
    dest = folder / "gate-up.i8"
    with dest.open("xb") as f:
        for branch in ("gate", "up"):
            raw = np.fromfile(c.verify(weights[branch]["weight"]), np.int8)
            w = matrix_view(raw, layouts[branch])
            packed = pack_matrix(w)
            require(np.array_equal(matrix_view(packed.ravel(), "nz-kn"), w), "nonreversible weight pack")
            packed.tofile(f)
    require(dest.stat().st_size == 2 * K * N, "wrong packed byte count")
    return c.identity(dest)


def custom_nodes(cls, branch, rows, packed_file, variant, prefix="pair"):
    width, packed_a, prefetch = VARIANTS[variant]
    nodes = []

    def const(stem, value):
        a = np.array(value, dtype="<i4")
        n = tensor_node(cls, prefix + "_" + stem, "Const", [], [("y", 7, a.shape)])
        n.attr["value"].t.desc.CopyFrom(n.output_desc[0])
        n.attr["value"].t.data = a.tobytes()
        nodes.append(n)
        return n.name + ":0"

    wshape = [2 * N // width, width // 32, K // 16, 16, 32]
    w = tensor_node(cls, prefix + "_panels", "FileConstant", [], [("y", 3, wshape)])
    w.attr["shape"].list.i[:] = wshape
    w.attr["shape"].list.val_type = 2
    w.attr["dtype"].dt = 2
    w.attr["file_path"].s = os.fsencode(packed_file)
    nodes.append(w)
    gate, up = branch["gate"], branch["up"]
    x = gate["dq"] + ":0"
    xshape = [1, rows, K]
    if packed_a:
        dims = const("dims", [rows, K // 32, 32])
        shape = [rows, K // 32, 32]
        view = tensor_node(
            cls,
            prefix + "_rows",
            "Reshape",
            [(x, "x", 3, xshape), (dims, "shape", 7, [3])],
            [("y", 3, shape)],
        )
        perm = const("permutation", [1, 0, 2])
        xshape = [K // 32, rows, 32]
        trans = tensor_node(
            cls,
            prefix + "_pack_a",
            "Transpose",
            [(view.name + ":0", "x", 3, shape), (perm, "perm", 7, [3])],
            [("y", 3, xshape)],
        )
        nodes.extend([view, trans])
        x = trans.name + ":0"
    yshape = [1, rows, N]
    n = tensor_node(
        cls,
        prefix + "_gate_up",
        OP,
        [
            (x, "x", 3, xshape),
            (w.name + ":0", "panels", 3, wshape),
            (gate["rows"] + ":0", "row_scale", 1, [rows]),
            (gate["scale"] + ":0", "gate_scale", 1, [1]),
            (up["scale"] + ":0", "up_scale", 1, [1]),
        ],
        [("gate", 2, yshape), ("up", 2, yshape)],
    )
    n.attr["cube_dequant"].i = int(variant != "pack128-vector")
    n.attr["prefetch_b"].i = int(prefetch)
    nodes.append(n)
    return nodes, n.name
