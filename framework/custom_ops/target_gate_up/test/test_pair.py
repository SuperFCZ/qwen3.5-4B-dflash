import os
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pair_graph import K, N, MM, DQ, pairs, protobuf_classes, single_graph, spec
from pair_reference import LAYOUTS, converted, ints, matrix_view


def fixture(tmp_path):
    descriptor = os.getenv("DFLASH_GE_DESCRIPTOR")
    if not descriptor:
        pytest.skip("CANN GE descriptor required")
    T = protobuf_classes(Path(descriptor).read_bytes())[0]
    m = T(name="target")
    g = m.graph.add(name="target")

    def op(name, kind, edges=(), dt=2):
        n = g.op.add(name=name, type=kind, id=len(g.op))
        n.input[:] = edges
        n.output_desc.add(dtype=dt)
        return n

    x = op("source", "Data", [""])
    x.attr["index"].i = 0
    g.input.append("source:0")
    op("norm", "Identity", ["source:0"])
    for i in range(32):
        for b in ("gate", "up"):
            root = f"body_layers_{i}_mlp_{b}_proj"
            for suffix, typ, dims in (("_W_q", 2, [K, N]), ("_scale", 0, [1])):
                name = root + suffix
                f = tmp_path / name
                with f.open("wb") as stream:
                    stream.truncate(K * N if typ == 2 else 4)
                n = op(name, "FileConstant")
                n.attr["file_path"].s = str(f).encode()
                n.attr["dtype"].dt = typ
                n.attr["shape"].list.i[:] = dims
            q = op(root + "dq", DQ, ["norm:0"], 3)
            q.output_desc.add(dtype=1)
            shape = op(root + "shape", "Const", dt=8)
            r = op(root + "row", "Reshape", [q.name + ":1", shape.name + ":0"], 1)
            op(root + "mm", MM, [q.name + ":0", root + "_W_q:0", root + "_scale:0", "", "", r.name + ":0"])
    return m, {"inputs": [spec("hidden", "float16", [1, 16, K])]}


def test_graph_boundaries_and_cse(tmp_path):
    m, abi = fixture(tmp_path)
    for mode, dq in (("prefix", 2), ("oracle", 2), ("native", 2), ("shared", 1)):
        n, a = single_graph(m, abi, 16, mode)
        ops = n.graph[0].op
        assert sum(p.type == DQ for p in ops) == dq and sum(p.type == MM for p in ops) == 2
        assert a["inputs"][0]["name"] == ("hidden" if mode == "prefix" else "x")
        assert all("body_layers_1_" not in p.name for p in ops)
        assert [x["name"] for x in a["outputs"]][:2] == ["gate", "up"]
    a = pairs(m.graph[0])[0]
    next(n for n in m.graph[0].op if n.name == a["up"]["dq"]).input[0] = "source:0"
    with pytest.raises(ValueError):
        pairs(m.graph[0])


def test_layout_inverse_and_exact_integer_reference():
    k, n = 64, 96
    w = np.random.default_rng(2).integers(-128, 128, (k, n), dtype=np.int8)
    raw = [
        w,
        w.T,
        w.reshape(k // 16, 16, n // 32, 32).transpose(2, 0, 1, 3),
        w.T.reshape(n // 16, 16, k // 32, 32).transpose(2, 0, 1, 3),
        w.T.reshape(n // 16, 16, k // 32, 32).transpose(0, 2, 1, 3),
    ]
    for mode, x in zip(LAYOUTS, raw, strict=True):
        assert np.array_equal(matrix_view(x.copy().ravel(), mode, k, n), w)
    q = np.full((3, 2560), -128, dtype=np.int8)
    v = np.full((2560, 16), 127, dtype=np.int8)
    assert np.array_equal(ints(q, v), q.astype(np.int64) @ v.astype(np.int64))
    assert converted(
        np.array([[100]], np.int32), np.float32(0.1), np.array([[0.2]], np.float32), "weight-first"
    )[0, 0] == np.float16(2).view(np.uint16)
