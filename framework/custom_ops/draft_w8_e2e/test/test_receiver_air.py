"""CPU contracts for reconstruction from real receiver graph port conventions."""

import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from receiver_air import TargetIR, OLD_LAYERS, UNION, DTYPE, const_int, port_map
from air import protobuf_classes


def fixture(types, rows, folder):
    g = types[0]().graph.add(name="source")
    old = {"inputs": [], "outputs": []}
    new = copy.deepcopy(old)
    table = {}
    tables = {}
    out = []

    def node(name, kind, ins=(), dtypes=(2,)):
        n = g.op.add(name=name, type=kind, id=len(g.op))
        table[name] = n
        for s in ins:
            n.input.append(s)
            if not s:
                n.input_desc.add()
                continue
            key, j = s.rsplit(":", 1)
            d = n.input_desc.add()
            d.CopyFrom(table[key].output_desc[int(j)])
        for dt in dtypes:
            n.output_desc.add(dtype=dt)
        return n

    def data(name, dt, shape, new_type=None, new_shape=None):
        i = len(old["inputs"])
        n = node(name, "Data", dtypes=[DTYPE[dt]])
        n.input.append("")
        n.output_desc[0].shape.dim.extend(shape)
        n.input_desc.add().CopyFrom(n.output_desc[0])
        n.attr["index"].i = i
        g.input.append(name + ":0")
        z = dict(name=name, dtype=dt, shape=shape)
        old["inputs"].append(z)
        new["inputs"].append(dict(z, dtype=new_type or dt, shape=new_shape or shape))
        return name + ":0"

    def pub(name, edge, dt, shape, new_type=None, new_shape=None):
        z = dict(name=name, dtype=dt, shape=shape)
        old["outputs"].append(z)
        new["outputs"].append(dict(z, dtype=new_type or dt, shape=new_shape or shape))
        out.append(edge)

    def literal(name, val):
        a = np.array(val, dtype="<i8")
        n = node(name, "Const", dtypes=[8])
        t = n.attr["value"].t
        t.desc.dtype = 8
        t.desc.shape.dim.extend(a.shape)
        t.data = a.tobytes()
        n.output_desc[0].CopyFrom(t.desc)
        return name + ":0"

    start = data("start_position", "int64", [1])
    valid = data("valid_rows", "int16", [1])
    z = literal("zero", 0)
    one = literal("one", 1)
    limit = literal("oldcols", 576)
    nr = literal("nrows", rows)
    col = node("cols", "Range", [z, limit, one], (8,)).name + ":0"
    pos = node("pos", "Range", [z, nr, one], (8,)).name + ":0"
    pos = node("positions", "Add", [start, pos], (8,)).name + ":0"
    end = node("end", "Add", [start, valid], (8,)).name + ":0"
    left = node("left", "LessEqual", [col, pos], (11,)).name + ":0"
    right = node("right", "Less", [col, end], (11,)).name + ":0"
    flag = node("mask_bool", "LogicalAnd", [left, right], (11,)).name + ":0"
    flag = node("mask_values", "SelectV2", [flag, z, one], (1,)).name + ":0"
    flag = node("mask", "Cast", [flag], (2,)).name + ":0"
    h0 = valid
    layers = []
    for i in range(32):
        a = node("res" + str(i), "Add", [h0, h0]).name + ":0"
        a = node("ff" + str(i), "Add", [a, a]).name + ":0"
        h0 = node("hidden" + str(i), "SelectV2", [valid, a, z]).name + ":0"
        layers.append(h0)
    clones = [node("source_clone" + str(i), "Identity", [layers[i]]).name + ":0" for i in OLD_LAYERS]
    cat = node("features_join", "ConcatV2", [*clones, literal("axis", -1)])
    cat.attr["N"].i = 8
    cat.attr["_input_name_key"].list.s.extend([b"concat_dim", *(f"x{i}".encode() for i in range(8))])
    cat.attr["_input_name_value"].list.i.extend([8, *range(8)])
    feat = cat.name + ":0"
    if rows != 64:
        feat = node("pad", "PadV3", [feat, z]).name + ":0"
    pub("features", feat, "float16", [1, 64, 20480], new_shape=[1, 64, 28160])
    accepted = node("accepted", "Cast", [valid], (8,)).name + ":0"
    committed_rows = node("commit_rows", "Add", [accepted, one], (5,)).name + ":0"
    if rows != 64:
        pub("accepted_count", accepted, "int64", [1])
    for i in range(24):
        s = f"t{i}_recurrent"
        sh = [1, 32, 128, 128]
        x = data(s, "float16", sh, "float32")
        cast = node("init" + s, "Cast", [x], (1,)).name + ":0"
        first = node("first" + s, "ChunkGatedDeltaRule", [h0] * 5 + [cast, valid], (2, 1))
        final = first
        if rows != 64:
            final = node("second" + s, "ChunkGatedDeltaRule", [h0] * 5 + [cast, committed_rows], (2, 1))
            pub("verify_discard_" + s, first.name + ":1", "float32", sh)
        y = node("out" + s, "Cast", [final.name + ":1"], (2,)).name + ":0"
        pub(s, y, "float16", sh, "float32")
    for i in range(8):
        pair = []
        for suf in ("key", "value"):
            s = f"t{i}_{suf}"
            osh = [9, 64, 64, 16]
            nsh = [33, 64, 64, 16]
            x = data(s, "float16", osh, new_shape=nsh)
            for j in range(1 if rows == 64 else 16):
                x = node(s + str(j), "CacheUpdate", [x, h0, z, z]).name + ":0"
            pub(s, x, "float16", osh, new_shape=nsh)
            pair.append(x)
        tab = node("tbl" + str(i), "FileConstant", dtypes=[1])
        tab.attr["dtype"].dt = 3
        tab.attr["shape"].list.i.extend([1, 9])
        f = folder / tab.name
        np.arange(9, dtype="<i4").tofile(f)
        tables[tab.name] = f
        ports = [h0, *pair, "", flag, nr, limit] + [""] * 7 + [tab.name + ":0", "", limit]
        node("att" + str(i), "AdnFusedInferAttention", ports)
    net = node("sink", "NetOutput", out, ())
    for i, v in enumerate(old["outputs"]):
        d = net.input_desc[i]
        d.shape.dim[:] = v["shape"]
        d.dtype = DTYPE[v["dtype"]]
    return g, old, new, tables


@unittest.skipUnless(os.getenv("DFLASH_GE_DESCRIPTOR"), "CANN descriptor")
class AirContracts(unittest.TestCase):
    def go(self, rows, corrupt=False):
        t = protobuf_classes(Path(os.environ["DFLASH_GE_DESCRIPTOR"]).read_bytes())
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            g, old, new, tabs = fixture(t, rows, p)
            ir = TargetIR(g, old, new)
            if corrupt:
                ir.nodes["source_clone9"].input[0] = "hidden8:0"
                with self.assertRaisesRegex(ValueError, "feature taps"):
                    ir.features(rows)
                return
            taps = ir.features(rows)
            r = ir.recurrent(rows == 16)
            a = ir.attention(rows, tabs, p)
            ir.finish()
            table = {n.name: n for n in g.op}
            self.assertEqual([taps[str(i)] for i in UNION], [f"hidden{i}" for i in UNION])
            self.assertEqual(port_map(table["features_join"])[b"concat_dim"], 11)
            self.assertEqual(len(r), 24)
            self.assertEqual(len(a["cache_update"]), 16)
            self.assertEqual(const_int(table["oldcols"]).item(), 2112)
            self.assertEqual(table["t0_recurrent"].output_desc[0].dtype, 1)
            self.assertEqual(table["firstt0_recurrent"].input[5:], ["t0_recurrent:0", "valid_rows:0"])
            self.assertEqual(
                table["sink"].input[ir.out_index["t0_recurrent"]],
                ("first" if rows == 64 else "second") + "t0_recurrent:1",
            )
            self.assertEqual(list(table["t0_key"].output_desc[0].shape.dim), [33, 64, 64, 16])
            np.testing.assert_array_equal(np.fromfile(p / "page-table-2112.i32", "<i4"), np.arange(33))

    def test_prefill(self):
        self.go(64)

    def test_verify(self):
        self.go(16)

    def test_wrong_old_tap(self):
        self.go(16, True)


if __name__ == "__main__":
    unittest.main()
