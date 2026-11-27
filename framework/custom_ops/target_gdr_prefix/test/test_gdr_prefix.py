import csv
import os
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gdr_graph import GDR, COMMIT, TYPES, STATE, specs, sites, prefix_graph, isolated
from pair_graph import protobuf_classes, tensor_node, spec
from gdr_measure import pass_times


def model():
    desc = os.getenv("DFLASH_GE_DESCRIPTOR")
    if not desc:
        pytest.skip("CANN GE schema required")
    m = protobuf_classes(Path(desc).read_bytes())[0](name="verify")
    g = m.graph.add(name="verify")
    cls = type(g.op.add())
    del g.op[:]
    seq = specs()
    seq[5]["name"] = "t0_recurrent"
    seq[6]["name"] = "valid_rows"
    seq.append(spec("accepted_count", "int64", [1]))
    for i, (s, dt) in enumerate(zip(seq, (*TYPES, 8), strict=True)):
        n = tensor_node(cls, "arg" + str(i), "Data", [], [("y", dt, s["shape"])])
        n.input.append("")
        n.attr["index"].i = i
        g.op.append(n)
        g.input.append(n.name + ":0")
    n = tensor_node(cls, "one", "Const", [], [("y", 8, [])])
    n.attr["value"].t.desc.CopyFrom(n.output_desc[0])
    n.attr["value"].t.data = np.int64(1).tobytes()
    g.op.append(n)
    n = tensor_node(cls, "plus", "Add", [("arg7:0", "x1", 8, [1]), ("one:0", "x2", 8, [])], [("y", 8, [1])])
    g.op.append(n)
    n = tensor_node(cls, "commit_rows", "Cast", [("plus:0", "x", 8, [1])], [("y", 5, [1])])
    g.op.append(n)
    for name, end in (("first", "arg6:0"), ("second", "commit_rows:0")):
        args = [
            ("arg" + str(i) + ":0", s["name"], dt, s["shape"])
            for i, (s, dt) in enumerate(zip(seq[:6], TYPES[:6], strict=True))
        ]
        n = tensor_node(
            cls,
            name,
            GDR,
            [*args, (end, "length", 5, [1])],
            [("core", 2, [1, 16, 32, 128]), ("state", 1, STATE)],
        )
        n.attr["chunk_size"].i = 64
        n.attr["use_qk_l2norm_in_kernel"].b = True
        n.attr["output_final_state"].b = True
        g.op.append(n)
    net = g.op.add(name="net", type="NetOutput")
    net.input[:] = ["arg7:0", "second:1", "first:1"]
    outs = [seq[-1], seq[5], spec("verify_discard_t0_recurrent", "float32", STATE)]
    for v, dt in zip(outs, (8, 1, 1), strict=True):
        net.input_desc.add(dtype=dt)
    return m, dict(inputs=seq, outputs=outs)


def test_split_reads_actual_operands_and_state_port():
    m, abi = model()
    loc = sites(m, abi)["t0_recurrent"]
    assert (loc["first"], loc["commit"]) == ("first", "second")
    p, a = prefix_graph(m, abi, loc)
    assert len(a["inputs"]) == len(a["outputs"]) == 5 and not any(n.type == GDR for n in p.graph[0].op)
    for native in (True, False):
        x, io = isolated(m, loc, chunk=1, custom=not native)
        op = x.graph[0].op[-2]
        assert op.type == (GDR if native else COMMIT)
        assert io["outputs"] == [spec("state", "float32", STATE)] and io["inputs"][-1]["dtype"] == "int16"
        assert x.graph[0].op[-1].input[0] == op.name + (":1" if native else ":0")
        if native:
            assert op.attr["chunk_size"].i == 1


@pytest.mark.parametrize("fault", ["wrong-length", "observable-core", "cross-layer"])
def test_refuses_unsafe_state_replacement(fault):
    m, abi = model()
    t = {n.name: n for n in m.graph[0].op}
    if fault == "wrong-length":
        t["second"].input[6] = "arg6:0"
    elif fault == "observable-core":
        t["net"].input.append("second:0")
    else:
        t["second"].input[5] = "arg0:0"
    with pytest.raises(ValueError):
        sites(m, abi)


def test_profile_matches_passes_by_port_and_requires_both_visits(tmp_path):
    f = tmp_path / "op_summary.csv"
    with f.open("w", newline="") as s:
        w = csv.writer(s)
        w.writerow(["Op Name", "Op Type", "Model ID", "Task Duration(us)"])
        for i in range(2):
            w.writerow(["g/a", GDR, "1", 100])
            w.writerow(["g/b", GDR, "1", 30])
    sites_ = {"t0_recurrent": {"first": "a", "commit": "b"}}
    r = pass_times(f, sites_)
    assert r["first"]["task_sum_ms"] == 0.1 and r["commit"]["task_sum_ms"] == 0.03
    with f.open("a") as s:
        s.write("b,ChunkGatedDeltaRule,1,4\n")
    with pytest.raises(ValueError):
        pass_times(f, sites_)
