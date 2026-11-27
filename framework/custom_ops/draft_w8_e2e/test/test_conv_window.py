"""Verify the receiver AIR pattern, arbitrary valid/accepted lengths and device profile accounting."""

import csv
import os
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from winners import c
from air import protobuf_classes
from conv_window_air import rewrite
from conv_window_check import op_summary, compare_bytes
from receiver_air import const_int, node_name
from run_conv_window import replace_models
from run_workspace import compare, request_summary
from context_measure import timings
from test_workspace import case as receiver_case


def graph(rows):
    desc = os.getenv("DFLASH_GE_DESCRIPTOR")
    if not desc:
        pytest.skip("CANN GE schema required")
    m = protobuf_classes(Path(desc).read_bytes())[0]()
    g = m.graph.add(name="windows")
    abi = dict(inputs=[], outputs=[])

    def add(name, kind, inputs=(), dtype=2):
        n = g.op.add(name=name, type=kind, id=len(g.op))
        n.input[:] = [e + ":0" for e in inputs]
        n.output_desc.add(dtype=dtype)
        return n

    def data(name, dt, typ, shp):
        n = add(name, "Data", dtype=dt)
        n.attr["index"].i = len(abi["inputs"])
        g.input.append(name + ":0")
        abi["inputs"].append(dict(name=name, dtype=typ, shape=shp))

    def const(name, value):
        n = add(name, "Const", dtype=8)
        a = np.asarray(value, dtype="<i8")
        t = n.attr["value"].t
        t.desc.dtype = 8
        t.desc.shape.dim[:] = a.shape
        t.data = a.tobytes()
        return name

    data("input_ids", 8, "int64", [1, rows])
    data("valid_rows", 5, "int16", [1])
    count = "valid_rows"
    outs = []
    one, last_axis = const("one", 1), const("axis", -1)
    if rows == 16:
        data("accepted", 8, "int64", [1])
        abi["outputs"].append(dict(name="accepted_count", dtype="int64", shape=[1]))
        outs.append("accepted")
        add("inc", "Add", ["accepted", one], 8)
        add("short", "Cast", ["inc"], 5)
        count = "short"
    add("length", "Cast", [count], 8)
    add("index", "Sub", ["length", one], 8)
    for layer in range(2):
        tag = str(layer)
        state, x = f"t{tag}_conv", "x" + tag
        data(state, 2, "float16", [1, 7, 4])
        data(x, 2, "float16", [1, 7, rows])
        history = add("history" + tag, "ConcatV2", [state, x, last_axis])
        slices = []
        for i in range(1, 5):
            label = f"slice{tag}_{i}"
            edges = [const(label + str(j), [z]) for j, z in enumerate((i, i + rows, 2, 1))]
            slices.append(add(label, "StridedSliceV2", [history.name, *edges]).name)
        pack = add("pack" + tag, "Pack", slices)
        pack.attr["axis"].i, pack.attr["N"].i = -1, 4
        flat = add("flat" + tag, "Reshape", [pack.name, const("fd" + tag, [7, rows, 4])])
        tr = add("transpose" + tag, "Transpose", [flat.name, const("perm" + tag, [1, 0, 2])])
        shaped = add("view" + tag, "Reshape", [tr.name, const("vd" + tag, [1, rows, 7, 4])])
        ident = add("identity" + tag, "Identity", [shaped.name])
        gather = add("gather" + tag, "GatherV2", [ident.name, "index", one])
        sq = add("squeeze" + tag, "Squeeze", [gather.name])
        sq.attr["axis"].list.i[:] = [1]
        abi["outputs"].append(dict(name=state, dtype="float16", shape=[1, 7, 4]))
        outs.append(sq.name)
    add("net", "NetOutput", outs)
    g.output[:] = [s + ":0" for s in outs]
    return m, abi


def execute(model, inputs):
    vals = {}
    for n in model.graph[0].op:
        xs = [vals[node_name(e)] for e in n.input]
        if n.type == "NetOutput":
            return xs
        if n.type == "Data":
            v = inputs[n.name]
        elif n.type == "Const":
            v = const_int(n)
        elif n.type == "ConcatV2":
            v = np.concatenate(xs[:2], axis=int(xs[2]))
        elif n.type == "StridedSliceV2":
            v = xs[0][:, :, int(xs[1][0]) : int(xs[2][0])]
        elif n.type == "Pack":
            v = np.stack(xs, axis=-1)
        elif n.type == "Reshape":
            v = xs[0].reshape(xs[1])
        elif n.type == "Transpose":
            v = xs[0].transpose(xs[1])
        elif n.type == "Cast":
            v = xs[0].astype({5: np.int16, 8: np.int64}[n.output_desc[0].dtype])
        elif n.type == "Sub":
            v = xs[0] - xs[1]
        elif n.type == "Add":
            v = xs[0] + xs[1]
        elif n.type == "GatherV2":
            v = np.take(xs[0], xs[1], axis=int(xs[2]))
        elif n.type == "Squeeze":
            v = np.squeeze(xs[0], axis=1)
        else:
            assert n.type == "Identity"
            v = xs[0]
        vals[n.name] = v
    raise AssertionError("missing outputs")


@pytest.mark.parametrize("rows", [16, 64])
def test_all_prefixes_and_bit_patterns(rows):
    old, abi = graph(rows)
    new = type(old).FromString(old.SerializeToString())
    proof = rewrite(new, abi)
    assert len(proof["branches"]) == 2
    assert not any(n.type in ("Pack", "Transpose", "StridedSliceV2") for n in new.graph[0].op)
    assert sum(n.name == "target_conv_indices" for n in new.graph[0].op) == 1
    assert list(new.graph[0].output) == list(new.graph[0].op[-1].input)
    rng = np.random.default_rng(3103)
    inputs = {v["name"]: rng.integers(0, 65536, size=v["shape"], dtype=np.uint16) for v in abi["inputs"]}
    for commit in range(1, rows + 1):
        inputs["valid_rows"] = np.array([rows if rows == 16 else commit], dtype=np.int16)
        inputs["accepted"] = np.array([commit - 1], dtype=np.int64)
        a, b = execute(old, inputs), execute(new, inputs)
        assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
        for i, got in enumerate(b[-2:]):
            history = np.concatenate([inputs[f"t{i}_conv"], inputs[f"x{i}"]], axis=-1)
            assert np.array_equal(got, history[..., commit : commit + 4])


@pytest.mark.parametrize("fault", ["wrong_layer", "wrong_length", "live_bank"])
def test_rejects_invalid_source(fault):
    m, abi = graph(16)
    nodes = {n.name: n for n in m.graph[0].op}
    if fault == "wrong_layer":
        nodes["history1"].input[0] = "t0_conv:0"
    elif fault == "wrong_length":
        nodes["length"].input[0] = "valid_rows:0"
    else:
        nodes["net"].input.append("transpose0:-1")
        del m.graph[0].output[:]
    with pytest.raises(ValueError):
        rewrite(m, abi)


def test_profile_needs_the_two_complete_target_executions(tmp_path):
    p = tmp_path / "profiler/op_summary.csv"
    p.parent.mkdir()
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Op Name", "Op Type", "Input Shapes", "Output Shapes", "Task Duration(us)"])
        for _ in range(2):
            for layer in range(249):
                w.writerow([f"p/{layer}", "QuantBatchMatmulV4444", "", "", 4])
            for layer in range(24):
                w.writerow([f"p/tr_{layer}", "Transpose", '"8192,16,4;3"', '"16,8192,4"', 100])
            w.writerow(["p/small", "Transpose", "16,8192;2", "8192,16", 10])
    z = op_summary(tmp_path, 16)
    assert z["large_nodes"] == 24 and z["large_ms"] == 2.4 and z["all_transpose_ms"] == 2.41
    with p.open("a") as f:
        f.write("p/0,QuantBatchMatmulV4444,,,4\n")
    with pytest.raises(ValueError):
        op_summary(tmp_path, 16)


def test_only_two_bindings_change_and_capture_checks_all_ref_bytes(tmp_path):
    def fake(name):
        path = tmp_path / name
        path.write_text(name)
        return dict(om=c.identity(path), abi={})

    before = {key: fake(key) for key in ("body", "verify")}
    after = {key: fake("new-" + key) for key in before}

    def graph(name, asset):
        return f'graph {name} "{asset["om"]["path"]}" {asset["om"]["sha256"]}\nend\n'

    oldtext = (
        "draft settings\n"
        + graph("target_prefill", before["body"])
        + graph("target_prefill_head", fake("head"))
        + graph("target_verify", before["verify"])
        + "done\n"
    )
    baseline = tmp_path / "old-plan"
    baseline.write_text(oldtext)
    target = c.verify(replace_models(tmp_path, c.identity(baseline), before, after)).read_text()
    for key, role in (("body", "target_prefill"), ("verify", "target_verify")):
        target = target.replace(graph(role, after[key]), graph(role, before[key]))
    assert target == oldtext
    x, y = {"t0_conv": {"sha256": "a"}}, {"t3_key": {"sha256": "b"}}
    compare_bytes((x, y), (x, y), "test")
    with pytest.raises(ValueError):
        compare_bytes((x, y), (x, {}), "test")


def test_six_om_suite_compares_to_split_head_receipt(tmp_path):
    prompts = [dict(id=k, group=k, tokens=[2] * size) for k, size in (("short", 28), ("long", 129))]
    plans = [{"sha256": name} for name in ("base", "candidate")]
    runs = [{"reports": {}}, {"reports": {}}]
    groups = {}
    for p in prompts:
        for i, (plan, run) in enumerate(zip(plans, runs, strict=True)):
            r = receiver_case(p, plan, 1)
            r["abi"].update(om_count=6, graph_count=6, split_target_prefill_head=True)
            for m in r["dflash"]["measurements"]:
                m["stage_ms"]["target_prefill_head"] = [3]
                m["stage_ms"]["target_prefill"] = [t - i for t in m["stage_ms"]["target_prefill"]]
                for metric in ("prefill", "model_total"):
                    m["latency_ms"][metric] += 3 - i * len(m["stage_ms"]["target_prefill"])
            f = tmp_path / f"{i}-{p['id']}.json"
            c.write(f, r)
            run["reports"][p["id"]] = c.identity(f)
            if not i:
                groups[p["group"]] = [
                    dict(x, prompt=p["id"], repetition=j)
                    for j, x in enumerate(timings(r, p, plan, context=True))
                ]
    groups["all"] = groups["short"] + groups["long"]
    archive = {k: {"split_head": request_summary(v)} for k, v in groups.items()}
    result = compare(tmp_path, prompts, *runs, plans[0], archive, window_plan=plans[1])
    assert result["same_workload"] and result["token_matches"] == result["round_matches"] == 6
    group = result["groups"]["all"]
    assert group["conv_window"]["prefill_head"]["count"] == 6
    assert group["previous"]["prefill_body"]["mean_ms"] - group["conv_window"]["prefill_body"]["mean_ms"] == 1
    assert group["conv_window"]["generation_tok_s"] > group["previous"]["generation_tok_s"]
