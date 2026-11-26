"""Host differential tests for the GE vocabulary cut, the production dispatch and its accounting."""

import copy
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from winners import c
from air import protobuf_classes
from prefill_split import split_model, ROW
from run_prefill_head import write_plan, check_blocks
from run_workspace import compare, request_summary
from context_measure import timings
from test_context_prefill import fake_plan
from test_workspace import case


def synthetic_graph(model_type):
    m = model_type(name="receiver")
    g = m.graph.add(name="receiver")

    def add(name, kind, ins, outs):
        n = g.op.add(name=name, type=kind, id=len(g.op))
        for edge, dt in ins:
            n.input.append(edge)
            n.input_desc.add(dtype=dt)
        for dt in outs:
            n.output_desc.add(dtype=dt)
        return n

    for i, name in enumerate(("hidden", "row")):
        n = add(name, "Data", [], [2 if i == 0 else 8])
        n.attr["index"].i = i
        g.input.append(name + ":0")
    add("pick", "GatherV2", [("hidden:0", 2), ("row:0", 8)], [2])
    add("quant", "DynamicQuant", [("pick:0", 2)], [3, 1])
    w = add("weight", "FileConstant", [], [3])
    w.attr["shape"].list.i[:] = [4, 8]
    add("scale", "FileConstant", [], [1])
    add("axis", "Const", [], [8])
    add("state", "fixture_state", [("hidden:0", 2), ("axis:0", 8)], [2, 1])
    add("scale_view", "Reshape", [("quant:1", 1)], [1])
    add(
        "vocab",
        "QuantBatchMatmulV4444",
        [("quant:0", 3), ("weight:0", 3), ("scale:0", 1), ("", 0), ("", 0), ("scale_view:0", 1)],
        [2],
    )
    add("ids", "ArgMaxV2", [("vocab:0", 2), ("axis:0", 8)], [8])
    net = add("end", "NetOutput", [("ids:0", 8), ("hidden:0", 2), ("state:1", 1)], [])
    net.attr["_input_name_key"].list.s[:] = [b"input0", b"input1", b"input2"]
    net.attr["_input_name_value"].list.i[:] = [0, 1, 2]
    abi = dict(
        inputs=[
            dict(name="hidden", dtype="float16", shape=[1, 64, 4]),
            dict(name="row", dtype="int64", shape=[1]),
        ],
        outputs=[
            dict(name="target_top1", dtype="int64", shape=[1, 1]),
            dict(name="features", dtype="float16", shape=[1, 64, 4]),
            dict(name="t0_recurrent", dtype="float32", shape=[1, 4]),
        ],
    )
    return m, abi


def execute(model, inputs, weights):
    vals = {}
    for n in model.graph[0].op:
        args = [vals[e] for e in n.input if e and not e.endswith(":-1")]
        if n.type == "NetOutput":
            return args
        if n.type == "Data":
            v = [inputs[n.name]]
        elif n.type == "Const":
            v = [-1]
        elif n.type == "FileConstant":
            v = [weights[n.name]]
        elif n.type == "GatherV2":
            v = [np.take(args[0], args[1], axis=1)]
        elif n.type == "fixture_state":
            v = [args[0], args[0].astype("f4").sum(axis=1)]
        elif n.type == "DynamicQuant":
            scale = np.maximum(np.max(np.abs(args[0].astype("f4")), axis=-1, keepdims=True), 1e-8) / 127
            v = [np.rint(args[0] / scale).astype("i1"), scale]
        elif n.type == "Reshape":
            v = [args[0]]
        elif n.type == "QuantBatchMatmulV4444":
            v = [((args[0].astype("i4") @ args[1].astype("i4")) * args[2] * args[3]).astype("f2")]
        elif n.type == "ArgMaxV2":
            v = [args[0].argmax(axis=args[1])]
        else:
            raise AssertionError(n.type)
        vals.update({f"{n.name}:{i}": t for i, t in enumerate(v)})
    raise AssertionError("no output")


def test_actual_ge_boundary_and_unrelated_nodes(tmp_path):
    descriptor = os.getenv("DFLASH_GE_DESCRIPTOR")
    if not descriptor:
        pytest.skip("CANN GE descriptor needed")
    m, abi = synthetic_graph(protobuf_classes(Path(descriptor).read_bytes())[0])
    saved = m.SerializeToString(deterministic=True)
    fragments, abis, proof = split_model(m, abi, vocab=8)
    assert m.SerializeToString(deterministic=True) == saved
    assert proof["shared_nodes"] == ["axis"]
    assert abis["body"]["outputs"][1:] == abi["outputs"][1:] and abis["head"]["outputs"] == abi["outputs"][:1]
    assert set(proof["removed_from_body"]) == {"ids", "quant", "scale_view", "scale", "vocab", "weight"}
    old = {n.name: n.SerializeToString(deterministic=True) for n in m.graph[0].op}
    for n in fragments["body"].graph[0].op:
        if n.type != "NetOutput":
            assert n.SerializeToString(deterministic=True) == old[n.name]
    rng = np.random.default_rng(310)
    w = dict(weight=rng.integers(-9, 10, (4, 8), dtype="i1"), scale=np.float32(0.25))
    for row in (0, 17, 63):
        ins = dict(hidden=rng.normal(size=(1, 64, 4)).astype("f2"), row=np.array([row]))
        a = execute(m, ins, w)
        z = execute(fragments["body"], ins, w)
        b = execute(fragments["head"], {ROW: z[0]}, w) + z[1:]
        assert z[0].shape == (1, 1, 4) and all(i.tobytes() == j.tobytes() for i, j in zip(a, b, strict=True))
    # An extra returned feature happens to consume the vocabulary result. It must never be cut.
    m.graph[0].op[-1].input[1] = "vocab:0"
    with pytest.raises(ValueError, match="retained"):
        split_model(m, abi, vocab=8)


def make_split_plan(folder, *, context=True):
    base, with_context = fake_plan(folder)
    plan = with_context if context else base
    old_om = folder / "target_prefill.om"
    abi = {"inputs": [], "outputs": []}
    for line in old_om.read_text().splitlines()[1:]:
        mark, name, dt, _rank, *shape = line.split()
        abi["inputs" if mark == "I" else "outputs"].append(
            dict(name=name, dtype=dt, shape=list(map(int, shape)))
        )
    original = dict(om=c.identity(old_om), abi=abi)
    hidden = dict(name=ROW, shape=[1, 1, 32], dtype="float16")
    parts = {}
    for name, role, ports in (
        ("body", "target_prefill", dict(inputs=abi["inputs"], outputs=[hidden, *abi["outputs"][1:]])),
        ("head", "target_prefill_head", dict(inputs=[hidden], outputs=abi["outputs"][:1])),
    ):
        lines = ["FAKE_CHUNK " + role]
        for side, mark in (("inputs", "I"), ("outputs", "O")):
            for x in ports[side]:
                lines.append(" ".join(map(str, [mark, x["name"], x["dtype"], len(x["shape"]), *x["shape"]])))
        p = folder / (name + ".om")
        p.write_text("\n".join([*lines, ""]))
        parts[name] = dict(om=c.identity(p), abi=ports)
    out = folder / "candidate"
    out.mkdir()
    return plan, write_plan(out, plan, original, parts), original, parts


@pytest.mark.parametrize("context", [False, True])
def test_route_only_when_all_consumers_allow_it(tmp_path, context):
    exe = os.getenv("QWEN35_CPP_TEST_RUNNER")
    if not exe:
        pytest.skip("host fake ACL runner needed")
    old, new, _, _ = make_split_plan(tmp_path, context=context)
    batch = tmp_path / "prompts"
    sizes = (63, 64, 65, 128, 129, 192)
    batch.write_text(
        "QWEN35_PROMPT_BATCH_V1\n" + "".join(f"p{n} " + ",".join(["2"] * n) + "\n" for n in sizes)
    )
    answers = []
    for plan in (old, new):
        out = tmp_path / (plan["sha256"] + ".json")
        p = subprocess.run(
            [
                exe,
                "--model-kind",
                "chunk",
                "--mode",
                "dflash",
                "--model",
                plan["path"],
                "--model-sha256",
                plan["sha256"],
                "--prompt-batch",
                batch,
                "--prompt-batch-sha256",
                c.sha(batch),
                "--low-memory",
                "--trace-rounds",
                "--device-id",
                "0",
                "--eos-token-ids",
                "63",
                "--pad-token-id",
                "0",
                "--max-new-tokens",
                "24",
                "--max-draft-tokens",
                "15",
                "--warmup",
                "0",
                "--repetitions",
                "2",
                "--output",
                out,
            ],
            capture_output=True,
            text=True,
        )
        assert p.returncode == 0, p.stderr
        results = []
        for n, entry in zip(sizes, c.read(out)["cases"], strict=True):
            r = c.read(entry["report"])
            assert r["abi"].get("split_target_prefill_head", False) == (plan == new)
            for m in r["dflash"]["measurements"]:
                stages = m["stage_ms"]
                assert len(stages["target_prefill"]) == (n + 63) // 64
                assert len(stages.get("target_prefill_head", [])) == (
                    (1 if context else (n + 63) // 64) if plan == new else 0
                )
                results.append((m["generated_token_ids"], m["rounds"], m["stop_reason"]))
        answers.append(results)
    assert answers[0] == answers[1]


def test_guarded_three_block_entry(tmp_path):
    exe = os.getenv("DFLASH_TEST_GUARDED_REPLAY")
    if not exe:
        pytest.skip("host guarded replay needed")
    _, _, original, parts = make_split_plan(tmp_path)
    folder = tmp_path / "captures"
    folder.mkdir()
    receipts = check_blocks(folder, [j % 32 for j in range(129)], exe, original, parts, os.environ)
    assert [r["valid"] for r in receipts] == [64, 64, 1]
    assert not any(folder.rglob("*actual-1.bin"))


def test_effective_last_call_includes_head_once():
    p = dict(id="long", tokens=[3] * 129)
    plan = {"sha256": "plan"}
    old = case(p, plan, 1)
    new = copy.deepcopy(old)
    new["abi"].update(split_target_prefill_head=True, om_count=6)
    for m in new["dflash"]["measurements"]:
        m["stage_ms"].update(target_prefill=[64, 64, 64], target_prefill_head=[4])
    r = request_summary(
        [dict(row, prompt="long", repetition=i) for i, row in enumerate(timings(new, p, plan, context=True))]
    )
    assert (r["prefill_head"]["count"], r["prefill_middle"]["count"]) == (3, 6)
    assert r["prefill_last"]["mean_ms"] == 68 and r["target_per_request"]["mean_ms"] == 296
    new["dflash"]["measurements"][0]["stage_ms"]["target_prefill_head"] *= 2
    with pytest.raises(ValueError, match="dispatch"):
        timings(new, p, plan, context=True)


def test_archived_comparison_charges_short_last_call_and_long_savings(tmp_path):
    prompts = [dict(id="s", group="short", tokens=[3] * 33), dict(id="l", group="long", tokens=[3] * 129)]
    plan, split = ({"sha256": k} for k in ("base", "new"))
    old, new = {"reports": {}}, {"reports": {}}
    rows = {"short": [], "long": []}
    for p in prompts:
        a = case(p, plan, 1)
        a["abi"]["graph_count"] = 5
        a["protocol"]["max_resident_models"] = 5
        b = copy.deepcopy(a)
        b.update(model=split)
        b["abi"].update(split_target_prefill_head=True, graph_count=6, om_count=6)
        b["protocol"]["max_resident_models"] = 6
        bcount = (len(p["tokens"]) + 63) // 64
        for m in b["dflash"]["measurements"]:
            m["stage_ms"].update(target_prefill=[65] * bcount, target_prefill_head=[6])
            for t in ("prefill", "model_total"):
                m["latency_ms"][t] += 6 - 5 * bcount
        for data, root, k in ((a, old, "base"), (b, new, "new")):
            f = tmp_path / (k + p["id"] + ".json")
            c.write(f, data)
            root["reports"][p["id"]] = c.identity(f)
        rows[p["group"]] = [
            dict(v, prompt=p["id"], repetition=i) for i, v in enumerate(timings(a, p, plan, context=True))
        ]
    rows["all"] = rows["short"] + rows["long"]
    ref = {k: {"workspace": request_summary(v)} for k, v in rows.items()}
    z = compare(tmp_path, prompts, old, new, plan, ref, split_plan=split)
    assert z["same_workload"] and z["round_matches"] == z["token_matches"] == 6
    s, l = [z["groups"][g]["split_head"] for g in ("short", "long")]
    assert s["prefill"]["mean_ms"] == 71 and l["prefill_middle"]["mean_ms"] == 65
    assert l["target_per_request"]["mean_ms"] == 65 * 3 + 6 + 100
