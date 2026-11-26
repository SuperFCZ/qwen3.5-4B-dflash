"""CPU graph equivalence and host-only ACL routing checks for cache-only Prefill."""

import os
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from winners import OPS, c, SELECT, UNION
from features import insert_selector
from context_graph import context_model, STATES
from context_measure import timings, summarize
from run_context import extend_plan


def test_full_cache_and_next_proposal(tmp_path):
    descriptor = os.getenv("DFLASH_GE_DESCRIPTOR")
    if not descriptor:
        pytest.skip("CANN GE descriptor needed for differential graph execution")
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(OPS / "draft_full_panels/test"))
    from test_full import fixture, execute, fe
    from air import protobuf_classes

    torch.manual_seed(31064)
    torch.set_num_threads(1)
    net, ops = fixture()
    net = fe.export_net(net, ops)
    ext = {}
    for name in ("embedding", "head"):
        file = tmp_path / name
        torch.randn(net.config.vocab_size, net.config.hidden_size).half().numpy().tofile(file)
        ext[name] = {"fp16": c.identity(file)}
    graph, abi = fe.trace_graph(
        protobuf_classes(Path(descriptor).read_bytes()), net, ops, {"target": ext}, 64, 256
    )
    abi, _ = insert_selector(graph.graph, abi, 64, net.config.hidden_size, UNION, SELECT)
    reduced, small = context_model(graph.model, abi, graph.projections)
    assert [v["name"] for v in small["inputs"]] == ["features", "start_position", "valid_rows", *STATES]
    assert [v["name"] for v in small["outputs"]] == STATES
    assert sum(n.type == fe.RAW for n in reduced.graph[0].op) == 6
    assert not any(
        n.type in ("FileConstant", "MatMul", "BatchMatMulV2", "ArgMaxV2") for n in reduced.graph[0].op
    )

    states = [torch.randn(v["shape"]).half() for v in small["outputs"]]
    for position in (0, 64, 128):
        x = torch.randn(abi["inputs"][0]["shape"]).half()
        controls = [
            torch.tensor([v], dtype=getattr(torch, t["dtype"]))
            for t, v in zip(abi["inputs"][1:5], [position, 64, 42 + position, 1])
        ]
        old = execute(graph, [x, *controls, *states])
        new = execute(SimpleNamespace(graph=reduced.graph[0]), [x, *controls[:2], *states])
        for before, expected, actual in zip(states, old[1:], new, strict=True):
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
            for region in (slice(0, position), slice(position + 64, None)):
                assert torch.equal(
                    actual[:, :, region].view(torch.int16), before[:, :, region].view(torch.int16)
                )
        states = new


def fake_plan(folder):
    feature = ("features", "float16", [1, 64, 32])
    start = ("start_position", "int64", [1])
    valid = ("valid_rows", "int16", [1])
    dstate = [(s, "float16", [1, 1, 320, 16]) for s in STATES]
    tstate = ("t0_recurrent", "float32", [1, 1, 1, 1])
    values = {
        "target_prefill": (
            [("input_ids", "int64", [1, 64]), start, valid, tstate],
            [("target_top1", "int64", [1, 1]), feature, tstate],
        ),
        "target_verify": (
            [("input_ids", "int64", [1, 16]), start, valid, tstate],
            [
                ("target_top1", "int64", [1, 16]),
                ("accepted_count", "int64", [1]),
                feature,
                tstate,
                ("verify_discard_t0_recurrent", "float32", [1, 1, 1, 1]),
            ],
        ),
        "draft": (
            [feature, start, valid, ("anchor", "int64", [1]), ("proposal_count", "int16", [1]), *dstate],
            [("draft_top1", "int64", [1, 15]), *dstate],
        ),
    }

    def ports(sig):
        return [
            " ".join(map(str, (marker, name, dtype, len(shape), *shape)))
            for marker, tensors in zip(("I", "O"), sig)
            for name, dtype, shape in tensors
        ]

    def model(name, role, sig):
        p = folder / (name + ".om")
        p.write_text("\n".join(["FAKE_CHUNK " + role, *ports(sig), ""]))
        return p

    lines = ["qwen35-dflash-chunk-v4", "capacity 256 64", "draft_prefill_policy static_draft16_64_oms"]
    for role, (ins, outs) in values.items():
        first = [("features", "float16", [1, 16, 32]), *ins[1:]] if role == "draft" else ins
        file = model(role, role, (first, outs))
        lines.append(f'graph {role} "{file}" {c.sha(file)}')
        if role == "draft":
            alt = model("draft64", role, (ins, outs))
            lines.append(f'static_gear64 "{alt}" {c.sha(alt)}')
        lines.extend([*ports((ins, outs)), "end"])
    base = folder / "original.plan"
    base.write_text("\n".join([*lines, "done", ""]))
    sig = ([feature, start, valid, *dstate], dstate)
    om = model("context", "draft_context64", sig)
    abi = {
        key: [dict(name=n, dtype=t, shape=s) for n, t, s in rows]
        for key, rows in zip(("inputs", "outputs"), sig)
    }
    return c.identity(base), extend_plan(folder, c.identity(base), {"om": c.identity(om)}, abi)


def test_dispatch_reset_and_both_banks(tmp_path):
    runner = os.getenv("QWEN35_CPP_TEST_RUNNER")
    if not runner:
        pytest.skip("host fake ACL runner required")
    base, new = fake_plan(tmp_path)
    batch = tmp_path / "batch"
    lengths = (63, 64, 65, 128, 129, 192)
    batch.write_text(
        "QWEN35_PROMPT_BATCH_V1\n" + "".join(f"p{n} " + ",".join(["2"] * n) + "\n" for n in lengths)
    )
    answers = []
    for plan in (base, new):
        out = tmp_path / (plan["sha256"] + ".json")
        cmd = [
            runner,
            "--model-kind",
            "chunk",
            "--mode",
            "dflash",
            "--model",
            plan["path"],
            "--model-sha256",
            plan["sha256"],
            "--prompt-batch",
            str(batch),
            "--prompt-batch-sha256",
            c.sha(batch),
            "--max-new-tokens",
            "24",
            "--max-draft-tokens",
            "15",
            "--pad-token-id",
            "0",
            "--eos-token-ids",
            "63",
            "--warmup",
            "0",
            "--repetitions",
            "2",
            "--low-memory",
            "--trace-rounds",
            "--output",
            str(out),
        ]
        sub = subprocess.run(cmd, capture_output=True, text=True)
        assert sub.returncode == 0, sub.stderr
        cases = []
        for n, case in zip(lengths, c.read(out)["cases"], strict=True):
            report = c.read(case["report"])
            assert report["abi"]["om_count"] == 4 + (plan == new)
            for m in report["dflash"]["measurements"]:
                times = m["stage_ms"]
                count = (n - 1) // 64
                assert len(times.get("draft_context64", [])) == (count if plan == new else 0)
                assert len(times["draft"]) == len(times["target_verify"]) + (0 if plan == new else count)
                cases.append((m["generated_token_ids"], m["rounds"], m["stop_reason"]))
        answers.append(cases)
    assert answers[0] == answers[1]


def test_prefill_savings_keep_decode_and_its_denominator():
    import copy

    prompt = dict(id="long", tokens=[3] * 129)
    m = dict(
        stage_ms=dict(draft=[16, 16, 15], target_verify=[51], target_prefill=[75] * 3),
        rounds=[dict(proposed_token_ids=[]), dict(proposed_token_ids=[1, 2])],
        latency_ms=dict(prefill=257, decode=66, model_total=323),
        generated_token_ids=[3, 1, 2],
        counters=dict(accepted_draft_tokens=2, drafted_tokens=2),
    )
    base = dict(
        status="PASS",
        cpu_fallback=False,
        device_id=0,
        model=dict(sha256="a"),
        prompt_token_ids=prompt["tokens"],
        eos_token_ids=[248044],
        limits=dict(max_new_tokens=128, max_draft_tokens=15),
        abi=dict(om_count=4),
        protocol=dict(warmup=1, repetitions=3, low_memory=True, round_trace_enabled=True),
        dflash=dict(measurements=[m] * 3),
    )
    new = copy.deepcopy(base)
    new["abi"].update(om_count=5, context_only_prefill=True)
    new_m = new["dflash"]["measurements"][0]
    new_m["stage_ms"].update(draft_context64=[2, 2], draft=[15])
    new_m["latency_ms"].update(prefill=229, model_total=295)
    old_stats = summarize(timings(base, prompt, base["model"], context=False))
    new_stats = summarize(timings(new, prompt, new["model"], context=True))
    assert old_stats["context"]["count"] == new_stats["context"]["count"] == 6
    assert old_stats["total"]["mean_ms"] - new_stats["total"]["mean_ms"] == 28
    assert old_stats["draft"] == new_stats["draft"] and new_stats["draft"]["mean_ms"] == 15
    assert new_stats["generation_tok_s"] == pytest.approx(3000 / 295)
    new_m["stage_ms"]["draft"].append(2)
    with pytest.raises(ValueError, match="dispatch"):
        timings(new, prompt, new["model"], context=True)
