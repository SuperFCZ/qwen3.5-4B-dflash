"""Reopen the actual direct-N64 winner, reference tensors and guarded runner without recompiling."""

from pathlib import Path

import kv_panel as kv
from kv_source import NZ, GN, PINS
from experiment import Stages
from bench import selected_image, read_om_json, embedded
from run_gate import pair_statistics


def winner(out, old, gear, choose):
    st = Stages(out, old)
    full = kv.c.read(old / "result.json")
    tag = str(gear)
    measured = full["gears"][tag]
    kv.require(
        full["status"] in ("MEASURED", "MEASURED_AND_PROFILED")
        and measured["selected"] == kv.DIRECT
        and measured["shape"] == list(kv.shape(gear)),
        "expected the measured first-pass direct KV winner",
    )
    src = st.completed("source-c" + tag)["result"]
    runner = st.completed("runner-c" + tag)["result"]
    pk = st.completed("build-" + kv.DIRECT)["result"]
    atc = st.completed("atc-c" + tag + "-" + kv.DIRECT)
    model = atc["result"]
    m, k, n = kv.shape(gear)
    kv.require(
        atc["contract"]["source"] == st.refs["source-c" + tag]
        and atc["contract"]["pkg"] == pk
        and src["kv_node"] == "proj_layer0_kv"
        and model["mode"] == kv.DIRECT
        and src["query"]["tiling_sha256"] == PINS[gear][0]
        and src["compiled_kv"]["executable_sections"] == [dict(bytes=PINS[gear][1], sha256=PINS[gear][2])]
        and [src["weights"][name]["sha256"] for name in ("w_nz.bin", "s_gn.bin")] == [NZ, GN],
        "crossed KV archive source, tiling or checkpoint",
    )
    kv.c.verify_tree(src["weights"])
    kv.c.verify_opp(pk["opp"])
    kv.c.verify(pk["source"])
    kv.c.verify(runner)
    kv.require(
        kv.plan(model["query"], src["query"], gear, kv.DIRECT) == model["contract"], "archived Cube drift"
    )
    nodes = read_om_json(kv.c.verify(model["dump"]))
    (node,) = (x for x in nodes if x["type"] == kv.OP)
    (image,) = embedded(kv.c.verify(model["om"]), [selected_image(node, 8, kv.OP)]).values()
    kv.require(
        image == model["selected"] and model["abi"] == kv.abi(gear), "different archived KV executable"
    )
    cases = {}
    for label in ("measured", "stress"):
        capture = st.completed(f"capture-c{gear}-{label}")
        kv.require(
            capture["contract"]["source"] == st.refs["source-c" + tag]
            and capture["contract"]["runner"] == runner
            and capture["contract"]["replay"] == src["replays"][label],
            "different archived KV replay",
        )
        cases[label] = capture["result"]["files"]
    cases["boundaries"] = st.completed("boundaries-c" + tag)["result"]
    kv.c.verify_tree(cases)
    kv.require(
        all(all(row[name] == v for name, v in src["weights"].items()) for row in cases.values()),
        "archived reference has other KV weights",
    )
    allowed = [v for v in kv.CANDIDATES if v not in measured["rejected"]]
    for mode in (kv.RAW, *allowed):
        name = f"atc-c{gear}-{mode}"
        bound = st.completed(name)["result"]
        sm = st.completed(f"smoke-c{gear}-{mode}")
        kv.require(
            sm["contract"]["runner"] == runner
            and sm["contract"]["cases"] == cases
            and sm["contract"]["model"] == bound["om"],
            "different guarded smoke",
        )

    def compare(a, b):
        key = a + "__" + b
        name = f"benchmark-c{gear}-" + key
        rec = st.completed(name)
        kv.require(
            rec["contract"]["smoke"] == [st.refs[f"smoke-c{gear}-{v}"] for v in (a, b)]
            and rec["contract"]["inputs"] == cases["measured"]
            and rec["contract"]["runner"] == runner
            and measured["pairs"][key]["report"] == st.refs[name],
            "different measured pair",
        )
        for row in rec["result"]["runs"]:
            execution = kv.c.read(kv.c.verify(row["execution"]))
            kv.require(
                Path(row["execution"]["path"]).parent == Path(row["folder"])
                and execution["status"] == "PASS"
                and execution["golden_match"] is True
                and execution["sync_completed"] is True
                and execution["cpu_fallback"] is False
                and execution["timing"]["execute_sync"]["samples_ms"]
                == row["validation"]["timing"]["samples_ms"],
                "changed archived KV samples",
            )
        stat = pair_statistics(
            rec["result"]["runs"], b, a, flops=2 * m * k * n, logical_bytes=kv.logical_bytes(gear)
        )
        kv.require(
            rec["result"]["statistics"] == stat and stat["decision"] == measured["pairs"][key]["decision"],
            "archived KV decision drift",
        )
        return stat

    kv.require(choose(allowed, compare) == kv.DIRECT, "archived winner failed recomputation")
    raw_record = st.completed(f"atc-c{gear}-{kv.RAW}")
    raw = raw_record["result"]
    (raw_node,) = (n for n in read_om_json(kv.c.verify(raw["dump"])) if n["type"] == kv.OLD_OP)
    (raw_image,) = embedded(kv.c.verify(raw["om"]), [selected_image(raw_node, 8, kv.OLD_OP)]).values()
    kv.require(
        raw_record["contract"]["pkg"] == src["pkg"]
        and raw_record["contract"]["source"] == st.refs["source-c" + tag]
        and raw_image == raw["selected"]
        and raw_image["executable_sections"] == src["compiled_kv"]["executable_sections"]
        and kv.plan(raw["query"], src["query"], gear, kv.RAW) == raw["contract"],
        "different archived raw KV",
    )
    return dict(
        raw=raw,
        source=src,
        package=pk,
        runner=runner,
        control=model,
        cases=cases,
        result=kv.c.identity(old / "result.json"),
        receipts=st.refs,
    )
