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
        return verified_pair(st, measured, runner, cases, gear, a, b)

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


def verified_pair(st, measured, runner, cases, gear, a, b):
    m, k, n = kv.shape(gear)
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


def wide_winner(out, old, gear, choose):
    kv.require(gear == 64, "fused A targets only C64")
    report = kv.c.read(old / "result.json")
    measured = report["gears"]["64"]
    chosen = "kv-n128-k1280"
    kv.require(
        report["status"] in ("MEASURED", "MEASURED_AND_PROFILED")
        and measured["selected"] == chosen
        and report["control"] == kv.DIRECT,
        "expected measured C64 N128/K1280 winner",
    )
    st = Stages(out, old)
    admission = st.completed("admit-c64")
    first = kv.c.verify(report["parent_result"]).parent
    kv.require(
        first.parent == old.parent
        and report["parent_run"] == str(first)
        and admission["contract"]["parent"] == report["parent_result"],
        "crossed original KV archive",
    )
    base = winner(out, first, 64, choose)
    kv.require(base == admission["result"], "different first-pass assets")
    pkg = st.completed("build-" + chosen)["result"]
    row = st.completed("atc-c64-" + chosen)
    model = row["result"]
    kv.require(
        row["contract"]["admit"] == st.refs["admit-c64"]
        and row["contract"]["width"] == 128
        and row["contract"]["build"] == st.refs["build-" + chosen]
        and model["mode"] == chosen
        and model["contract"] == kv.plan(model["query"], base["source"]["query"], 64, chosen),
        "different wide KV model or plan",
    )
    kv.c.verify_opp(pkg["opp"])
    kv.c.verify(pkg["source"])
    (node,) = (n for n in read_om_json(kv.c.verify(model["dump"])) if n["type"] == kv.OP)
    (image,) = embedded(kv.c.verify(model["om"]), [selected_image(node, 8, kv.OP)]).values()
    kv.require(image == model["selected"], "different embedded N128 KV image")
    admitted = [mode for mode in kv.WIDE if mode not in measured["rejected"]]
    for mode in (kv.DIRECT, kv.RAW, *admitted):
        om = (
            base["control" if mode == kv.DIRECT else "raw"]["om"]
            if mode in (kv.DIRECT, kv.RAW)
            else st.completed("atc-c64-" + mode)["result"]["om"]
        )
        sm = st.completed("smoke-c64-" + mode)["contract"]
        kv.require(
            sm["model"] == om and sm["cases"] == base["cases"] and sm["runner"] == base["runner"],
            "crossed wide KV smoke",
        )

    def compare(a, b):
        return verified_pair(st, measured, base["runner"], base["cases"], 64, a, b)

    kv.require(
        choose(admitted, compare, kv.DIRECT) == chosen and compare(kv.RAW, chosen)["decision"] == "GO",
        "wrong remeasured N128 winner",
    )
    return dict(base, control=model, package=pkg, result=kv.c.identity(old / "result.json"), receipts=st.refs)
