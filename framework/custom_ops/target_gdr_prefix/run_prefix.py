"""Measure the deployed GDR commit over real Verify operands and test alternate short-prefix paths."""

import argparse
import os
from pathlib import Path
import re
import shutil
import traceback

from gdr_graph import ROOT, sites, prefix_graph, isolated
from gdr_measure import (
    state_files,
    pass_times,
    full_profile,
    length_files,
    freeze,
    equal_state,
    pair_times,
    histogram,
)
from pair_graph import c, require, load_model, save_graph
from pair_reference import deployment, compile_om
from experiment import Stages
from panel_profiler import bind_msprof
from run_prefill_head import capture
from run_q import run_path


def source_outline(env):
    """Report interface/phase boundaries of the selected installation, without saving its source."""
    dirs = [Path(v) for v in env["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep) if v]
    paths = [
        f for d in dirs for f in (d / "op_impl/ai_core/tbe").glob("*/dynamic/chunk_gated_delta_rule.cpp")
    ]
    require(len(paths) == 1, "expected one effective-length GDR source")
    src = paths[0]
    text = src.read_text()
    lines = text.splitlines()
    print(
        "GDR source",
        c.sha(src),
        "tiling fields",
        sorted(set(re.findall(r"tiling\s*->\s*(\w+)", text))),
        "headers",
        re.findall(r"^\s*#include\s+([^\n]+)", text, re.M)[:12],
        flush=True,
    )
    # Emit dataflow labels for just the bounded chunk readout/state interval. The
    # build and all generated vendor-dependent artifacts stay on the receiver.
    if c.sha(src) == "93bfea89114a408b3e189b25207de157f4eb155b34d1e7500164ff627ea9c601":
        labels = []
        for i in range(1247, 1601):
            s = lines[i - 1].split("//", 1)[0].strip()
            call = re.match(r"(\w+)(?:<[^;()]+>)?\s*\((.*)", s)
            if call and call[1] not in ("if", "for", "PipeBarrier", "SetFlag", "WaitFlag"):
                # Tensor identifiers suffice to see where buffers are repurposed.
                operands = re.findall(
                    r"\b(?:ub_temp(?:_half_)?\d+|[qkv]Global|[a-zA-Z_]+Global|[a-zA-Z]*L[01][A-C]_[a-z]+|lsGlobal|isGlobal)\b",
                    call[2],
                )
                labels.append(f"{i}:{call[1]}(" + ",".join(operands[:4]) + ")")
        for i in range(0, len(labels), 12):
            print("chunk phases", " ".join(labels[i : i + 12]), flush=True)
        # Include the kernel class template argument and workspace handoff, not its implementation.
        print(
            "entry",
            [
                (i + 1, s.strip()[:140])
                for i, s in enumerate(lines)
                if i >= 1882
                and any(
                    k in s
                    for k in (
                        "Kernel<",
                        "op.Init",
                        "Init(",
                        "GetUserWorkspace",
                        "Process(",
                        "GET_TILING_DATA",
                    )
                )
            ][:14],
            flush=True,
        )
        print(
            "workspace bindings",
            [(i + 1, s.strip()[:140]) for i, s in enumerate(lines[243:253], 243)],
            flush=True,
        )
    return c.identity(src)


def run(out, args, state):
    base = run_path(out.parent, args.from_run, "target-conv-window-")
    old = c.read(base / "result.json")
    cann = Path(os.environ["CANN_ROOT"]).resolve(strict=True)
    assets, env = deployment(out, base, cann)
    model = assets["models"]["verify"]
    runner = c.verify(assets["tools"]["ac_om_runner"])
    atc = Path(shutil.which("atc", path=env["PATH"])).resolve(strict=True)
    desc = c.identity(out / "cann-ge.desc")
    src = load_model(desc, model["air"])
    site_map = sites(src, model["abi"])
    require(len(site_map) == 24, "expected 24 GDN layers")
    src0 = site_map["t0_recurrent"]
    samples, csv = state_files(out, base, assets, old)
    prev = Stages(out, base)
    hist = histogram(old, prev.completed("generation")["result"])
    stages = Stages(out, run_path(out.parent, args.resume, "target-gdr-prefix-") if args.resume else None)

    def stage(name, key, action):
        state["phase"] = name
        return stages.run(name, dict(base=assets["baseline"], inputs=key), action)

    graph_code = c.sha(ROOT / "gdr_graph.py")
    meas = c.sha(ROOT / "gdr_measure.py")
    cli = bind_msprof(out, cann, env)
    state.update(baseline=assets["baseline"], histogram=hist, native_source=source_outline(env))
    full = {}
    for length in (1, 3, 16):
        v = (
            pass_times(csv, site_map)
            if length == 16
            else stage(
                f"native-full-r{length}",
                [model["om"], samples[length], cli["file"], meas],
                lambda d: full_profile(d, length, model, samples[length], site_map, runner, env, cli),
            )
        )
        full[str(length)] = v
        print(
            f"Native full M16, committed {length}: first {v['first']['task_sum_ms']:.3f} ms, "
            f"24 commit tasks {v['commit']['task_sum_ms']:.3f} ms; 24+24 matched twice",
            flush=True,
        )
    state["native_full_profiles"] = full

    def make_prefix(d):
        m, abi = prefix_graph(src, model["abi"], src0)
        return save_graph(d, m, abi)

    graph = stage("air-operands", [model["air"], graph_code], make_prefix)
    prefix = stage(
        "atc-operands",
        [graph["air"], c.identity(atc), assets["binding"]],
        lambda d: compile_om(d, graph, atc, env),
    )
    ins = {s["name"]: samples[16]["inputs"][s["name"]] for s in prefix["abi"]["inputs"]}

    def collect(d):
        values, mods = capture(d / "device", runner, prefix, ins, env)
        require(not mods, "prefix modified state inputs")
        return values

    operands = dict(
        stage("real-operands", [prefix["om"], ins], collect), initial=samples[16]["inputs"]["t0_recurrent"]
    )
    state["operands"] = operands
    models = {}
    for label, chunk in (("chunk64", 64), ("recurrent1", 1)):
        graph = stage(
            "air-" + label,
            [model["air"], src0, chunk, graph_code],
            lambda d: save_graph(d, *isolated(src, src0, chunk)),
        )
        try:
            models[label] = stage(
                "atc-" + label,
                [graph["air"], c.identity(atc), assets["binding"]],
                lambda d: compile_om(d, graph, atc, env, dump=True),
            )
        except (RuntimeError, ValueError) as e:
            if label == "chunk64":
                raise
            print(label, "unavailable", e, flush=True)
    golden, inputs, trial = {}, {}, {}
    for length in range(1, 17):
        key = str(length)
        ins = stage(f"inputs-r{length}", [operands, length], lambda d: length_files(d, operands, length))
        inputs[key] = ins
        ref = stage(
            f"native-state-r{length}",
            [models["chunk64"]["om"], ins],
            lambda d: freeze(d, models["chunk64"], ins, runner, env),
        )
        if length in samples:
            want = samples[length]["golden"]["t0_recurrent"]
            require(
                ref["sha256"] == want["sha256"],
                f"isolated state disagrees with sealed full Verify at r={length}",
            )
        golden[key] = ref
        if "recurrent1" in models:
            try:
                val = stage(
                    f"recurrent1-state-r{length}",
                    [models["recurrent1"]["om"], ins],
                    lambda d: freeze(d, models["recurrent1"], ins, runner, env),
                )
                trial[key] = equal_state(val, ref)
            except (ValueError, RuntimeError) as e:
                trial[key] = dict(exact=False, error=str(e))
            print(f"Native chunk_size=1 vs 64, r={length}: {trial[key]}", flush=True)
    times = {}
    for length in (1, 2, 3, 4, 8, 16):
        key = str(length)
        if trial.get(key, {}).get("exact"):
            times[key] = stage(
                f"recurrent1-pair-r{length}",
                [models, inputs[key], golden[key], meas],
                lambda d: pair_times(
                    d,
                    models["recurrent1"],
                    models["chunk64"],
                    inputs[key],
                    golden[key],
                    runner,
                    assets["workspace"],
                    env,
                    env,
                ),
            )
            a, b = times[key]["native_us"], times[key]["candidate_us"]
            print(
                f"Exact recurrent r={length} single-layer Execute+Sync median {a['p50']:.2f} -> {b['p50']:.2f} us",
                flush=True,
            )
    state.update(
        models=models, golden=golden, inputs=inputs, recurrent_comparison=trial, recurrent_times=times
    )
    return "NATIVE_PREFIX_CHARACTERIZED"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-run", required=True)
    p.add_argument("--resume")
    a = p.parse_args()
    out = Path(os.environ["DFLASH_OUTPUT"])
    s = {"status": "RUNNING"}
    try:
        s["status"] = run(out, a, s)
    except Exception as e:
        s.update(status="FAILED", error=str(e))
        traceback.print_exc()
    c.write(out / "result.json", s)
    print(s["status"], "REPORT", out / "result.json", flush=True)
    raise SystemExit(s["status"] == "FAILED")
