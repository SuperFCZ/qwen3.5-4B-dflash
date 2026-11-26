"""Measure full generation with the original Target vocabulary Head invoked only for the final block."""

import argparse
from collections import Counter
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import traceback

import numpy as np

from winners import c, require, HERE, OPS, REPO
from prefill_split import ROW, export
from experiment import Stages
from run_q import run_path
from run_context import build_tools
from run_workspace import archive_inputs, memory, compare
from diagnose_calls import receiver_runtime
from probe_dynamic import inventory
from cache_backend import audit_compiled
from full_replay import prepare_case
from package import command
from measure import launch


def implementation(fn):
    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()


def compile_part(d, side, part, original, atc, env, cache_vendor):
    args = list(original["argv"])
    require(
        Path(args[0]).resolve() == atc
        and args.count("--model=" + original["source_air"]["path"]) == 1
        and "--soc_version=Ascend310P3" in args
        and "--precision_mode=must_keep_origin_dtype" in args,
        "different original Target ATC or precision",
    )
    c.verify_tree(part["payloads"])
    air = c.verify(part["air"])
    prefix = d / ("target_prefill_" + side)
    (output,) = (x for x in args if x.startswith("--output="))
    args[args.index("--model=" + original["source_air"]["path"])] = "--model=" + str(air)
    args[args.index(output)] = "--output=" + str(prefix)
    cache = d / "cache"
    cache.mkdir()
    e = dict(env, ASCEND_CACHE_PATH=str(cache), ASCEND_WORK_PATH=str(d))
    c.write(d / "atc-command.json", args)
    command(args, d / "atc.log", e, air.parent)
    om = c.identity(prefix.with_suffix(".om"))
    if side == "body":
        backend = audit_compiled(d, "target_prefill", Path(om["path"]), atc, e, cache_vendor)
        dump = c.read(c.verify(backend))["dump"]
    else:
        path = d / "compiled-target.json"
        command([atc, "--mode=1", "--om=" + om["path"], "--json=" + str(path)], d / "inspect.log", e, d)
        dump = c.identity(path)
    return dict(om=om, dump=dump, abi=part["abi"])


def audit_parts(d, original, parts):
    ref = c.read(c.verify(original["cache_backend"]))
    before, _ = inventory(dict(om=original["om"], dump=ref["dump"]))
    bodies, bc = inventory(parts["body"])
    heads, hc = inventory(parts["head"])

    def backend(records):
        return Counter((v["role"], v["vendor"], v["engine"]) for v in records.values())

    require(
        backend(bodies) + backend(heads) == backend(before),
        "changed Target provider, native family or engine",
    )
    require(
        Counter(v["role"] for v in heads.values()) == {"DynamicQuant": 1, "QuantBatchMatmulV4444": 1}
        and any("argmax" in k.lower() for k in hc)
        and not any("argmax" in k.lower() for k in bc),
        "compiled vocabulary Head was not isolated",
    )
    evidence = {}
    for label, selected in (("body", bodies), ("head", heads)):
        known = sum(
            bool(v["text"])
            and n in before
            and v["text"] == before[n]["text"]
            and v["block"] == before[n]["block"]
            for n, v in selected.items()
        )
        evidence[label] = dict(
            nodes=len(selected),
            same_named_executable=known,
            families=dict(Counter(v["role"] for v in selected.values())),
        )
    print(
        "Compiled Target native provider counts match. Body/head vocabulary projections 0/1; "
        f"body/head quantized matmuls {bc.get('QuantBatchMatmulV4444')}/{hc.get('QuantBatchMatmulV4444')}",
        flush=True,
    )
    c.write(d / "providers.json", evidence)
    return evidence


def write_plan(d, old_plan, original, parts):
    lines = c.verify(old_plan).read_text().splitlines()
    (start,) = (i for i, line in enumerate(lines) if shlex.split(line)[:2] == ["graph", "target_prefill"])
    end = lines.index("end", start)
    require(
        shlex.split(lines[start])[2:] == [original["om"]["path"], original["om"]["sha256"]]
        and lines[-1] == "done",
        "baseline Prefill binding differs",
    )

    def ports(abi):
        return [
            " ".join(map(str, (mark, v["name"], v["dtype"], len(v["shape"]), *v["shape"])))
            for mark, k in (("I", "inputs"), ("O", "outputs"))
            for v in abi[k]
        ]

    require(lines[start + 1 : end] == ports(original["abi"]), "baseline Prefill ABI differs")

    def block(name, part):
        om = part["om"]
        return [
            "graph " + name + " " + json.dumps(str(c.verify(om))) + " " + om["sha256"],
            *ports(part["abi"]),
            "end",
        ]

    lines[start : end + 1] = block("target_prefill", parts["body"])
    lines[-1:] = [*block("target_prefill_head", parts["head"]), "done"]
    path = d / "chunk-plan.txt"
    path.write_text("\n".join([*lines, ""]))
    return c.identity(path)


def capture(folder, exe, model, files, env):
    abi = model["abi"]
    app = folder / "case"
    require(set(files) == {v["name"] for v in abi["inputs"]}, "missing reference input")
    prepare_case(app, abi, {"files": files}, None)
    command(
        [exe, "0", c.verify(model["om"]), app, app / "inputs", "0", "huge-first", "normal", "--capture"],
        folder / "capture.log",
        env,
        folder,
    )
    e = c.read(app / "execution.json")
    require(
        e["status"] == "PASS"
        and not e["cpu_fallback"]
        and e["reference_capture"]
        and all(e[k] for k in ("guards", "readonly", "repeat"))
        and e["timing"]["repetitions"] == 0,
        "invalid two-poison device capture",
    )
    return {v["name"]: c.identity(app / (v["name"] + "-actual-0.bin")) for v in abi["outputs"]}


def check_blocks(d, tokens, exe, original, parts, env):
    require(len(tokens) >= 129, "real prompt needs two complete blocks and a partial last block")
    abi = original["abi"]
    wanted = {v["name"]: v for v in abi["inputs"]}
    state = {}
    for name, spec in wanted.items():
        if name in ("input_ids", "start_position", "valid_rows"):
            continue
        p = d / (name + "-initial.bin")
        with p.open("xb") as f:
            f.truncate(math.prod(spec["shape"]) * np.dtype(spec["dtype"]).itemsize)
        state[name] = c.identity(p)
    require(set(state) == {v["name"] for v in abi["outputs"][2:]}, "Target state inventory differs")
    proofs = []
    for pos, valid in ((0, 64), (64, 64), (128, 1)):
        block = d / f"block-{pos}"
        block.mkdir()
        ids = list(tokens[pos : pos + valid]) + [0] * (64 - valid)
        files = dict(state)
        for name, values in (("input_ids", ids), ("start_position", [pos]), ("valid_rows", [valid])):
            spec = wanted[name]
            p = block / (name + ".bin")
            np.asarray(values, dtype=np.dtype(spec["dtype"]).newbyteorder("<")).reshape(spec["shape"]).tofile(
                p
            )
            files[name] = c.identity(p)
        old = capture(block / "original", exe, original, files, env)
        body = capture(block / "body", exe, parts["body"], files, env)
        head = capture(block / "head", exe, parts["head"], {ROW: body[ROW]}, env)
        same = {**{k: v for k, v in body.items() if k != ROW}, **head}
        require(set(same) == set(old), "missing recovered Target result")
        errors = [k for k in old if old[k]["sha256"] != same[k]["sha256"]]
        require(not errors, f"Target outputs differ at position {pos} valid={valid}: {errors}")
        state = {k: body[k] for k in state}
        proofs.append(dict(start=pos, valid=valid, reference=old, candidate=same, hidden=body[ROW]))
        print(
            f"Target block start={pos} valid={valid}: Top1, features and all {len(state)} complete states bitwise PASS",
            flush=True,
        )
    return proofs


def run(out, args, state):
    base = run_path(out.parent, args.from_run, "workspace-e2e-")
    baseline = c.identity(base / "result.json")
    result = c.read(c.verify(baseline))
    require(
        result["status"] == "MEASURED"
        and result["token_matches"] == result["round_matches"] == 60
        and result["same_workload"]
        and result["policies"] == dict(shared_workspace="HUGE_FIRST", io="NORMAL_ONLY"),
        "use the output-matched workspace-only full-generation baseline",
    )
    plan = result["plan"]
    c.verify(plan)
    control = Stages(out, base).completed("generation")["result"]
    require(control["index"] == result["capture"], "wrong baseline batch")
    history = archive_inputs(out, control, plan)
    oldmem = memory(control, True)
    ctx = c.read(c.verify(result["baseline"]))
    original = c.verify(ctx["baseline"]).parent
    underlying = c.read(original / "result.json")
    target = c.read(original / "target.json")
    src = target["models"]["target_prefill"]
    source_root = c.verify(src["source_air"]).parent.parent
    receipt = Stages(out, source_root).completed("reconstruct-target_prefill")["result"]
    require(
        receipt["air"] == src["source_air"]
        and receipt["abi"] == src["abi"]
        and target["models"] == underlying["targets"],
        "wrong Target Prefill AIR/ABI or providers",
    )
    cann = Path(os.environ["CANN_ROOT"]).resolve(strict=True)
    env, binding = receiver_runtime(original, underlying, cann)
    atc = Path(shutil.which("atc", path=env["PATH"])).resolve(strict=True)
    desc = c.identity(out / "cann-ge.desc")
    stages = Stages(out, run_path(out.parent, args.resume, "prefill-head-") if args.resume else None)

    def stage(name, key, action):
        state["phase"] = name
        return stages.run(name, dict(baseline=baseline, inputs=key), action)

    state.update(baseline=baseline, previous_plan=plan, baseline_generation_executions=0)
    air = stage(
        "split-air",
        [receipt["air"], receipt["payloads"], desc["sha256"], c.sha(HERE / "prefill_split.py")],
        lambda d: export(d, desc, receipt),
    )
    parts = {}
    for side in ("head", "body"):
        part = air["parts"][side]
        parts[side] = stage(
            "atc-" + side,
            [
                part["air"],
                part["payloads"],
                binding,
                c.identity(atc),
                src["argv"],
                implementation(compile_part),
            ],
            lambda d: compile_part(d, side, part, src, atc, env, target["cache_provider"]["path"]),
        )
    state["models"] = {k: v["om"] for k, v in parts.items()}
    stage(
        "compiled-audit",
        [state["models"], src["cache_backend"], implementation(audit_parts)],
        lambda d: audit_parts(d, src, parts),
    )
    candidate_plan = stage(
        "plan",
        [plan, state["models"], air["proof"], implementation(write_plan)],
        lambda d: write_plan(d, plan, src, parts),
    )
    cpp = REPO / "framework/runtime/cpp"
    paths = [
        *cpp.glob("src/*.cpp"),
        *cpp.glob("include/qwen35_dflash/*.hpp"),
        cpp / "CMakeLists.txt",
        OPS / "draft_ac/test/runner.cpp",
        OPS / "draft_ac/test/CMakeLists.txt",
        OPS / "draft_quant/test/runner_common.h",
    ]
    tools = stage(
        "runners", [binding, {str(p): c.sha(p) for p in paths}], lambda d: build_tools(d, cann, env)
    )
    long_prompt = next(p["tokens"] for p in history["prompts"] if len(p["tokens"]) >= 129)
    stage(
        "target-block-check",
        [
            long_prompt[:129],
            tools["ac_om_runner"],
            src["om"],
            state["models"],
            implementation(check_blocks),
            implementation(capture),
        ],
        lambda d: check_blocks(d, long_prompt, c.verify(tools["ac_om_runner"]), src, parts, env),
    )
    ref = c.read(c.verify(control["command"]))
    for flag, value in (("--model", "path"), ("--model-sha256", "sha256")):
        i = ref.index(flag) + 1
        require(ref[i] == plan[value], "different historical generation plan")
        ref[i] = candidate_plan[value]
    cap = stage(
        "generation",
        [
            candidate_plan,
            history["batch"]["sha256"],
            tools["qwen35_dflash_acl_runner"],
            binding,
            stages.refs["target-block-check"],
            c.sha(HERE / "measure.py"),
        ],
        lambda d: launch(
            d,
            history,
            candidate_plan,
            {"file": tools["qwen35_dflash_acl_runner"]},
            env,
            reference_command=ref,
        ),
    )
    mem = memory(cap, True, model_count=6)

    def weight_bytes(m, role):
        s = c.verify(m["log"]).read_text()
        (v,) = re.findall(rf"om-memory graph={role} query_status=0 weight_bytes=(\d+)", s)
        return int(v)

    ow = weight_bytes(oldmem, "target_prefill")
    bw = weight_bytes(mem, "target_prefill")
    hw = weight_bytes(mem, "target_prefill_head")
    print(
        f"Prefill OM weights {ow:,} -> {bw:,} + {hw:,} bytes; shared workspace {oldmem['shared_bytes']:,} -> {mem['shared_bytes']:,} bytes",
        flush=True,
    )
    state.update(
        plan=candidate_plan,
        workspace=mem,
        capture=cap["index"],
        prefill_weight_bytes=dict(previous=ow, body=bw, head=hw),
    )
    metrics = stage(
        "comparison",
        [
            control["index"],
            cap["index"],
            c.sha(HERE / "context_measure.py"),
            c.sha(HERE / "run_workspace.py"),
        ],
        lambda d: compare(
            d, history["prompts"], control, cap, plan, result["groups"], split_plan=candidate_plan
        ),
    )
    state.update(metrics)
    return (
        "MEASURED"
        if metrics["same_workload"]
        and metrics["token_matches"] == metrics["round_matches"] == metrics["measurements"]
        else "MEASURED_WITH_OUTPUT_OR_TRACE_DIFFERENCES"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-run", required=True)
    p.add_argument("--resume")
    args = p.parse_args()
    out = Path(os.environ["DFLASH_OUTPUT"])
    state = {"status": "RUNNING"}
    try:
        state["status"] = run(out, args, state)
    except Exception as error:
        state.update(status="FAILED", error=str(error))
        traceback.print_exc()
    c.write(out / "result.json", state)
    print(state["status"], "REPORT", out / "result.json", flush=True)
    raise SystemExit(state["status"] != "MEASURED")
