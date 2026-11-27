"""Measure four-column convolution state selection in the six-OM DFlash receiver."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shlex
import shutil
import traceback

from winners import c, require, HERE
from experiment import Stages
from run_q import run_path
from run_prefill_head import compile_part
from run_workspace import archive_inputs, compare, memory
from diagnose_calls import receiver_runtime
from measure import launch
from panel_profiler import bind_msprof
from probe_dynamic import FAMILIES
from c_profile_graph import read_om_json
from conv_window_air import export
from conv_window_check import archived, boundaries, op_summary, profiled, validate_round


ROLES = {"body": "target_prefill", "verify": "target_verify"}


def replace_models(d, old_plan, before, after):
    rows = c.verify(old_plan).read_text().splitlines()
    for key, role in ROLES.items():
        (i,) = (i for i, line in enumerate(rows) if shlex.split(line)[:2] == ["graph", role])
        old, new = before[key], after[key]
        require(
            old["abi"] == new["abi"] and shlex.split(rows[i])[2:] == [old["om"]["path"], old["om"]["sha256"]],
            "previous OM or public ABI differs: " + role,
        )
        rows[i] = f"graph {role} {json.dumps(str(c.verify(new['om'])))} {new['om']['sha256']}"
    path = d / "chunk-plan.txt"
    path.write_text("\n".join([*rows, ""]))
    return c.identity(path)


def engines(model):
    counts = Counter()
    for n in read_om_json(c.verify(model["dump"])):
        if n["type"] in FAMILIES:
            attr = {a["key"]: a.get("value") or {} for a in n["attributes"]}
            counts[
                (
                    n["type"],
                    attr.get("_ge_attr_op_kernel_lib_name", {}).get("s"),
                    attr.get("custom_op_file_path", {}).get("s"),
                )
            ] += 1
    return counts


def run(out, args, state):
    base = run_path(out.parent, args.from_run, "prefill-head-")
    bench = run_path(out.parent, args.from_profile, "target-task-sample-")
    baseline = c.identity(base / "result.json")
    old = c.read(base / "result.json")
    require(
        old["status"] == "MEASURED"
        and old["same_workload"]
        and old["token_matches"] == old["round_matches"] == old["measurements"] == 60,
        "expected the output-matched split-Head 20-prompt receiver",
    )
    prior = Stages(out, base)
    control = prior.completed("generation")["result"]
    require(control["index"] == old["capture"], "previous full generation differs")
    tools = prior.completed("runners")["result"]
    runner = c.verify(tools["ac_om_runner"])
    old_plan = old["plan"]
    history = archive_inputs(out, control, old_plan)
    old_memory = memory(control, True, model_count=6)
    ws = c.read(c.verify(old["baseline"]))
    ctx = c.read(c.verify(ws["baseline"]))
    full = c.verify(ctx["baseline"]).parent
    previous = c.read(full / "result.json")
    target = c.read(full / "target.json")
    verify = target["models"]["target_verify"]
    vsrc = Stages(out, c.verify(verify["source_air"]).parent.parent).completed("reconstruct-target_verify")[
        "result"
    ]
    require(vsrc["air"] == verify["source_air"] and vsrc["abi"] == verify["abi"], "wrong original Verify AIR")
    sources = dict(body=prior.completed("split-air")["result"]["parts"]["body"], verify=vsrc)
    body = prior.completed("atc-body")["result"]
    require(body["om"] == old["models"]["body"] and body["abi"] == sources["body"]["abi"], "wrong body OM")
    original = dict(
        body=dict(
            body,
            source_air=sources["body"]["air"],
            argv=c.read(Path(body["om"]["path"]).parent / "atc-command.json"),
        ),
        verify=dict(verify, dump=c.read(c.verify(verify["cache_backend"]))["dump"]),
    )
    proof = prior.completed("target-block-check")["result"]
    cann = Path(os.environ["CANN_ROOT"]).resolve(strict=True)
    env, binding = receiver_runtime(full, previous, cann)
    atc = Path(shutil.which("atc", path=env["PATH"])).resolve(strict=True)
    desc = c.identity(out / "cann-ge.desc")
    require(desc["sha256"] == c.sha(base / "cann-ge.desc"), "different CANN GE schema")
    snapshots = {k: archived(bench / k, v["abi"]) for k, v in original.items()}
    validate_round(snapshots, c.read(c.verify(control["reports"]["zh_explain"])))
    previous_profiles = {k: op_summary(bench / k, 64 if k == "body" else 16) for k in ROLES}
    require(
        all(p["large_nodes"] == 24 for p in previous_profiles.values()), "wrong reference transpose profile"
    )
    cli = bind_msprof(out, cann, env)
    stages = Stages(out, run_path(out.parent, args.resume, "target-conv-window-") if args.resume else None)
    scripts = {
        n: c.sha(HERE / n)
        for n in (
            "run_conv_window.py",
            "conv_window_air.py",
            "conv_window_check.py",
            "run_prefill_head.py",
            "run_workspace.py",
            "measure.py",
            "context_measure.py",
        )
    }
    state.update(
        baseline=baseline,
        profiler_baseline=str(bench),
        formal_control_runs=0,
        runner=tools["qwen35_dflash_acl_runner"],
    )

    def stage(name, inputs, action):
        state["phase"] = name
        return stages.run(name, dict(baseline=baseline, code=scripts, inputs=inputs), action)

    models = {}
    for k, role in ROLES.items():
        source = sources[k]
        edited = stage("air-" + k, [source["air"], source["abi"], desc], lambda d: export(d, desc, source))
        models[k] = stage(
            "atc-" + k,
            [edited, original[k]["argv"], binding, c.identity(atc)],
            lambda d: compile_part(d, k, edited, original[k], atc, env, target["cache_provider"]["path"]),
        )
        counts = engines(original[k])
        require(
            set(op for op, _, _ in counts) == set(FAMILIES) and counts == engines(models[k]),
            "Target native family/engine/vendor changed: " + role,
        )
    state["models"] = {k: v["om"] for k, v in models.items()}
    plan = stage("plan", [old_plan, state["models"]], lambda d: replace_models(d, old_plan, original, models))
    state["plan"] = plan
    state["boundaries"] = stage(
        "state-check",
        [
            state["models"],
            original["verify"]["om"],
            prior.refs["target-block-check"],
            tools["ac_om_runner"],
            snapshots["verify"]["inputs"],
        ],
        lambda d: boundaries(d, runner, models, original["verify"], proof, snapshots["verify"], env),
    )
    profiles = {}
    for k, rows in (("body", 64), ("verify", 16)):
        s, ref = snapshots[k], previous_profiles[k]
        profiles[k] = stage(
            "profile-" + k,
            [models[k]["om"], s, ref["csv"], tools["ac_om_runner"], cli["file"]],
            lambda d: profiled(d, runner, models[k], s, rows, ref, cli, env),
        )
    state["profiles"] = profiles
    # Match the archived command and runner byte-for-byte apart from the plan and result paths.
    cmd = c.read(c.verify(control["command"]))
    exe = c.verify(tools["qwen35_dflash_acl_runner"])
    require(cmd[0] == str(exe), "different archived generation runner")
    for flag, key in (("--model", "path"), ("--model-sha256", "sha256")):
        index = cmd.index(flag) + 1
        require(cmd[index] == old_plan[key], "different archived generation plan")
        cmd[index] = plan[key]
    cap = stage(
        "generation",
        [plan, state["runner"], history["batch"]["sha256"], binding, cmd],
        lambda d: launch(d, history, plan, {"file": state["runner"]}, env, reference_command=cmd),
    )
    state["capture"] = cap["index"]
    state["workspace"] = memory(cap, True, model_count=6)
    print(
        f"Six-OM shared workspace {old_memory['shared_bytes']} -> {state['workspace']['shared_bytes']} bytes",
        flush=True,
    )
    metrics = stage(
        "comparison",
        [control["index"], cap["index"], plan],
        lambda d: compare(d, history["prompts"], control, cap, old_plan, old["groups"], window_plan=plan),
    )
    state.update(metrics)
    if not (metrics["same_workload"] and metrics["token_matches"] == metrics["round_matches"] == 60):
        return "MEASURED_WITH_DIFFERENCES"
    if any("candidate" not in p or p["candidate"]["unshaped_transposes"] for p in profiles.values()):
        return "MEASURED_PROFILE_INCOMPLETE"
    state["large_transposes_removed_in_both"] = all(p["candidate"]["large_nodes"] == 0 for p in profiles.values())
    return "MEASURED_AND_PROFILED"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-run", required=True)
    parser.add_argument("--from-profile", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    out = Path(os.environ["DFLASH_OUTPUT"])
    state = {"status": "RUNNING"}
    try:
        state["status"] = run(out, args, state)
    except Exception as error:
        state.update(status="FAILED", error=str(error))
        traceback.print_exc()
    c.write(out / "result.json", state)
    print(state["status"], "REPORT", out / "result.json", flush=True)
    raise SystemExit(state["status"] != "MEASURED_AND_PROFILED")
