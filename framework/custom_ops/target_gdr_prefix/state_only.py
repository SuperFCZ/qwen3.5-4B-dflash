"""Validate a state-only Chunk commit, profile it in Verify and compare the 20 real generations."""

import argparse
from collections import Counter
import os
from pathlib import Path
import shutil
import traceback

import numpy as np

from gdr_graph import ROOT, GDR, COMMIT, specs, isolated, replace_commit, sites
from pair_graph import c, require, load_model, save_graph
from pair_reference import deployment, compile_om
from pair_kernel import use_operator
from gdr_package import build
from gdr_measure import freeze, equal_state, length_files, state_files, pair_times, full_profile, histogram
from run_conv_window import replace_models
from run_prefill_head import compile_part
from run_workspace import archive_inputs, compare, memory
from measure import launch
from c_profile_graph import read_om_json
from experiment import Stages
from panel_profiler import bind_msprof
from run_q import run_path


def stress(d):
    rng = np.random.default_rng(310303)
    records = {}
    for s in specs()[:-1]:
        name = s["name"]
        a = rng.normal(0, 0.08 if name == "initial" else 0.45, s["shape"]).astype("<f4")
        if name == "g":
            a = -np.abs(a)
        if name == "beta":
            a = rng.uniform(0, 1, s["shape"]).astype("<f4")
            a[:, ::3, 0] = 0
            a[:, ::4, 1] = 1
        if name in ("key", "query"):
            a[:, :, 1::2] = a[:, :, ::2]
        p = d / (name + ".bin")
        a.astype(np.dtype(s["dtype"]).newbyteorder("<")).tofile(p)
        records[name] = c.identity(p)
    return records


def inventory(model):
    out = {}
    for n in read_om_json(c.verify(model["dump"])):
        if n["type"] not in (GDR, COMMIT, "CacheUpdate"):
            continue
        a = {v["key"]: v.get("value") or {} for v in n["attributes"]}
        name = n["name"]
        require(name not in out, "ambiguous compiled node " + name)
        out[name] = dict(
            kind=n["type"],
            engine=a.get("_ge_attr_op_kernel_lib_name", {}).get("s"),
            vendor=a.get("custom_op_file_path", {}).get("s"),
            blocks=a.get("tvm_blockdim", {}).get("i"),
            binary=a.get("_kernel_bin_id", {}).get("s", "")[-64:],
        )
    return out


def verify_bindings(old, new, mapping, vendor):
    a, b = inventory(old), inventory(new)
    kinds = Counter(n["kind"] for n in b.values())
    require(kinds == {GDR: 24, COMMIT: 24, "CacheUpdate": 256}, "wrong compiled GDR/cache inventory")
    identical = 0
    for v in mapping.values():
        first = a[v["first"]]
        same = b[v["first"]]
        commit = b[v["new"]]
        require(
            first["vendor"] == same["vendor"]
            and same["engine"] == "AIcoreEngine"
            and commit["engine"] == "AIcoreEngine"
            and (commit["blocks"] is None or int(commit["blocks"]) == 8)
            and (
                not commit["vendor"]
                or Path(commit["vendor"]).resolve().is_relative_to(Path(vendor).resolve())
            ),
            "wrong compiled GDR provider or engine",
        )
        identical += bool(first["binary"] and first["binary"] == same["binary"])
    return dict(counts=dict(kinds), first_same_binary=identical)


def run(out, args, state):
    parent = run_path(out.parent, args.from_probe, "target-gdr-prefix-")
    check = c.identity(parent / "result.json")
    ref = c.read(c.verify(check))
    require(
        ref["status"] == "NATIVE_PREFIX_CHARACTERIZED" and set(ref["golden"]) == set(map(str, range(1, 17))),
        "need the completed 16-length native reference run",
    )
    base = c.verify(ref["baseline"]).parent
    orig = c.read(base / "result.json")
    cann = Path(os.environ["CANN_ROOT"]).resolve(strict=True)
    assets, env = deployment(out, base, cann)
    require(assets["baseline"] == ref["baseline"], "different full Verify baseline")
    impl = c.verify(ref["native_source"])
    require(
        any(
            impl.is_relative_to(Path(d).resolve())
            for d in env["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep)
            if d
        ),
        "different GDR provider",
    )
    desc = c.identity(out / "cann-ge.desc")
    model = assets["models"]["verify"]
    receiver = load_model(desc, model["air"])
    location = sites(receiver, model["abi"])["t0_recurrent"]
    native = ref["models"]["chunk64"]
    runner = c.verify(assets["tools"]["ac_om_runner"])
    require(
        c.verify(native["om"]).is_file()
        and native["abi"]["outputs"] == [dict(name="state", dtype="float32", shape=[1, 32, 128, 128])],
        "wrong native single-layer reference",
    )
    atc = Path(shutil.which("atc", path=env["PATH"])).resolve(strict=True)
    st = Stages(out, run_path(out.parent, args.resume, "target-gdr-state-") if args.resume else None)

    def stage(name, inputs, action):
        state["phase"] = name
        return st.run(name, dict(reference=check, inputs=inputs), action)

    script = c.sha(ROOT / "state_only.py")
    gc = c.sha(ROOT / "gdr_graph.py")
    mc = c.sha(ROOT / "gdr_measure.py")
    code = {
        str(f.relative_to(ROOT)): c.sha(f)
        for f in [
            ROOT / (COMMIT + ".json"),
            ROOT / "kernel_entry.inc",
            ROOT / "gdr_package.py",
            *sorted((ROOT / "op_host").iterdir()),
        ]
    }
    pkg = stage(
        "private-operator", [code, ref["native_source"]], lambda d: build(d, ref["native_source"], cann, env)
    )
    private = use_operator(env, pkg)
    state.update(
        reference=check,
        baseline=assets["baseline"],
        package=pkg,
        histogram=ref["histogram"],
        formal_control_runs=0,
    )
    print("Private commit specialization", pkg["proof"], flush=True)

    stress_data = stage("stress-inputs", [script, 16], stress)
    data = {"real": ref["inputs"], "stress": {}}
    gold = {"real": ref["golden"], "stress": {}}
    for r in range(1, 17):
        k = str(r)
        data["stress"][k] = stage(
            f"stress-input-r{r}", [stress_data, r, mc], lambda d: length_files(d, stress_data, r)
        )
        gold["stress"][k] = stage(
            f"stress-native-r{r}",
            [native["om"], data["stress"][k]],
            lambda d: freeze(d, native, data["stress"][k], runner, env),
        )
    models = {}
    errors = {}
    cloned = False
    for name, level in (("clone", 0), ("terminal", 1), ("readout", 2), ("short-mask", 3)):
        air = stage(
            "air-" + name,
            [model["air"], level, gc],
            lambda d: save_graph(d, *isolated(receiver, location, custom=True, level=level)),
        )
        candidate = stage(
            "atc-" + name,
            [air["air"], pkg, assets["binding"], c.identity(atc)],
            lambda d: compile_om(d, air, atc, private, dump=True),
        )
        nodes = inventory(candidate)
        require(
            len(nodes) == 1
            and next(iter(nodes.values()))["kind"] == COMMIT
            and next(iter(nodes.values()))["engine"] == "AIcoreEngine",
            "state candidate did not compile on AI Core",
        )
        failures = {}
        checks = 0
        for kind, files in data.items():
            for r in (1, 3, 16) if name == "clone" else range(1, 17):
                k = str(r)
                got = stage(
                    f"{name}-{kind}-r{r}",
                    [candidate["om"], files[k], mc],
                    lambda d: freeze(d, candidate, files[k], runner, private),
                )
                diff = equal_state(got, gold[kind][k])
                checks += 1
                if not diff["exact"]:
                    failures[f"{kind}/r{r}"] = diff
        errors[name] = failures
        print(
            f"{name}: {checks - len(failures)}/{checks} complete FP32 states bitwise matched; "
            f"first difference {next(iter(failures.items()), None)}",
            flush=True,
        )
        if name == "clone":
            cloned = not failures
        elif not failures:
            models[name] = (candidate, level)
    state["state_differences"] = errors
    state["clone_exact"] = cloned
    if not models:
        return "NO_BITWISE_CANDIDATE"

    hist = {int(k): v for k, v in ref["histogram"].items()}
    timing = {}
    score = {}
    for name, (cand, level) in models.items():
        ts = {}
        for r in (1, 2, 3, 4, 8, 16):
            k = str(r)
            ins = data["real"][k]
            g = gold["real"][k]
            z = stage(
                f"pair-{name}-r{r}",
                [cand["om"], native["om"], ins, g, mc],
                lambda d: pair_times(d, cand, native, ins, g, runner, assets["workspace"], env, private),
            )
            ts[k] = z
            a, b = z["native_us"], z["candidate_us"]
            print(
                f"{name} r={r} single-layer Execute+Sync Median/P95 {a['p50']:.2f}/{a['p95']:.2f} -> "
                f"{b['p50']:.2f}/{b['p95']:.2f} us; ratios {[round(v, 3) for v in z['round_ratios']]}",
                flush=True,
            )
        # Use the observed distribution of the short-prefix population where every length was timed.
        score[name] = sum(hist[r] * ts[str(r)]["candidate_us"]["p50"] for r in range(1, 5)) / sum(
            hist[r] * ts[str(r)]["native_us"]["p50"] for r in range(1, 5)
        )
        timing[name] = ts
    best = min(score, key=score.get)
    level = models[best][1]
    state.update(single_layer=timing, short_relative_time=score, selected=best)
    print("Selected state kernel", best, "short-prefix relative time", score[best], flush=True)

    prior = Stages(out, base)
    original_verify = prior.completed("atc-verify")["result"]
    air0 = prior.completed("air-verify")["result"]
    require(
        original_verify["om"] == model["om"] and air0["air"] == model["air"], "different Verify artifacts"
    )
    srcinfo = dict(
        source_air=model["air"], argv=c.read(Path(model["om"]["path"]).parent / "atc-command.json")
    )

    def export(d):
        src = load_model(desc, model["air"])
        mapping = replace_commit(src, model["abi"], level)
        require(len(mapping) == 24, "wrong replacement inventory")
        air = save_graph(d, src, model["abi"])
        return dict(air, payloads=air0["payloads"], mapping=mapping)

    new_air = stage("air-full-verify", [model["air"], level, gc], export)
    (scatter,) = set(
        v["vendor"] for v in inventory(model).values() if v["kind"] == "CacheUpdate" and v["vendor"]
    )
    new = stage(
        "atc-full-verify",
        [new_air["air"], pkg, c.identity(atc), assets["binding"], srcinfo],
        lambda d: compile_part(d, "verify", new_air, srcinfo, atc, private, scatter),
    )
    state["verify"] = new["om"]
    state["compiled"] = verify_bindings(model, new, new_air["mapping"], pkg["vendor"])
    print("Compiled Verify", state["compiled"], flush=True)
    cli = bind_msprof(out, cann, private)
    samples, _ = state_files(out, base, assets, orig)
    profs = {}
    for r, sample in samples.items():
        obs = stage(
            f"profile-full-r{r}",
            [new["om"], sample, cli["file"], mc],
            lambda d: full_profile(d, r, new, sample, new_air["mapping"], runner, private, cli, custom=True),
        )
        old = ref["native_full_profiles"][str(r)]
        profs[str(r)] = dict(previous=old, candidate=obs)
        print(
            f"Full Verify r={r}, 91 outputs and ref writes bitwise PASS; 24 first-pass tasks "
            f"{old['first']['task_sum_ms']:.3f} -> {obs['first']['task_sum_ms']:.3f} ms; 24 commit tasks "
            f"{old['commit']['task_sum_ms']:.3f} -> {obs['commit']['task_sum_ms']:.3f} ms",
            flush=True,
        )
    state["full_profiles"] = profs
    plan = stage(
        "plan",
        [assets["plan"], new["om"]],
        lambda d: replace_models(
            d, assets["plan"], assets["models"], dict(body=assets["models"]["body"], verify=new)
        ),
    )
    control = prior.completed("generation")["result"]
    require(
        control["index"] == orig["capture"] and histogram(orig, control) == hist,
        "different generation archive",
    )
    history = archive_inputs(out, control, assets["plan"])
    cmd = c.read(c.verify(control["command"]))
    exe = assets["tools"]["qwen35_dflash_acl_runner"]
    require(cmd[0] == str(c.verify(exe)), "different C++ generation runner")
    for flag, key in (("--model", "path"), ("--model-sha256", "sha256")):
        pos = cmd.index(flag) + 1
        require(cmd[pos] == assets["plan"][key], "wrong baseline plan")
        cmd[pos] = plan[key]
    run = stage(
        "generation",
        [plan, exe, history["batch"]["sha256"], pkg, cmd],
        lambda d: launch(d, history, plan, dict(file=exe), private, reference_command=cmd),
    )
    state.update(plan=plan, capture=run["index"], workspace=memory(run, True, model_count=6))
    res = stage(
        "comparison",
        [control["index"], run["index"], plan, c.sha(ROOT.parent / "draft_w8_e2e/run_workspace.py")],
        lambda d: compare(
            d,
            history["prompts"],
            control,
            run,
            assets["plan"],
            orig["groups"],
            window_plan=plan,
            labels=("gdr_prefix", "conv_window"),
        ),
    )
    state.update(res)
    return (
        "MEASURED_AND_PROFILED"
        if res["token_matches"] == res["round_matches"] == 60 and res["same_workload"]
        else "MEASURED_WITH_DIFFERENCES"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-probe", required=True)
    p.add_argument("--resume")
    args = p.parse_args()
    out = Path(os.environ["DFLASH_OUTPUT"])
    state = {"status": "RUNNING"}
    try:
        state["status"] = run(out, args, state)
    except Exception as e:
        state.update(status="FAILED", error=str(e))
        traceback.print_exc()
    c.write(out / "result.json", state)
    print(state["status"], "REPORT", out / "result.json", flush=True)
    raise SystemExit(state["status"] != "MEASURED_AND_PROFILED")
