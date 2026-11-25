"""One manual 310P3 run for the preregistered W8A16 transfers. There is no full Draft execution."""

import argparse
import csv
from dataclasses import asdict
import os
from pathlib import Path
import shutil
import sys
import traceback

from recipes import HERE, OPS, CONTROL, SINGLE, c, require, candidates, header
from kernels import build, device, host
from incumbents import archived, down_source, GATE_RUN, GATE_RESULT, DOWN_SCOPE
from atc_transfer import compiled
from gate_input import DEPLOYMENT, FULL
from air import protobuf_classes
from experiment import Stages
from fresh_dso import seal
from fresh_run import environment
from run_gate import boundary_case, pair_statistics
from run_q import run_path
from run_kv import numeric_rejection
from panel_profiler import bind_msprof
from fc_runtime import pipeline_times
from packed_run import charged_span
import o_runtime


class DeviceFailure(RuntimeError):
    """Stop on an unclassified execute, synchronization, mapping or guard failure."""


def code(*paths):
    return {str(p): c.sha(p) for p in paths}


def boundaries(shape, recipes):
    _, k, _ = shape
    ends = [16, 32, 128, 256]
    for r in recipes:
        ends.extend((r.base_k, r.chunk, r.panel, k - r.panel, k - r.chunk))
    return [(0, k - 1), *((i - 1, i) for i in dict.fromkeys(ends) if 0 < i < k)]


def fresh_pair(d, left, right, call, inputs, shape, reverse):
    rows = []
    for i in range(3):
        order = (right, left) if (i + int(reverse)) % 2 else (left, right)
        for mode in order:
            rows.append(call(d / str(i) / mode, mode, inputs, True, i))
    m, k, n = shape
    stat = pair_statistics(
        rows, right, left, flops=2 * m * k * n, logical_bytes=k * n + 2 * (k // 128) * n + 2 * m * (k + n)
    )
    return dict(runs=rows, statistics=stat)


def choose(control, modes, compare):
    passed = [m for m in modes if compare(control, m)["decision"] == "GO"]
    if not passed:
        return control
    best = passed[0]
    for other in passed[1:]:
        if compare(best, other)["decision"] == "GO":
            best = other
    return best


def summary(out, state):
    rows = []
    profiles = []
    for key, grp in state["groups"].items():
        ctrl = grp["incumbent"]
        for mode, prof in grp["profiles"].items():
            base = dict(
                group=key,
                mode=mode,
                incumbent=mode == ctrl,
                selected=grp.get("selected") == mode,
                core_p50_us=prof.get("core_us"),
                core_p95_us=prof.get("core_p95_us"),
                prep_p50_us=prof.get("pack_us"),
                span_p50_us=prof.get("span_us"),
                span_p95_us=prof.get("p95_us"),
                attribution=prof.get("attribution"),
                counter_matches=prof.get("joined"),
            )
            for field in (
                "aicore_time(us)",
                "mac_time(us)",
                "mte1_time(us)",
                "mte2_time(us)",
                "mte3_time(us)",
                "vec_time(us)",
                "scalar_time(us)",
                "cube_utilization(%)",
            ):
                base[field] = prof.get("counters", {}).get(field)
            profiles.append(base)
        for name, entry in grp["candidates"].items():
            pair = grp["pairs"].get(ctrl + "__" + name, {})
            profile = grp["profiles"].get(name, {})
            rows.append(
                dict(
                    group=key,
                    incumbent=ctrl,
                    candidate=name,
                    state=entry["status"],
                    decision=pair.get("decision"),
                    control_p50_ms=pair.get("control_ms"),
                    candidate_p50_ms=pair.get("candidate_ms"),
                    control_p95_ms=pair.get("control_p95_ms"),
                    candidate_p95_ms=pair.get("candidate_p95_ms"),
                    ratio=pair.get("ratio"),
                    core_us=profile.get("core_us"),
                    pack_us=profile.get("pack_us"),
                    device_span_us=profile.get("span_us"),
                    device_p95_us=profile.get("p95_us"),
                    attribution=profile.get("attribution"),
                    counter_matches=profile.get("joined"),
                    selected=grp.get("selected") == name,
                )
            )
    for name, items in (("summary", rows), ("profiles", profiles)):
        if not items:
            continue
        tmp = out / (name + ".tmp")
        with tmp.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(items[0]))
            writer.writeheader()
            writer.writerows(items)
        tmp.replace(out / (name + ".csv"))
    tmp = out / "result.tmp"
    c.write(tmp, state)
    tmp.replace(out / "result.json")


def compact(state):
    print("Single-OM Execute+Sync, ms (fresh paired Median/P95). Device time includes required preparation.")
    for key, grp in state["groups"].items():
        for name, entry in grp["candidates"].items():
            p = grp["pairs"].get(grp["incumbent"] + "__" + name)
            z = grp["profiles"].get(name, {})
            if p:
                dev = f"{z['span_us']:.2f}" if z.get("span_us") is not None else "—"
                print(
                    f"{key:5s} {name:24s} {p['control_ms']:.6f}/{p['control_p95_ms']:.6f} → "
                    f"{p['candidate_ms']:.6f}/{p['candidate_p95_ms']:.6f}  {p['ratio']:.4f}x "
                    f"{p['decision']:5s}  device {dev} us"
                )
            else:
                print(key, name, entry["status"], entry.get("error", ""))
        selected = grp.get("selected")
        print("SELECTED", key, selected or grp["status"])
        for mode in dict.fromkeys((grp["incumbent"], selected)):
            z = grp["profiles"].get(mode, {})
            if z.get("span_us") is not None:
                print(
                    f"  {mode} device Median/P95 {z['span_us']:.2f}/{z['p95_us']:.2f} us, "
                    f"kernel {z['core_us']:.2f} us, {z['attribution']}, counters {z['joined']}/30"
                )


class Sweep:
    def __init__(self, out, args, state):
        self.out = out
        self.args = args
        self.state = state
        self.runs = Path(os.environ["DFLASH_RUN_ROOT"]) / "runs"
        old = run_path(self.runs, args.resume, "w8-transfer-") if args.resume else None
        self.st = Stages(out, old)
        self.cann = Path(os.environ["CANN_ROOT"]).resolve(strict=True)
        self.tools = {n: c.identity(shutil.which(n)) for n in ("cmake", "msopgen", "atc")}
        self.plain = environment(self.cann, self.tools)
        desc = out / "cann-ge.desc"
        self.types = protobuf_classes(desc.read_bytes())
        self.schema = c.sha(desc)
        self.atc = [
            self.tools["atc"]["path"],
            "--mode=0",
            "--framework=1",
            "--model=INPUT",
            "--output=OUTPUT",
            "--soc_version=Ascend310P3",
            "--precision_mode=must_keep_origin_dtype",
            "--deterministic=0",
        ]
        self.cli = None
        self.runtime = code(
            HERE / "run_transfer.py",
            OPS / "draft_o_bandwidth/o_runtime.py",
            OPS / "draft_gate_tile_nd/fresh_run.py",
            OPS / "draft_full_panels/full_launch.py",
        )

    def stage(self, name, key, action):
        self.state["phase"] = name
        return self.st.run(name, key, action)

    def down(self):
        b = CONTROL["down"]
        key = "down/"
        st = lambda text, contract, action: self.stage(key + text, contract, action)
        src = st(
            "source",
            dict(
                indices={n: c.identity(self.runs / n / "stages.json") for n in (GATE_RUN, DEPLOYMENT, FULL)},
                gate_result=GATE_RESULT,
                schema=self.schema,
                code=code(HERE / "incumbents.py", OPS / "draft_gate_next_n/gate_input.py"),
            ),
            lambda d: down_source(d, self.runs, self.types),
        )
        for key_ in ("reference_air", "air", "packing", "replays", "weights"):
            c.verify_tree(src[key_])
        exe = st(
            "runner",
            dict(
                abis=[src["reference_abi"], b.abi],
                scope=DOWN_SCOPE,
                cmake=self.tools["cmake"],
                code=self.runtime,
            ),
            lambda d: o_runtime.runner(
                d, self.cann, self.plain, src["reference_abi"], projection_abi=b.abi, reference=DOWN_SCOPE
            ),
        )
        pkg = src["control_package"]
        raw = src["pkg"]
        for p in (pkg, raw):
            c.verify_opp(p["opp"])
        env = environment(self.cann, self.tools, pkg["opp"]["install"])
        ref_env = environment(self.cann, self.tools, raw["opp"]["install"])
        anchor = dict(air=src["air"], query=src["query"], selected=src["selected"])
        mod = st(
            "control-om",
            dict(
                source=self.st.refs["down/source"],
                atc=self.tools["atc"],
                opp=pkg["opp"],
                code=code(HERE / "atc_transfer.py"),
            ),
            lambda d: compiled(
                d, self.types, b, b, anchor, src["weights"], pkg, env, self.atc, recreate_control=True
            ),
        )

        def check(plans):
            tag = "x".join(map(str, b.shape))
            require(
                plans[tag]["tiling_sha256"] == src["source_plans"][tag]["tiling_sha256"],
                "raw Down reference tiling changed",
            )

        ref = st(
            "reference-om",
            dict(source=self.st.refs["down/source"], atc=self.tools["atc"], code=self.runtime),
            lambda d: o_runtime.compile_prefix(d, src, ref_env, self.atc, validate_site=check),
        )
        cases = {}
        if not self.args.compile_only:
            dsos = seal(self.out / "down/dso-reference", raw, exe, ref_env)
            for label, files in src["replays"].items():
                try:
                    row = st(
                        "capture-" + label,
                        dict(source=self.st.refs["down/source"], om=ref["om"], runner=exe, inputs=files),
                        lambda d: o_runtime.capture(
                            d, exe, ref, ref_env, dsos, src, label, input_shape=b.shape[:2]
                        ),
                    )
                except Exception as e:
                    raise DeviceFailure("Down reference capture: " + str(e)) from e
                cases[label] = row["files"]
        return dict(model=mod, package=pkg, runner=exe, scope=DOWN_SCOPE, cases=cases, weights=src["weights"])

    def measure(self, key, grp):
        root = self.out / key
        root.mkdir()
        b = CONTROL[key]
        trials = candidates(key)
        m, k, n = b.shape
        if key == "down":
            a = self.down()
        else:
            old = GATE_RUN if key == "gate" else SINGLE[b.gear][b.role].root
            a = self.stage(
                key + "/admit",
                dict(
                    index=c.identity(self.runs / old / "stages.json"),
                    result=c.identity(self.runs / old / "result.json"),
                    code=code(HERE / "incumbents.py", OPS / "draft_all_w8/archive.py"),
                ),
                lambda d: archived(d, self.runs, key),
            )
        cases = dict(a["cases"])
        weights = a.get("weights") or {name: cases["measured"][name] for name in ("w_nz.bin", "s_gn.bin")}
        c.verify_tree(weights)
        if not self.args.compile_only:
            ends = boundaries(b.shape, [b, *trials])
            cases["boundaries"] = self.stage(
                key + "/boundaries",
                dict(
                    weights=weights,
                    shape=b.shape,
                    pairs=ends,
                    code=c.sha(OPS / "draft_gate_bandwidth/run_gate.py"),
                ),
                lambda d: boundary_case(d / "inputs", weights, shape=b.shape, pairs=ends),
            )
            c.verify_tree(cases)
        exe = a["runner"]
        scope = a["scope"]
        c.verify(exe)
        models = {b.name: a["model"]}
        pkgs = {b.name: a["package"]}
        c.verify_opp(pkgs[b.name]["opp"])
        c.verify(models[b.name]["om"])
        c.verify(models[b.name]["dump"])
        c.verify(models[b.name]["air"])
        grp["control_om"] = models[b.name]["om"]
        grp["runner"] = exe
        envs = {b.name: environment(self.cann, self.tools, pkgs[b.name]["opp"]["install"])}
        valid = []
        dsos = {}
        compiled_plans = [(b, models[b.name])]

        def stage(tag, contract, action):
            return self.stage(key + "/" + tag, contract, action)

        def call(d, mode, files, timed, i):
            return o_runtime.execute(
                d, exe, models[mode], envs[mode], dsos[mode], files, timed, i, abi=b.abi, reference=scope
            )

        def smoke(mode):
            dsos[mode] = seal(root / ("dso-" + mode), pkgs[mode], exe, envs[mode])

            def check(d):
                try:
                    for_case = {
                        label: call(d / label, mode, files, False, -1) for label, files in cases.items()
                    }
                except Exception as e:
                    why = numeric_rejection(d) if mode != b.name else None
                    if why:
                        return dict(accepted=False, reason=why)
                    raise DeviceFailure(f"{key}/{mode} guarded smoke: {e}") from e
                return dict(accepted=True, cases=for_case)

            row = stage(
                "smoke-" + mode,
                dict(om=models[mode]["om"], runner=exe, inputs=cases, code=self.runtime),
                check,
            )
            if row["accepted"]:
                for item in row["cases"].values():
                    c.verify(item["execution"])
            return row

        if not self.args.compile_only:
            smoke(b.name)
        for r in trials:
            entry = grp["candidates"][r.name]
            try:
                pkg = stage(
                    "build-" + r.name,
                    dict(
                        recipe=asdict(r),
                        tools=self.tools,
                        host=qhash(host(r)),
                        device=qhash(device(r)),
                        header=qhash(header(r)),
                        helper=c.sha(OPS / "draft_gate_bandwidth/panel_vector.h"),
                        code=code(HERE / "kernels.py", OPS / "draft_gate_tile_nd/package.py"),
                    ),
                    lambda d: build(d, self.cann, self.plain, r),
                )
                c.verify_opp(pkg["opp"])
                pkgs[r.name] = pkg
                envs[r.name] = environment(self.cann, self.tools, pkg["opp"]["install"])
                mod = stage(
                    "atc-" + r.name,
                    dict(
                        build=self.st.refs[key + "/build-" + r.name],
                        weights=weights,
                        base=grp["control_om"],
                        air=models[b.name]["air"],
                        atc=self.tools["atc"],
                        schema=self.schema,
                        code=code(HERE / "atc_transfer.py"),
                    ),
                    lambda d: compiled(
                        d, self.types, r, b, models[b.name], weights, pkg, envs[r.name], self.atc
                    ),
                )
                c.verify(mod["om"])
                c.verify(mod["dump"])
                for other, oldmodel in compiled_plans:
                    if other.cube_geometry == r.cube_geometry:
                        require(
                            bytes.fromhex(mod["query"]["schedule"]["tiling_hex"])[24:]
                            == bytes.fromhex(oldmodel["query"]["schedule"]["tiling_hex"])[24:],
                            "same-geometry sibling Cube differs: " + other.name,
                        )
                    require(
                        oldmodel["selected"]["executable_sections"] != mod["selected"]["executable_sections"],
                        "duplicate machine implementation: " + other.name,
                    )
                compiled_plans.append((r, mod))
                models[r.name] = mod
                entry.update(status="COMPILED", om=mod["om"])
            except Exception as e:
                entry.update(status="BUILD_OR_ATC_REJECTED", error=repr(e), phase=self.state["phase"])
                (root / (r.name + "-compile-error.txt")).write_text(traceback.format_exc())
                print("REJECTED before device", r.name, repr(e), flush=True)
                summary(self.out, self.state)
                continue
            if self.args.compile_only:
                continue
            result = smoke(r.name)
            if not result["accepted"]:
                entry.update(status="BITWISE_REJECTED", evidence=result["reason"])
                print("REJECTED", r.name, result["reason"], flush=True)
            else:
                entry["status"] = "SMOKE_PASS"
                valid.append(r.name)
                print(f"{r.name} bitwise + guards PASS, {m*n:,} outputs, 3 inputs × 2 poisons", flush=True)
        if self.args.compile_only:
            grp["status"] = "COMPILED"
            return
        logical = k * n + 2 * (k // 128) * n + 2 * m * (k + n)

        def compare(left, right):
            label = left + "__" + right

            def timed(d):
                try:
                    return fresh_pair(
                        d, left, right, call, cases["measured"], b.shape, (list(models).index(right) % 2) == 0
                    )
                except Exception as e:
                    raise DeviceFailure("paired measurement: " + str(e)) from e

            rec = stage(
                "pair-" + label,
                dict(
                    smoke=[self.st.refs[key + "/smoke-" + v] for v in (left, right)],
                    runner=exe,
                    inputs=cases["measured"],
                    protocol=self.runtime,
                    stats=c.sha(OPS / "draft_gate_bandwidth/run_gate.py"),
                ),
                timed,
            )
            stat = pair_statistics(rec["runs"], right, left, flops=2 * m * k * n, logical_bytes=logical)
            require(stat == rec["statistics"], "paired sample mismatch")
            for row in rec["runs"]:
                ev = c.read(c.verify(row["execution"]))
                require(
                    ev["golden_match"]
                    and ev["sync_completed"]
                    and ev["timing"]["execute_sync"]["samples_ms"]
                    == row["validation"]["timing"]["samples_ms"],
                    "archived pair changed",
                )
            l, h = (stat["variants"][v] for v in (left, right))
            grp["pairs"][label] = dict(
                report=self.st.refs[key + "/pair-" + label],
                decision=stat["decision"],
                ratio=stat["speedup"],
                round_ratios=stat["round_ratios"],
                control_ms=l["median_ms"],
                candidate_ms=h["median_ms"],
                control_p95_ms=l["p95_ms"],
                candidate_p95_ms=h["p95_ms"],
            )
            if left == b.name:
                grp["candidates"][right]["status"] = "MEASURED"
            print(
                f"{key} {left} → {right}: {l['median_ms']:.6f}/{l['p95_ms']:.6f} → "
                f"{h['median_ms']:.6f}/{h['p95_ms']:.6f} ms {stat['speedup']:.4f}x {stat['decision']}",
                flush=True,
            )
            summary(self.out, self.state)
            return stat

        grp["selected"] = choose(b.name, valid, compare)
        grp["status"] = "MEASURED" if valid else "NO_ADMITTED_CANDIDATE"
        print("Selected", key, grp["selected"], flush=True)
        summary(self.out, self.state)
        if self.args.no_profile:
            return
        if self.cli is None:
            self.cli = bind_msprof(self.out, self.cann, envs[b.name])
        cli = self.cli
        for mode in (b.name, *valid):
            model = models[mode]
            role = "fc" if b.role == "fc" else "layer0." + b.role

            def capture(d):
                try:
                    cap = o_runtime.collect_profile(
                        d,
                        exe,
                        model,
                        envs[mode],
                        dsos[mode],
                        cases["measured"],
                        cli,
                        abi=b.abi,
                        reference=scope,
                    )
                except Exception as e:
                    raise DeviceFailure("msprof application: " + str(e)) from e
                cap["exports"] = [
                    c.identity(p)
                    for p in sorted(Path(cap["profiler"]).rglob("*"))
                    if p.is_file()
                    and (
                        (p.name.startswith("msprof") and p.suffix == ".json")
                        or (p.name.startswith(("op_summary", "step_trace")) and p.suffix == ".csv")
                    )
                ]
                return cap

            cap = stage(
                "profile-" + mode,
                dict(
                    om=model["om"],
                    runner=exe,
                    cli=cli["file"],
                    flags=cli["flags"],
                    smoke=self.st.refs[key + "/smoke-" + mode],
                    code=self.runtime,
                ),
                capture,
            )

            def analyze(d):
                c.verify_tree(cap["exports"])
                c.verify(cap["execution"])
                rep = o_runtime.analyze_profile(d, model, cap, role=role, logical_bytes=logical)
                rep["pipeline"] = pipeline_times(d, rep, module=role, profiler_folder=Path(cap["profiler"]))
                rep["charged"] = charged_span(d, model, rep, module=role, logical_bytes=logical)
                return rep

            try:
                rep = stage(
                    "analyze-" + mode,
                    dict(
                        capture=self.st.refs[key + "/profile-" + mode],
                        code=code(
                            OPS / "draft_o_bandwidth/o_runtime.py",
                            OPS / "draft_fc_bandwidth/fc_runtime.py",
                            OPS / "draft_fc_bandwidth/packed_run.py",
                            OPS / "draft_full_panels/module_profile.py",
                        ),
                    ),
                    analyze,
                )
                span = rep["charged"]
                pipe = rep["pipeline"]
                us = rep["kernel"]["p50_task_sum_ms"] * 1000
                grp["profiles"][mode] = dict(
                    attribution=span["status"],
                    core_us=us,
                    core_p95_us=rep["kernel"]["p95_task_sum_ms"] * 1000,
                    pack_us=span.get("pack_task_median_us"),
                    span_us=span.get("median_us"),
                    p95_us=span.get("p95_us"),
                    joined=pipe["joined"],
                    counters=pipe["medians"],
                    report=self.st.refs[key + "/analyze-" + mode],
                )
                print(
                    f"{mode} msprof {us:.2f} us kernel, prep→end {span.get('median_us')} us, {span['status']}, "
                    f"counters {pipe['joined']}/30",
                    flush=True,
                )
            except Exception as e:
                grp["profiles"][mode] = dict(attribution="UNRESOLVED", error=repr(e))
                (root / (mode + "-analysis-error.txt")).write_text(traceback.format_exc())
                print("Profile analysis unresolved", mode, repr(e), flush=True)
            summary(self.out, self.state)
        if valid:
            grp["status"] = (
                "MEASURED_AND_PROFILED"
                if all(
                    v["attribution"] != "UNRESOLVED" and v.get("joined") == 30
                    for v in grp["profiles"].values()
                )
                else "MEASURED_PROFILE_INCOMPLETE"
            )


def qhash(value):
    import hashlib

    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", nargs="+", choices=tuple(CONTROL), default=list(CONTROL))
    p.add_argument("--resume", metavar="PREVIOUS_W8_TRANSFER_RUN")
    flag = p.add_mutually_exclusive_group()
    flag.add_argument("--no-profile", action="store_true")
    flag.add_argument("--compile-only", action="store_true")
    args = p.parse_args()
    if len(set(args.only)) != len(args.only):
        p.error("duplicate projection group")
    out = Path(os.environ["DFLASH_OUTPUT"])
    state = dict(status="RUNNING", phase="setup", groups={})
    try:
        sweep = Sweep(out, args, state)
        for key in args.only:
            grp = state["groups"][key] = dict(
                status="STARTING",
                incumbent=CONTROL[key].name,
                candidates={r.name: dict(status="PENDING", recipe=asdict(r)) for r in candidates(key)},
                pairs={},
                profiles={},
            )
            try:
                sweep.measure(key, grp)
            except DeviceFailure:
                raise
            except Exception as e:
                grp.update(status="FAILED", error=repr(e), phase=state["phase"])
                (out / (key + "-error.txt")).write_text(traceback.format_exc())
                print("FAILED", key, state["phase"], repr(e), flush=True)
            summary(out, state)
        failed = any(
            g["status"] in ("FAILED", "MEASURED_PROFILE_INCOMPLETE")
            or any(v["status"] in ("PENDING", "BUILD_OR_ATC_REJECTED") for v in g["candidates"].values())
            for g in state["groups"].values()
        )
        rejected = any(
            v["status"] == "BITWISE_REJECTED"
            for g in state["groups"].values()
            for v in g["candidates"].values()
        )
        state["status"] = (
            "PARTIAL"
            if failed
            else (
                "COMPILED_NOT_EXECUTED"
                if args.compile_only
                else (
                    "COMPLETED_WITH_BITWISE_REJECTIONS"
                    if rejected
                    else "MEASURED" if args.no_profile else "MEASURED_AND_PROFILED"
                )
            )
        )
    except Exception as e:
        state.update(status="FAILED", error=repr(e))
        traceback.print_exc()
        (out / "traceback.txt").write_text(traceback.format_exc())
    summary(out, state)
    compact(state)
    print(state["status"], "REPORT", out / "result.json", flush=True)
    return int(state["status"] in ("FAILED", "PARTIAL"))


if __name__ == "__main__":
    sys.exit(main())
