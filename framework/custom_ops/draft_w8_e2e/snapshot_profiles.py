"""Measure allocation policies on one archived Draft at a time, with real receiver inputs and outputs."""

from pathlib import Path
import re
import subprocess

from winners import OPS, c, require
from package import command
from full_replay import prepare_case
from call_latency_results import dist, task_contrasts


POLICIES = {
    "stock": ("normal", "normal"),
    "work-only": ("huge-first", "normal"),
    "io-only": ("normal", "huge-first"),
    "both": ("huge-first", "huge-first"),
}


class ProfileLoadFailure(RuntimeError):
    """The profiler launched the application but all its resident OMs did not load."""


def check_application(rc, evidence, log):
    if rc == 0 and evidence:
        return
    useful = [
        line
        for line in log.splitlines()
        if any(
            text in line
            for text in (
                "phase=before_models",
                "phase=load_failed",
                "PROBE FAILED:",
                "FAIL:",
                "[ERROR]",
                "Script params are invalid",
            )
        )
    ]
    detail = "\n".join(useful[:2] + useful[-5:] if len(useful) > 7 else useful)
    message = f"application rc={rc}, completion record={evidence}\n{detail or log[-2000:]}"
    if "phase=load_failed" in log and "PROBE FAILED: aclmdlLoadFromFileWithMem failed" in log:
        raise ProfileLoadFailure(message)
    raise RuntimeError(message)


def workspace_size(log):
    sizes = re.findall(r"workspace policy=shared_serial shared_bytes=(\d+)", log)
    require(len(sizes) == 1 and int(sizes[0]) > 0, "no unique measured shared workspace")
    return int(sizes[0])


def build(d, cann, env):
    target = d / "build"
    command(
        [
            "cmake",
            "-S",
            OPS / "draft_ac/test",
            "-B",
            target,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCANN_ROOT=" + str(cann),
        ],
        d / "configure.log",
        env,
        d,
    )
    command(
        ["cmake", "--build", target, "--target", "ac_om_runner", "--parallel", "2"], d / "build.log", env, d
    )
    exe = target / "ac_om_runner"
    libs = subprocess.run(["ldd", str(exe)], env=env, capture_output=True, check=True, text=True).stdout
    require(
        "libascendcl.so" in libs and "not found" not in libs and "/devlib/" not in libs,
        "single-OM runner needs real AscendCL",
    )
    return c.identity(exe)


def replay(d, exe, case, capacity, policy, env, cli=None):
    f = d / "case"
    prepare_case(f, case["abi"], case["io"], case["golden"])
    work, io = POLICIES[policy]
    args = list(map(str, [c.verify(exe), 0, c.verify(case["om"]), f, f / "inputs", capacity, work, io]))
    if cli:
        require(c.identity(cli["alias"]) == cli["file"], "profiler executable changed")
        args = [cli["alias"], *cli["flags"], "--output=" + str(d / "profiler"), *args]
    c.write(d / "command.json", args)
    log = d / "runner.log"
    with log.open("xb") as out:
        rc = subprocess.run(args, cwd=d, env=env, stdout=out, stderr=subprocess.STDOUT).returncode
    receipt = f / "execution.json"
    check_application(rc, receipt.is_file(), log.read_text(errors="replace"))
    v = c.read(receipt)
    allocation = v["allocation"]
    require(
        v["status"] == "PASS"
        and v["cpu_fallback"] is False
        and all(v[k] for k in ("guards", "readonly", "repeat"))
        and allocation
        == dict(
            workspace=work,
            io=io,
            workspace_bytes=capacity,
            ge_weight_bytes=case["weight_bytes"],
            golden_outputs=11,
        )
        and len(case["golden"]) == 11,
        "single-OM allocation or full snapshot validation differs",
    )
    t = v["timing"]
    require(t["protocol"] == "continuous-v1", "different timing mode")
    require(
        t["warmup"] == 5
        and t["repetitions"] == 30
        and len(t["execute_sync"]["samples_ms"]) == 30
        and t["correctness_before_calls"] == 2
        and t["correctness_after_calls"] == 1
        and t["timed_tail_checked"],
        "incomplete single-OM run",
    )
    return dict(
        time=dist(t["execute_sync"]["samples_ms"]),
        allocation=allocation,
        execution=c.identity(receipt),
        root=str(d),
        profiled=bool(cli),
    )


def collect(stage, cann, env, cli, paired, cases, export_csv):
    log = c.verify(paired["log"]).read_text(errors="replace")
    capacity = workspace_size(log)
    for g, case in cases.items():
        role = "draft" if g == "16" else "draft_static64"
        rows = re.findall(rf"om-memory graph={role} query_status=0 weight_bytes=(\d+) work_bytes=\d+", log)
        require(len(rows) == 1, "missing original model memory requirement for " + role)
        case["weight_bytes"] = int(rows[0])
    sources = [
        Path(__file__),
        OPS / "draft_ac/test/runner.cpp",
        OPS / "draft_ac/test/CMakeLists.txt",
        OPS / "draft_quant/test/runner_common.h",
    ]
    src = {str(f): c.sha(f) for f in sources}
    exe = stage("snapshot-runner", src, lambda d: build(d, cann, env))
    result = dict(
        scope="single resident full Draft OM, frozen Target-produced inputs, GE-owned weights, "
        "receiver-sized workspace, guarded I/O; Execute+Sync excludes transfers",
        workspace_bytes=capacity,
        runner=exe,
        gears={},
        complete=True,
    )
    for g, case in cases.items():

        def observe(label, policy, profiler=None):
            key = [
                exe,
                case["om"],
                case["io"],
                case["golden"],
                capacity,
                policy,
                profiler and profiler["file"],
            ]
            return stage(
                f"snapshot-c{g}-{label}", key, lambda d: replay(d, exe, case, capacity, policy, env, profiler)
            )

        measures = {}
        order = ("work-only", "io-only", "both") if g == "16" else ("both", "io-only", "work-only")
        for label in ("stock", *order, "stock-end"):
            run = observe(label, "stock" if label == "stock-end" else label)
            measures[label] = run["time"]
        print(
            f"C{g} one-OM Execute+Sync P50/P95 ms "
            + " ".join(f"{k}={t['p50']:.3f}/{t['p95']:.3f}" for k, t in measures.items()),
            flush=True,
        )
        profiles = {}
        for policy in ("stock", "both"):
            run = observe("profile-" + policy, policy, cli)
            folder = Path(run["root"])
            export_csv(folder, cli, env)
            seq = {g: [(i - 7 if 7 <= i < 37 else -1, policy) for i in range(38)]}
            evidence = task_contrasts(folder / "profiler", {g: case["spec"]}, seq, single=g)
            profiles[policy] = evidence
            scope = evidence.get("gears", {}).get(g, {})
            sums = scope.get("task_sums_ms", {})
            result["complete"] = result["complete"] and scope.get("status") == "COMPLETE"
            span = scope.get("fc_to_head_span_ms", {}).get(policy)
            print(
                f"C{g} one-OM msprof {policy} {scope.get('status', 'UNRESOLVED')} "
                + " ".join(f"{k}={z[policy]['p50']:.3f} ms" for k, z in sums.items())
                + (f" FC-start-to-Head-end={span['p50']:.3f} ms" if span else ""),
                flush=True,
            )
        result["gears"][g] = dict(unprofiled=measures, profile=profiles)
    return result
