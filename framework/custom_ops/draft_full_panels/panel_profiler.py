"""Locate the installed CANN CLI and profile optimized OMs using the admitted full runner."""

import os
from pathlib import Path
import shutil

from full_replay import run_om
from joint import c, HERE, OPS, require
from module_profile import analyze
from package import command
from profile_common import FLAGS


def find_msprof(cann, path):
    root = Path(cann).resolve(strict=True)
    candidates = [
        root / "tools/profiler/bin/msprof",
        root / "bin/msprof",
        root / "aarch64-linux/tools/profiler/bin/msprof",
        root / "aarch64-linux/bin/msprof",
    ]
    on_path = shutil.which("msprof", path=path)
    if on_path:
        candidates.append(Path(on_path))
    for alias in candidates:
        if alias.is_file() and os.access(alias, os.X_OK) and root in alias.resolve(strict=True).parents:
            return alias
    raise ValueError(f"No executable CANN msprof under {root}; checked " + ", ".join(map(str, candidates)))


def bind_msprof(out, cann, env):
    alias = find_msprof(cann, env["PATH"])
    tool = c.identity(alias)
    print("msprof", alias, "SHA256", tool["sha256"], flush=True)
    help_log = out / "msprof-help.log"
    command([alias, "--help"], help_log, env, out)
    help_text = help_log.read_text(errors="replace")
    missing = [f.split("=")[0] for f in FLAGS if f.split("=")[0] not in help_text]
    require(not missing, f"{alias} --help is missing {missing}; see {help_log}")
    require(c.identity(alias) == tool, "msprof changed during capability check")
    binding = dict(alias=str(alias), file=tool, flags=list(FLAGS), help=c.identity(help_log))
    c.write(out / "msprof-tool.json", binding)
    return binding


def profile_modules(stages, cann, env, pkg, runner, dsos, cases, *, concise=False):
    cli = bind_msprof(stages.folder, cann, env)
    capture_sources = {
        str(p): c.sha(p)
        for p in (
            HERE / "full_launch.py",
            HERE / "full_replay.py",
            OPS / "draft_a_core_scheduler_profile/exec_shim.py",
            OPS / "draft_a_core_scheduler_profile/collect_profile.py",
        )
    }
    for gear, case in cases.items():
        case_env, case_pkg, case_dsos = case.get("runtime", (env, pkg, dsos))
        key = dict(
            benchmark=stages.refs[f"benchmark-c{gear}"],
            cli=cli["file"],
            flags=cli["flags"],
            om=case["model"]["om"],
            runner=runner,
            code=capture_sources,
        )

        def profile(d):
            require(c.identity(cli["alias"]) == cli["file"], "bound msprof changed before collection")
            c.verify_opp(case_pkg["opp"])
            row = run_om(
                d,
                case["graph"]["abi"],
                case["io"],
                case["model"],
                runner,
                case_env,
                case_dsos,
                "benchmark",
                case["reference"],
                profiler=cli["alias"],
            )
            raw = d / "profiler"
            files = [
                c.identity(p)
                for p in sorted(raw.rglob("*"))
                if p.is_file()
                and (
                    (p.name.startswith("msprof") and p.suffix == ".json")
                    or (p.name.startswith(("op_summary", "step_trace")) and p.suffix == ".csv")
                )
            ]
            return dict(row, profiler=str(raw), exported=files)

        capture = stages.run(f"profile-c{gear}", key, profile)
        for item in capture["exported"]:
            c.verify(item)
        name = f"modules-c{gear}"
        summary = stages.run(
            name,
            dict(
                capture=stages.refs[f"profile-c{gear}"],
                code=c.sha(HERE / "module_profile.py"),
                owners=case["graph"]["modules"],
                verified=case["graph"].get("verified_projections", {}),
            ),
            lambda d: analyze(d, capture, case["model"], case["graph"]),
        )
        print(
            f"C{gear} optimized | per Execute task sums (ms) | {summary['scope']} | "
            f"{summary['unmapped']}/{summary['timed_tasks']} unmapped",
            flush=True,
        )
        if not concise:
            print("  module / layer                       Median       P95    calls")
            for row in [*summary["modules"], *summary["layers"]]:
                p50, p95 = row["p50_task_sum_ms"], row["p95_task_sum_ms"]
                timing = f"{p50:9.4f} {p95:9.4f}" if p50 is not None else "        -         -"
                print(f"  {row['module']:32s} {timing}   {row['calls_observed']}/30")
        print("Module CSV", Path(stages.refs[name]["path"]).parent / "module-latency.csv", flush=True)
