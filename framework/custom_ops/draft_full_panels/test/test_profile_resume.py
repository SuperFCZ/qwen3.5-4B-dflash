"""Host-only regression tests for SDK discovery and the measured archive left before profiling."""

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiment
from experiment import Stages
from full_replay import compare_rounds
from joint import c
from panel_profiler import bind_msprof, find_msprof, FLAGS
from profile_archive import read_gear
from raw_protocol import OUTPUTS, verify_run


def file(folder, name, data):
    p = folder / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return c.identity(p)


class ProfilerCli(unittest.TestCase):
    def test_installed_layout_help_and_file_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "cann"
            logs = Path(d) / "logs"
            logs.mkdir()
            tool = root / "tools/profiler/bin/msprof"
            tool.parent.mkdir(parents=True)
            flags = " ".join(f.split("=")[0] for f in FLAGS)
            tool.write_text("#!/bin/sh\n[ \"$1\" = '--help' ] || exit 7\n" + "echo '" + flags + "'\n")
            tool.chmod(0o755)
            result = bind_msprof(logs, root, {"PATH": "/usr/bin:/bin"})
            self.assertEqual(result["file"], c.identity(tool))
            self.assertEqual(c.read(logs / "msprof-tool.json"), result)
            self.assertIn("--task-time", c.verify(result["help"]).read_text())
            tool.write_text("#!/bin/sh\necho --help\n")
            with self.assertRaisesRegex(ValueError, "missing"):
                bind_msprof(Path(d), root, {"PATH": "/usr/bin:/bin"})

    def test_path_fallback_checks_executable_and_toolkit_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve() / "cann"
            other = Path(d).resolve() / "external/bin"
            alt = root / "other/bin"
            for folder in (root / "bin", other, alt):
                folder.mkdir(parents=True)
            p = other / "msprof"
            p.write_text("other")
            p.chmod(0o755)
            alias = root / "bin/msprof"
            alias.symlink_to(p)
            with self.assertRaisesRegex(ValueError, "No executable CANN msprof"):
                find_msprof(root, str(other))
            alias.unlink()
            real = alt / "msprof"
            real.write_text("sdk")
            real.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "checked"):
                find_msprof(root, str(alt))
            real.chmod(0o755)
            self.assertEqual(find_msprof(root, str(alt)), real)
            alias.symlink_to(real)
            self.assertEqual(find_msprof(root, str(other)), alias)


def replay(folder, golden, timed, base_ms, runner, om):
    folder = (folder / "application").resolve()
    folder.mkdir(parents=True)
    argv = [
        runner["path"],
        "0",
        om["path"],
        str(folder),
        str(folder / "inputs"),
        str(folder / "golden"),
        "benchmark" if timed else "smoke",
        "7",
    ]
    c.write(folder.parent / "command.json", dict(runner=runner, om=om, argv=argv, application=argv))
    inventory = {}
    for name, row in golden.items():
        b = c.verify(row).read_bytes()
        inventory[name] = [file(folder, f"{name}-{i}.bin", b) for i in range(4 if timed else 2)]
    e = dict(
        status="PASS",
        guards=True,
        readonly=True,
        repeat=True,
        cpu_fallback=False,
        golden_match=True,
        poison_verified=True,
        sync_completed=True,
        timed_log_stable=True,
        device=0,
        verification_calls=4 if timed else 2,
        row_permutation_checked=False,
        workspace_bytes=8,
        model_weight_bytes=16,
    )
    t = dict(
        status="MEASURED" if timed else "NOT_RUN",
        protocol="continuous-v1" if timed else "checked-v1",
        warmup=5 if timed else 0,
        repetitions=30 if timed else 0,
        correctness_before_calls=2,
        correctness_after_calls=int(timed),
        timed_tail_checked=timed,
        per_timed_call_readback=not timed,
        per_timed_call_poison=not timed,
        prepare={"samples_ms": []},
        execute_sync={"samples_ms": [base_ms] * 30 if timed else []},
    )
    c.write(folder / "execution.json", dict(e, timing=t))
    row = dict(
        execution=c.identity(folder / "execution.json"),
        outputs=inventory,
        profiled=False,
        launch_authorized=True,
    )
    row["validation"] = verify_run(row, {"golden": golden}, 0, timed)
    return row


class ProfileOnly(unittest.TestCase):
    def test_sealed_8013254_measurement_and_crossed_predecessor(self):
        with tempfile.TemporaryDirectory() as d:
            prior = Path(d) / "original"
            new = Path(d) / "profile"
            prior.mkdir()
            new.mkdir()
            s = Stages(prior, None)
            runner = file(prior, "runner-bin", b"runner")
            golden = {n: file(prior, f"golden/{n}", n.encode()) for n in OUTPUTS}
            s.run("private-opp", {}, lambda _: {})
            graph = {"abi": {"inputs": ["x"], "outputs": OUTPUTS}}
            models = {}
            for mode in ("baseline", "optimized"):
                graph[mode] = file(prior, mode + ".air", mode.encode())
                models[mode] = s.run(
                    "atc-c16-" + mode,
                    dict(pkg=s.refs["private-opp"], air=graph[mode]),
                    lambda path: dict(
                        air=graph[mode],
                        om=file(path, "draft.om", mode.encode()),
                        dump=file(path, "dump.json", b"{}"),
                    ),
                )
            io = s.run(
                "inputs-c16", {"abi": graph["abi"]}, lambda f: dict(files={"x": file(f, "x.bin", b"x")})
            )
            key = dict(
                io=s.refs["inputs-c16"],
                models=[s.refs["atc-c16-" + m] for m in ("baseline", "optimized")],
                runner=runner,
                code={"experiment.py": "4266cd8e3a44ef6a037ad15a31e4a4dc736ec707ad4fd4fa01a2c0c7862471e2"},
            )
            s.run("reference-c16", key, lambda f: dict(reference=golden))

            def pair(folder, timed):
                return {
                    m: replay(folder / m, golden, timed, v, runner, models[m]["om"])
                    for m, v in (("baseline", 28.5), ("optimized", 19.1))
                }

            sm = dict(key, reference=golden)
            s.run("smoke-c16", sm, lambda f: pair(f, False))

            def benchmark(folder):
                rounds = [pair(folder / str(i), True) for i in range(3)]
                return dict(rounds=rounds, statistics=compare_rounds(rounds))

            b = s.run("benchmark-c16", dict(sm, smoke=s.refs["smoke-c16"]), benchmark)
            resumed = Stages(new, prior)
            resumed.completed("private-opp")
            with patch.object(Stages, "run", side_effect=AssertionError("re-executed a measured stage")):
                case, timing = read_gear(resumed, graph, 16, runner)
                self.assertEqual(timing, b["statistics"])
                self.assertEqual(case["io"], io)
                self.assertEqual(resumed.refs, s.refs)
                self.assertFalse(any(new.iterdir()))
                # A valid but unrelated report cannot stand in for the sealed smoke predecessor.
                p = c.verify(s.refs["benchmark-c16"])
                row = c.read(p)
                row["contract"]["smoke"] = s.refs["inputs-c16"]
                row["contract_sha256"] = hashlib.sha256(
                    json.dumps(row["contract"], sort_keys=True).encode()
                ).hexdigest()
                c.write(p, row)
                resumed.previous["benchmark-c16"] = c.identity(p)
                with self.assertRaisesRegex(ValueError, "predecessors"):
                    read_gear(resumed, graph, 16, runner)

    def test_cli_enters_profile_only_before_checkpoint_or_build_tools(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run = root / "runs/full-draft-panels-old"
            run.mkdir(parents=True)
            out = root / "new"
            out.mkdir()
            c.write(run / "stages.json", {})
            env = {"DFLASH_OUTPUT": str(out), "DFLASH_RUN_ROOT": d, "CANN_ROOT": d}
            with ExitStack() as ctx:
                ctx.enter_context(patch.dict(os.environ, env, clear=True))
                ctx.enter_context(patch.object(sys, "argv", ["experiment.py", "--profile-only", run.name]))
                capture = ctx.enter_context(patch.object(experiment, "profile_completed"))
                ctx.enter_context(
                    patch.object(experiment.shutil, "which", side_effect=AssertionError("toolchain used"))
                )
                experiment.main()
            self.assertEqual(capture.call_args.args[1:], (run.resolve(), root.resolve(), [16, 64]))
            with self.assertRaisesRegex(ValueError, "requires a completed stage"):
                Stages(out, run).completed("benchmark-c16")


if __name__ == "__main__":
    unittest.main()
