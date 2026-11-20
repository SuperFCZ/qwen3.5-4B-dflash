"""Gate8 tool provenance and sealed Source/Build/ATC continuation on CPU."""

from contextlib import ExitStack, redirect_stdout
import hashlib
import io
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import run_scale as flow
import scale_reuse as reuse
import scale_review as review
from scale_contracts import CANDIDATE, c, layout_contract


class CpuToolsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        tools = {}
        for name, real in (("cmake", "real-cmake"), ("c++", "real-g++")):
            p = self.root / real
            p.write_bytes(real.encode())
            p.chmod(0o755)
            alias = self.root / name
            alias.symlink_to(p.name)
            tools[name] = dict(alias=str(alias), file=c.identity(p))
        self.tools = tools
        self.raw = {"compiler": {"command_path_tools": tools}}

    def test_run_cpu_uses_verified_gate8_alias_and_target(self):
        prior = self.root / "host.json"
        c.write(prior, {"binding": {"prior": {}}})
        src = {"host_source": c.identity(prior), "cann": "/selected-cann"}
        out = self.root / "sdk-cpu"
        with patch.object(
            review.host.envs, "for_prior", return_value=({"ASCEND_CUSTOM_OPP_PATH": ""}, {})
        ), patch.object(
            review, "command", side_effect=RuntimeError("stop before host compilation")
        ) as call, self.assertRaisesRegex(RuntimeError, "stop before host compilation"):
            review.run_cpu(out, src, self.raw)
        args = call.call_args.args[0]
        self.assertEqual(args[0], self.tools["cmake"]["file"]["path"])
        self.assertIn("-DCMAKE_CXX_COMPILER=" + self.tools["c++"]["file"]["path"], args)
        self.assertEqual(call.call_args.kwargs["cwd"], out)
        self.assertEqual(review.cpu_build_tools(self.raw), self.tools)

    def test_never_falls_back_to_the_candidate_schema(self):
        with self.assertRaisesRegex(ValueError, "command_path_tools"):
            review.cpu_build_tools({"compiler": {"tools": {k: v["file"] for k, v in self.tools.items()}}})
        del self.tools["c++"]
        with self.assertRaisesRegex(ValueError, "c\\+\\+"):
            review.cpu_build_tools(self.raw)

    def test_retarget_even_byte_identical_or_drift_is_blocked(self):
        entry = self.tools["c++"]
        alias = Path(entry["alias"])
        target = Path(entry["file"]["path"])
        other = self.root / "same-contents"
        shutil.copy2(target, other)
        alias.unlink()
        alias.symlink_to(other)
        with self.assertRaisesRegex(ValueError, "alias/permission"):
            review.cpu_build_tools(self.raw)
        alias.unlink()
        alias.symlink_to(target)
        target.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "alias/permission"):
            review.cpu_build_tools(self.raw)
        target.chmod(0o755)
        target.write_bytes(b"x" * entry["file"]["bytes"])
        with self.assertRaisesRegex(ValueError, "identity changed"):
            review.cpu_build_tools(self.raw)

    def test_replay_rejects_substituted_compiler_journal(self):
        model = self.root / "build/scale_cpu"
        model.parent.mkdir()
        model.touch()
        scale = c.identity(model)
        r = dict(
            binary=scale,
            scope="SDK_CPU_MODEL_ONLY",
            frozen_scale=scale,
            compiler_tools=self.tools,
            build_logs=[],
        )
        for name, argv in review.cpu_build_commands(self.root, "/sdk", self.tools).items():
            p = self.root / (name + ".command.json")
            c.write(p, dict(argv=["cmake", *argv[1:]], cwd=str(self.root)))
            r["build_logs"].append(c.identity(p))
        with patch.dict(review.FIRST_HASHES, {"s_gn.bin": scale["sha256"]}), self.assertRaisesRegex(
            ValueError, "compiler invocation drift"
        ):
            review.validate_cpu(r, self.root, {"cann": "/sdk"}, scale, self.raw)

    @unittest.skipUnless(
        os.environ.get("SCALE_GATE8_COMPILE_REPORT"), "requires the real archived CANN ARM64 compiler record"
    )
    def test_real_archived_gate8_tools_build_and_run_vendor_checker(self):
        hist = c.read(Path(os.environ["SCALE_GATE8_COMPILE_REPORT"]))["compilers"]["raw"]
        self.assertNotIn("tools", hist)
        cann = os.environ["SCALE_CANN_ROOT"]
        env = dict(
            PATH="/usr/bin:/bin",
            HOME=str(self.root),
            TMPDIR=str(self.root),
            LC_ALL="C",
            ASCEND_CUSTOM_OPP_PATH="",
            CANN_ROOT=cann,
            ASCEND_HOME_PATH=cann,
            ASCEND_TOOLKIT_HOME=cann,
            LD_LIBRARY_PATH=cann + "/lib64",
        )
        bound, exe = review.compile_cpu_checker(self.root, cann, {"compiler": hist}, env)
        self.assertEqual(bound, {k: hist["command_path_tools"][k] for k in ("c++", "cmake")})
        words = self.root / "64-words.bin"
        words.write_bytes(bytes(range(128)))
        log = self.root / "model.log"
        review.command([c.verify(exe), words], log, env, cwd=self.root)
        self.assertIn(review.expected_line(64), log.read_text().splitlines())


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.old = self.base / "sealed"
        self.new = self.base / "new"
        self.archive = self.base / "gate8"
        self.archive.mkdir()
        self.checkout = self.base / "module"
        self.checkout.mkdir()

        def put(name, value):
            p = self.checkout / name
            p.write_bytes(value)
            return c.identity(p)

        self.original_code = b"old read code"
        self.unchanged = put("kernel.inc", b"original payload")
        old_lock = [self.unchanged, put("reader.py", self.original_code)]
        self.current = [
            self.unchanged,
            put("reader.py", b"repaired read code"),
            put("resume.py", b"continuation"),
        ]
        base = dict(
            schema=1,
            root=str(self.old),
            archive=str(self.archive),
            sources=old_lock,
            scope="C16_LAYER0_GATE_SCALE_EXPERIMENT",
        )
        refs = {}
        self.binding = {"frozen": {"s_gn.bin": self.unchanged}}
        self.candidate = {"opp": {"install": str(self.old / "build/package/opp")}}
        for s in ("source", "build", "atc", "review"):
            folder = self.old / s
            folder.mkdir(parents=True)
            r = dict(
                **base,
                phase=s,
                folder=str(folder),
                predecessors=dict(refs),
                error=None,
                integrity_error=None,
                status=flow.STATUS[s],
                atc="NOT_RUN",
                device_execution="NOT_RUN",
                benchmark="NOT_RUN",
            )
            if s == "source":
                r.update(binding=self.binding, layout=layout_contract(), sdk={})
            if s == "build":
                w = folder / "worker.json"
                c.write(w, dict(status="SCALE_PRIVATE_BUILD_RECORDED", candidate=self.candidate))
                r.update(candidate=self.candidate, worker=c.identity(w))
            if s == "atc":
                p = folder / CANDIDATE / "om-graph.json"
                p.parent.mkdir()
                p.write_bytes(b"archive graph")
                r.update(
                    candidate={"dump": c.identity(p)},
                    safety={"cpu_fixture": True},
                    atc="COMPILED_NOT_EXECUTED",
                )
            if s == "review":
                r.update(status="BLOCKED", error="KeyError('tools')")
            path = folder / "report.json"
            c.write(path, r)
            refs[s] = c.identity(path)
        self.refs = refs
        st = ExitStack()
        self.addCleanup(st.close)
        st.enter_context(patch.object(reuse, "HERE", self.checkout))
        st.enter_context(patch.dict(reuse.REVISED, {"reader.py": self.current[1]["sha256"]}, clear=True))
        st.enter_context(patch.object(reuse, "ADDED", {"resume.py"}))
        st.enter_context(patch.object(reuse, "original_bytes", return_value=self.original_code))
        st.enter_context(patch.object(reuse, "FAILED_SHA", refs["review"]["sha256"]))
        st.enter_context(patch.object(reuse, "FAILED_BYTES", refs["review"]["bytes"]))
        st.enter_context(
            patch.dict(reuse.PREFIX, {s: refs[s]["sha256"] for s in ("source", "build", "atc")}, clear=True)
        )
        st.enter_context(patch.object(flow, "sources", return_value=self.current))
        st.enter_context(
            patch.object(flow, "read_baseline", return_value=(self.binding, {"cann": "sdk"}, {}, {}))
        )
        st.enter_context(patch.object(flow, "sdk_files", return_value={}))
        self.build_replay = st.enter_context(patch.object(flow, "validate_package"))
        self.atc_replay = st.enter_context(
            patch.object(flow, "check_model", return_value={"cpu_fixture": True})
        )
        self.cpu = st.enter_context(patch.object(flow, "run_cpu", return_value={}))
        st.enter_context(patch.object(flow, "validate_cpu"))
        self.builder = st.enter_context(
            patch.object(flow, "command", side_effect=AssertionError("recompiled"))
        )
        self.atc = st.enter_context(
            patch.object(flow, "compile_candidate", side_effect=AssertionError("ran ATC"))
        )
        self.runner = st.enter_context(
            patch.object(flow, "case_run", side_effect=AssertionError("ran device"))
        )

    def step(self, phase="review", allow=False):
        with redirect_stdout(io.StringIO()):
            return flow.run(
                SimpleNamespace(
                    root=self.new, archive=self.archive, phase=phase, allow_device=allow, resume_from=self.old
                )
            )

    def test_continues_at_review_and_links_old_reports_without_copying(self):
        self.assertEqual(self.step(), 0)
        r = c.read(self.new / "review/report.json")
        self.assertEqual(r["status"], "SCALE_REVIEW_PASS")
        self.assertEqual(r["reuse"]["failed_review"], self.refs["review"])
        self.assertEqual(r["predecessors"], {s: self.refs[s] for s in ("source", "build", "atc")})
        self.assertEqual(r["sources"], self.current)
        self.assertEqual([p.name for p in self.new.iterdir()], ["review"])
        self.assertEqual(r["device_execution"], "NOT_RUN")
        self.assertTrue(all(c.verify(x).is_file() for x in self.refs.values()))
        self.assertEqual(c.read(self.old / "review/report.json")["status"], "BLOCKED")
        self.build_replay.assert_called_once()
        self.atc_replay.assert_called_once()
        self.cpu.assert_called_once()
        self.builder.assert_not_called()
        self.atc.assert_not_called()
        self.runner.assert_not_called()
        # The new review is a live, revalidated prerequisite for any manual device invocation.
        r["status"] = "BLOCKED"
        c.write(self.new / "review/report.json", r)
        with self.assertRaises(ValueError):
            self.step("smoke", True)
        self.runner.assert_not_called()

    def test_rejects_wrong_root_upstream_drift_and_recompilation(self):
        self.new = self.old
        with self.assertRaises(ValueError):
            self.step()
        self.new = self.base / "new"
        with self.assertRaises(ValueError):
            self.step("atc")
        self.assertFalse(self.new.exists())
        p = self.old / "atc" / CANDIDATE / "om-graph.json"
        p.write_bytes(b"changed graph")
        self.assertEqual(self.step(), 1)
        self.cpu.assert_not_called()
        self.runner.assert_not_called()

    def test_next_stage_replays_new_review_and_uses_original_model(self):
        self.assertEqual(self.step(), 0)
        launched = []

        def cpu_transport(out, phase, src, raw, control, chain, report, *, launch=False):
            self.assertEqual(phase, "smoke")
            self.assertEqual(chain["atc"]["root"], str(self.old))
            self.assertEqual(chain["review"]["root"], str(self.new))
            if launch:
                launched.append(str(out))
                report["device_execution"] = "COMPLETED"
                report["runs"] = []
            return None

        with patch.object(flow, "activity", side_effect=cpu_transport):
            self.assertEqual(self.step("smoke", True), 0)
        smoke = c.read(self.new / "smoke/report.json")
        self.assertEqual(smoke["predecessors"]["atc"], self.refs["atc"])
        self.assertEqual(smoke["predecessors"]["review"], c.identity(self.new / "review/report.json"))
        self.assertEqual(launched, [str(self.new / "smoke")])
        self.runner.assert_not_called()
        self.builder.assert_not_called()
        self.atc.assert_not_called()

    def test_refuses_modified_receipt_or_failure_hash(self):
        with (self.old / "build/report.json").open("ab") as f:
            f.write(b" ")
        self.assertEqual(self.step(), 1)
        self.cpu.assert_not_called()
        self.new = self.base / "other"
        with (self.old / "review/report.json").open("ab") as f:
            f.write(b" ")
        self.assertEqual(self.step(), 1)
        self.cpu.assert_not_called()

    def test_cannot_accept_arbitrary_source_or_different_git_blob(self):
        data = c.read(self.old / "review/report.json")["sources"]
        with patch.object(reuse, "original_bytes", return_value=b"wrong git bytes"), self.assertRaises(
            ValueError
        ):
            reuse.verify_transition(data, self.current)
        p = Path(self.unchanged["path"])
        p.write_bytes(b"drift")
        with self.assertRaises(ValueError):
            reuse.verify_transition(data, [c.identity(p), *self.current[1:]])


class RepairSourceLockTests(unittest.TestCase):
    def test_published_fix_only_authorizes_exact_operational_changes(self):
        for rel, digest in reuse.REVISED.items():
            self.assertEqual(c.sha(HERE / rel), digest)
        self.assertTrue(all((HERE / n).is_file() for n in reuse.ADDED))
        if shutil.which("git"):
            for rel in reuse.REVISED:
                self.assertNotEqual(
                    hashlib.sha256(reuse.original_bytes(HERE / rel)).hexdigest(), reuse.REVISED[rel]
                )


if __name__ == "__main__":
    unittest.main()
