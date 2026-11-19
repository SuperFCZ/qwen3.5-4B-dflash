"""Offline contracts; fabricated test binaries/listings are never registered as trusted decoders."""

import copy
from contextlib import ExitStack
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import retry_decoder as retry
from w16_common import DOUBLE, FIRST, c


def fixture_elf(symbol, text):
    names = b"\0.text\0.shstrtab\0"
    data = text + symbol.encode() + b"\0" + names
    off = 64 + len(data)
    header = struct.pack(
        "<16sHHIQQQIHHHHHH", b"\x7fELF\x02\x01\x01" + bytes(9), 2, 0x1029, 1, 0, 0, off, 0, 64, 56, 0, 64, 3, 2
    )
    sh = lambda *v: struct.pack("<IIQQQQIIQQ", *v)
    return (
        header
        + data
        + bytes(64)
        + sh(1, 1, 6, 0, 64, len(text), 0, 0, 4, 0)
        + sh(7, 3, 0, 0, off - len(names), len(names), 0, 0, 1, 0)
    )


class ReplayTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dest = Path(temp.name).resolve()
        self.raw = b"abcdefgh"
        self.symbol = "fixture_only"
        self.blob = fixture_elf(self.symbol, self.raw)
        self.table = retry.sections(self.blob)
        self.header = "fixture: file format elf64-hiipu\nDisassembly of section .text:\n"
        self.placeholder = self.header + "0: <not available>\n4: <not available>\n"
        self.full = self.header + "0: 61 62 63 64  movi x0, 1\n4: 65 66 67 68  ret\n"
        self.sdk = self.dest / "sdk"
        self.tool = self.file("sdk/aarch64-linux/ccec_compiler/bin/llvm-objdump", b"unit test identity")
        Path(self.tool["path"]).chmod(0o755)
        self.cap = self.capture("case", self.blob)

    def file(self, name, data):
        p = self.dest / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, bytes):
            p.write_bytes(data)
        else:
            c.write(p, data)
        return c.identity(p)

    def capture(self, name, blob):
        elf = self.file(name + "/selected.elf", blob)
        om = self.file(name + "/original.om", b"IMOD" + blob)
        code = dict(bytes=len(blob), sha256=elf["sha256"], symbol=self.symbol, block_dim=8, offset=4)
        table = self.file(name + "/sections.json", retry.sections(blob))
        attempts = []
        for i, flags in enumerate(retry.LEGACY_FLAGS):
            f = self.file(f"{name}/log-{i}", self.placeholder.encode())
            attempts.append(dict(argv=[self.tool["path"], *flags, elf["path"]], returncode=0, listing=f))
        return dict(
            elf=elf,
            om=om,
            node=code,
            sections=table,
            tool=self.tool,
            attempts=attempts,
            status="BLOCKED_STATIC_ADMISSION",
            **attempts[-1],
        )

    def test_real_size_placeholder_and_false_zero_return(self):
        raw = b"x" * 25964
        table = [dict(flags=6, name=".text", va=0, offset=0, bytes=len(raw))]
        listing = self.header + "".join(f"{pc:8x}: <not available>\n" for pc in range(0, len(raw), 4))
        verdict, rows = retry.listing_evidence(raw, table, listing, 0)
        self.assertIsNone(rows)
        self.assertEqual(
            (verdict["pc_rows"], verdict["placeholder_rows"], verdict["verified_bytes"]), (6491, 6491, 0)
        )
        self.assertEqual(verdict["outcome"], "ALL_PC_ROWS_UNDECODED")
        for text, rc, status in (
            (self.header, 0, "NO_EXECUTABLE_PC_ROWS"),
            (self.full, 1, "NONZERO_RETURN"),
            (self.full, None, "NONZERO_RETURN"),
            (self.full.replace("movi x0, 1", "word 0x64636261"), 0, "DATA_DIRECTIVE_IN_EXECUTABLE_SECTION"),
            (self.full.replace("61 62", "65 66"), 0, "INCOMPLETE_OR_UNVERIFIED_LISTING"),
            (self.full.replace("4:", "8:"), 0, "INCOMPLETE_OR_UNVERIFIED_LISTING"),
            (self.full.split("4:")[0], 0, "INCOMPLETE_OR_UNVERIFIED_LISTING"),
            (self.full.replace("elf64-hiipu", "elf64-littleaarch64"), 0, "UNBOUND_DEVICE_LISTING_FORMAT"),
            (self.full.replace("65 66 67 68  ret", "<unknown>"), 0, "INCOMPLETE_OR_UNVERIFIED_LISTING"),
        ):
            v, rows = retry.listing_evidence(self.blob, self.table, text, rc)
            self.assertEqual(v["outcome"], status)
            self.assertEqual(v["verified_bytes"], 0)
            self.assertIsNone(rows)
        good, _ = retry.listing_evidence(self.blob, self.table, self.full, 0)
        self.assertEqual(good["verified_bytes"], 8)

    def test_unchanged_decoder_reuses_sealed_attempts(self):
        with patch.object(retry, "run_tool", side_effect=AssertionError("should not retry")):
            result, new = retry.retry_one(self.dest / "out", self.cap, self.tool, {}, False)
        self.assertEqual(result["outcome"], "NO_DECODED_INSTRUCTIONS")
        self.assertIsNone(new)
        self.assertEqual(len(result["archived_attempts"]), 2)
        self.assertEqual(result["attempts"], [])
        self.assertEqual(c.identity(self.cap["elf"]["path"]), self.cap["elf"])

    def test_supplemental_aicore_argv_is_not_a_canonical_proof(self):
        calls = []

        def fake(tool, flags, elf, log, env):
            calls.append(flags)
            log.write_text(self.header if flags == retry.AICORE else self.full)
            return dict(argv=[tool["path"], *flags, str(elf)], returncode=0, listing=c.identity(log))

        with patch.object(retry, "run_tool", side_effect=fake):
            record, new = retry.retry_one(self.dest / "out", self.cap, self.tool, {}, True)
        self.assertEqual(calls, [retry.AICORE, retry.AICORE_TARGET])
        self.assertIsNone(new)
        self.assertEqual(record["outcome"], "DECODED_INVOCATION_OUTSIDE_FROZEN_PROOF_CONTRACT")
        self.assertTrue((self.dest / "out/instructions.json").is_file())
        self.assertNotEqual(record["probe_elf"]["path"], self.cap["elf"]["path"])

    def test_different_tool_requires_exact_coverage_and_original_elf(self):
        other = self.file("sdk/bin/ccec-objdump", b"other test identity")

        def fake(tool, flags, elf, log, env):
            log.write_text(self.full if flags == retry.TARGET else self.placeholder)
            return dict(argv=[tool["path"], *flags, str(elf)], returncode=0, listing=c.identity(log))

        with patch.object(retry, "run_tool", side_effect=fake):
            rec, new = retry.retry_one(self.dest / "out", self.cap, other, {}, True)
        self.assertEqual(rec["outcome"], "COMPLETE_BYTE_MATCH")
        self.assertEqual(new["status"], "DECODED_REQUIRES_PC_PROOF")
        self.assertEqual(new["elf"]["sha256"], self.cap["node"]["sha256"])
        self.assertEqual(new["om"], self.cap["om"])
        self.assertEqual(new["argv"], [other["path"], *retry.TARGET, new["elf"]["path"]])
        blob, table = retry.bound_image(new)
        self.assertEqual(retry.archived_attempts(new, blob, table)[-1]["verified_bytes"], 8)

    def test_sha_root_and_alias(self):
        self.assertEqual(retry.select_tool(self.sdk, self.tool, None, None), self.tool)
        p = Path(self.tool["path"])
        alias = self.sdk / "bin/llvm-objdump"
        alias.parent.mkdir()
        alias.symlink_to(p)
        self.assertEqual(retry.select_tool(self.sdk, self.tool, alias, self.tool["sha256"]), self.tool)
        with self.assertRaises(ValueError):
            retry.select_tool(self.sdk, self.tool, alias, "0" * 64)
        with self.assertRaises(ValueError):
            retry.select_tool(self.sdk, self.tool, None, "a" * 64)
        external = self.file("foreign/ccec-objdump", b"foreign")
        escape = self.sdk / "bin/ccec-objdump"
        escape.symlink_to(external["path"])
        with self.assertRaises(ValueError):
            retry.select_tool(self.sdk, self.tool, escape, external["sha256"])

    def test_crossed_om_drift_and_archive_arguments_rejected(self):
        other = self.capture("another", fixture_elf(self.symbol, b"ABCDEFGH"))
        self.assertEqual(retry.compare_text(self.cap, other)["different_bytes_in_common_extent"], 8)
        with self.assertRaisesRegex(ValueError, "OM"):
            retry.bound_image(dict(self.cap, om=other["om"]))
        with self.assertRaisesRegex(ValueError, "section"):
            retry.bound_image(dict(self.cap, sections=other["sections"]))
        bad = copy.deepcopy(self.cap)
        bad["attempts"][0]["argv"][-1] = other["elf"]["path"]
        with self.assertRaises(ValueError):
            retry.archived_attempts(bad, self.blob, self.table)
        bad = copy.deepcopy(self.cap)
        bad["attempts"].pop(0)
        with self.assertRaises(ValueError):
            retry.archived_attempts(bad, self.blob, self.table)
        Path(self.cap["listing"]["path"]).write_text(self.full)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            retry.archived_attempts(self.cap, self.blob, self.table)

    def test_incomplete_cannot_publish_a_launchable_flow(self):
        captures = {m: {"node": self.cap} for m in DOUBLE}
        with self.assertRaisesRegex(ValueError, "incomplete"):
            retry.publish_flow(self.dest, {}, {}, {}, self.tool, captures, {}, None)
        self.assertFalse((self.dest / "flow").exists())

    def test_handoff_replays_passed_phases_and_stops_before_static(self):
        cap2 = self.capture("c2", fixture_elf(self.symbol, b"ABCDEFGH"))
        prior_caps = {DOUBLE[0]: {"gate": self.cap}, DOUBLE[1]: {"gate": cap2}}
        new_tool = self.file("sdk/bin/ccec-objdump", b"alternate fixture")

        def fake(tool, flags, elf, log, env):
            blob = elf.read_bytes()
            t = next(s for s in retry.sections(blob) if s["name"] == ".text")
            code = blob[t["offset"] : t["offset"] + t["bytes"]]
            log.write_text(self.header + f"0: {code[:4].hex(' ')}  movi x0, 1\n4: {code[4:].hex(' ')}  ret\n")
            return dict(argv=[tool["path"], *flags, str(elf)], returncode=0, listing=c.identity(log))

        captures = {}
        with patch.object(retry, "run_tool", side_effect=fake):
            for mode, nodes in prior_caps.items():
                _, ready = retry.retry_one(self.dest / mode, nodes["gate"], new_tool, {}, False)
                captures[mode] = {"gate": ready}
        original = self.dest / "archive"
        common = dict(history_roots=[], sources=retry.flow.sources(), atc="NOT_RUN", root=str(original))
        chain = dict(
            source=dict(common, binding={"tool": self.tool}),
            build=dict(common, variants={}, workers={}),
            atc=dict(common, models={}, captures={FIRST: prior_caps}, baseline_images={FIRST: {}}),
        )
        refs = {phase: self.file("archive/" + phase + "/report.json", data) for phase, data in chain.items()}
        witness = self.file("witness.json", {"scope": "CPU_FIXTURE"})

        def validate(path, row, accepted, assets):
            self.assertEqual(list(accepted), list(row["predecessors"]))
            self.assertEqual(row["replay"]["original"], refs[row["phase"]])
            self.assertEqual(row["device_execution"], "NOT_RUN")

        with ExitStack() as stack:
            checks = stack.enter_context(patch.object(retry.flow, "validate", side_effect=validate))
            admission = stack.enter_context(patch.object(retry.flow, "admit", return_value={"tool": new_tool}))
            result = retry.publish_flow(self.dest, chain, None, refs, new_tool, captures, {}, witness)
        self.assertEqual(checks.call_count, 3)
        admission.assert_called_once_with(new_tool["path"])
        self.assertEqual(c.read(self.dest / "flow/source/report.json")["binding"]["tool"], new_tool)
        self.assertEqual(c.read(refs["source"]["path"])["binding"]["tool"], self.tool)
        self.assertEqual(result["static"], "NOT_RUN")
        self.assertFalse((self.dest / "flow/static/report.json").exists())
        with self.assertRaises(ValueError):
            retry.flow.previous(self.dest / "flow", "static")
        # Even a successful-looking capture with an instruction JSON from another ELF must fail.
        captures[DOUBLE[0]]["gate"]["instructions"] = captures[DOUBLE[1]]["gate"]["instructions"]
        with self.assertRaisesRegex(ValueError, "supplemental machine listing"):
            retry.publish_flow(self.dest, chain, None, refs, new_tool, captures, {}, witness)

    def test_blocked_probe_does_not_create_stage_receipts(self):
        base = self.dest / "out"
        base.mkdir()
        cap2 = self.capture("c2", fixture_elf(self.symbol, b"ABCDEFGH"))
        chain = {
            "atc": {
                "status": "W16_ATC_PASS",
                "captures": {FIRST: {DOUBLE[0]: {"gate": self.cap}, DOUBLE[1]: {"gate": cap2}}},
            },
            "source": {"status": "W16_SOURCE_PASS", "binding": {"tool": self.tool}},
        }
        src = {"cann": str(self.sdk), "host_source": self.file("host.json", {"binding": {"prior": {}}})}

        def fake(tool, flags, elf, log, env):
            self.assertEqual(flags, ["--help"])
            log.write_text("fixture help (without AI Core selector)")
            return dict(returncode=0, listing=c.identity(log), argv=[tool["path"], *flags])

        with ExitStack() as stack:
            stack.enter_context(patch.object(retry, "load_archive", return_value=({}, chain, (src, {}, {}))))
            stack.enter_context(
                patch.object(retry.host.envs, "for_prior", return_value=({"ASCEND_CUSTOM_OPP_PATH": ""}, {}))
            )
            stack.enter_context(patch.object(retry, "run_tool", side_effect=fake))
            stack.enter_context(patch.object(retry, "publish_flow", side_effect=AssertionError("published")))
            r = {}
            retry.replay(base, self.dest, None, None, True, r)
        self.assertEqual(r["reason_code"], "SDK_OFFSETS_ONLY")
        self.assertEqual(len(r["variants"]), 2)
        self.assertFalse((base / "flow").exists())

    @unittest.skipUnless(os.environ.get("W16_LOCAL_BUILD_REPORT"), "optional real ARM64 CANN package ELF")
    def test_installed_decoder_capability(self):
        r = json.loads(Path(os.environ["W16_LOCAL_BUILD_REPORT"]).read_text())
        for mode in DOUBLE:
            original = r["variants"][mode]
            tool = original["objdump"]
            self.assertEqual(tool["sha256"], "2b15d0cbe2eebdf8d230b76550c6d713e8c4855acf9cfb9f2a6089b113881b3c")
            # Explicitly check a package ELF only, not a substitute for ATC or a proof.
            pkg = c.verify(original["installed_kernel"])
            elf = self.dest / (mode + ".elf")
            elf.write_bytes(pkg.read_bytes())
            table = retry.sections(elf.read_bytes())
            for i, flags in enumerate((*retry.LEGACY_FLAGS, retry.AICORE, retry.AICORE_TARGET)):
                rec = retry.run_tool(tool, flags, elf, self.dest / f"{mode}-{i}.txt", dict(os.environ))
                measure, decoded = retry.listing_evidence(
                    elf.read_bytes(), table, c.verify(rec["listing"]).read_text(), rec["returncode"]
                )
                self.assertEqual(rec["returncode"], 0)
                self.assertIsNone(decoded)
                self.assertEqual(measure["verified_bytes"], 0)
                self.assertEqual(
                    measure["outcome"],
                    "NO_EXECUTABLE_PC_ROWS" if flags == retry.AICORE else "ALL_PC_ROWS_UNDECODED",
                )

    def test_original_manifest_excludes_offline_retry(self):
        self.assertFalse(any(HERE in Path(x["path"]).parents for x in retry.flow.sources()))
        self.assertIn(c.identity(HERE / "retry_decoder.py"), retry.sources())


if __name__ == "__main__":
    unittest.main()
