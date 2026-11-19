"""CPU checks of evidence accounting. Listings here are fixtures, never device evidence."""

import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import audit_isa as a
import inventory


class TestEvidence(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.blob = bytes(range(8))
        self.table = [
            dict(
                name=".text",
                va=32,
                offset=0,
                bytes=8,
                flags=6,
                sha256=hashlib.sha256(self.blob).hexdigest(),
            )
        ]
        path = self.root / "cpu.fixture"
        path.write_bytes(self.blob)
        self.image = dict(elf=inventory.identity(path), table=self.table)
        self.lines = "x: file format elf64-hiipu\nDisassembly of section .text:\n"
        self.good = "20: 00 01 02 03 add r0,r1\n24: 04 05 06 07 nop\n"

    def coverage(self, rows, rc=0):
        return a.text_coverage(self.blob, self.table, self.lines + rows, rc)["bytes"]

    def test_explicit_pc_hex_opcode_operands(self):
        self.assertEqual(self.coverage(self.good), 8)
        self.assertEqual(self.coverage(self.good.replace("add r0,r1", "add")), 4)
        self.assertEqual(self.coverage(self.good.replace("06 07", "ff ff")), 4)
        self.assertEqual(self.coverage(self.good.replace("24:", "40:")), 4)
        self.assertEqual(self.coverage(self.good, 1), 0)
        self.assertEqual(self.coverage("20: <not available>\n24: <unknown>"), 0)
        self.assertEqual(
            self.coverage(self.good.replace("add r0,r1", "word 0x42").replace("nop", ".word 0")),
            0,
        )
        self.assertEqual(self.coverage("20: 00 01 02 03 add r0,r1\n" * 2), 4)
        self.lines = self.lines.replace("elf64-hiipu", "elf64-aarch64")
        self.assertEqual(self.coverage(self.good), 0)

    def test_control_rejects_generic_mnemonics_even_with_all_bytes(self):
        path = self.root / "listing"
        path.write_text(self.lines + self.good)
        with patch.object(
            a,
            "command",
            return_value=dict(returncode=0, listing=inventory.identity(path)),
        ):
            rec = a.attempt(None, (), self.image, self.root, "probe", {}, True)
        self.assertEqual(rec["text_byte_witnesses"]["bytes"], 8)
        self.assertEqual(rec["outcome"], "REFERENCE_SEMANTICS_UNPROVEN")
        self.assertNotIn("instructions", rec)
        witnesses = a.reference_semantics(
            {
                ".text": [
                    dict(pc=i, encoding="00", asm=asm)
                    for i, asm in enumerate(["copy_gm_to_ubuf x1,x0", "mmad x2,x3", "pipe_barrier PIPE_ALL"])
                ]
            }
        )
        self.assertTrue(all(x["count"] == 1 for x in witnesses.values()))

    def test_control_binds_tool_and_invocation(self):
        controls = {a.HEAD: "ref", **{m: m for m in a.DOUBLE}}
        tool = {"path": "sdk"}
        other = {"path": "host"}
        probes = [({"file": v}, self.root, {}, [a.retry.PROBE]) for v in (tool, other)]

        def no_control(t, flags, image, folder, tag, env, check_control=False):
            return dict(
                outcome="ALL_PC_ROWS_UNDECODED" if check_control else "COMPLETE_BYTE_MATCH",
                verified_bytes=0 if check_control else 8,
                listing=tag,
            )

        with patch.object(a, "attempt", side_effect=no_control):
            trials, pairs = a.checked_audit_tools(probes, controls, {}, tool, tool)
        self.assertFalse(pairs)
        for m in a.DOUBLE:
            self.assertEqual(trials[0]["candidate_attempts"][m][0]["qualified_bytes"], 0)
            self.assertFalse(trials[1]["candidate_attempts"][m])

    def test_machine_evidence_is_required(self):
        self.assertTrue(all(v == {"status": "UNPROVEN", "pcs": []} for v in a.claims_for().values()))
        r = {
            "reason": "NO_DECODER_PASSED_INDEPENDENT_CONTROL",
            "status": "BLOCKED_STATIC_ADMISSION",
        }
        a.prove(self.root, r, {}, [], None)
        self.assertEqual(r["status"], "BLOCKED_STATIC_ADMISSION")
        self.assertIn(
            "UNPROVEN",
            a.markdown(
                dict(
                    status=r["status"],
                    reason=r["reason"],
                    images={a.HEAD: self.image},
                    control={},
                    timing={m: dict(claims=a.claims_for()) for m in a.DOUBLE},
                )
            ),
        )

    def test_architecture_and_bound_bytes(self):
        with self.assertRaises(ValueError):
            a.copy_elf(self.root / "out", self.blob + b"!", self.image["elf"])
        with self.assertRaises((ValueError, RuntimeError)):
            a.reference_image(self.image["elf"], self.root / "invalid.elf")

    def test_inventory_is_not_authority(self):
        self.assertEqual(inventory.role("dav_m200-disasm"), "possible_isa_tool")
        self.assertEqual(inventory.role("llvm-mc-15"), "llvm_mc")
        self.assertEqual(inventory.role("msobjdump"), "metadata_or_debug_dump")
        self.assertEqual(inventory.role("libLLVM-15.so.1"), "library_inspect_only")
        self.assertEqual(inventory.role("bisheng"), "compiler")
        self.assertIsNone(inventory.role("normal.sh"))
        alias = self.root / "objdump"
        alias.symlink_to(self.image["elf"]["path"])
        Path(self.image["elf"]["path"]).chmod(0o744)
        self.assertEqual(inventory.identity(alias), self.image["elf"])
        self.assertEqual(
            a.source_trust(
                dict(file=inventory.identity(alias), scopes=["PATH"]),
                self.root / "sdk",
                None,
            ),
            "INVENTORY_ONLY",
        )

    @unittest.skipUnless(os.environ.get("DAV_M200_KNOWN_DIR"), "retained SDK fixture on ARM64 Linux")
    def test_real_preexisting_dav_object(self):
        p = Path(os.environ["DAV_M200_KNOWN_DIR"]) / "probe.o"
        rid = inventory.identity(p)
        self.assertEqual(
            rid["sha256"],
            "f96a0c6ef3cfa824ff1095da08936ae4105317304d66ac5e36d2aa23e22a0d00",
        )
        obj = a.reference_image(rid, self.root / "known.elf")
        self.assertEqual(
            (obj["machine"], obj["elf_type"], obj["table"][0]["bytes"]),
            (0x1029, 1, 116),
        )
        sdk = Path("/usr/local/Ascend/cann-9.0.0/tools/bisheng_compiler/bin/llvm-objdump")
        tool = inventory.identity(sdk)
        self.assertEqual(
            tool["sha256"],
            "2b15d0cbe2eebdf8d230b76550c6d713e8c4855acf9cfb9f2a6089b113881b3c",
        )
        res = a.attempt(tool, a.ARCH, obj, self.root, "real-arch", dict(os.environ))
        self.assertEqual(
            (
                res["returncode"],
                res["pc_rows"],
                res["placeholder_rows"],
                res["verified_bytes"],
            ),
            (0, 29, 29, 0),
        )


if __name__ == "__main__":
    unittest.main()
