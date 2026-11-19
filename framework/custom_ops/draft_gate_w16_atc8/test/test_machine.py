"""Byte/parser and certificate fixtures only. No fixture is eligible as a published ATC proof."""

import copy
from pathlib import Path
import struct
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import w16_static as static
from w16_common import CUBE_SHA, OVERLAP, SERIAL, c
from w16_operands import ACCESSES, consumed_address


def elf(code):
    names = b"\0.text\0.shstrtab\0.symtab\0.strtab\0"
    sym = bytes(24) + struct.pack("<IBBHQQ", 1, 18, 0, 1, 0, len(code))
    strings = b"\0fixture\0"
    syms = 64 + len(code) + len(names)
    strs = syms + len(sym)
    start = strs + len(strings)
    hdr = struct.pack(
        "<16sHHIQQQIHHHHHH",
        b"\x7fELF\x02\x01\x01" + b"\0" * 9,
        2,
        0x1029,
        1,
        0,
        0,
        start,
        0,
        64,
        56,
        0,
        64,
        5,
        2,
    )
    sh = lambda *args: struct.pack("<IIQQQQIIQQ", *args)
    return (
        hdr
        + code
        + names
        + sym
        + strings
        + bytes(64)
        + sh(1, 1, 6, 0, 64, len(code), 0, 0, 4, 0)
        + sh(7, 3, 0, 0, 64 + len(code), len(names), 0, 0, 1, 0)
        + sh(names.index(b".symtab"), 2, 0, 0, syms, len(sym), 4, 1, 8, 24)
        + sh(names.index(b".strtab"), 3, 0, 0, strs, len(strings), 0, 0, 1, 0)
    )


def build_fixture(folder, mode):
    instructions = []

    def add(asm):
        pc = 4 * len(instructions)
        instructions.append(asm)
        return dict(pc=pc, asm=asm)

    definitions = {}
    for key, (region, opcode, operand) in ACCESSES.items():
        base = static.REGIONS[region][0]
        first = add("movi x0, " + hex(base))
        mem = add(opcode + (" x0, x2" if operand == 0 else " x2, x0"))
        definitions[key] = dict(blocks=[[first["pc"], mem["pc"]]], pc=mem["pc"])
    steady = ["set_s", "set_v", "signal_ready", "ready", "read_b", "cube", "raw_s", "raw_v"]
    d = ["cast"] * 8 + (["broadcast"] * 8 + ["mul"]) * 2
    tail = d + ["all"] if mode == OVERLAP else ["all"] + d
    paths = {}
    asm = {
        "ready": "wait_flag pipe_v, pipe_s, x0",
        "read_b": "copy_ubuf_to_cbuf x2, x0",
        "cube": "mmad x2",
        "raw_s": "wait_flag pipe_mte2, pipe_s, x0",
        "raw_v": "wait_flag pipe_mte2, pipe_v, x1",
        "cast": "vconv_s82f16 x0, x1",
        "broadcast": "vtranspose x2, x3",
        "mul": "vmul x0, x0, x1",
        "set_s": "set_flag pipe_mte2, pipe_s, x0",
        "set_v": "set_flag pipe_mte2, pipe_v, x1",
        "signal_ready": "set_flag pipe_v, pipe_s, x0",
        "all": "pipe_barrier all",
        "output": "copy_ubuf_to_gm x3, x0",
        "release": "sbitset0 x3, x4",
        "end": "ret",
    }
    branches = {}
    for key, actions in {
        "first": ["set_s", "set_v", "raw_s", "raw_v"] + d + steady + tail,
        "odd": steady + tail,
        "even": steady + tail,
        "last": ["signal_ready", "ready", "read_b", "cube", "all", "output", "all"],
        "rebind": ["all", "signal_ready", "ready", "read_b", "cube"],
        "false_next": ["raw_s", "raw_v", "all", "release", "release", "end"],
        "false_last": ["all", "release", "release", "end"],
    }.items():
        if key == "false_next":
            pc = 4 * len(instructions)
            target = pc + 4 * (1 + len(actions))
            branches[pc] = target
            add(f"b.eq 0x{target:x} <false_last>")
        events = [dict(action=a, walk_index=i, **add(asm[a])) for i, a in enumerate(actions)]
        paths[key] = dict(blocks=[[events[0]["pc"], events[-1]["pc"]]], events=events)
    words = [struct.pack("<I", i + 1) for i in range(len(instructions))]
    blob = elf(b"".join(words))
    ep = folder / "selected.elf"
    ep.write_bytes(blob)
    listing = "Disassembly of section .text:\n" + "".join(
        f"{i * 4:x}: " + w.hex(" ") + "  " + v + "\n" for i, (w, v) in enumerate(zip(words, instructions))
    )
    log = folder / "out.asm"
    log.write_text(listing)
    ip = folder / "instructions.json"
    c.write(ip, static.decode(blob, static.sections(blob), listing))
    kernel = dict(sha256=static.digest(blob), bytes=len(blob), symbol="fixture")
    src = dict(sha256="source")
    plan = dict(parent_sha256="parent")
    table = folder / "sections.json"
    c.write(table, static.sections(blob))
    cap = dict(
        node=kernel,
        elf=c.identity(ep),
        tool=dict(sha256="fixture-only", path="/fixture"),
        listing=c.identity(log),
        instructions=c.identity(ip),
        sections=c.identity(table),
        returncode=0,
        argv=["/fixture", "--disassemble", str(ep.resolve(strict=True))],
    )
    bind = dict(
        elf=kernel["sha256"],
        decoder="fixture-only",
        sources="source",
        instructions=cap["instructions"]["sha256"],
        cube=CUBE_SHA,
        parent="parent",
        sdk={},
        mode=mode,
        entry="fixture",
    )
    claim = dict(derivation="synthetic negative-test fixture", witnesses=[dict(pc=0, asm=instructions[0])])
    cfg = [
        dict(
            first=4 * i,
            last=4 * i,
            derivation="fixture",
            successors=[]
            if op == "ret"
            else [4 * i + 4, branches[4 * i]]
            if 4 * i in branches
            else [4 * i + 4],
        )
        for i, op in enumerate(instructions)
    ]
    packet = dict(
        schema=1,
        binding=bind,
        cfg=cfg,
        regions={k: dict(claim, byte_interval=list(span)) for k, span in static.REGIONS.items()},
        obligations={k: claim for k in static.OBLIGATIONS},
        accesses=definitions,
        observed_placement="before_fence" if mode == OVERLAP else "after_fence",
        cube_drained_before_next_d=mode != OVERLAP,
        paths=paths,
    )
    return packet, cap, mode, src, plan, {}


class MachineProofTests(unittest.TestCase):
    def test_whole_cfg_and_set_wait_identity(self):
        from w16_cfg import reachable_pcs
        from w16_operands import paired_flags

        ins = [
            dict(pc=4 * i, encoding="00000000", asm=op)
            for i, op in enumerate(
                ("set_flag pipe_mte2, pipe_s, x3", "movi x3, 4", "wait_flag pipe_mte2, pipe_s, x3", "ret")
            )
        ]
        with self.assertRaisesRegex(ValueError, "definition changed"):
            paired_flags(ins, [dict(action="set_s", walk_index=0), dict(action="raw_s", walk_index=2)])
        cfg = [dict(first=0, last=12, derivation="fixture", successors=[])]
        self.assertEqual(reachable_pcs(cfg, ins, 0), {0, 4, 8, 12})
        ins[1]["asm"] = "call indirect_helper"
        with self.assertRaisesRegex(ValueError, "unreviewed"):
            reachable_pcs(cfg, ins, 0)

    def test_both_orders_and_fail_closed_mutations(self):
        with tempfile.TemporaryDirectory() as tmp:
            for mode in (SERIAL, OVERLAP):
                args = build_fixture(Path(tmp), mode)
                static.check_proof(*args)
                tests = (
                    "region",
                    "elf",
                    "cube",
                    "scope",
                    "path",
                    "address",
                    "fence",
                    "lost_op",
                    "generation",
                    "unbound",
                )
                for fault in tests:
                    copy_args = copy.deepcopy(args)
                    p = copy_args[0]
                    if fault == "region":
                        p["regions"]["w1"]["byte_interval"][0] -= 32768
                    if fault == "elf":
                        p["binding"]["elf"] = "other"
                    if fault == "cube":
                        p["binding"]["cube"] = "different"
                    if fault == "scope":
                        p["obligations"].pop("all_reachable_blocks_and_relocations")
                    if fault == "path":
                        p["paths"]["odd"]["events"].reverse()
                    if fault == "address":
                        p["accesses"]["w1_b_read"] = p["accesses"]["w0_b_read"]
                    if fault == "fence":
                        p["paths"]["last"]["events"][3]["asm"] = "nop"
                    if fault == "lost_op":
                        p["paths"]["first"]["events"].pop(3)
                    if fault == "generation":
                        p["paths"].pop("even")
                    if fault == "unbound":
                        p["accesses"]["q0_dma"]["blocks"][0][0] += 4
                    with self.subTest(mode=mode, fault=fault), self.assertRaises(ValueError):
                        static.check_proof(*copy_args)
                if mode == OVERLAP:
                    args[0]["cube_drained_before_next_d"] = True
                    with self.assertRaises(static.NoOverlap):
                        static.check_proof(*args)

    def test_register_clobber_cannot_be_paid_for_by_expected_address(self):
        walk = [dict(asm="movi x3, 0x8000"), dict(asm="addi x4, x3, 512"), dict(asm="copy_ubuf_to_cbuf x3, x4")]
        self.assertEqual(consumed_address(walk, "copy_ubuf_to_cbuf", 1), 33280)
        walk.insert(-1, dict(asm="unknown_instruction x4, x7"))
        with self.assertRaises(ValueError):
            consumed_address(walk, "copy_ubuf_to_cbuf", 1)
