import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from elf_identity import equivalent
from elf_layout import EH, PH, SH, SYM, table

SYMBOL = "te_example_" + "b" * 64 + "__kernel0"


def elf(pid="607982", *, info=4, source_va=0, alias=False, load_string=False):
    names = [
        "",
        ".text",
        ".data",
        ".rela.text",
        ".ascend.meta.entry",
        ".comment",
        ".symtab",
        ".shstrtab",
        ".strtab",
    ]
    strings = b"\0" + b"\0".join(k.encode() for k in names[1:]) + b"\0"
    source = SYMBOL.split("__kernel")[0] + "_" + pid + "_kernel.cpp"
    labels = b"\0" + source.encode() + b"\0" + SYMBOL.encode() + b"\0"
    file_sym = SYM.pack(1, info, 0, 0xFFF1, 0, 0)
    puboff = labels.index(SYMBOL.encode())
    pubs = SYM.pack(labels.index(pid.encode()) if alias else puboff, 18, 0, 1, 0, 4)
    pieces = [
        b"",
        b"ABCD",
        b"12345678",
        b"RELADATA",
        b"metadata",
        b"ver",
        bytes(24) + file_sym + pubs,
        strings,
        labels,
    ]
    types = [0, 1, 1, 4, 7, 1, 2, 3, 3]
    flags = [0, 6, 3, 64, 0, 48, 0, 0, 0]
    out = bytearray(EH.size + PH.size)
    headers = []
    for i, (name, chunk) in enumerate(zip(names, pieces)):
        off = len(out) if i else 0
        headers.append(
            SH.pack(
                strings.index(name.encode() + b"\0") if i else 0,
                types[i],
                flags[i],
                source_va if i == 8 else 0,
                off,
                len(chunk),
                8 if i == 6 else 0,
                2 if i == 6 else 0,
                1,
                SYM.size if i == 6 else 0,
            )
        )
        out.extend(chunk)
    end = len(out)
    out.extend(bytes((-len(out)) % 8))
    shoff = len(out)
    out += b"".join(headers)
    ident = b"\x7fELF\x02\x01\x01" + bytes(9)
    EH.pack_into(
        out, 0, ident, 2, 0x1029, 1, 0, EH.size, shoff, 0x8A0080, EH.size, PH.size, 1, SH.size, len(names), 7
    )
    PH.pack_into(
        out, EH.size, 1, 5, EH.size + PH.size, 0, 0, (end - EH.size - PH.size) if load_string else 4, 4, 4
    )
    return bytes(out)


class FileSymbols(unittest.TestCase):
    def test_exact_metadata_only_and_variable_pid_width(self):
        a = elf()
        self.assertTrue(equivalent(a, a, SYMBOL)["exact_elf_equal"])
        for pid in ("629406", "9", "1234567"):
            b = elf(pid)
            r = equivalent(a, b, SYMBOL)
            self.assertEqual(r["status"], "LOCAL_STT_FILE_TAG_ONLY")
            self.assertFalse(r["exact_elf_equal"])
            self.assertNotEqual(r["raw_sha256"], r["candidate_sha256"])
            self.assertEqual(r["candidate_tag"]["tag"], pid)

    def test_full_byte_contract(self):
        a, b = elf(), elf("629406")
        for section in (".text", ".data", ".rela.text", ".ascend.meta.entry", ".comment", ".strtab"):
            s = next(s for s in table(b)[1] if s["name"] == section)
            d = bytearray(b)
            d[s["offset"] + (s["bytes"] - 3 if section == ".strtab" else 0)] ^= 2
            with self.subTest(section=section), self.assertRaises(ValueError):
                equivalent(a, bytes(d), SYMBOL)
        bad = bytearray(b)
        bad[48] ^= 1
        with self.assertRaises(ValueError):
            equivalent(a, bytes(bad), SYMBOL)

    def test_digit_string_must_be_one_unmapped_local_file_symbol(self):
        for kwargs in ({"info": 0}, {"source_va": 16}, {"alias": True}, {"load_string": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                equivalent(elf(**kwargs), elf("629406", **kwargs), SYMBOL)
        with self.assertRaises(ValueError):
            equivalent(elf(), elf("629406"), "other__kernel0")
