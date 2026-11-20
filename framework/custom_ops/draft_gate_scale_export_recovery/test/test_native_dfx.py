import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from elf_layout import EH, SH, SYM, table
from native_dfx import compare, records, arguments, runtime_argument_count

CAST = "te_fused_op_cast_mul_0f1ae9a466224384a342327842ae85af777e3d492cc66742556f824ab0a191d7__kernel0"
SLICE = "te_stridedsliced_34f36fc7fe228ae678e7b4ee8502b76747508afd202bc2446c8e50c2590f7788__kernel0"
# Exact function-note bytes returned for the sealed 310P3 models.
CAST_NOTES = tuple(
    bytes.fromhex(x)
    for x in (
        "010004000100000006000400000000000400b400000500070000000000010001000000000001000000000000000000040000000000000001000000000000001000000000000000080000000000000080000500070000000000010001000000000000000100000000000000040000000000000001000000000000000100000000000000010000000000000080000500070000000000010001000000000000800000000000000000040000000000000001000000000000001000000000000000080000000000000080",
        "0100040001000000060004000000000004003c00000500020000000000010001ffffffffffffffff000500020000000000010001ffffffffffffffff000500020000000000010001ffffffffffffffff",
    )
)
SLICE_NOTES = tuple(
    bytes.fromhex(x)
    for x in (
        "010004000100000006000400000000000400480000050004000000000001000100000000000000800000000000000001000000000000001000050004000000000001000100000000000000780000000000000001000000000000000f",
        "0100040001000000060004000000000004002800000500020000000000010001ffffffffffffffff000500020000000000010001ffffffffffffffff",
    )
)


def obj(symbol, note):
    names = [
        "",
        ".strtab",
        ".text",
        ".ascend.meta." + symbol,
        ".comment",
        ".note.GNU-stack",
        ".llvm_addrsig",
        ".symtab",
    ]
    strings = b"\0" + b"\0".join(n.encode() for n in names[1:]) + b"\0" + symbol.encode() + b"\0"
    sym = (
        bytes(24)
        + SYM.pack(1, 1, 0, 3, 0, len(note))
        + SYM.pack(strings.index(symbol.encode()), 18, 0, 2, 0, 4)
    )
    chunks = [b"", strings, b"code", note, b"compiler", b"", b"x", sym]
    data = bytearray(64)
    headers = []
    for i, (n, chunk) in enumerate(zip(names, chunks)):
        off = len(data) if i else 0
        headers.append(
            SH.pack(
                strings.index(n.encode() + b"\0") if i else 0,
                [0, 3, 1, 7, 1, 1, 0x6FFF4C03, 2][i],
                [0, 0, 6, 0, 48, 0, 0x80000000, 0][i],
                0,
                off,
                len(chunk),
                1 if i == 7 else 0,
                0,
                1,
                24 if i == 7 else 0,
            )
        )
        data.extend(chunk)
    data.extend(bytes((-len(data)) % 8))
    off = len(data)
    data += b"".join(headers)
    EH.pack_into(
        data, 0, b"\x7fELF\x02\x01\x01" + bytes(9), 1, 4137, 1, 0, 0, off, 9043968, 64, 0, 0, 64, 8, 1
    )
    return bytes(data)


class NativeMetadata(unittest.TestCase):
    def test_actual_both_record_sets_and_runtime_receiver(self):
        for name, notes, argc in ((CAST, CAST_NOTES, 3), (SLICE, SLICE_NOTES, 2)):
            old, new = (obj(name, n) for n in notes)
            p = compare(old, new, name)
            self.assertEqual(p["user_args_num"], argc)
            self.assertEqual(p["changed_argument_positions"], list(range(argc)))
            self.assertFalse(p["exact_elf_equal"])
            for note in notes:
                arg = records(note)[2][1]
                self.assertEqual(len(arguments(arg)), runtime_argument_count(arg))

    def test_runtime_metadata_count_and_concrete_drift_block(self):
        n = bytearray(CAST_NOTES[1])
        for offset in (4, 12, 22, 28, 38):
            bad = n.copy()
            bad[offset] ^= 2
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                compare(obj(CAST, CAST_NOTES[0]), obj(CAST, bytes(bad)), CAST)
        bad = bytearray(CAST_NOTES[0])
        bad[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "concrete"):
            compare(obj(CAST, CAST_NOTES[0]), obj(CAST, bytes(bad)), CAST)
        with self.assertRaises(ValueError):
            compare(obj(CAST, CAST_NOTES[0]), obj(CAST, SLICE_NOTES[1]), CAST)
        with self.assertRaises(ValueError):
            arguments(records(CAST_NOTES[0])[2][1][:-1])

    def test_text_symbol_tables_notes_headers_all_stay_exact(self):
        a, b = [obj(SLICE, n) for n in SLICE_NOTES]
        for key in (".text", ".strtab", ".comment", ".llvm_addrsig", ".symtab"):
            s = next(q for q in table(b)[1] if q["name"] == key)
            tmp = bytearray(b)
            tmp[s["offset"] + s["bytes"] - 1] ^= 2
            with self.subTest(key=key), self.assertRaises(ValueError):
                compare(a, bytes(tmp), SLICE)
        tmp = bytearray(b)
        tmp[48] ^= 1
        with self.assertRaises(ValueError):
            compare(a, bytes(tmp), SLICE)
        with self.assertRaises(ValueError):
            compare(a, b, "te_matmul_" + ("6" * 64) + "__kernel0")
