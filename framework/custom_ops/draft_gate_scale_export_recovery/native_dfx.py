"""Validate the two observed ET_REL DFX descriptor forms. Executable ABI is checked independently."""

import hashlib
import re
import struct

from archive import require
from elf_layout import EH, SH, SYM, table

FAMILIES = {"te_fused_op_cast_mul": 3, "te_stridedsliced": 2}
DFX_HEADER = bytes.fromhex("0000000000010001")


def records(note):
    pos = 0
    result = []
    while pos < len(note):
        require(pos + 4 <= len(note), "truncated function TLV")
        typ, size = struct.unpack_from("<HH", note, pos)
        pos += 4
        require(pos + size <= len(note), "invalid function TLV length")
        result.append((typ, note[pos : pos + size]))
        pos += size
    require([typ for typ, _ in result] == [1, 6, 4], "unreviewed native function metadata")
    return result


def arguments(payload):
    pos = 0
    result = []
    while pos < len(payload):
        require(pos + 4 <= len(payload), "truncated DFX argument")
        typ, words = struct.unpack_from(">HH", payload, pos)
        pos += 4
        end = pos + 8 * words
        require(
            typ == 5 and words >= 2 and end <= len(payload) and payload[pos : pos + 8] == DFX_HEADER,
            "unknown DFX argument kind/qualifier/extent",
        )
        data = struct.unpack_from(">" + str(words - 1) + "q", payload, pos + 8)
        if data == (-1,):
            result.append(dict(form="DFX_UNSPECIFIED", descriptor=list(data)))
        else:
            require(
                len(data) >= 3
                and 0 < data[1] <= 32
                and len(data) == 2 + data[1]
                and data[0] >= 0
                and all(d > 0 for d in data[2:]),
                "invalid concrete DFX descriptor",
            )
            result.append(dict(form="DFX_CONCRETE", descriptor=list(data)))
        pos = end
    return result


def runtime_argument_count(payload):
    """Match v9.0.0 ElfParseKernelArgNum, including its reserved argument and remainLen initialization."""
    require(4 <= len(payload) <= 65535, "DFX TLV extent")
    pos = 0
    remaining = len(payload) - 4
    count = 1
    while remaining > 4:
        typ, words = struct.unpack_from(">HH", payload, pos)
        extent = 4 + 8 * words
        if extent > remaining:
            break
        if typ == 5:
            count += 1
        pos += extent
        remaining -= extent
    return count


def normalized_object(blob, symbol):
    h, rows = table(blob)
    names = [s["name"] for s in rows]
    note = ".ascend.meta." + symbol
    require(
        h[1:4] == (1, 4137, 1)
        and h[5] == h[9] == h[10] == 0
        and names
        == ["", ".strtab", ".text", note, ".comment", ".note.GNU-stack", ".llvm_addrsig", ".symtab"],
        "unsupported native diagnostic object layout",
    )
    meta, syms = rows[3], rows[7]
    require(
        meta["flags"] == meta["va"] == 0
        and meta["type"] == 7
        and syms["type"] == 2
        and syms["link"] == 1
        and syms["entry_size"] == SYM.size
        and syms["bytes"] % SYM.size == 0,
        "unbound native metadata/symbol table",
    )
    lo = meta["offset"]
    hi = lo + meta["bytes"]
    require(EH.size <= lo < hi <= h[6], "invalid metadata range")
    for i, s in enumerate(rows):
        if i != 3 and s["bytes"]:
            require(s["offset"] + s["bytes"] <= lo or hi <= s["offset"], "overlapping native metadata")
    record = records(blob[lo:hi])
    # Rebase file positions around the independent note. No output binary is written.
    data = bytearray(blob)
    struct.pack_into("<Q", data, 40, h[6] - meta["bytes"])
    for i, s in enumerate(rows):
        where = h[6] + i * SH.size
        if i == 3:
            struct.pack_into("<Q", data, where + 32, 0)
        elif s["offset"] >= hi:
            struct.pack_into("<Q", data, where + 24, s["offset"] - meta["bytes"])
        elif lo < s["offset"] < hi:
            raise ValueError("native section inside DFX record")
    owned = []
    for idx, (name, info, other, section, value, size) in enumerate(
        SYM.iter_unpack(blob[syms["offset"] : syms["offset"] + syms["bytes"]])
    ):
        if section == 3 and size:
            require((info, other, value, size) == (1, 0, 0, meta["bytes"]), "wrong ELF symbol owns DFX note")
            struct.pack_into("<Q", data, syms["offset"] + idx * SYM.size + 16, 0)
            owned.append(idx)
        elif section == 3:
            require(value == 0, "symbol points inside DFX data")
    require(len(owned) == 1, "unbound DFX STT_OBJECT")
    del data[lo:hi]
    return bytes(data), record, owned[0]


def compare(left, right, symbol):
    family = re.fullmatch(r"(te_fused_op_cast_mul|te_stridedsliced)_[a-f0-9]{64}__kernel0", symbol)
    require(family is not None, "unsupported native diagnostic symbol: " + symbol)
    a, ta, ia = normalized_object(left, symbol)
    b, tb, ib = normalized_object(right, symbol)
    require(a == b and ia == ib and ta[:2] == tb[:2], "native bytes outside DFX shape records changed")
    av, bv = arguments(ta[2][1]), arguments(tb[2][1])
    argc = FAMILIES[family[1]]
    require(
        len(av) == len(bv) == runtime_argument_count(ta[2][1]) == runtime_argument_count(tb[2][1]) == argc,
        "Runtime user argument count changed",
    )
    changes = []
    for i, (l, r) in enumerate(zip(av, bv)):
        if l != r:
            require(
                {l["form"], r["form"]} == {"DFX_CONCRETE", "DFX_UNSPECIFIED"},
                "concrete DFX descriptor changed",
            )
            changes.append(i)
    require(bool(changes), "unexplained native metadata difference")
    return dict(
        status="ET_REL_DFX_ARGUMENT_DESCRIPTORS_ONLY",
        exact_elf_equal=False,
        raw_sha256=hashlib.sha256(left).hexdigest(),
        candidate_sha256=hashlib.sha256(right).hexdigest(),
        non_dfx_comparison_sha256=hashlib.sha256(a).hexdigest(),
        metadata_symbol_index=ia,
        raw_dfx=av,
        candidate_dfx=bv,
        changed_argument_positions=changes,
        user_args_num=argc,
    )
