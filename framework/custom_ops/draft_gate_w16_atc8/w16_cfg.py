"""Conservative control-flow and symbol checks on the decoded, graph-selected ELF."""

import re
import struct
from w16_common import require


def local_entry(blob, table, name):
    found = set()
    for number, symtab in enumerate(table):
        if symtab["type"] not in (2, 11):
            continue
        require(symtab["entry_size"] == 24 and symtab["link"] < len(table), "unsupported symbol table")
        names = table[symtab["link"]]
        require(names["type"] == 3, "invalid symbol string table")
        strings = blob[names["offset"] : names["offset"] + names["bytes"]]
        data = blob[symtab["offset"] : symtab["offset"] + symtab["bytes"]]
        symbols = list(struct.iter_unpack("<IBBHQQ", data))
        for off, info, _, sec, value, _size in symbols:
            end = strings.find(b"\0", off)
            require(0 <= off <= end, "invalid symbol name")
            if strings[off:end] == name.encode():
                require(
                    0 < sec < len(table) and table[sec]["name"] == ".text" and (info & 15) in (0, 2),
                    "wrong entry section",
                )
                found.add(value)
        for reloc in (s for s in table if s["type"] in (4, 9) and s["link"] == number):
            require(reloc["info"] < len(table), "bad relocation target")
            if not table[reloc["info"]]["flags"] & 4:
                continue
            data = blob[reloc["offset"] : reloc["offset"] + reloc["bytes"]]
            for row in struct.iter_unpack("<QQq" if reloc["type"] == 4 else "<QQ", data):
                index = row[1] >> 32
                require(index < len(symbols) and symbols[index][3] != 0, "unresolved external text relocation")
    require(len(found) == 1, "missing or ambiguous defined symbol for graph entry")
    return found.pop()


# Closed vocabulary. New SDK dialects require evidence of their control effects.
REGULAR = re.compile(
    r"(?:mov[iak]?|add[i]?|sub[i]?|mul[i]?|and[i]?|or[i]?|xor[i]?|lsl|lsr|shr|shl|"
    r"cmp|sel|nop|sff0|sbitset[01]|bclr|bic|ld|st|ldr|str|pipe_barrier|barrier|"
    r"wait_flag|set_flag|mmad|mad|load_cbuf_to_c[a-z0-9_]+|copy_[a-z0-9_]+|"
    r"vconv_[a-z0-9_]+|vmul|vadds|vtranspose|vector_dup|set_vector_mask|dcci)"
)
BRANCH = re.compile(r"(?:b|br|bra|jmp|j|b[._]?(?:eq|ne|lt|le|gt|ge|z|nz)|j(?:eq|ne|lt|le|gt|ge|z|nz))")


def successors(row):
    asm = row["asm"].lower()
    op = asm.split()[0]
    nxt = row["pc"] + len(bytes.fromhex(row["encoding"]))
    if op in ("ret", "task_end", "block_end", "end"):
        return set()
    if BRANCH.fullmatch(op):
        target = re.search(r"(?:^|[\s,])(?:0x)?([0-9a-f]+)\s*<[^<>]+>\s*$", asm)
        require(target is not None, "unresolved branch target")
        return {int(target[1], 16)} if op in ("b", "br", "bra", "jmp", "j") else {nxt, int(target[1], 16)}
    require(REGULAR.fullmatch(op) is not None, "unreviewed instruction/control semantics: " + op)
    return {nxt}


def reachable_pcs(blocks, decoded, entry):
    require(isinstance(blocks, list) and 0 < len(blocks) <= len(decoded), "missing complete CFG review")
    ins = {x["pc"]: x for x in decoded}
    starts = {b["first"] for b in blocks}
    require(len(starts) == len(blocks) and entry in starts, "missing/duplicate entry block")
    owned = {}
    edges = {}
    for b in blocks:
        lo, hi = b["first"], b["last"]
        require(lo in ins and hi in ins and lo <= hi and b["derivation"].strip(), "unreviewed block")
        pc = lo
        while True:
            require(pc in ins and pc not in owned and pc <= hi, "overlap/gap in reviewed machine CFG")
            owned[pc] = lo
            row = ins[pc]
            targets = successors(row)
            if pc == hi:
                break
            pc += len(bytes.fromhex(row["encoding"]))
            require(targets == {pc}, "omitted control edge within basic block")
        require(set(b["successors"]) == targets and targets <= starts, "wrong basic block successors")
        edges[lo] = targets
    require(set(owned) == set(ins), "machine CFG does not cover executable text")
    reached = set()
    todo = [entry]
    while todo:
        block = todo.pop()
        if block not in reached:
            reached.add(block)
            todo.extend(edges[block] - reached)
    return {pc for pc, block in owned.items() if block in reached}
