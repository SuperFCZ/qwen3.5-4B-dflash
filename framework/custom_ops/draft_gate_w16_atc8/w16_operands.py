"""Small fail-closed verifier for address-bearing operands of a decoded dav_2002 straight-line/CFG witness.

No initial register values are accepted from the proof author. Unmodelled scalar
instructions invalidate their named registers. Expand the walk across a loop or
previous block if a definition is needed. This only proves the cited access;
whole-path ownership/coverage is reviewed separately.
"""

import re

REG = re.compile(r"(?:x|r)[0-9]+", re.I)
IMM = re.compile(r"#?(-?(?:0x[0-9a-f]+|[0-9]+))", re.I)
MEMORY = {
    "vconv_s82f16",
    "vadds",
    "vtranspose",
    "vmul",
    "vector_dup",
    "copy_ubuf_to_cbuf",
    "copy_gm_to_ubuf",
    "copy_matrix_cc_to_ubuf",
    "copy_ubuf_to_gm",
}


def operands(asm):
    fields = asm.lower().strip().split(None, 1)
    args = fields[1].strip().strip("()").split(",") if len(fields) == 2 else []
    return fields[0], [s.strip() for s in args]


def value(arg, registers):
    m = IMM.fullmatch(arg)
    if m:
        v = m[1]
        return int(v, 16 if "0x" in v else 10)
    return registers.get(arg)


def consumed_address(walk, opcode, index):
    registers = {}
    for row in walk[:-1]:
        op, args = operands(row["asm"])
        if op in MEMORY or op in ("barrier", "pipe_barrier", "set_flag", "wait_flag", "nop"):
            continue
        vals = [value(arg, registers) for arg in args[1:]]
        dest = args[0] if args else ""
        computed = None
        if REG.fullmatch(dest) and None not in vals:
            if op in ("mov", "movi") and len(vals) == 1:
                computed = vals[0]
            elif len(vals) == 2:
                a, b = vals
                if op in ("add", "addi"):
                    computed = a + b
                elif op in ("sub", "subi"):
                    computed = a - b
                elif op in ("mul", "muli"):
                    computed = a * b
                elif op in ("shl", "lsl") and 0 <= b < 64:
                    computed = a << b
                elif op in ("shr", "lsr") and 0 <= b < 64 and a >= 0:
                    computed = a >> b
                elif op in ("or", "ori"):
                    computed = a | b
                elif op in ("and", "andi"):
                    computed = a & b
                elif op in ("xor", "xori"):
                    computed = a ^ b
        if computed is None:
            # No inference from numeric PC, register naming, prose or expected layout.
            for reg in REG.findall(row["asm"].lower()):
                registers.pop(reg, None)
        else:
            registers[dest] = computed if 0 <= computed < 2**64 else None
    op, args = operands(walk[-1]["asm"])
    if op != opcode or index >= len(args):
        raise ValueError("wrong UB memory instruction or operand position")
    result = value(args[index], registers)
    if result is None:
        raise ValueError("no decoded reaching definition for UB memory operand")
    return result


# For W0 and W1, prove BOTH the actual Vector destination and the SDK's actual
# UB source of the first aligned B-plane transfer. The scratch consumer here is
# the selected CO1 -> local CO2 output path, at offset zero for this fixed tile.
ACCESSES = {
    "q0_dma": ("q0", "copy_gm_to_ubuf", 0),
    "q1_dma": ("q1", "copy_gm_to_ubuf", 0),
    "s0_dma": ("s0", "copy_gm_to_ubuf", 0),
    "s1_dma": ("s1", "copy_gm_to_ubuf", 0),
    "w0_cast": ("w0", "vconv_s82f16", 0),
    "w1_cast": ("w1", "vconv_s82f16", 0),
    "w0_b_read": ("w0", "copy_ubuf_to_cbuf", 1),
    "w1_b_read": ("w1", "copy_ubuf_to_cbuf", 1),
    "expanded_write": ("expanded", "vtranspose", 0),
    "fixed_tmp": ("brcb_fixed", "vadds", 0),
    "sdk_co2": ("sdk", "copy_matrix_cc_to_ubuf", 0),
}


def paired_flags(walk, events):
    """Same symbolic ID at signal/wait; no hard-coded pool slot.

    This restricted proof accepts a literal ID or a preserved scalar register.
    A reassignment/spill in the interval needs a reviewed verifier extension.
    """
    pending = {}
    pairs = {"set_s": "raw_s", "set_v": "raw_v", "signal_ready": "ready"}
    for step in events:
        action = step["action"]
        if action not in pairs and action not in pairs.values():
            continue
        idx = step["walk_index"]
        _, args = operands(walk[idx]["asm"])
        require_id = args[-1] if len(args) >= 3 else ""
        if not REG.fullmatch(require_id) and not IMM.fullmatch(require_id):
            raise ValueError("unproven flag ID operand")
        if action in pairs:
            key = pairs[action]
            if key in pending:
                raise ValueError("reserved signal reused before Wait")
            pending[key] = (idx, require_id)
        elif action in pending:
            start, signal = pending.pop(action)
            if signal != require_id:
                raise ValueError("flag Set/Wait ID differs")
            if REG.fullmatch(signal):
                for row in walk[start + 1 : idx]:
                    op, values = operands(row["asm"])
                    if (
                        op not in MEMORY
                        and op not in ("set_flag", "wait_flag", "pipe_barrier", "barrier", "nop")
                        and values
                        and values[0] == signal
                    ):
                        raise ValueError("flag ID reaching definition changed")
    if pending:
        raise ValueError("unconsumed signal in reviewed path")
