"""Specialize the receiver's installed Chunk arithmetic in a private, separately named OPP."""

import hashlib
import re
import shutil

from gdr_graph import ROOT, COMMIT, require
from pair_graph import c
from package import generate, write_preset, build_install


SOURCE_SHA = "93bfea89114a408b3e189b25207de157f4eb155b34d1e7500164ff627ea9c601"
# Source coordinates refer only to the pinned installation. Original source text and
# generated code remain on the receiver. Level 1 suppresses the final readout, level
# 2 also suppresses the query-side attention products. 0 is the exact-arithmetic clone.
READOUT_MM = {1385: 2, 1525: 1, 1531: 1}
VECTOR_WORK = {
    1356: "Duplicate",
    1357: "L2normDim128FloatTrans",
    1359: "L2normDim128FloatTrans",
    1362: "Duplicate",
    1364: "Cast",
    1366: "Cast",
    1368: "Transpose_64_128",
    1372: "Muls",
    1373: "Cast",
    1375: "UB2L1_ND2NZ",
    1387: "Mul",
    1389: "Cast",
    1411: "Mul",
    1423: "Cast",
    1508: "UB2L1_ND2NZ",
    1518: "UB2L1_ND2NZ",
    1535: "Add",
    1548: "Cast",
    1553: "StoreAttnHalfWithTail",
}
FIELDS = dict(
    batchSize=1,
    seqLen=16,
    numHead=32,
    headDimQK=128,
    headDimV=128,
    chunkSize=64,
    hasInitState=1,
    outputFinalState=1,
    useQKL2normInKernel=1,
)
EVENT = re.compile(r"\b(?:SetFlag|WaitFlag|set_flag|wait_flag|SyncAll|CrossCore\w*(?:Set|Wait)\w*)\b", re.I)
FLAG_PAIR = re.compile(r"\b(SetFlag|WaitFlag)\s*<\s*HardEvent::(\w+)\s*>\s*\(\s*(\w+)\s*\)")


def code_only(text):
    return re.sub(r"/\*[\s\S]*?\*/|//[^\n]*", lambda m: re.sub(r"[^\n]", " ", m[0]), text)


def methods(code):
    bodies = {}
    for m in re.finditer(r"::(\w+)\s*\([^;{}]*\)\s*(?:const\s*)?\{", code):
        start = m.end()
        level = 1
        for b in re.finditer(r"[{}]", code[start:]):
            level += 1 if b[0] == "{" else -1
            if level == 0:
                bodies.setdefault(m[1], []).append(code[start : start + b.start()])
                break
    return {name: "\n".join(values) for name, values in bodies.items()}


def helper_barriers(name, bodies):
    """Return barrier kinds for a pure helper, or None if it has an explicit event handoff."""
    seen = set()
    todo = [name]
    barriers = set()
    while todo:
        k = todo.pop()
        if k not in bodies or k in seen:
            continue
        seen.add(k)
        body = bodies[k]
        if EVENT.search(body):
            # Skip only straight-line, internally balanced flags. An incoming
            # wait cannot be discarded, even if a later set uses the same ID.
            if EVENT.search(FLAG_PAIR.sub("", body)) or re.search(
                r"\b(if|else|for|while|switch|goto|return|break|continue)\b", body
            ):
                return None
            outstanding = set()
            for op, direction, slot in FLAG_PAIR.findall(body):
                tag = (direction, slot)
                if op == "SetFlag" and tag not in outstanding:
                    outstanding.add(tag)
                elif op == "WaitFlag" and tag in outstanding:
                    outstanding.remove(tag)
                else:
                    return None
            if outstanding:
                return None
        barriers.update(re.findall(r"PipeBarrier\s*<\s*(PIPE_\w+)\s*>", body))
        todo.extend(
            callee for callee in re.findall(r"\b(\w+)\s*(?:<[^;(){}<>]*>)?\s*\(", body) if callee in bodies
        )
    return sorted(barriers)


def guard_call(lines, active, index, condition, barriers=()):
    """Brace the complete statement so an enclosing unbraced if/else stays associated."""
    end = index
    while end < len(lines) and ";" not in active[end]:
        end += 1
    require(end < len(lines), "unterminated receiver arithmetic statement")
    fallback = " ".join(f"PipeBarrier<{kind}>();" for kind in barriers)
    at = active[end].index(";") + 1
    lines[end] = (
        lines[end][:at] + " }" + (" else { " + fallback + " }" if fallback else "") + " }" + lines[end][at:]
    )
    at = len(lines[index]) - len(lines[index].lstrip())
    lines[index] = lines[index][:at] + "{ if (" + condition + ") { " + lines[index][at:]


def generate_kernel(raw, entry):
    require(hashlib.sha256(raw).hexdigest() == SOURCE_SHA, "unrecognized private GDR source revision")
    return specialize(raw.decode(), entry)


def specialize(text, entry):
    marker = r'extern\s+"C"\s+__global__\s+__aicore__\s+void\s+chunk_gated_delta_rule\s*\('
    entries = list(re.finditer(marker, text))
    require(len(entries) == 1, "missing unique native kernel boundary")
    prefix = text[: entries[0].start()]
    code = code_only(prefix)
    require(set(re.findall(r"tiling\s*->\s*(\w+)", code)) == set(FIELDS), "different native shape contract")
    require(
        re.search(r"InitWorkSpace\(workspace\)\s*;", code)
        and set(re.findall(r'(?m)^\s*#include\s+"([^"]+)"', code))
        == {"kernel_operator.h", "lib/matmul_intf.h"},
        "unexpected native workspace or include contract",
    )
    bodies = methods(code)
    require(
        "MatMul" in bodies and EVENT.search(bodies["MatMul"]), "missing externally synchronized Cube helper"
    )
    require(
        {"L0C2UB_NZ2ND", "L2normDim128FloatTrans", "Transpose_64_128", "UB2L1_ND2NZ", "StoreAttnHalfWithTail"}
        <= bodies.keys(),
        "could not resolve synchronization scope of native helpers",
    )
    lines = prefix.splitlines()
    active = code.splitlines()
    require(len(active) == len(lines), "line map differs")
    call = re.compile(r"\bMatMul(\s*(?:<[^<>(){};]*>)?\s*\()")
    declarations = 0
    guarded_cube = 0
    protected = []
    cut = []
    seen_mm = set()
    for i, line in enumerate(active, 1):
        if i == 1310:
            # The causal inverse uses the active top-left submatrix; the final
            # state also uses decay from the last padded row. Keep that row even
            # for r=1. Untouched padded mask rows retain the original zero fill.
            require(
                re.search(r"\bDuplicate(?:<[^>]+>)?\s*\(ub_temp2\b", active[1295])
                and re.search(r"\bfor\s*\(\s*int\s+l\s*=\s*1\s*;\s*l\s*<\s*chunkSize\s*;", active[1308])
                and re.match(r"\s*Exp(?:<[^>]+>)?\s*\(ub_temp2\b", line),
                "unrecognized causal decay mask",
            )
            guard_call(lines, active, i - 1, "tiling->pruneLevel < 3 || l < seqLen || l + 1 == chunkSize")
        elif i in VECTOR_WORK:
            op = VECTOR_WORK[i]
            require(
                re.match(r"^\s*" + op + r"\s*(?:<[^;()]*>)?\s*\(", line),
                "unexpected readout operation at " + str(i),
            )
            barriers = helper_barriers(op, bodies)
            if barriers is None:
                protected.append(i)
                continue
            guard_call(lines, active, i - 1, f"tiling->pruneLevel < {1 if i >= 1535 else 2}", barriers)
            cut.append(i)
        elif call.search(line):
            position = call.search(line).end()
            if re.search(r"\bvoid\s+(?:ChunkGatedDeltaRuleKernel<Mode>::)?MatMul\s*\(", line):
                value = "bool prefixCompute, "
                declarations += 1
            else:
                value = "true, "
                if i in READOUT_MM:
                    seen_mm.add(i)
                    value = f"tiling->pruneLevel < {READOUT_MM[i]}, "
            lines[i - 1] = lines[i - 1][:position] + value + lines[i - 1][position:]
        elif 538 <= i < 568:
            # Keep the exact template-controlled incoming MTE3 wait, all L0A/L0B
            # loads/reuse, and both paired MTE1/M and M/V handoffs, even when the
            # discarded matrix arithmetic is disabled.
            if re.match(r"^\s*(?:AscendC::)?(?:Mmad|mad|mmad)\s*(?:<[^;()]*>)?\s*\(", line):
                guard_call(lines, active, i - 1, "prefixCompute")
                guarded_cube += 1
            elif re.match(r"^\s*L0C2UB_NZ2ND\s*(?:<[^;()]*>)?\s*\(", line):
                barriers = helper_barriers("L0C2UB_NZ2ND", bodies)
                if barriers is not None:
                    guard_call(lines, active, i - 1, "prefixCompute", barriers)
    require(
        declarations == 2 and guarded_cube >= 1 and seen_mm == set(READOUT_MM),
        f"receiver MatMul mapping changed: declarations={declarations}, MMAD={guarded_cube}, readout={sorted(seen_mm)}",
    )
    edited = "\n".join(lines).replace("ChunkGatedDeltaRuleTilingData", "PrefixNativeConfig")
    cfg = (
        "struct PrefixNativeConfig { unsigned int pruneLevel;\n"
        + "\n".join(f"static constexpr unsigned int {key} = {val};" for key, val in FIELDS.items())
        + "\n};\n"
    )
    header = '#include "lib/matmul_intf.h"'
    require(edited.count(header) == 1, "ambiguous configuration insertion")
    edited = edited.replace(header, header + "\n" + cfg)
    # The public entry is ours. Never copy the native symbol into this OPP.
    entry_code = code_only(text[entries[0].end() :])
    depth = 0
    close = None
    for b in re.finditer(r"[{}]", entry_code):
        depth += 1 if b[0] == "{" else -1
        if depth == 0:
            close = entries[0].end() + b.end()
            break
    require(close is not None, "unterminated native entry")
    return edited + "\n" + entry + "\n" + text[close:], dict(
        disabled_vector_lines=cut,
        kept_event_helpers=protected,
        guarded_cube_calls=sorted(seen_mm),
        guarded_mmad_sites=guarded_cube,
    )


def build(d, original, cann, env):
    source = c.verify(original)
    text, proof = generate_kernel(source.read_bytes(), (ROOT / "kernel_entry.inc").read_text())
    project = generate(d, ROOT / (COMMIT + ".json"), env)
    stem = "target_gdr_prefix_state"
    for f in (ROOT / "op_host").iterdir():
        shutil.copyfile(f, project / "op_host" / f.name)
    cpp = project / "op_kernel" / (stem + ".cpp")
    cpp.write_text(text)
    preset = project / "CMakePresets.json"
    write_preset(preset, preset, cann)
    root = build_install(d, project, cann, env) / "vendors/customize"
    sdk = root / "op_impl/ai_core/tbe"
    binaries = sorted((sdk / "kernel/ascend310p" / stem).glob("*.o"))
    require(
        binaries and all(c.read(f.with_suffix(".json"))["sha256"] == c.sha(f) for f in binaries),
        "unbound state kernel",
    )
    (impl,) = sdk.glob("*/dynamic/" + stem + ".cpp")
    require(c.sha(impl) == c.sha(cpp) and c.sha(source) == SOURCE_SHA, "private implementation drift")
    c.write(d / "specialization.json", proof)
    return dict(
        vendor=str(root),
        implementation=c.identity(impl),
        binaries=[c.identity(f) for f in binaries],
        native=original,
        proof=proof,
    )
