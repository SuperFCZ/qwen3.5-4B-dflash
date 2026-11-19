"""Static evidence from the *actual embedded ELF*. Never infer completion from source markers."""

import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess

from w16_common import CUBE_SHA, DOUBLE, HERE, OVERLAP, SERIAL, c, require
from platform_contracts import unique_object
from w16_operands import ACCESSES, consumed_address, paired_flags
from w16_cfg import local_entry, reachable_pcs


class StaticBlocked(ValueError):
    pass


class NoOverlap(ValueError):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def disassembler(cann, selected=None):
    root = Path(cann).resolve(strict=True)
    path = Path(selected) if selected else root / "aarch64-linux/ccec_compiler/bin/llvm-objdump"
    target = path.resolve(strict=True)
    require(
        root in target.parents and target.name in ("llvm-objdump", "ccec-objdump"), "not a selected SDK decoder"
    )
    return c.identity(target)


def sections(blob):
    require(len(blob) >= 64 and blob[:7] == b"\x7fELF\x02\x01\x01", "not ELF64 LE")
    head = struct.unpack_from("<16sHHIQQQIHHHHHH", blob)
    start, stride, count, names = head[6], head[11], head[12], head[13]
    require(
        head[1] == 2
        and head[2] == 0x1029
        and stride == 64
        and 0 <= names < count
        and start + stride * count == len(blob),
        "unexpected dav_m200 ELF machine/type/extent",
    )
    headers = list(struct.iter_unpack("<IIQQQQIIQQ", blob[start:]))
    a, length = headers[names][4:6]
    require(a + length <= len(blob) and headers[names][1] == 3, "invalid section-name table")
    table = blob[a : a + length]
    result = []
    for name, kind, flags, va, offset, size, link, info, align, ent in headers:
        end = table.find(b"\0", name)
        require(0 <= name <= end and (kind == 8 or offset + size <= len(blob)), "invalid ELF section")
        result.append(
            dict(
                name=table[name:end].decode("ascii"),
                type=kind,
                flags=flags,
                va=va,
                offset=offset,
                bytes=size,
                link=link,
                info=info,
                alignment=align,
                entry_size=ent,
                sha256=None if kind == 8 else digest(blob[offset : offset + size]),
            )
        )
    require(any(s["flags"] & 4 and s["bytes"] for s in result), "no executable sections")
    return result


def decode(blob, table, listing):
    """Require exact contiguous raw instruction encodings for every byte of every executable section."""
    execs = {s["name"]: s for s in table if s["flags"] & 4 and s["bytes"]}
    decoded = {}
    name = None
    bad = re.compile(r"<not\s+available>|<unknown>|unknown instruction|unsupported|invalid instruction", re.I)
    if bad.search(listing):
        raise StaticBlocked("SDK disassembler returned undecoded instructions")
    for line in listing.splitlines():
        label = re.fullmatch(r"\s*Disassembly of section (.+):\s*", line)
        if label:
            name = label[1]
            if name in execs:
                require(name not in decoded, "duplicate disassembly section")
                decoded[name] = []
            continue
        if name not in decoded or not re.match(r"^\s*[0-9a-fA-F]+:", line):
            continue
        match = re.fullmatch(
            r"\s*([0-9a-fA-F]+):\s+((?:(?:[0-9a-fA-F]{2}|[0-9a-fA-F]{4}|[0-9a-fA-F]{8}|[0-9a-fA-F]{16})\s+)+)([A-Za-z][\w.]*)(?:\s+(.*?))?\s*",
            line,
        )
        if not match:
            raise StaticBlocked("unsupported or byte-free instruction: " + line[:160])
        pc = int(match[1], 16)
        raw = bytes.fromhex(match[2])
        sec = execs[name]
        prev = decoded[name]
        next_pc = prev[-1]["pc"] + len(bytes.fromhex(prev[-1]["encoding"])) if prev else sec["va"]
        offset = sec["offset"] + pc - sec["va"]
        actual = blob[offset : offset + len(raw)]
        words = match[2].split()
        if raw != actual and all(len(x) > 2 for x in words):
            raw = b"".join(bytes.fromhex(x)[::-1] for x in words)
        if pc != next_pc or not raw or actual != raw:
            raise StaticBlocked("disassembly has a gap or does not describe the extracted bytes")
        prev.append(dict(pc=pc, encoding=raw.hex(), asm=match[3] + (" " + match[4] if match[4] else "")))
    if set(decoded) != set(execs) or any(
        sum(len(x["encoding"]) // 2 for x in ins) != execs[n]["bytes"] for n, ins in decoded.items()
    ):
        raise StaticBlocked("incomplete executable-section coverage")
    return decoded


def collect(folder, om, node, tool):
    folder.mkdir(parents=True)
    with c.verify(om).open("rb") as reader:
        reader.seek(node["offset"])
        blob = reader.read(node["bytes"])
    require(
        digest(blob) == node["sha256"] and node["symbol"].encode() in blob and node["block_dim"] == 8,
        "wrong embedded binary for node",
    )
    elf = folder / "selected.elf"
    elf.write_bytes(blob)
    table = sections(blob)
    c.write(folder / "sections.json", table)
    result = dict(
        om=om,
        node=node,
        elf=c.identity(elf),
        sections=c.identity(folder / "sections.json"),
        tool=tool,
        attempts=[],
    )
    for filename, options in (("selected.disassembly", []), ("dav-m200.disassembly", ["--mcpu=dav-m200"])):
        asm = folder / filename
        argv = [str(c.verify(tool)), "--disassemble", *options, str(elf.resolve(strict=True))]
        with asm.open("xb") as stream:
            proc = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
        attempt = dict(argv=argv, returncode=proc.returncode, listing=c.identity(asm))
        result["attempts"].append(attempt)
        result.update(attempt)
        try:
            if proc.returncode:
                raise StaticBlocked("disassembler exit " + str(proc.returncode))
            decoded = decode(blob, table, asm.read_text(errors="replace"))
            c.write(folder / "instructions.json", decoded)
            result.update(
                status="DECODED_REQUIRES_PC_PROOF", instructions=c.identity(folder / "instructions.json")
            )
            result.pop("reason", None)
            break
        except StaticBlocked as exc:
            result.update(status="BLOCKED_STATIC_ADMISSION", reason=str(exc))
    return result


REGIONS = dict(
    q0=(0, 16384),
    q1=(16384, 32768),
    s0=(32768, 33024),
    s1=(33024, 33280),
    w0=(33280, 66048),
    w1=(66048, 98816),
    expanded=(98816, 101120),
    sdk=(101120, 253952),
    reserved=(253952, 262144),
    brcb_fixed=(254976, 256000),
)
OBLIGATIONS = {
    "direct_same_core_no_deferred_client",
    "all_reachable_blocks_and_relocations",
    "actual_tpipe_base_offsets_and_failure_branch",
    "all_ub_read_write_intervals",
    "local_b_final_mte3_read_before_fence",
    "raw_event_id_reservation_pair_drain_release",
    "alternate_bank_and_no_rebind_before_fence",
    "two_group128_roundings",
    "partial_false_then_true_same_l0c",
    "last_k_no_speculative_dma_or_dequant",
    "one_final_fp16_copyout",
    "false_iterate_no_output_or_pending_event",
    "n_boundary_fence_before_reuse",
    "no_intervening_full_drain_before_lookahead",
}
# Each witness quotes fully decoded, byte-verified instructions. Global CFG/operand
# semantics require independent review of the exact machine image, including relocation data.
OPS = dict(
    set_s=r"set.*mte2.*(?:pipe_s\b|,\s*s\b|_s\b)",
    set_v=r"set.*mte2.*(?:pipe_v\b|,\s*v\b|_v\b)",
    signal_ready=r"set.*(?:pipe_v\b|\bv\b|v_s).*s",
    raw_s=r"wait.*mte2.*(?:pipe_s\b|,\s*s\b|_s\b)",
    raw_v=r"wait.*mte2.*(?:pipe_v\b|,\s*v\b|_v\b)",
    ready=r"wait.*(?:pipe_v\b|\bv\b|v_s).*s",
    read_b=r"copy_ubuf_to_cbuf",
    cube=r"(?:mmad|mad\b)",
    cast=r"vconv_s82f16",
    broadcast=r"vtranspose\b",
    mul=r"vmul\b",
    all=r"(?:pipe_barrier|barrier).*all",
    output=r"copy_(?:ubuf|cbuf|cc)_to_gm(?:\b|_)",
    release=r"\b(?:sbitset0|bclr|bic|and|andi)\b",
    end=r"\b(?:ret|task_end|block_end|end)\b",
)


def reviewed_packet(commit, elf_hash):
    """Read immutable review data from this project's feature history; working-tree JSON is never authority."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise StaticBlocked("expected an exact review commit")
    repo = c.REPO
    args = ["git", "-C", str(repo)]
    remote = "refs/remotes/origin/feature/ascend310p-custom-ops"
    if subprocess.run(args + ["merge-base", "--is-ancestor", commit, remote]).returncode:
        raise StaticBlocked("review commit not in fetched feature history")
    rel = (HERE / "machine_reviews" / (elf_hash + ".json")).relative_to(repo).as_posix()
    result = subprocess.run(args + ["show", commit + ":" + rel], capture_output=True)
    if result.returncode:
        raise StaticBlocked("no reviewed register/CFG proof for this ATC ELF: " + elf_hash)
    return json.loads(result.stdout, object_pairs_hook=unique_object), digest(result.stdout)


def review_template(capture, mode, source, plan, sdk):
    """Index the evidence a reviewer actually received; empty witnesses cannot be admitted."""
    return dict(
        schema=1,
        binding=dict(
            elf=capture["elf"]["sha256"],
            decoder=capture["tool"]["sha256"],
            sources=source["sha256"],
            instructions=capture.get("instructions", {}).get("sha256"),
            cube=CUBE_SHA,
            parent=plan["parent_sha256"],
            sdk={k: v["sha256"] for k, v in sdk.items()},
            mode=mode,
            entry=capture["node"]["symbol"],
        ),
        regions={
            name: dict(byte_interval=list(span), derivation="", witnesses=[]) for name, span in REGIONS.items()
        },
        obligations={name: dict(derivation="", witnesses=[]) for name in sorted(OBLIGATIONS)},
        cfg=[],
        accesses={key: dict(blocks=[], pc=None) for key in ACCESSES},
        observed_placement=None,
        cube_drained_before_next_d=None,
        paths={
            k: dict(blocks=[], events=[])
            for k in ("first", "odd", "even", "rebind", "last", "false_next", "false_last")
        },
    )


def walk_instructions(blocks, decoded):
    """Expand a reviewed basic-block walk; jumps need an explicit decoded target.

    Only direct, statically resolved edges are supported. A dossier cannot reorder
    arbitrary PC snippets to manufacture the requested before/after relationship.
    """
    require(isinstance(blocks, list) and 0 < len(blocks) <= 10000, "no bounded CFG witness")
    items = {r["pc"]: r for r in decoded}
    output = []
    for index, span in enumerate(blocks):
        lo, hi = span
        require(
            type(lo) is int and type(hi) is int and lo in items and hi in items and lo <= hi,
            "invalid CFG block",
        )
        pc = lo
        while pc <= hi:
            row = items[pc]
            asm = row["asm"].lower()
            op = asm.split()[0]
            # Calls/indirect transfers require another verifier. Refuse to treat
            # them as a fallthrough proving same-core completion.
            terminal = index == len(blocks) - 1 and pc == hi
            require(
                not re.match(r"^(?:call|jal|jr|brind|blr|bl$)", op) and (op != "ret" or terminal),
                "unresolved call/return inside witness",
            )
            branch = bool(re.fullmatch(r"(?:b|br|bra|jmp|j|b[._]?[a-z]{1,3}|j(?:eq|ne|lt|le|gt|ge|z|nz))", op))
            if branch:
                require(pc == hi, "omitted control transfer inside CFG block")
            output.append(row)
            pc += len(bytes.fromhex(row["encoding"]))
        require(output[-1]["pc"] == hi, "CFG ends inside instruction")
        if index + 1 == len(blocks):
            continue
        target = blocks[index + 1][0]
        if not branch:
            require(target == pc, "non-branch cannot change PC")
        else:
            # Accept byte-address targets printed with the symbol by llvm-objdump.
            m = re.search(r"(?:^|[\s,])(?:0x)?([0-9a-f]+)\s*<[^<>]+>\s*$", asm)
            require(m is not None, "unresolved direct branch target")
            choices = {int(m[1], 16)}
            if op not in ("b", "br", "bra", "jmp", "j"):
                choices.add(pc)
            require(target in choices, "CFG successor disagrees with decoded branch")
    return output


def check_proof(packet, capture, mode, source, plan, sdk):
    """Recheck provenance, actual PCs, bank ranges, edge claims and witness order. Never accept a PASS flag."""
    blob = c.verify(capture["elf"]).read_bytes()
    require(digest(blob) == capture["node"]["sha256"] and len(blob) == capture["node"]["bytes"], "ELF identity")
    table = sections(blob)
    require(
        c.read(c.verify(capture["sections"])) == table
        and capture["returncode"] == 0
        and capture["argv"][0] == capture["tool"]["path"]
        and capture["argv"][-1] == capture["elf"]["path"]
        and capture["argv"][1:-1] in (["--disassemble"], ["--disassemble", "--mcpu=dav-m200"]),
        "section table/decoder binding",
    )
    decoded = decode(blob, table, c.verify(capture["listing"]).read_text())
    require(decoded == c.read(c.verify(capture["instructions"])), "decoded inventory mismatch")
    expected = dict(
        elf=digest(blob),
        decoder=capture["tool"]["sha256"],
        sources=source["sha256"],
        instructions=capture["instructions"]["sha256"],
        cube=CUBE_SHA,
        parent=plan["parent_sha256"],
        sdk={k: v["sha256"] for k, v in sdk.items()},
        mode=mode,
        entry=capture["node"]["symbol"],
    )
    require(
        packet["schema"] == 1
        and packet["binding"] == expected
        and mode in DOUBLE
        and type(packet["cube_drained_before_next_d"]) is bool,
        "machine proof binding",
    )
    require(set(decoded) == {".text"}, "multiple executable sections require another reviewed decoder")
    entry = local_entry(blob, sections(blob), capture["node"]["symbol"])
    reachable = reachable_pcs(packet["cfg"], decoded[".text"], entry)
    ins = {v["pc"]: v["asm"] for v in decoded[".text"]}

    def witness(row):
        pc = row["pc"]
        require(
            type(pc) is int and pc in reachable and ins[pc] == row["asm"], "not an exact reachable PC witness"
        )
        return ins[pc].lower()

    def argument(row):
        require(
            isinstance(row["derivation"], str) and row["derivation"].strip() and row["witnesses"],
            "missing dataflow derivation",
        )
        for rec in row["witnesses"]:
            witness(rec)

    require(
        set(packet["regions"]) == set(REGIONS) and set(packet["obligations"]) == OBLIGATIONS,
        "incomplete actual address/CFG/ownership review",
    )
    for name, span in REGIONS.items():
        region = packet["regions"][name]
        require(region["byte_interval"] == list(span), "incorrect physical interval: " + name)
        argument(region)
    for row in packet["obligations"].values():
        argument(row)
    require(set(packet["accesses"]) == set(ACCESSES), "incomplete address-bearing machine accesses")
    for key, (region, op, operand) in ACCESSES.items():
        record = packet["accesses"][key]
        walk = walk_instructions(record["blocks"], decoded[".text"])
        require(
            all(row["pc"] in reachable for row in walk)
            and walk[-1]["pc"] == record["pc"]
            and consumed_address(walk, op, operand) == REGIONS[region][0],
            "actual machine memory operand disagrees with bank: " + key,
        )
    # The dossier specifies execution-path order, not numeric PC order across branch/loop backedges.
    steady = ["set_s", "set_v", "signal_ready", "ready", "read_b", "cube", "raw_s", "raw_v"]
    compute = ["cast"] * 8 + (["broadcast"] * 8 + ["mul"]) * 2
    order = packet["observed_placement"]
    require(order in ("after_fence", "before_fence"), "unresolved placement")
    seq = compute + ["all"] if order == "before_fence" else ["all"] + compute
    expected_paths = {
        "first": ["set_s", "set_v", "raw_s", "raw_v"] + compute + steady + seq,
        "odd": steady + seq,
        "even": steady + seq,
        "rebind": ["all", "signal_ready", "ready", "read_b", "cube"],
        "last": ["signal_ready", "ready", "read_b", "cube", "all", "output", "all"],
        "false_next": ["raw_s", "raw_v", "all", "release", "release", "end"],
        "false_last": ["all", "release", "release", "end"],
    }
    require(packet["paths"].keys() == expected_paths.keys(), "incomplete first/steady/tail/failure paths")
    for key, actions in expected_paths.items():
        path = packet["paths"][key]
        walk = walk_instructions(path["blocks"], decoded[".text"])
        require(all(row["pc"] in reachable for row in walk), "unreachable device path")
        entries = path["events"]
        require([x["action"] for x in entries] == actions, "invalid reviewed path order: " + key)
        positions = [step["walk_index"] for step in entries]
        require(
            all(type(i) is int and 0 <= i < len(walk) for i in positions)
            and positions == sorted(set(positions)),
            "unordered/out-of-range machine witnesses",
        )
        for step in entries:
            require(
                walk[step["walk_index"]]["pc"] == step["pc"]
                and bool(re.match(OPS[step["action"]], witness(step))),
                "wrong instruction on actual path for " + step["action"],
            )
        paired_flags(walk, entries)
        if key in ("first", "odd", "even") and order == "before_fence":
            issued = next(x["walk_index"] for x in entries if x["action"] == "cube")
            begins = next(
                x["walk_index"] for x in entries if x["action"] == "cast" and x["walk_index"] > issued
            )
            fence = entries[-1]["walk_index"]
            gap = [x["asm"].lower() for x in walk[issued + 1 : begins]]
            if any(re.match(OPS["all"], asm) for asm in gap):
                raise NoOverlap("whole core drained before first lookahead Cast")
            if any(re.match(r"wait.*(?:pipe_m|\bm)\b.*(?:pipe_[sv]|\b[sv]\b)", asm) for asm in gap):
                raise StaticBlocked("unresolved Cube completion wait before lookahead")
            require(
                not any(re.match(OPS["all"], x["asm"].lower()) for x in walk[begins:fence]),
                "extra whole-core fence within lookahead; partial overlap unresolved",
            )
        if key.startswith("false_"):
            require(
                not any(
                    re.search(r"\b(?:vconv_s82f16|mmad|vmul|copy_ubuf_to_gm)\b", x["asm"].lower()) for x in walk
                ),
                "speculative compute or output on Iterate failure",
            )
    if mode == OVERLAP and (order == "after_fence" or packet["cube_drained_before_next_d"] is True):
        raise NoOverlap("reviewed final machine schedule is serialized")
    require(
        packet["cube_drained_before_next_d"] is False if mode == OVERLAP else order == "after_fence",
        "wrong or ambiguous serial/overlap machine path",
    )


def proof_index(path, expected):
    require(
        path.is_absolute() and re.fullmatch(r"[a-f0-9]{64}", expected or ""),
        "explicit external proof index SHA256 required",
    )
    item = c.identity(path)
    require(item["sha256"] == expected, "review index digest changed")
    index = json.loads(c.verify(item).read_bytes(), object_pairs_hook=unique_object)
    require(
        index.get("schema") == 1
        and index.get("kind") == "W16_EXACT_ATC_MACHINE_EVIDENCE"
        and isinstance(index.get("reports"), dict)
        and index["reports"],
        "invalid technical review index",
    )
    for name, ref in index["reports"].items():
        require(
            re.fullmatch(r"[0-9a-f]{64}", name) and set(ref) == {"path", "bytes", "sha256"},
            "bad reviewed ELF/file identity",
        )
        c.verify(ref)
    return dict(file=item, reports=index["reports"])


def admit_static(records, models, selected, sdk, commit, external=None):
    result = {}
    for key, variants in records.items():
        if set(variants) == set(DOUBLE) and all(
            variants[SERIAL][n]["elf"]["sha256"] == variants[OVERLAP][n]["elf"]["sha256"]
            for n in variants[SERIAL]
        ):
            raise NoOverlap("B/C selected the same complete ATC machine image")
        for mode, nodes in variants.items():
            model = models[mode][key]
            require(set(nodes) == set(model["gate_nodes"].values()), "wrong static node inventory")
            for name, cap in nodes.items():
                require(
                    cap["om"] == model["om"] and cap["node"] == model["atc_kernels"][name],
                    "static/model crossing",
                )
                if cap["status"] != "DECODED_REQUIRES_PC_PROOF":
                    raise StaticBlocked(cap.get("reason", "SDK decode incomplete") + ": " + cap["elf"]["path"])
                if external is None:
                    packet, packet_hash = reviewed_packet(commit, cap["node"]["sha256"])
                    provenance = dict(commit=commit)
                else:
                    index = proof_index(c.verify(external["file"]), external["file"]["sha256"])
                    require(index == external, "external review identity drift")
                    ref = index["reports"].get(cap["node"]["sha256"])
                    require(ref, "external dossier missing for " + cap["node"]["sha256"])
                    packet = json.loads(c.verify(ref).read_bytes(), object_pairs_hook=unique_object)
                    packet_hash = ref["sha256"]
                    provenance = dict(index=index["file"], packet=ref)
                source = selected[mode]["generated_sources"]["op_kernel/d_flash_group_quant_linear.cpp"]["file"]
                check_proof(packet, cap, mode, source, model["plans"]["16x2560x19456"]["contract"], sdk)
                result["/".join((key, mode, name))] = dict(
                    elf=cap["elf"], provenance=provenance, proof_sha256=packet_hash
                )
    require(result, "empty static admission")
    return result
