#!/usr/bin/env python3
"""Read-only CANN toolchain survey, independent ELF control, and frozen OM machine audit."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
OPS = HERE.parent
sys.path.insert(0, str(OPS / "draft_gate_w16_decoder_retry"))
import retry_decoder as retry
from w16_common import DOUBLE, FIRST, c, historical, host, one, require
from w16_static import NoOverlap, admit_static, proof_index, review_template
from inventory import discover

sys.path.insert(0, str(OPS / "draft_quant/simulator"))
from bottleneck_elf import elf_layout

HEAD = "DFlashDraftLmHeadTop1"
HEAD_SHA = "47ffde8366f14aa97267a8d52b3d55fc7fec7684126c578831152ade2d5c343b"
PINS = {
    DOUBLE[0]: (
        "7de3096e7697b9056366321c2585d3e77079651957a9f9f44611c4d4a7e0f323",
        "6ec3564e76c5e716ed2bae4c906dee958fac8dec5ac8b8d9fdc7a285e44e1aa4",
    ),
    DOUBLE[1]: (
        "195d076c225e54db57d50dda7a550df65a5c6fcf3842b4f256ecaa0b1cf90eb2",
        "458e9ebf0ec3b1fd9701aed0cc1c194f7b076a3c6bdfe7ecc40f41c5cf2cded9",
    ),
}
TRIPLE = (
    "--disassemble",
    "--triple=hiipu64-hisilicon-cce",
    "--mcpu=dav-m200",
    "--disassemble-zeroes",
)
ARCH = ("--disassemble", "--arch-name=hiipu64", "--mcpu=dav-m200")
CLAIMS = {
    "SetTensorB / SDK last W read": (
        "local_b_final_mte3_read_before_fence",
        "all_ub_read_write_intervals",
    ),
    "Iterate(k), success and failure": (
        "direct_same_core_no_deferred_client",
        "false_iterate_no_output_or_pending_event",
    ),
    "Dequant(k+1) relative to F(k)": (
        "no_intervening_full_drain_before_lookahead",
        "two_group128_roundings",
    ),
    "original PIPE_ALL(k) retained": (
        "n_boundary_fence_before_reuse",
        "all_reachable_blocks_and_relocations",
    ),
    "next SetTensorA/B after fence": ("alternate_bank_and_no_rebind_before_fence",),
    "W0/W1 actual addresses and consumers": ("actual_tpipe_base_offsets_and_failure_branch",),
    "UB intervals incl. SDK/Brcb scratch": ("all_ub_read_write_intervals",),
    "k=0, steady odd/even, k=9, N boundary": (
        "last_k_no_speculative_dma_or_dequant",
        "n_boundary_fence_before_reuse",
    ),
    "FP32 K order / sole final FP16 copy": (
        "partial_false_then_true_same_l0c",
        "one_final_fp16_copyout",
    ),
}


def copy_elf(path, blob, identity):
    require(
        len(blob) == identity["bytes"] and hashlib.sha256(blob).hexdigest() == identity["sha256"],
        "ELF identity mismatch",
    )
    with path.open("xb") as dst:
        dst.write(blob)
    path.chmod(0o444)
    item = c.identity(path)
    require(item["sha256"] == identity["sha256"], "ELF copy changed")
    return item


def reference_image(original, destination):
    """An independently pinned control may be a genuine ET_REL or linked ET_EXEC."""
    blob = c.verify(original).read_bytes()
    file = copy_elf(destination, blob, original)
    layout = elf_layout(c.verify(file))
    require(
        layout["elf_type"] in (1, 2) and layout["machine"] == 0x1029,
        "control is not a HiIPU object",
    )
    table = [
        dict(
            s,
            va=s["address"],
            sha256=hashlib.sha256(blob[s["offset"] : s["offset"] + s["bytes"]]).hexdigest(),
        )
        for s in layout["executable_sections"]
        if s["bytes"]
    ]
    require(any(s["name"] == ".text" for s in table), "empty reference text")
    return dict(
        elf=file,
        table=table,
        elf_type=layout["elf_type"],
        machine=layout["machine"],
        elf_flags=struct.unpack_from("<I", blob, 48)[0],
    )


def archive_images(out, chain, assets):
    src, raw, _ = assets
    ref_build = c.read(c.verify(src["flow_build"]))["variants"]["A"]

    def head(variant):
        return one(
            (row for row in variant["opp"]["preflight"]["kernels"] if row["op"] == HEAD),
            "unique independent Head",
        )

    parent, control = head(ref_build), head(raw)
    require(
        all(head(v)["sha256"] == HEAD_SHA for v in chain["build"]["variants"].values())
        and parent["sha256"] == control["sha256"] == HEAD_SHA
        and parent["bytes"] == control["bytes"] == 18632,
        "independent control differs from frozen Head",
    )
    original = c.identity(control["binary"])
    meta = c.identity(control["json"])
    info = c.read(c.verify(meta))
    domain = (
        Path(raw["opp"]["install"]).resolve()
        / "vendors/customize/op_impl/ai_core/tbe/kernel/ascend310p/d_flash_draft_lm_head_top1"
    )
    require(
        Path(original["path"]).parent == domain
        and Path(meta["path"]) == Path(original["path"]).with_suffix(".json")
        and original["sha256"] == HEAD_SHA
        and meta["sha256"] == control["json_sha256"]
        and info["sha256"] == HEAD_SHA
        and info["binFileName"] == Path(original["path"]).stem,
        "unbound independent control metadata",
    )
    require(original["bytes"] == 18632, "reference byte length drift")
    images = {HEAD: reference_image(original, out / "independent-head.elf")}
    provenance = {
        HEAD: dict(
            original=original,
            meta=meta,
            flow_build=src["flow_build"],
            op=HEAD,
            source=c.identity(OPS / "draft_head/op_kernel/head_kernel.h"),
            kind="FROZEN_ASCEND310P_OPP_CONTROL",
        )
    }
    caps = chain["atc"]["captures"][FIRST]
    for mode in DOUBLE:
        node, cap = one(caps[mode].items(), "single Gate required")
        blob, table = retry.bound_image(cap)  # includes a byte-for-byte comparison with the actual OM slice
        text = one((s for s in table if s["name"] == ".text"), "unique text")
        require(
            (cap["elf"]["sha256"], text["sha256"]) == PINS[mode]
            and len(blob) == 32304
            and text["bytes"] == 25964,
            "not the two EqxSwk images",
        )
        images[mode] = dict(elf=copy_elf(out / (mode + ".elf"), blob, cap["elf"]), table=table)
        provenance[mode] = dict(
            om=cap["om"],
            elf=cap["elf"],
            graph_node=node,
            graph_binding=cap["node"],
            sections=cap["sections"],
            archived_attempts=retry.archived_attempts(cap, blob, table),
        )
    return images, provenance


def command(tool, args, folder, tag, env, stdin=None):
    log = folder / (tag + ".log")
    argv = [str(c.verify(tool)), *map(str, args)]
    with log.open("xb") as dst:
        try:
            proc = subprocess.run(
                argv,
                env=env,
                cwd=folder,
                input=stdin,
                stdout=dst,
                stderr=subprocess.STDOUT,
                timeout=60,
            )
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = None
    c.verify(tool)
    return dict(argv=argv, returncode=rc, listing=c.identity(log))


def source_trust(row, cann, additional):
    path = Path(row["file"]["path"])
    if not os.access(path, os.X_OK):
        return "INVENTORY_ONLY"
    if additional and row["file"] == additional:
        return "EXPLICITLY_PINNED_SDK"
    if cann in path.parents and "CANN" in row["scopes"]:
        return "SELECTED_SDK"
    if path.parent == Path("/usr/bin") or re.fullmatch(r"/usr/lib(?:64)?/llvm-[\d.]+/bin", str(path.parent)):
        return "SYSTEM_INSPECTION_ONLY"
    return "INVENTORY_ONLY"


def read_log(rec):
    return c.verify(rec["listing"]).read_text(errors="replace")


def probe_inventory(inv, cann, out, env, reference, extra):
    dumpers = []
    for index, row in enumerate(inv["tools"]):
        roles = set(row["roles"])
        tool = row["file"]
        trust = source_trust(row, cann, extra)
        row["trust"] = trust
        if (
            "library_inspect_only" in roles
            and Path(tool["path"]).name in ("libcamodel.so", "lib_pvmodel.so")
            and Path("/usr/bin/nm").is_file()
        ):
            folder = out / f"model-{index:03}"
            folder.mkdir()
            inspection = command(
                c.identity("/usr/bin/nm"),
                ["-D", "--defined-only", "-C", c.verify(tool)],
                folder,
                "exported-symbols",
                {"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            )
            lines = read_log(inspection).splitlines()
            row["model_exports"] = dict(
                command=inspection,
                model_disassembler=[
                    x
                    for x in lines
                    if re.search(r"\s(?:disassembler_t::disassemble\(|processor_t::disasm\()", x)
                ],
                public_c_api=[x for x in lines if re.search(r"\sT pv_\w+$", x)],
            )
            row["capability"] = "MODEL_IMPLEMENTATION_SYMBOLS_ONLY"
            continue
        if "library_inspect_only" in roles and Path(tool["path"]).name.lower().startswith("libllvm"):
            # Inspect only public LLVM-C exports. No dlopen, private ABI calls or simulation.
            nm = Path("/usr/bin/nm")
            if nm.is_file():
                folder = out / f"lib-{index:03}"
                folder.mkdir()
                nmi = c.identity(nm)
                result = command(
                    nmi,
                    ["-D", "--defined-only", c.verify(tool)],
                    folder,
                    "public-symbols",
                    {"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
                )
                log = read_log(result)
                targets = sorted(set(re.findall(r"\bLLVMInitialize(\w+)Disassembler\b", log)))
                row["public_llvm_api"] = dict(
                    nm=nmi,
                    command=result,
                    targets=targets,
                    disasm_instruction=bool(re.search(r"\bLLVMDisasmInstruction\b", log)),
                )
                row["capability"] = (
                    "PUBLIC_DAV_BACKEND_NEEDS_CONTROL"
                    if any(re.search(r"dav|hiipu|aicore", t, re.I) for t in targets)
                    else "NO_PUBLIC_DAV_DISASSEMBLER_EXPORT_OBSERVED"
                    if result["returncode"] == 0
                    else "SYMBOLS_UNAVAILABLE"
                )
            continue
        active_roles = {
            "compiler",
            "llvm_mc",
            "debugger",
            "objdump",
            "metadata_or_debug_dump",
            "possible_isa_tool",
        }
        if trust == "INVENTORY_ONLY" or (not (active_roles & roles) and trust != "EXPLICITLY_PINNED_SDK"):
            continue
        folder = out / f"tool-{index:03}"
        folder.mkdir()
        system = trust == "SYSTEM_INSPECTION_ONLY"
        child_env = (
            {
                "PATH": "/usr/bin:/bin",
                "LC_ALL": "C",
                "DEBUGINFOD_URLS": "",
                "TMPDIR": str(folder),
            }
            if system
            else env
        )
        base = Path(tool["path"]).name
        if "metadata_or_debug_dump" in roles:
            # These are SDK container/metadata tools. Retain their actual advertised operations.
            if system or base not in ("msobjdump", "show_kernel_debug_data"):
                continue
            r = command(tool, ["--help"], folder, "help", child_env)
            row["help"] = r
            row["advertised_operations"] = [s.strip()[:200] for s in read_log(r).splitlines() if "--" in s][
                :20
            ]
            row["capability"] = "METADATA_HELP_CAPTURED"
            continue
        if "possible_isa_tool" in roles or (trust == "EXPLICITLY_PINNED_SDK" and "objdump" not in roles):
            # A vendor may ship a dedicated static disassembler under a different name.
            # Do not treat source/trace readers or dump-collection programs as executables.
            static_name = re.search(r"disasm|disassem|decoder", base, re.I)
            if not (
                "objdump" in roles
                or trust == "EXPLICITLY_PINNED_SDK"
                or (static_name and row["format"].startswith("ELF") and "bin" in Path(tool["path"]).parts)
            ):
                row["capability"] = "SOURCE_OR_UNCLASSIFIED_READER"
                continue
            h = command(tool, ["--help"], folder, "dedicated-help", child_env)
            row["help"] = h
            usage = read_log(h)
            if "--disassemble" in usage:
                dumpers.append((row, folder, child_env, [retry.PROBE]))
                row["capability"] = "DEDICATED_CONTROL_REQUESTED"
            elif re.search(r"(?im)^\s*-d\b[^\n]*disassembl", usage):
                dumpers.append((row, folder, child_env, [("-d",)]))
                row["capability"] = "DEDICATED_SHORT_OPTION_CONTROL_REQUESTED"
            else:
                row["capability"] = "CLI_OPERANDS_NEED_DOCUMENTATION"
            continue
        if not row["format"].startswith("ELF"):
            continue  # do not invoke unresolved debugger/front-end wrappers
        version = command(tool, ["--version"], folder, "version", child_env)
        version_text = read_log(version)
        row.update(version=version, version_excerpt=version_text.splitlines()[:4])
        llvm = "LLVM" in version_text or "clang version" in version_text
        if "compiler" in roles and llvm and base != "llc":
            row["targets"] = command(tool, ["--print-targets"], folder, "targets", child_env)
            if not system and base in ("bisheng", "ccec"):
                front = command(tool, ["-cc1", "--help"], folder, "cc1-help", child_env)
                row["internal_frontend"] = dict(
                    command=front,
                    disasm_options=[
                        s.strip()[:240] for s in read_log(front).splitlines() if "disassem" in s.lower()
                    ],
                )
                row["capability"] = "COMPILER_TARGET_AND_CC1_HELP"
        elif "objdump" in roles and "GNU" in version_text and not llvm:
            row["targets"] = command(tool, ["--info"], folder, "targets", child_env)
        targets = read_log(row["targets"]) if "targets" in row else version_text
        row["hiipu_registered"] = bool(re.search(r"^\s*hiipu(?:64)?\s+[-:]", targets, re.I | re.M))
        if "objdump" in roles:
            help_rec = command(tool, ["--help"], folder, "help", child_env)
            row["help"] = help_rec
            helptext = read_log(help_rec)
            if "--disassemble" not in helptext or not (llvm or "GNU" in version_text):
                row["capability"] = "UNREVIEWED_CLI"
                continue
            flags = [retry.PROBE]
            if llvm and row["hiipu_registered"]:
                cpu = command(
                    tool,
                    ["--disassemble", "--mcpu=help", reference["elf"]["path"]],
                    folder,
                    "cpus",
                    child_env,
                )
                row["cpu_help"] = cpu
                row["dav_m200_advertised"] = bool(re.search(r"^\s*dav-m200\s+-", read_log(cpu), re.M))
                flags += [retry.TARGET, ARCH, TRIPLE]
                if "--disassemble-aicore" in helptext:
                    flags.extend([retry.AICORE, retry.AICORE_TARGET])
            dumpers.append((row, folder, child_env, flags))
        elif "llvm_mc" in roles:
            row["capability"] = "RAW_STREAM_INTERFACE"
            if row["hiipu_registered"]:
                t = one(
                    (s for s in reference["table"] if s["name"] == ".text"),
                    "reference text",
                )
                data = c.verify(reference["elf"]).read_bytes()[t["offset"] : t["offset"] + t["bytes"]]
                tokens = " ".join(f"0x{x:02x}" for x in data) + "\n"
                row["mc_reference"] = command(
                    tool,
                    [
                        "--disassemble",
                        "--show-encoding",
                        "--triple=hiipu64-hisilicon-cce",
                        "--mcpu=dav-m200",
                    ],
                    folder,
                    "reference-stream",
                    child_env,
                    tokens.encode(),
                )
                row["capability"] = "REVIEW_RAW_STREAM_OUTPUT_FOR_EXPLICIT_ELF_PC_BINDING"
        elif "debugger" in roles:
            # File-only sessions, no target process, no startup scripts or auto-loaded code.
            elf = c.verify(reference["elf"])
            if re.fullmatch(r"gdb(?:-[\d.]+)?", base):
                text = one(
                    (s for s in reference["table"] if s["name"] == ".text"),
                    "reference text",
                )
                row["file_target"] = command(
                    tool,
                    [
                        "--nx",
                        "--nh",
                        "--batch",
                        "-iex",
                        "set auto-load off",
                        "-ex",
                        "file " + json.dumps(str(elf)),
                        "-ex",
                        "show architecture",
                        "-ex",
                        f"disassemble /r 0x{text['va']:x},0x{text['va'] + text['bytes']:x}",
                    ],
                    folder,
                    "target",
                    dict(child_env, DEBUGINFOD_URLS=""),
                )
            elif re.fullmatch(r"lldb(?:-[\d.]+)?", base):
                text = one(
                    (s for s in reference["table"] if s["name"] == ".text"),
                    "reference text",
                )
                row["file_target"] = command(
                    tool,
                    [
                        "--no-lldbinit",
                        "--batch",
                        "-O",
                        "settings set target.load-script-from-symbol-file false",
                        "-o",
                        "target create -- " + json.dumps(str(elf)),
                        "-o",
                        "image list -t",
                        "-o",
                        f"disassemble --bytes --start-address 0x{text['va']:x} --end-address 0x{text['va'] + text['bytes']:x}",
                    ],
                    folder,
                    "target",
                    child_env,
                )
            if "file_target" in row:
                row["capability"] = "FILE_TARGET_INSPECTED"
                if re.search(r"hiipu|dav[-_]m?200", read_log(row["file_target"]), re.I):
                    row["capability"] = "REVIEW_DEBUGGER_TARGET"
                    row["target_excerpt"] = read_log(row["file_target"])[:1000]
            else:
                row["help"] = command(tool, ["--help"], folder, "help", child_env)
                row["capability"] = "NO_REVIEWED_FILE_ONLY_DEBUG_COMMAND"
    return dumpers


def reference_semantics(decoded):
    # The independent released Head stages FP16 hidden from GM, executes a Cube
    # dot, and explicitly fences it. Bind the exact operand-carrying PCs.
    groups = {
        "GM_to_UB": r"copy_gm_to_ubuf",
        "Cube": r"(?:mmad|mad)",
        "all_fence": r"(?:pipe_barrier|barrier)",
    }
    evidence = {}
    for role, pattern in groups.items():
        matches = []
        for section, rows in decoded.items():
            for r in rows:
                asm = r["asm"].lower().split(None, 1)
                if (
                    len(asm) == 2
                    and re.fullmatch(pattern, asm[0])
                    and asm[1].strip()
                    and (role != "all_fence" or "all" in asm[1])
                ):
                    matches.append(dict(section=section, **r))
        evidence[role] = dict(count=len(matches), examples=matches[:2])
    return evidence


def attempt(tool, flags, image, folder, tag, env, check_control=False):
    path = c.verify(image["elf"])
    r = command(tool, [*flags, path], folder, tag, env)
    blob = c.verify(image["elf"]).read_bytes()
    listing = read_log(r)
    cover, decoded = retry.listing_evidence(blob, image["table"], listing, r["returncode"])
    result = dict(
        r,
        **cover,
        text_byte_witnesses=text_coverage(blob, image["table"], listing, r["returncode"]),
    )
    text = result["text_byte_witnesses"]
    if decoded is not None and text["bytes"] != text["total"]:
        result.update(outcome="INCOMPLETE_PC_ENCODING_MNEMONIC_OPERAND_COVERAGE", verified_bytes=0)
        decoded = None
    if decoded is not None and check_control:
        witnesses = reference_semantics(decoded)
        result["independent_witnesses"] = witnesses
        if not all(w["count"] for w in witnesses.values()):
            result["outcome"] = "REFERENCE_SEMANTICS_UNPROVEN"
            decoded = None
    if decoded is not None:
        path = folder / (tag + "-instructions.json")
        c.write(path, decoded)
        result["instructions"] = c.identity(path)
    return result


def text_coverage(blob, table, listing, rc):
    """Union of in-bounds .text byte ranges printed next to an explicit PC and instruction.

    Even a partial witness must match actual bytes. This diagnostic cannot grant
    admission; the original sequential decoder rejects gaps, unknowns and overlaps.
    """
    sec = one(
        (s for s in table if s["name"] == ".text" and s["flags"] & 4),
        "executable .text",
    )
    seen = bytearray(sec["bytes"])
    active = False
    if (
        type(rc) is int
        and rc == 0
        and re.findall(r"^.*:\s+file format (\S+)\s*$", listing, re.M) == ["elf64-hiipu"]
    ):
        for line in listing.splitlines():
            label = re.fullmatch(r"\s*Disassembly of section (.+):\s*", line)
            if label:
                active = label[1] == ".text"
                continue
            if not active:
                continue
            m = re.fullmatch(
                r"\s*([0-9a-fA-F]+):\s+((?:(?:[0-9a-fA-F]{2}|[0-9a-fA-F]{4}|[0-9a-fA-F]{8}|[0-9a-fA-F]{16})\s+)+)([A-Za-z][\w.]*)(?:\s+(.*?))?\s*",
                line,
            )
            if not m:
                continue
            op = m[3].lower()
            if (
                op
                in (
                    "byte",
                    "short",
                    "long",
                    "quad",
                    "word",
                    "data",
                    "inst",
                    "db",
                    "dw",
                    "dd",
                    "dq",
                    "unimp",
                    "invalid",
                    "undefined",
                )
                or "unknown" in op
            ):
                continue
            if not m[4] and op not in ("ret", "nop", "end", "task_end", "block_end"):
                continue
            start = int(m[1], 16) - sec["va"]
            code = bytes.fromhex(m[2])
            end = start + len(code)
            if start < 0 or end > len(seen):
                continue
            actual = blob[sec["offset"] + start : sec["offset"] + end]
            words = m[2].split()
            if code == actual or (
                all(len(x) > 2 for x in words) and b"".join(bytes.fromhex(x)[::-1] for x in words) == actual
            ):
                seen[start:end] = b"\x01" * len(code)
    return dict(bytes=seen.count(1), total=len(seen))


def checked_audit_tools(tools, images, caps, bound_tool, optional_pin):
    """Only the SAME decoder + flags that passed a separate Head may sponsor a candidate proof."""
    probes = []
    pairs = []
    for row, folder, env, options in tools:
        tool = row["file"]
        attempts = []
        records = {mode: [] for mode in DOUBLE}
        usable = {mode: None for mode in DOUBLE}
        for idx, flags in enumerate(options):
            ref = attempt(tool, flags, images[HEAD], folder, f"control-{idx}", env, True)
            attempts.append(ref)
            ref_ok = ref["outcome"] == "COMPLETE_BYTE_MATCH" and "instructions" in ref
            # Always re-observe the archive-bound CANN decoder on both real images,
            # even if the control failed. Other unsuccessful tools stop at the control.
            if not ref_ok and tool != bound_tool:
                continue
            for mode in DOUBLE:
                hit = attempt(tool, flags, images[mode], folder, f"{mode}-{idx}", env)
                hit["control_listing"] = ref["listing"]
                hit["control_outcome"] = ref["outcome"]
                hit["qualified_bytes"] = hit["verified_bytes"] if ref_ok else 0
                records[mode].append(hit)
                if (
                    ref_ok
                    and "instructions" in hit
                    and flags in retry.LEGACY_FLAGS
                    and (tool == bound_tool or tool == optional_pin)
                    and usable[mode] is None
                ):
                    cap = caps[mode]
                    capture = dict(
                        om=cap["om"],
                        node=cap["node"],
                        sections=cap["sections"],
                        tool=tool,
                        elf=images[mode]["elf"],
                        status="DECODED_REQUIRES_PC_PROOF",
                        instructions=hit["instructions"],
                        **{k: hit[k] for k in ("argv", "returncode", "listing")},
                    )
                    usable[mode] = capture
        probes.append(
            dict(
                tool=tool,
                folder=str(folder),
                reference_attempts=attempts,
                candidate_attempts=records,
            )
        )
        if all(usable.values()):
            pairs.append(usable)
    return probes, pairs


def claims_for(packet=None):
    if packet is None:
        return {name: {"status": "UNPROVEN", "pcs": []} for name in CLAIMS}
    return {
        name: {
            "status": "REVIEWED_AND_REPLAYED",
            "derivations": {k: packet["obligations"][k] for k in keys},
        }
        for name, keys in CLAIMS.items()
    }


def prove(out, r, chain, pairs, args):
    for i, pair in enumerate(pairs):
        captures = {FIRST: {}}
        for mode, cap in pair.items():
            source = chain["build"]["variants"][mode]["generated_sources"][
                "op_kernel/d_flash_group_quant_linear.cpp"
            ]["file"]
            plan = chain["atc"]["models"][mode][FIRST]["plans"]["16x2560x19456"]["contract"]
            sdk = chain["source"]["binding"]["sdk"]
            p = out / f"review-input-{i}-{mode}.json"
            c.write(p, review_template(cap, mode, source, plan, sdk))
            captures[FIRST][mode] = {cap["node"]["node"]: dict(cap, review_input=c.identity(p))}
        try:
            external = (
                None if args.proof_index is None else proof_index(args.proof_index, args.proof_index_sha256)
            )
            commit = subprocess.check_output(
                ["git", "-C", str(c.REPO), "rev-parse", "HEAD"], text=True
            ).strip()
            proofs = admit_static(
                captures,
                chain["atc"]["models"],
                chain["build"]["variants"],
                sdk,
                commit,
                external,
            )
        except NoOverlap as exc:
            r.update(status="NO_TESTABLE_OVERLAP", reason=str(exc))
            return
        except (ValueError, OSError, KeyError) as exc:
            r["proof_failures"].append(str(exc))
            continue
        from w16_static import reviewed_packet

        verified_timing = {}
        for mode, cap in pair.items():
            sha = cap["elf"]["sha256"]
            if external is None:
                packet, packet_hash = reviewed_packet(commit, sha)
            else:
                ref = external["reports"][sha]
                packet = c.read(c.verify(ref))
                packet_hash = ref["sha256"]
            proof = proofs["/".join((FIRST, mode, cap["node"]["node"]))]
            require(packet_hash == proof["proof_sha256"], "proof changed after verification")
            verified_timing[mode] = dict(
                claims=claims_for(packet),
                placement=packet["observed_placement"],
                bank_accesses=packet["accesses"],
                byte_intervals=packet["regions"],
                paths=packet["paths"],
            )
        r.update(
            status="W16_STATIC_PASS",
            reason="BOTH_MACHINE_PROOFS_REPLAYED",
            proofs=proofs,
            timing=verified_timing,
        )
        return
    if pairs:
        r["reason"] = "COMPLETE_DECODE_BUT_MACHINE_PROOF_NOT_VERIFIED"


def markdown(report):
    lines = [
        "# EqxSwk W16 指令审计",
        "",
        f"**{report['status']}**　{report['reason']}",
        "",
        "本次为离线审计。Source/Build/ATC 仅重验封存证据。NPU、Smoke、Benchmark、Profile 均为 NOT_RUN。",
    ]
    if "control" not in report:
        return "\n".join(lines) + "\n"
    lines += [
        "",
        "## 独立控制及可执行段覆盖",
        "",
        "| 对象 | ELF SHA256 | .text SHA256 | 核验的 .text 字节 / 总字节 | 通过控制校验的工具覆盖 |",
        "|---|---|---|---:|---:|",
    ]
    for name, row in report["images"].items():
        text = one((s for s in row["table"] if s["name"] == ".text"), "one .text")
        cov = report.get("coverage", {}).get(name)
        cells = (
            f"{cov['observed_text_bytes']}/{text['bytes']} | {cov['qualified_text_bytes']}/{text['bytes']}"
            if cov
            else "UNPROVEN | UNPROVEN"
        )
        lines.append(f"| {name} | `{row['elf']['sha256']}` | `{text['sha256']}` | {cells} |")
    lines += [
        "",
        "分母为实际 ELF 的 .text 长度。字节依据来自单次完整保留的工具调用。各次调用的局部片段不相加。",
        "有效覆盖需要 PC、实际编码、助记符和适用的操作数。独立对照还需出现带操作数的 GM→UB、Cube dot 和全流水 fence。",
        "",
        "## 指令级时序对照",
        "",
        "| 所需证明 | double-w-serial | double-w-overlap |",
        "|---|---|---|",
    ]
    for name in CLAIMS:
        values = []
        for mode in DOUBLE:
            row = report["timing"][mode]["claims"][name]
            if row["status"] == "UNPROVEN":
                values.append("UNPROVEN")
            else:
                pcs = sorted({w["pc"] for d in row["derivations"].values() for w in d["witnesses"]})
                values.append("、".join(map(hex, pcs)) + "（完整路径见 JSON）")
        lines.append(f"| {name} | {values[0]} | {values[1]} |")
    lines += [
        "",
        "## 工具身份",
        "",
        "| 工具实际路径 | SHA256 | 角色 |",
        "|---|---|---|",
    ]
    for t in report.get("inventory", {}).get("tools", []):
        lines.append(f"| `{t['file']['path']}` | `{t['file']['sha256']}` | {', '.join(t['roles'])} |")
    lines += [
        "",
        "各工具的版本、目标、已探测范围、原始 stdout/stderr、项目解析器源码和 SDK 后端行号见同目录 JSON。",
        "设备入口只在原 W16 工作流正式 static PASS 后使用。当前实际许可见 status。",
    ]
    return "\n".join(lines) + "\n"


def audit_tool(cann, original, path, pin, extra_roots):
    canonical = retry.select_tool(cann, original, None, None)
    if path is None:
        require(pin is None, "decoder digest needs a tool path")
        return canonical, canonical
    require(
        path.is_absolute() and re.fullmatch(r"[a-f0-9]{64}", pin or ""),
        "alternate decoder needs an absolute path and independent digest",
    )
    item = c.identity(path)
    real = c.verify(item)
    require(
        item["sha256"] == pin
        and os.access(real, os.X_OK)
        and any(root.resolve(strict=True) in real.parents for root in (cann, *extra_roots)),
        "alternate decoder digest/installation/permissions changed",
    )
    # Other documented CLIs may be tested on the independent ELF, but the old
    # formal flow only accepts these two SDK tool names within its CANN root.
    formal = (
        retry.select_tool(cann, original, path, pin)
        if cann in real.parents and real.name in ("llvm-objdump", "ccec-objdump")
        else None
    )
    return item, formal


def run(out, old, args, r):
    refs, chain, assets = retry.load_archive(old)
    src, _, _ = assets
    cann = Path(src["cann"]).resolve(strict=True)
    src_tool = chain["source"]["binding"]["tool"]
    tool, formal_tool = audit_tool(cann, src_tool, args.decoder, args.decoder_sha256, args.extra_tool_root)
    require(
        bool(args.proof_index) == bool(args.proof_index_sha256),
        "external proof index and expected SHA must be supplied together",
    )
    images, origin = archive_images(out, chain, assets)
    prior = c.read(c.verify(src["host_source"]))["binding"]["prior"]
    env, env_id = host.envs.for_prior(prior, None)
    require(
        not env["ASCEND_CUSTOM_OPP_PATH"],
        "unexpected private OPP in offline tool environment",
    )
    env.update(
        TMPDIR=str(out),
        XDG_CACHE_HOME=str(out / "cache"),
        PYTHONDONTWRITEBYTECODE="1",
        DEBUGINFOD_URLS="",
    )
    inv = discover(cann, c.REPO, os.environ.get("PATH", ""), args.extra_tool_root)
    if not any(t["file"] == tool for t in inv["tools"]):
        require(args.decoder is not None, "source-bound decoder absent from scoped inventory")
        from inventory import file_type

        inv["tools"].append(
            dict(
                file=tool,
                format=file_type(c.verify(tool)),
                aliases=[str(args.decoder)],
                roles=["possible_isa_tool"],
                scopes=["EXPLICIT"],
            )
        )
    r.update(
        upstream=refs,
        archived_status={k: x["status"] for k, x in chain.items()},
        images=images,
        control=origin[HEAD],
        image_provenance=origin,
        environment=env_id,
        inspection_env=env,
        chosen_tool=tool,
        inventory=inv,
        proof_failures=[],
    )
    dumpers = probe_inventory(inv, cann, out, env, images[HEAD], tool if args.decoder else None)
    require(
        any(g[0]["file"] == src_tool for g in dumpers),
        "original SDK decoder not probed",
    )
    caps = {m: next(iter(chain["atc"]["captures"][FIRST][m].values())) for m in DOUBLE}
    probes, pairs = checked_audit_tools(dumpers, images, caps, src_tool, formal_tool)
    r["tool_trials"] = probes
    r["comparison"] = retry.compare_text(*(caps[m] for m in DOUBLE))
    r["coverage"] = {}
    for name in images:
        tests = [
            a
            for t in probes
            for a in (t["reference_attempts"] if name == HEAD else t["candidate_attempts"][name])
        ]
        r["coverage"][name] = dict(
            observed_text_bytes=max((a["text_byte_witnesses"]["bytes"] for a in tests), default=0),
            qualified_text_bytes=max(
                (
                    a["text_byte_witnesses"]["bytes"]
                    for a in tests
                    if (
                        a["outcome"] == "COMPLETE_BYTE_MATCH"
                        if name == HEAD
                        else a.get("qualified_bytes", 0) > 0
                    )
                ),
                default=0,
            ),
            text_bytes=one(
                (s["bytes"] for s in images[name]["table"] if s["name"] == ".text"),
                "one .text",
            ),
            attempts=len(tests),
        )
    r["reason"] = (
        "NO_DECODER_PASSED_INDEPENDENT_CONTROL"
        if not r["coverage"][HEAD]["qualified_text_bytes"]
        else "NO_QUALIFIED_CANONICAL_PAIR"
    )
    prove(out, r, chain, pairs, args)
    historical.verify_tree(r, {})


def main(args):
    old = args.root.resolve(strict=True)
    protected = [
        c.REPO,
        old,
        Path("/usr/local/Ascend"),
        *map(Path, c.read(old / "static/report.json")["history_roots"]),
    ]
    parent = (args.output_parent or old.parent).resolve(strict=True)
    require(
        parent.is_dir()
        and all(parent != p.resolve() and p.resolve() not in parent.parents for p in protected),
        "output parent inside read-only evidence/SDK/source",
    )
    out = Path(tempfile.mkdtemp(prefix="w16-isa-audit-", dir=parent))
    r = dict(
        schema=1,
        status="BLOCKED_STATIC_ADMISSION",
        reason="AUDIT_INCOMPLETE",
        root=str(old),
        output=str(out),
        process_scope="CPU_OFFLINE_AUDIT",
        atc="NOT_RUN",
        build="NOT_RUN",
        device_execution="NOT_RUN",
        device_launch_authorized=False,
        benchmark="NOT_RUN",
        errors=[],
        timing={m: dict(claims=claims_for()) for m in DOUBLE},
        sources=[
            c.identity(p) for p in sorted(HERE.rglob("*")) if p.is_file() and "__pycache__" not in p.parts
        ],
    )
    try:
        c.clean()
        run(out, old, args, r)
        c.clean()
    except (
        OSError,
        ValueError,
        RuntimeError,
        IndexError,
        KeyError,
        TypeError,
        struct.error,
        subprocess.SubprocessError,
    ) as exc:
        r.update(status="BLOCKED_STATIC_ADMISSION", reason="AUDIT_EVIDENCE_INCOMPLETE")
        r["timing"] = {m: dict(claims=claims_for()) for m in DOUBLE}
        r.pop("proofs", None)
        r["errors"].append(repr(exc))
    c.write(out / "static-audit.json", r)
    (out / "static-audit.md").write_text(markdown(r))
    print(r["status"], r["reason"], r["errors"])
    for t in r.get("inventory", {}).get("tools", []):
        if set(t["roles"]) & {
            "compiler",
            "objdump",
            "llvm_mc",
            "possible_isa_tool",
            "debugger",
            "metadata_or_debug_dump",
            "library_inspect_only",
        }:
            print(
                "TOOL",
                t["file"]["path"],
                t["file"]["sha256"],
                t.get("trust"),
                "hiipu64=",
                t.get("hiipu_registered"),
                t.get("capability", ""),
            )
    if "control" in r:
        print("CONTROL ELF", r["control"]["original"]["path"])
    for name, cov in r.get("coverage", {}).items():
        print(
            "COVERAGE",
            name,
            cov["qualified_text_bytes"],
            "/",
            cov["text_bytes"],
            r["images"][name]["elf"]["sha256"],
        )
    for trial in r.get("tool_trials", []):
        for a in trial["reference_attempts"]:
            print(
                "CONTROL",
                Path(trial["tool"]["path"]).name,
                a["argv"][1:-1],
                a["returncode"],
                a["outcome"],
                a["pc_rows"],
                a["placeholder_rows"],
            )
    if "comparison" in r:
        print("B/C TEXT", r["comparison"])
    for mode in DOUBLE:
        for trial in r.get("tool_trials", []):
            if trial["tool"] == r["chosen_tool"]:
                print(
                    "ELF TRIALS",
                    mode,
                    [
                        (
                            a["argv"][1:-1],
                            a["returncode"],
                            a["pc_rows"],
                            a["placeholder_rows"],
                            a["qualified_bytes"],
                        )
                        for a in trial["candidate_attempts"][mode]
                    ],
                )
    print("REPORT", out / "static-audit.json")
    print("TIMING", out / "static-audit.md")
    return 0 if r["status"] == "W16_STATIC_PASS" else 3 if r["status"] == "NO_TESTABLE_OVERLAP" else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--root", type=Path, default=retry.DEFAULT_ROOT)
    p.add_argument("--output-parent", type=Path)
    p.add_argument(
        "--extra-tool-root",
        action="append",
        type=Path,
        default=[],
        help="additional read-only installation search root",
    )
    p.add_argument("--decoder", type=Path)
    p.add_argument("--decoder-sha256")
    p.add_argument("--proof-index", type=Path)
    p.add_argument("--proof-index-sha256")
    sys.exit(main(p.parse_args()))
