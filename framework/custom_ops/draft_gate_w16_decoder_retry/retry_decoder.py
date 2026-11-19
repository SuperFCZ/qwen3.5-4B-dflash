#!/usr/bin/env python3
"""Offline recovery of decoder evidence for the frozen first Gate. No ATC, model load or NPU calls."""

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
sys.path.insert(0, str(HERE.parent / "draft_gate_w16_atc8"))
import run_w16 as flow
from w16_common import DOUBLE, FIRST, SINGLE, c, historical, host, one, require
from w16_source import recovered
from w16_static import decode, disassembler, review_template, sections


PROBE = ("--disassemble",)
TARGET = ("--disassemble", "--mcpu=dav-m200")
AICORE = ("--disassemble-aicore",)
AICORE_TARGET = ("--disassemble", "--disassemble-aicore", "--mcpu=dav-m200")
LEGACY_FLAGS = (PROBE, TARGET)
DEFAULT_ROOT = Path("/home/w00949577/qwen35-dflash-run/gate-w16-atc8-EqxSwk")


def sources():
    return [c.identity(p) for p in sorted(HERE.rglob("*")) if p.is_file() and "__pycache__" not in p.parts]


def load_archive(root):
    """Run the unchanged formal source, build and ATC validators against the original evidence."""
    failed_path = root / "static/report.json"
    failed_id = c.identity(failed_path)
    failed = c.read(c.verify(failed_id))
    require(
        failed["schema"] == 1
        and failed["root"] == str(root)
        and failed["folder"] == str(failed_path.parent)
        and failed["phase"] == "static"
        and failed["status"] == "BLOCKED_STATIC_ADMISSION"
        and failed["error"]
        and not failed["integrity_error"]
        and failed["device_execution"] == failed["atc"] == failed["benchmark"] == "NOT_RUN",
        "expected a sealed host-only static failure",
    )
    lock = flow.sources()
    history, refs, chain, seen = failed["history_roots"], {}, {}, {}
    require(failed["sources"] == lock, "archived W16 sources differ from checkout")
    assets = None
    for stage in flow.requirements("static"):
        path, row = flow.previous(root, stage)
        identity = c.identity(path)
        require(c.read(c.verify(identity)) == row, "archived report changed while opening: " + stage)
        require(
            row["schema"] == 1
            and row["status"] == flow.passed(stage)
            and row["phase"] == stage
            and row["root"] == str(root)
            and row["folder"] == str(path.parent)
            and row["sources"] == lock
            and row["predecessors"] == refs
            and row["history_roots"] == history
            and not row["error"]
            and not row["integrity_error"],
            "invalid W16 predecessor: " + stage,
        )
        historical.verify_tree(row, seen)
        flow.validate(path.parent, row, chain, assets)
        require(flow.status(row) == row["status"], "archived stage status mismatch")
        c.verify(identity)
        refs[stage] = identity
        chain[stage] = row
        if stage == "source":
            assets = recovered(row["binding"])
    require(refs == failed["predecessors"], "static failure refers to different ATC artifacts")
    historical.verify_tree(failed, seen)
    c.verify(failed_id)
    refs["blocked_static"] = failed_id
    return refs, chain, assets


def listing_evidence(blob, table, listing, rc):
    exec_sections = [s for s in table if s["flags"] & 4 and s["bytes"]]
    executable = {s["name"] for s in exec_sections}
    require(executable and len(executable) == len(exec_sections), "ambiguous/missing executable sections")
    section, pc_rows, placeholders, first = None, 0, 0, None
    for line in listing.splitlines():
        label = re.fullmatch(r"\s*Disassembly of section (.+):\s*", line)
        if label:
            section = label[1]
            continue
        match = re.match(r"^\s*([0-9a-fA-F]+):\s*(.*)$", line)
        if section in executable and match:
            pc_rows += 1
            if re.search(r"<not\s+available>|<unknown>", match[2], re.I):
                placeholders += 1
                if first is None:
                    first = int(match[1], 16)
    formats = re.findall(r"^.*:\s+file format (\S+)\s*$", listing, flags=re.M)
    evidence = dict(
        pc_rows=pc_rows,
        placeholder_rows=placeholders,
        first_placeholder_pc=first,
        verified_bytes=0,
        reported_formats=formats,
    )
    if type(rc) is not int or rc != 0:
        return dict(evidence, outcome="NONZERO_RETURN"), None
    try:
        instructions = decode(blob, table, listing)
    except ValueError as exc:
        reason = (
            "NO_EXECUTABLE_PC_ROWS"
            if not pc_rows
            else "ALL_PC_ROWS_UNDECODED"
            if placeholders == pc_rows
            else "INCOMPLETE_OR_UNVERIFIED_LISTING"
        )
        return dict(evidence, outcome=reason, detail=str(exc)), None
    if formats != ["elf64-hiipu"]:
        return dict(evidence, outcome="UNBOUND_DEVICE_LISTING_FORMAT"), None
    data_directives = {"byte", "short", "long", "quad", "word", "data", "inst", "db", "dw", "dd", "dq"}
    if any(ins["asm"].split()[0].lower() in data_directives for rows in instructions.values() for ins in rows):
        return dict(evidence, outcome="DATA_DIRECTIVE_IN_EXECUTABLE_SECTION"), None
    evidence.update(
        outcome="COMPLETE_BYTE_MATCH", verified_bytes=sum(s["bytes"] for s in table if s["name"] in executable)
    )
    return evidence, instructions


def archived_attempts(cap, blob, table):
    rows = []
    for attempt in cap["attempts"]:
        argv = attempt["argv"]
        flags = tuple(argv[1:-1])
        require(
            len(rows) < len(LEGACY_FLAGS)
            and flags == LEGACY_FLAGS[len(rows)]
            and argv[0] == cap["tool"]["path"]
            and argv[-1] == cap["elf"]["path"]
            and type(attempt["returncode"]) is int,
            "unbound archived decoder invocation",
        )
        evidence, _ = listing_evidence(
            blob, table, c.verify(attempt["listing"]).read_text(errors="replace"), attempt["returncode"]
        )
        rows.append(dict(attempt, **evidence))
    require(
        rows and all(cap[k] == rows[-1][k] for k in ("argv", "returncode", "listing")),
        "capture/last attempt mismatch",
    )
    require(
        (rows[-1]["outcome"] == "COMPLETE_BYTE_MATCH") == (cap["status"] == "DECODED_REQUIRES_PC_PROOF"),
        "recorded and independently replayed decoder outcome differ",
    )
    require(
        cap["status"] in ("BLOCKED_STATIC_ADMISSION", "DECODED_REQUIRES_PC_PROOF")
        and (cap["status"] != "BLOCKED_STATIC_ADMISSION" or len(rows) == len(LEGACY_FLAGS))
        and all(row["outcome"] != "COMPLETE_BYTE_MATCH" for row in rows[:-1]),
        "invalid archived attempt sequence",
    )
    return rows


def bound_image(cap):
    blob = c.verify(cap["elf"]).read_bytes()
    node = cap["node"]
    require(
        len(blob) == node["bytes"]
        and hashlib.sha256(blob).hexdigest() == node["sha256"]
        and node["symbol"].encode() in blob,
        "not the node-bound complete ELF",
    )
    with c.verify(cap["om"]).open("rb") as stream:
        stream.seek(node["offset"])
        require(stream.read(len(blob)) == blob, "archived ELF no longer equals selected OM bytes")
    table = sections(blob)
    require(c.read(c.verify(cap["sections"])) == table, "archived ELF section table changed")
    return blob, table


def run_tool(tool, flags, elf, log, env):
    argv = [str(c.verify(tool)), *flags, str(elf)] if elf else [str(c.verify(tool)), *flags]
    with log.open("xb") as stream:
        p = subprocess.run(argv, cwd=log.parent, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
    c.verify(tool)
    return dict(argv=argv, returncode=p.returncode, listing=c.identity(log))


def retry_one(out, cap, tool, env, has_aicore):
    """Never count a help banner, an address, a word dump or a partial decode as evidence."""
    out.mkdir()
    blob, table = bound_image(cap)
    archived = archived_attempts(cap, blob, table)
    # Never expose an archived ELF as a writable tool input or working directory.
    elf = out / "selected.elf"
    with elf.open("xb") as stream:
        stream.write(blob)
    elf.chmod(0o444)
    exact = c.identity(elf)
    require(
        (exact["sha256"], exact["bytes"]) == (cap["node"]["sha256"], cap["node"]["bytes"]), "copy/OM mismatch"
    )
    record = dict(
        archived_elf=cap["elf"],
        probe_elf=exact,
        om=cap["om"],
        node=cap["node"],
        sections=cap["sections"],
        archived_tool=cap["tool"],
        archived_attempts=archived,
        attempts=[],
    )
    same = tool == cap["tool"]
    result = None
    if same and cap["status"] == "DECODED_REQUIRES_PC_PROOF":
        decoded = decode(blob, table, c.verify(cap["listing"]).read_text())
        require(decoded == c.read(c.verify(cap["instructions"])), "archived decoded inventory drift")
        result = cap
    trials = [] if same else list(LEGACY_FLAGS)
    if result is None and has_aicore:
        trials.extend((AICORE, AICORE_TARGET))
    # Same pinned binary/ELF and canonical arguments already have immutable observations.
    # Extra documented selectors are capability probes; keep their exact argv.
    formal_attempts = []
    for index, flags in enumerate(trials):
        attempt = run_tool(tool, flags, c.verify(exact), out / f"attempt-{index}.disassembly", env)
        c.verify(exact)
        evidence, instructions = listing_evidence(
            blob, table, c.verify(attempt["listing"]).read_text(errors="replace"), attempt["returncode"]
        )
        record["attempts"].append(dict(attempt, **evidence))
        if flags in LEGACY_FLAGS:
            formal_attempts.append(attempt)
        if instructions is not None:
            path = out / "instructions.json"
            c.write(path, instructions)
            if flags in LEGACY_FLAGS:
                result = dict(
                    om=cap["om"],
                    node=cap["node"],
                    elf=exact,
                    sections=cap["sections"],
                    tool=tool,
                    attempts=formal_attempts,
                    **attempt,
                    status="DECODED_REQUIRES_PC_PROOF",
                    instructions=c.identity(path),
                )
            else:
                record.update(
                    outcome="DECODED_INVOCATION_OUTSIDE_FROZEN_PROOF_CONTRACT", instructions=c.identity(path)
                )
            break
    c.verify(cap["elf"])
    c.verify(tool)
    if result is not None:
        record.update(outcome="COMPLETE_BYTE_MATCH", verified_capture=result)
    elif "outcome" not in record:
        outcomes = {a["outcome"] for a in (archived if same else []) + record["attempts"]}
        record["outcome"] = (
            "NO_DECODED_INSTRUCTIONS"
            if outcomes and outcomes <= {"ALL_PC_ROWS_UNDECODED", "NO_EXECUTABLE_PC_ROWS"}
            else "NO_COMPLETE_BYTE_VERIFIED_DECODE"
        )
    return record, result


def compare_text(left, right):
    data = []
    for cap in (left, right):
        blob, table = bound_image(cap)
        sec = one((s for s in table if s["name"] == ".text"), "expected a unique .text section")
        data.append(blob[sec["offset"] : sec["offset"] + sec["bytes"]])
    a, b = data
    return dict(
        text_sha256=[hashlib.sha256(x).hexdigest() for x in data],
        bytes=[len(a), len(b)],
        different_bytes_in_common_extent=sum(x != y for x, y in zip(a, b)),
        length_difference=abs(len(a) - len(b)),
        identical_bytes=a == b,
        interpretation="BYTE_COMPARISON_ONLY",
    )


def tool_inventory(cann, bound):
    root = Path(cann).resolve(strict=True)
    paths = [
        root / d / n
        for d in (
            "bin",
            "tools/bisheng_compiler/bin",
            "aarch64-linux/ccec_compiler/bin",
            "compiler/ccec_compiler/bin",
        )
        for n in ("llvm-objdump", "ccec-objdump")
    ]
    result = {}
    for path in paths:
        if path.is_file():
            if root not in path.resolve(strict=True).parents:
                result[str(path)] = dict(aliases=[str(path)], inventory_status="EXTERNAL_ALIAS_NOT_SELECTED")
                continue
            item = disassembler(root, path)
            row = result.setdefault(item["path"], dict(file=item, aliases=[], same_as_source=item == bound))
            row["aliases"].append(str(path))
    return list(result.values())


def select_tool(cann, bound, path, expected):
    require(disassembler(cann, c.verify(bound)) == bound, "archived decoder outside selected CANN")
    if path is None:
        require(expected is None, "decoder pin supplied without an alternate decoder")
        return bound
    require(
        path.is_absolute() and re.fullmatch(r"[0-9a-f]{64}", expected or ""),
        "alternate SDK decoder requires absolute path and independent SHA256",
    )
    item = disassembler(cann, path)
    require(
        item["sha256"] == expected and os.access(item["path"], os.X_OK),
        "alternate decoder identity/execute permission mismatch",
    )
    return item


def publish_flow(out, chain, assets, refs, tool, captures, baseline, witness):
    """Re-admit already successful upstream work. Publish ATC last; never publish a static verdict."""
    require(
        set(captures) == set(DOUBLE)
        and all(
            nodes and all(cap["status"] == "DECODED_REQUIRES_PC_PROOF" for cap in nodes.values())
            for nodes in captures.values()
        ),
        "incomplete B/C capture cannot be handed to static",
    )
    for mode, nodes in captures.items():
        require(set(nodes) == set(chain["atc"]["captures"][FIRST][mode]), "missing measured variant")
        for name, cap in nodes.items():
            orig = chain["atc"]["captures"][FIRST][mode][name]
            require(
                cap["node"] == orig["node"]
                and cap["om"] == orig["om"]
                and cap["tool"] == tool
                and cap["elf"]["sha256"] == orig["elf"]["sha256"],
                "supplemental capture substitution",
            )
            blob, table = bound_image(cap)
            archived_attempts(cap, blob, table)
            require(
                decode(blob, table, c.verify(cap["listing"]).read_text())
                == c.read(c.verify(cap["instructions"])),
                "supplemental machine listing no longer describes the OM",
            )
    old_source = chain["source"]
    binding = flow.admit(*(Path(p) for p in old_source["history_roots"]), tool["path"])
    require(
        binding == dict(old_source["binding"], tool=tool), "source re-admission changed more than the decoder"
    )
    root = out / "flow"
    root.mkdir()
    payloads = {
        "source": dict(binding=binding),
        "build": {k: chain["build"][k] for k in ("variants", "workers")},
        "atc": dict(
            models=chain["atc"]["models"], captures={FIRST: captures}, baseline_images={FIRST: baseline}
        ),
    }
    accepted, predecessors = {}, {}
    for phase in flow.requirements("static"):
        folder = root / phase
        folder.mkdir()
        old = chain[phase]
        row = dict(
            schema=1,
            phase=phase,
            root=str(root),
            folder=str(folder),
            history_roots=old["history_roots"],
            predecessors=dict(predecessors),
            sources=old["sources"],
            device_execution="NOT_RUN",
            benchmark="NOT_RUN",
            atc=old["atc"],
            production="NOT_MODIFIED",
            status=flow.passed(phase),
            error=None,
            integrity_error=None,
            replay=dict(
                action="REVALIDATED_ARCHIVED_STAGE",
                original=refs[phase],
                witness=witness,
                implementation=sources(),
                new_build=False,
                new_atc=False,
                new_npu=False,
            ),
            **payloads[phase],
        )
        # Same phase validators, complete package/OM/ELF replay and original source seal.
        require(flow.sources() == row["sources"], "original source seal changed")
        historical.verify_tree(row, {})
        flow.validate(folder, row, accepted, assets)
        require(flow.status(row) == row["status"], "replayed stage evidence/status mismatch")
        tmp = folder / "validated.json.tmp"
        c.write(tmp, row)
        dest = folder / "report.json"
        tmp.rename(dest)
        predecessors[phase] = c.identity(dest)
        accepted[phase] = row
    return dict(root=str(root), reports=predecessors, static="NOT_RUN")


def replay(out, old, decoder, decoder_sha256, prepare_flow, r):
    refs, chain, assets = load_archive(old)
    src, raw, _ = assets
    binding = chain["source"]["binding"]
    tool = select_tool(src["cann"], binding["tool"], decoder, decoder_sha256)
    prior = c.read(c.verify(src["host_source"]))["binding"]["prior"]
    env, env_binding = host.envs.for_prior(prior, None)
    require(not env["ASCEND_CUSTOM_OPP_PATH"], "decoder process must be toolkit-only")
    help_rec = run_tool(tool, ["--help"], None, out / "decoder-help.txt", env)
    helptext = c.verify(help_rec["listing"]).read_text(errors="replace")
    has_aicore = (
        help_rec["returncode"] == 0
        and re.search(r"^\s*--disassemble-aicore(?:\s|$)", helptext, re.M) is not None
    )
    r.update(
        upstream=refs,
        archived_status=dict(
            {phase: report["status"] for phase, report in chain.items()}, static="BLOCKED_STATIC_ADMISSION"
        ),
        decoder=tool,
        decoder_env=env_binding,
        decoder_help=help_rec,
        inventory=tool_inventory(src["cann"], binding["tool"]),
        variants={},
    )
    case = chain["atc"]["captures"][FIRST]
    require(
        any(cap["status"] == "BLOCKED_STATIC_ADMISSION" for mode in DOUBLE for cap in case[mode].values()),
        "all original candidate listings were already decoded; inspect the original machine-proof failure",
    )
    r["comparison"] = compare_text(*(next(iter(case[mode].values())) for mode in DOUBLE))
    copies = {mode: {} for mode in DOUBLE}
    sealed_sources = sources()
    for mode in DOUBLE:
        r["variants"][mode] = {}
        for index, (name, cap) in enumerate(case[mode].items()):
            folder = out / (mode + "-" + str(index))
            observed, new = retry_one(folder, cap, tool, env, has_aicore)
            r["variants"][mode][name] = observed
            if new is not None:
                src_id = chain["build"]["variants"][mode]["generated_sources"][
                    "op_kernel/d_flash_group_quant_linear.cpp"
                ]["file"]
                plan = chain["atc"]["models"][mode][FIRST]["plans"]["16x2560x19456"]["contract"]
                template = folder / "review-input.json"
                c.write(template, review_template(new, mode, src_id, plan, binding["sdk"]))
                observed["review_input"] = c.identity(template)
                copies[mode][name] = dict(new, review_input=observed["review_input"])
    if any(set(copies[m]) != set(case[m]) for m in DOUBLE):
        outcomes = {row["outcome"] for group in r["variants"].values() for row in group.values()}
        if outcomes == {"NO_DECODED_INSTRUCTIONS"}:
            r["reason_code"] = "SDK_OFFSETS_ONLY"
        elif "DECODED_INVOCATION_OUTSIDE_FROZEN_PROOF_CONTRACT" in outcomes:
            r["reason_code"] = "NEEDS_REVIEW_OF_ACTUAL_DECODER_INVOCATION"
        else:
            r["reason_code"] = "DECODER_EVIDENCE_INCOMPLETE"
        return
    if not prepare_flow:
        r.update(status="W16_DECODER_COVERAGE_VERIFIED_PROOF_PENDING", reason_code=None)
        return
    # The single-W baseline has independent historical acceptance. A fresh binding
    # for its capture is still required by the original ATC inventory validator.
    baseline = {}
    for index, (name, cap) in enumerate(chain["atc"]["baseline_images"][FIRST].items()):
        folder = out / (SINGLE + "-" + str(index))
        observed, new = retry_one(folder, cap, tool, env, False)
        if new is None:
            # Its already accepted single-W execution has no new overlap to prove;
            # preserve an incomplete control capture exactly as the frozen ATC gate does.
            if tool == cap["tool"]:
                new = cap
            else:
                last = observed["attempts"][-1]
                new = dict(
                    om=cap["om"],
                    node=cap["node"],
                    elf=c.identity(folder / "selected.elf"),
                    sections=cap["sections"],
                    tool=tool,
                    attempts=observed["attempts"],
                    **{k: last[k] for k in ("argv", "returncode", "listing")},
                    status="BLOCKED_STATIC_ADMISSION",
                    reason=observed["outcome"],
                )
        baseline[name] = new
    witness = out / "decoder-binding.json"
    c.write(
        witness,
        dict(
            scope="OFFLINE_REPLAY_ONLY",
            archive=refs,
            implementation=sealed_sources,
            original_tool=binding["tool"],
            decoder=tool,
            source_selection="SUPPLEMENTAL_DECODER_ONLY",
            tool_help=help_rec,
            decoder_env=env_binding,
            variants=r["variants"],
            text_comparison=r["comparison"],
            baseline=baseline,
            captures=copies,
            atc="NOT_RUN",
            device_execution="NOT_RUN",
        ),
    )
    require(
        sources() == sealed_sources and flow.sources() == chain["source"]["sources"],
        "source/decoder replay drift",
    )
    historical.verify_tree(c.read(witness), {})
    r["flow"] = publish_flow(out, chain, assets, refs, tool, copies, baseline, c.identity(witness))
    r.update(status="W16_DECODER_COVERAGE_VERIFIED_PROOF_PENDING", reason_code=None)


def main(args):
    old = args.root.resolve(strict=True)
    parent = (args.output_parent or old.parent).resolve(strict=True)
    # Only a fresh sibling/temp directory may be written; protect all historical
    # roots, including the Gate8 baseline, profile, source checkout and toolkit.
    static = c.read(old / "static/report.json")
    protected = [old, c.REPO, Path("/usr/local/Ascend"), *(Path(p) for p in static["history_roots"])]
    require(
        parent.is_dir() and all(parent != p.resolve() and p.resolve() not in parent.parents for p in protected),
        "output parent overlaps read-only evidence, checkout or CANN",
    )
    out = Path(tempfile.mkdtemp(prefix="gate-w16-decoder-", dir=parent))
    r = dict(
        status="BLOCKED_STATIC_ADMISSION",
        reason_code=None,
        error=None,
        root=str(old),
        output=str(out),
        sources=sources(),
        build="NOT_RUN",
        atc="NOT_RUN",
        formal_static="NOT_RUN",
        device_execution="NOT_RUN",
        benchmark="NOT_RUN",
    )
    try:
        c.clean()
        replay(out, old, args.decoder, args.decoder_sha256, args.prepare_flow, r)
        require(sources() == r["sources"], "changed replay implementation")
        historical.verify_tree(r, {})
        c.clean()
    except (
        OSError,
        ValueError,
        KeyError,
        IndexError,
        TypeError,
        RuntimeError,
        struct.error,
        OverflowError,
        subprocess.SubprocessError,
    ) as exc:
        r.update(status="BLOCKED_STATIC_ADMISSION", reason_code="REPLAY_VALIDATION_FAILED", error=repr(exc))
        if (out / "flow").is_dir():
            # Preserve partial evidence at a path that cannot be used as a W16 predecessor.
            (out / "flow").rename(out / "incomplete-flow")
            r.pop("flow", None)
    c.write(out / "report.json", r)
    print(r["status"], r["reason_code"], r["error"] or "")
    if "decoder" in r:
        print("DECODER", r["decoder"]["path"], r["decoder"]["sha256"])
        for item in r["inventory"]:
            if item.get("same_as_source") is False:
                print("OTHER SDK TOOL (independent pin required)", item["file"]["path"], item["file"]["sha256"])
    for mode, nodes in r.get("variants", {}).items():
        for name, rec in nodes.items():
            print(mode, name, rec["outcome"], rec["archived_elf"]["sha256"])
            for attempt in rec["archived_attempts"] + rec["attempts"]:
                print(
                    " ",
                    attempt["argv"][1:-1],
                    "rc",
                    attempt["returncode"],
                    "PC rows",
                    attempt["pc_rows"],
                    "placeholders",
                    attempt["placeholder_rows"],
                    "verified bytes",
                    attempt["verified_bytes"],
                )
    if "comparison" in r:
        print("TEXT", json.dumps(r["comparison"], separators=(",", ":")))
    if r["status"] == "W16_DECODER_COVERAGE_VERIFIED_PROOF_PENDING" and "flow" in r:
        print("REVALIDATED FLOW", r["flow"]["root"], "NEXT: manual static with per-ELF machine proofs")
    print("REPORT", out / "report.json")
    return int(r["status"] == "BLOCKED_STATIC_ADMISSION")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT, help="read-only original W16 experiment root"
    )
    parser.add_argument(
        "--output-parent",
        type=Path,
        help="parent for a new uniquely named output; default: original root's parent",
    )
    parser.add_argument(
        "--decoder", type=Path, help="optional vendor-provided compatible objdump in the selected CANN"
    )
    parser.add_argument(
        "--decoder-sha256", help="independently obtained expected SHA256 of the optional decoder"
    )
    parser.add_argument(
        "--prepare-flow",
        action="store_true",
        help="only after complete canonical B/C decoding, re-admit frozen source/build/ATC for the original manual static gate",
    )
    raise SystemExit(main(parser.parse_args()))
