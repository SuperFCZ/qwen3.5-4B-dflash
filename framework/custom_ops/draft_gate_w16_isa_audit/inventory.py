"""Read-only tool discovery. Candidate names/registered targets alone never qualify a decoder."""

import hashlib
import os
from pathlib import Path
import re
import struct


def identity(path):
    real = Path(path).resolve(strict=True)
    sha = hashlib.sha256()
    with real.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    return dict(path=str(real), bytes=real.stat().st_size, sha256=sha.hexdigest())


def file_type(path):
    with path.open("rb") as stream:
        head = stream.read(96)
    if len(head) >= 20 and head[:4] == b"\x7fELF" and head[5] in (1, 2):
        machine = struct.unpack_from("<H" if head[5] == 1 else ">H", head, 18)[0]
        return f"ELF{ {1: 32, 2: 64}.get(head[4], '?') } machine=0x{machine:04x}"
    if head.startswith(b"#!"):
        return head.splitlines()[0].decode(errors="replace")
    return "FILE"


def role(name, internal=False):
    n = name.lower()
    if n in ("msobjdump", "show_kernel_debug_data", "asc_dumper"):
        return "metadata_or_debug_dump"
    if re.fullmatch(r"(?:[\w.-]+-)?objdump(?:-[0-9.]+)?", n):
        return "objdump"
    if re.fullmatch(r"llvm-mc(?:-[0-9.]+)?", n):
        return "llvm_mc"
    if n in ("bisheng", "ccec") or re.fullmatch(r"(?:clang|clang\+\+|llc)(?:-[0-9.]+)?", n):
        return "compiler"
    if n in ("cce-ld", "ld.lld", "lld", "lld-link"):
        return "linker"
    if re.fullmatch(
        r"(?:llvm-|[\w.-]+-)?(?:readelf|readobj|objcopy|symbolizer|addr2line|strip|link|ar)(?:-[0-9.]+)?",
        n,
    ):
        return "elf_or_link_utility"
    if re.fullmatch(r"(?:msdebug|lldb|gdb)(?:-[0-9.]+)?", n):
        return "debugger"
    if re.search(
        r"(?:disassem|disasm|(?:dav[_-]?m?200|hiipu|aicore|instr|isa)[_-]?(?:decoder?|dump))",
        n,
    ):
        return "possible_isa_tool"
    if re.fullmatch(
        r"lib(?:llvm(?:-[\d.]+)?|opcodes[^/]*|capstone|camodel|_pvmodel)\.so(?:\.[\w.-]+)?",
        n,
    ):
        return "library_inspect_only"
    return "compiler_internal" if internal else None


SDK_READERS = {
    "python/site-packages/msopgen/simulator/parse_objdump.py": (
        "objdump client",
        ("llvm-objdump",),
    ),
    "python/site-packages/msopgen/simulator/instr_dump_parser.py": (
        "simulator trace parser",
        ("class ",),
    ),
    "python/site-packages/msopgen/simulator/instr_dump_record.py": (
        "existing trace records",
        ("class ",),
    ),
    "python/site-packages/msobjdump/msobjdump_main.py": (
        "offload ELF metadata/extraction",
        ("--dump-elf", "--list-elf", "--extract-elf"),
    ),
    "python/site-packages/show_kernel_debug_data/dump_parser.py": (
        "debug data parser",
        ("class ",),
    ),
    "tools/simulator/include/model_v200/pv_model/inc/tik_pv_wrapper.h": (
        "dynamic PV model public API",
        ("pv_step(", "pv_get_mem_access("),
    ),
}
PROJECT_READERS = {
    "draft_quant/simulator/bottleneck_elf.py": (
        "ELF sections and symbolization",
        ("def elf_layout", "llvm_frames"),
    ),
    "draft_quant/simulator/bottleneck_parse.py": ("simulator trace reader", ("def ",)),
    "draft_w8a8_output_dequant/stage4g/inspect_binary.py": (
        "invokes external disassembler",
        ("def disassemble", "--disassemble"),
    ),
    "draft_gate_w16_atc8/w16_static.py": (
        "byte/listing and proof verifier",
        ("def decode(", "--disassemble"),
    ),
    "draft_gate_w16_atc8/w16_cfg.py": (
        "CFG verifier using decoded instructions",
        ("def reachable_pcs",),
    ),
    "draft_gate_w16_atc8/w16_operands.py": (
        "decoded operand verifier",
        ("def consumed_address",),
    ),
    "draft_gate_w16_decoder_retry/retry_decoder.py": (
        "SDK decoder replay",
        ("def listing_evidence", "def retry_one"),
    ),
}


def reader_evidence(base, names):
    result = {}
    for rel, (purpose, markers) in names.items():
        path = base / rel
        if not path.is_file():
            result[rel] = {"status": "ABSENT"}
            continue
        lines = path.read_text(errors="replace").splitlines()
        result[rel] = dict(
            file=identity(path),
            purpose=purpose,
            evidence={m: [n for n, line in enumerate(lines, 1) if m in line][:4] for m in markers},
        )
    return result


def project_candidates(ops, errors):
    # Look for actual APIs/call sites, including local project tools absent from Git.
    # Run no project file. Generated packages, run data and test doubles are excluded.
    hits = []
    pattern = re.compile(rb"LLVMDisasmInstruction|MCDisassembler|llvm-objdump|ccec-objdump|capstone\.")
    for parent, dirs, files in os.walk(ops, onerror=lambda e: errors.append(str(e))):
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".")
            and d
            not in (
                "__pycache__",
                "test",
                "tests",
                "build",
                "build_out",
                "node_modules",
                "dist",
            )
        ]
        for name in files:
            p = Path(parent) / name
            if p.is_symlink() or not p.is_file():
                continue
            try:
                evidence = []
                if (
                    p.suffix in (".py", ".sh", ".c", ".cc", ".cpp", ".h", ".rs")
                    and p.stat().st_size <= 2 * 1024 * 1024
                ):
                    evidence = [
                        (i, m[0].decode())
                        for i, line in enumerate(p.read_bytes().splitlines(), 1)
                        for m in [pattern.search(line)]
                        if m
                    ][:8]
                if evidence or role(name) in (
                    "objdump",
                    "possible_isa_tool",
                    "llvm_mc",
                ):
                    hits.append(dict(file=identity(p), matches=evidence))
            except OSError as exc:
                errors.append(str(exc))
    return hits


def discover(cann, repo, path_value, extra_roots=()):
    cann = Path(cann).resolve(strict=True)
    tools, errors = {}, []

    def record(alias, scope, kind):
        try:
            real = alias.resolve(strict=True)
            if not real.is_file():
                return
            row = tools.get(str(real))
            if row is None:
                row = dict(
                    file=identity(real),
                    format=file_type(real),
                    roles=[],
                    scopes=[],
                    aliases=[],
                )
                tools[str(real)] = row
            for field, value in (
                ("roles", kind),
                ("scopes", scope),
                ("aliases", str(alias)),
            ):
                if value not in row[field]:
                    row[field].append(value)
        except (OSError, RuntimeError) as exc:
            errors.append(f"{alias}: {exc}")

    def scan(path, recursive, scope):
        if not path.is_dir():
            return

        def failed(err):
            errors.append(str(err))

        for parent, dirs, files in os.walk(path, followlinks=False, onerror=failed):
            dirs[:] = [d for d in dirs if recursive and d not in (".git", "__pycache__")]
            folder = Path(parent)
            internal = (
                scope == "CANN"
                and folder.name in ("bin", "libexec")
                and any(d in ("bisheng_compiler", "ccec_compiler", "compiler") for d in folder.parts)
            )
            for name in files:
                kind = role(name, internal)
                if name in ("libcamodel.so", "lib_pvmodel.so") and folder.resolve() not in sim_dirs:
                    continue
                if kind and not name.endswith((".pyc", ".whl")):
                    record(folder / name, scope, kind)

    roots = [
        (cann, True, "CANN"),
        *((Path(p).resolve(strict=True), True, "EXTRA") for p in extra_roots),
    ]
    # Enumerate actual install, then known alias directories and visible host tools.
    # Never recurse through an arbitrary user PATH entry.
    sim_dirs = {(cann / "tools/simulator" / s / "lib").resolve() for s in ("Ascend310P3", "dav_2002")}
    for p in path_value.split(os.pathsep):
        if p and Path(p).is_absolute():
            roots.append((Path(p), False, "PATH"))
    for rel in (
        "bin",
        "tools/ccec_compiler/bin",
        "aarch64-linux/ccec_compiler/bin",
        "tools/bisheng_compiler/bin",
    ):
        roots.append((cann / rel, False, "CANN_ALIAS"))
    for p in (
        Path("/usr/bin"),
        Path("/usr/local/bin"),
        *Path("/usr/lib").glob("llvm-*/bin"),
        *Path("/usr/lib64").glob("llvm-*/bin"),
        *Path("/opt").glob("llvm*/bin"),
    ):
        roots.append((p, False, "HOST"))
    # A public LLVM C disassembler can be installed as a library without llvm-mc.
    for p in (
        Path("/usr/lib"),
        Path("/usr/lib64"),
        Path("/usr/lib/aarch64-linux-gnu"),
        *Path("/usr/lib").glob("llvm-*/lib"),
        *Path("/usr/lib64").glob("llvm-*/lib"),
    ):
        roots.append((p, False, "HOST_LIB"))
    for p in (
        Path("/usr/local/Ascend/driver/tools"),
        Path("/opt/mindstudio"),
        Path("/usr/local/Ascend/mindstudio"),
    ):
        if p.is_dir():
            roots.append((p, True, "ALTERNATE_INSTALL"))
    for path, recursive, scope in sorted(set(roots), key=lambda x: (str(x[0]), x[2])):
        scan(path, recursive, scope)
    ops = repo / "framework/custom_ops"
    candidates = project_candidates(ops, errors)
    return dict(
        roots=[dict(path=str(p), scope=s, recursive=r, exists=p.is_dir()) for p, r, s in roots],
        errors=errors,
        tools=sorted(tools.values(), key=lambda r: r["file"]["path"]),
        sdk_readers=reader_evidence(cann, SDK_READERS),
        project_readers=reader_evidence(ops, PROJECT_READERS),
        project_decoder_candidates=candidates,
    )
