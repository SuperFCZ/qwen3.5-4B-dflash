"""Place the six measured device sources in a single private msopgen project for each gear."""

from pathlib import Path
import re

from selection import OPS, c, require, selections
from package import generate, write_preset, build_install

INCLUDE = re.compile(rb'^\s*#\s*include\s*"([^"]+)"', re.M)


def local_headers(source):
    """Follow the project's quoted headers. SDK paths are supplied by the same CANN install."""
    result = {}

    def walk(file):
        for token in INCLUDE.findall(file.read_bytes()):
            name = token.decode()
            other = file.parent / name
            if not other.is_file():
                require("/" in name or name == "kernel_operator.h", "missing local header: " + str(other))
                continue
            require(Path(name).name == name, "unexpected nested private include: " + name)
            if name not in result:
                result[name] = other
                walk(other)

    walk(source)
    return result


def build(folder, cann, env, gear, assets):
    specs = selections(gear)
    (row,) = c.read(OPS / "draft_gate_tile_nd/DFlashGateTileNd.json")
    schema = folder / "operators.json"
    c.write(schema, [dict(row, op=s.op) for s in specs.values()])
    proj = generate(folder, schema, env, operators=tuple(s.op for s in specs.values()))
    seen = {}
    devices = {}
    inputs = {}

    def put(folder, name, content):
        key = (folder, name)
        if key in seen:
            require(seen[key] == content, "conflicting joint header: " + name)
        else:
            (folder / name).write_bytes(content)
            seen[key] = content

    for kind, spec in specs.items():
        source = assets[kind]
        inputs[kind] = {}
        for role, sub in (("host", "op_host"), ("device", "op_kernel")):
            file = c.verify(source[role])
            require(file.name == spec.stem + ".cpp", "wrong source for " + kind)
            dest = proj / sub
            src = file.read_bytes()
            headers = local_headers(file)
            inputs[kind][role] = dict(
                source=source[role], headers={k: c.identity(v) for k, v in headers.items()}
            )
            for name, path in headers.items():
                # Wide KV and packed FC emit different column widths in the scheduling receipt.
                # Bind the archived implementation per translation unit to avoid C++ inline symbol collisions.
                target = (
                    spec.stem + "_scheduler.h" if role == "host" and name == "scheduler_evidence.h" else name
                )
                put(dest, target, path.read_bytes())
            if role == "host":
                old = b'#include "scheduler_evidence.h"'
                require(src.count(old) == 1, "missing measured scheduler for " + kind)
                new = f'#include "{spec.stem}_scheduler.h"'.encode()
                src = f"#define dflash_scheduler {spec.stem}_scheduler\n".encode() + src.replace(old, new)
            else:
                devices[kind] = (file, headers)
            (dest / file.name).write_bytes(src)
    preset = proj / "CMakePresets.json"
    write_preset(preset, preset, cann)
    root = build_install(folder, proj, cann, env)
    vendor = root / "vendors/customize"
    files = [
        dict(path=str(p), sha256=c.sha(p))
        for p in sorted(vendor.rglob("*"))
        if p.is_file() and (p.suffix in (".h", ".json", ".so") or ".so." in p.name)
    ]
    binaries = []
    for kind, spec in specs.items():
        original, headers = devices[kind]
        (dyn,) = vendor.rglob(spec.stem + ".cpp")
        require(
            c.sha(dyn) == c.sha(original)
            and all(c.sha(dyn.parent / name) == c.sha(p) for name, p in headers.items()),
            "private dynamic device source differs from the measured implementation: " + kind,
        )
        (elf,) = (vendor / "op_impl/ai_core/tbe/kernel/ascend310p" / spec.stem).glob("*.o")
        meta = elf.with_suffix(".json")
        binary = c.identity(elf)
        require(c.read(meta)["sha256"] == binary["sha256"], "metadata does not bind " + spec.op)
        binaries.append(
            dict(
                op=spec.op,
                binary=str(elf),
                sha256=binary["sha256"],
                bytes=binary["bytes"],
                json=str(meta),
                json_sha256=c.sha(meta),
                json_bytes=meta.stat().st_size,
            )
        )
    opp = dict(
        kind=f"PRIVATE_ALL_W8_C{gear}",
        install=str(root),
        preflight=dict(vendor=str(vendor), files=files, kernels=binaries),
    )
    c.verify_opp(opp)
    return dict(opp=opp, inputs=inputs)
