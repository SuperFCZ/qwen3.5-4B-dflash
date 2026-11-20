"""Check the real, 64-bit CANN 9 IMOD partition table used by the two frozen Draft OMs."""

import hashlib
import json
import os
import struct
from pathlib import Path

from evidence import require

KINDS = (8, 0, 1, 3, 2)  # Observed order. Type 1 is WEIGHTS_DATA.


def weight_partition(path):
    path = Path(path)
    with path.open("rb") as f:
        before = os.fstat(f.fileno())
        header = f.read(256)
        require(
            len(header) == 256 and header[:12] == b"IMOD" + struct.pack("<II", 256, 0x20000000),
            "unsupported OM header or version",
        )
        count = f.read(8)
        require(count == struct.pack("<Q", len(KINDS)), "unsupported OM partition count/layout")
        end = 0
        sections = {}
        for expected in KINDS:
            entry = f.read(24)
            require(len(entry) == 24, "truncated OM partition table")
            kind, pos, size = struct.unpack("<QQQ", entry)
            require(kind == expected and pos == end and size > 0, "OM partition order/extent mismatch")
            sections[kind] = (pos, size)
            end += size
        start = f.tell()
        require(start + end == before.st_size, "overlap, truncation or trailing bytes in OM")
        off, length = sections[1]
        f.seek(start + off)
        remaining = length
        digest = hashlib.sha256()
        while remaining:
            block = f.read(min(8 << 20, remaining))
            require(bool(block), "truncated model weights")
            digest.update(block)
            remaining -= len(block)
        after = os.fstat(f.fileno())
        require(
            (before.st_size, before.st_mtime_ns, before.st_ino)
            == (after.st_size, after.st_mtime_ns, after.st_ino),
            "OM changed while reading weights",
        )
    return dict(partition_type=1, absolute_offset=start + off, bytes=length, sha256=digest.hexdigest())


def inline_constants(path):
    """Keep Const payloads the common graph summarizer intentionally elides, plus external weight references."""
    path = Path(path)
    require(path.stat().st_size <= 256 << 20, "unbounded mode-1 graph dump")
    nodes = {}

    def normalized_const(value, key=""):
        if isinstance(value, dict):
            return {k: normalized_const(v, k) for k, v in value.items()}
        if isinstance(value, list):
            entries = [normalized_const(v) for v in value]
            if key in ("attr", "attributes") and all(isinstance(v, dict) and "key" in v for v in entries):
                # Preserve every GE entry and its multiplicity. Tensor axes, ports and data stay ordered.
                entries.sort(key=lambda v: json.dumps(v, sort_keys=True, separators=(",", ":")))
            return entries
        return value

    def visit(value):
        if isinstance(value, list):
            for row in value:
                visit(row)
        if isinstance(value, dict):
            ops = value.get("op", [])
            for node in ops if isinstance(ops, list) else ():
                if isinstance(node, dict) and node.get("type") in ("Const", "Constant"):
                    name = node["name"]
                    require(name not in nodes, "duplicate Const in OM graph: " + name)
                    encoded = json.dumps(
                        normalized_const(node), sort_keys=True, separators=(",", ":")
                    ).encode()
                    nodes[name] = hashlib.sha256(encoded).hexdigest()
            for k, child in value.items():
                if k not in ("op", "data", "tensor_content"):
                    visit(child)

    visit(json.loads(path.read_bytes()))
    require(bool(nodes), "no Const inventory in mode-1 OM graph")
    return nodes


def compare_constants(left, right):
    a, b = weight_partition(left["om"]["path"]), weight_partition(right["om"]["path"])
    require((a["bytes"], a["sha256"]) == (b["bytes"], b["sha256"]), "compiled OM weight bytes differ")
    nodes = inline_constants(left["dump"]["path"])
    require(nodes == inline_constants(right["dump"]["path"]), "inline constant or weight-reference drift")
    return dict(raw_partition=a, transpose_partition=b, const_nodes=nodes)
