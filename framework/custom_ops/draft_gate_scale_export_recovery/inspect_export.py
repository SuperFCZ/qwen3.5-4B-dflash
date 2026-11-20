#!/usr/bin/env python3
"""Bounded, read-only diagnostic of the existing C16 full export. No admission or device execution."""

import argparse
from collections import Counter
from pathlib import Path
import hashlib
import struct

from archive import ARCHIVE, archived, c, require
from graph_data import nodes, compare_for_diagnosis, consumer_keys, short
from w16_export import native_images
from w16_static import sections
from om_constants import weight_partition


def part_data(model):
    p = Path(model["om"]["path"])
    weights = weight_partition(p)
    with p.open("rb") as f:
        f.seek(256)
        (count,) = struct.unpack("<Q", f.read(8))
        table = list(struct.iter_unpack("<QQQ", f.read(24 * count)))
        base = f.tell()
        result = {}
        for typ, off, size in table:
            if typ == 1:
                result[typ] = (size, weights["sha256"])
                continue
            f.seek(base + off)
            h = hashlib.sha256()
            remaining = size
            while remaining:
                data = f.read(min(8 << 20, remaining))
                require(bool(data), "truncated partition")
                h.update(data)
                remaining -= len(data)
            result[typ] = (size, h.hexdigest())
    return result


def blob(stream, record):
    stream.seek(record["offset"])
    data = stream.read(record["bytes"])
    require(
        len(data) == record["bytes"] and hashlib.sha256(data).hexdigest() == record["sha256"],
        "wrong embedded ELF",
    )
    return data


def sections_by_name(data):
    all_ = sections(data)
    require(len({s["name"] for s in all_}) == len(all_), "duplicate ELF section")
    return {s["name"]: s for s in all_}


def native_delta(a, b):
    la, lb = (native_images(m) for m in (a, b))
    print("NATIVE INVENTORY", len(la), len(lb), "unpaired", sorted(la.keys() ^ lb.keys())[:10])
    counts, examples = Counter(), {}
    with open(a["om"]["path"], "rb") as fa, open(b["om"]["path"], "rb") as fb:
        for name in sorted(la.keys() & lb.keys()):
            x, y = la[name], lb[name]
            if x["sha256"] == y["sha256"]:
                continue
            l, r = blob(fa, x), blob(fb, y)
            ls, rs = sections_by_name(l), sections_by_name(r)
            delta = tuple(
                k
                for k in sorted(ls.keys() | rs.keys())
                if ls.get(k, {}).get("sha256") != rs.get(k, {}).get("sha256")
            )
            key = (x["executable_sections"] == y["executable_sections"], delta)
            counts[key] += 1
            if key not in examples:
                changes = []
                for k in (".strtab", ".comment"):
                    if k not in delta or k not in ls or k not in rs:
                        continue
                    strings = [
                        set(v[t[k]["offset"] : t[k]["offset"] + t[k]["bytes"]].split(b"\0"))
                        for v, t in ((l, ls), (r, rs))
                    ]
                    changes.append(
                        (
                            k,
                            [
                                [z[:130].decode(errors="replace") for z in sorted(d)[:4]]
                                for d in (strings[0] - strings[1], strings[1] - strings[0])
                            ],
                        )
                    )
                examples[key] = (name, x["symbol"], changes)
    for k, num in counts.items():
        print("NATIVE GROUP (text_equal,sections)", k, "nodes", num, examples[k])


def main(archive):
    _, _, a, b = archived(archive)
    for label, model in (("Raw", a), ("Transpose", b)):
        print("PARTITIONS", label, part_data(model))
    l, r = (nodes(c.verify(m["dump"])) for m in (a, b))
    aliases, miss, count, first = compare_for_diagnosis(l, r)
    print(
        "GRAPH",
        len(l),
        len(r),
        "direct unpaired",
        len(l.keys() - r.keys()),
        len(r.keys() - l.keys()),
        "dynamic pairs by consumer ports",
        len(aliases),
        "unpaired generated",
        list(map(len, miss)),
    )
    for t, m, side in ((l, miss[0], "Raw"), (r, miss[1], "Transpose")):
        ports = consumer_keys(t)
        for n in m[:3]:
            print("UNPAIRED", side, n, t[n].get("type"), short(ports.get(n, ())), list(t[n].keys())[:25])
    mapped = {aliases.get(k, k) for k in r}
    print("UNPAIRED AFTER BINDING", sorted(l.keys() - mapped)[:8], sorted(mapped - l.keys())[:8])
    print("GENERATED EXAMPLES", list(aliases.items())[:3])
    for k, num in count.most_common(64):
        print("DELTA", k, num, *first[k])
    print("DISTINCT DELTA CLASSES", len(count))
    native_delta(a, b)
    print("DIAGNOSTIC_ONLY ATC=NOT_RUN DEVICE=NOT_RUN")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--archive", type=Path, default=ARCHIVE)
    main(p.parse_args().archive)
