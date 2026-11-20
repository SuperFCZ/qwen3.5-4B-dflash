"""Read mode-1 GE nodes and pair generated constants using actual consumer ports."""

from collections import Counter, defaultdict
import hashlib
import json
import re

from archive import c, require

GENERATED = re.compile(r"dynamic_const_[0-9]+_[0-9]+\Z")


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def nodes(path):
    require(path.stat().st_size <= 256 << 20, "mode-1 graph exceeds analysis bound")
    out = {}

    def visit(value):
        if isinstance(value, list):
            for row in value:
                visit(row)
        elif isinstance(value, dict):
            for row in value.get("op", []) if isinstance(value.get("op"), list) else ():
                require(
                    isinstance(row, dict) and isinstance(row.get("name"), str) and row["name"] not in out,
                    "invalid or duplicate compiled node",
                )
                out[row["name"]] = row
            for key, child in value.items():
                if key not in ("op", "data", "tensor_content"):
                    visit(child)

    visit(c.read(path))
    require(bool(out), "empty compiled graph")
    return out


def edge(value):
    if not isinstance(value, str):
        return None, None
    name, sep, port = value.rpartition(":")
    return (name, port) if sep and re.fullmatch(r"-?\d+", port) else (value, None)


def consumer_keys(table):
    consumers = defaultdict(list)
    for name, node in table.items():
        for i, value in enumerate(node.get("input", [])):
            parent, port = edge(value)
            if parent in table and GENERATED.fullmatch(parent):
                consumers[parent].append((name, i, port))
    return {name: tuple(sorted(refs)) for name, refs in consumers.items()}


def const_mapping(left, right):
    """Return candidate->control names. Ambiguous/unreferenced/unexpected nodes remain unresolved."""
    ports = [consumer_keys(x) for x in (left, right)]
    names = [{k for k in x if GENERATED.fullmatch(k)} for x in (left, right)]
    records = []
    for table, group, refs in zip((left, right), names, ports):
        buckets = defaultdict(list)
        for n in sorted(group):
            key = refs.get(n, ())
            if (
                key
                and table[n]["type"] == "Const"
                and all(k in left and k in right and not GENERATED.fullmatch(k) for k, _, _ in key)
            ):
                buckets[key].append(n)
        records.append(buckets)
    mapping = {}
    for key, l in records[0].items():
        r = records[1].get(key, [])
        if len(l) == len(r) == 1:
            mapping[r[0]] = l[0]
    missing = [sorted(names[0] - set(mapping.values())), sorted(names[1] - set(mapping))]
    return mapping, missing


def mapped_string(value, mapping):
    name, port = edge(value)
    if name not in mapping:
        return value
    return mapping[name] if port is None else mapping[name] + ":" + port


def unorder_attrs(value, key=""):
    """Diagnostic multiset view. Preserve positional arrays and every attribute entry, including duplicates."""
    if isinstance(value, dict):
        return {k: unorder_attrs(v, k) for k, v in value.items()}
    if isinstance(value, list):
        if key in ("attr", "attributes") and all(
            isinstance(r, dict) and isinstance(r.get("key"), str) for r in value
        ):
            groups = defaultdict(list)
            for row in value:
                groups[row["key"]].append(unorder_attrs(row))
            return {k: sorted(v, key=encode) for k, v in groups.items()}
        return [unorder_attrs(row) for row in value]
    return value


def for_diagnosis(value, mapping):
    if isinstance(value, dict):
        return {k: for_diagnosis(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [for_diagnosis(v, mapping) for v in value]
    return mapped_string(value, mapping) if isinstance(value, str) else value


def differences(a, b, path=""):
    if type(a) is type(b) is dict:
        for k in sorted(a.keys() | b.keys()):
            if k in a and k in b:
                yield from differences(a[k], b[k], path + "/" + k)
            else:
                yield path + "/" + k, k in a, k in b
    elif type(a) is type(b) is list and len(a) == len(b):
        for x, y in zip(a, b):
            yield from differences(x, y, path + "[]")
    elif type(a) != type(b) or a != b:
        yield path, a, b


def compare_for_diagnosis(left, right):
    mapping, missing = const_mapping(left, right)
    mapped = {mapping.get(k, k): v for k, v in right.items()}
    counts, first = Counter(), {}
    for name in sorted(left.keys() & mapped.keys()):
        a, b = left[name], mapped[name]
        for path, x, y in differences(unorder_attrs(a), unorder_attrs(for_diagnosis(b, mapping))):
            group = ("Const" if a["type"] == "Const" else "Op", path)
            counts[group] += 1
            if group not in first:
                first[group] = (name, short(x), short(y))
    return mapping, missing, counts, first


def short(v):
    s = encode(v)
    return s if len(s) <= 120 else s[:75] + "… sha256=" + hashlib.sha256(s.encode()).hexdigest()[:16]
