"""Compare CANN 9 mode-1 GE node summaries while keeping positional fields ordered."""

import json
from pathlib import Path

from w16_common import GATE, require
from c_profile_graph import read_om_json


class _AttributeEntries(dict):
    """Keep a GE entry array distinct from a JSON object, including when both are empty."""


def _short(value):
    if isinstance(value, (dict, list)):
        return f"{type(value).__name__}({len(value)} entries)"
    return json.dumps(value, ensure_ascii=False)[:240]


def _checked_vendor(value, root, node):
    require(
        root is not None and Path(root).is_absolute() and ".." not in Path(root).parts,
        f"{node}: missing or invalid bound private OPP root",
    )
    expected = Path(root) / "vendors/customize"
    require(
        value == {"s": str(expected)},
        f"{node}.attributes.custom_op_file_path: expected {expected}, got {_short(value.get('s') if isinstance(value, dict) else value)}",
    )
    require(
        expected.is_dir()
        and expected.resolve(strict=True) == Path(root).resolve(strict=True) / "vendors/customize",
        f"{node}.attributes.custom_op_file_path: missing or redirected private vendor {expected}",
    )
    return {"s": "vendors/customize"}


def _descriptor_pair(rows, key, path, op_type):
    """Only reverse a conflicting same-key pair in the descriptor cases observed on the 310P3."""
    if len(rows) != 2:
        return rows
    direct = len(path) == 3 and path[0] in ("input_desc", "output_desc") and path[-1] == "attr"
    nested = path[-2:] in (("desc", "attr"), ("td", "attr")) or (
        path[-3:-2] == ("td",) and isinstance(path[-2], int) and path[-1] == "attr"
    )
    if key == "origin_shape_initialized" and (direct or nested):
        field, expected = "b", (False, True)
    elif key == "format_for_int" and op_type == GATE and path == ("input_desc", 1, "attr"):
        field, expected = "i", (29, 100663325)
    else:
        return rows
    # An unexpected type, third entry or extra field stays subject to ordered comparison.
    if all(
        set(row) == {"key", "value"}
        and isinstance(row["value"], dict)
        and set(row["value"]) == {field}
        and type(row["value"][field]) is type(expected[0])
        for row in rows
    ):
        ordered = sorted(rows, key=lambda row: row["value"][field])
        if tuple(row["value"][field] for row in ordered) == expected:
            return ordered
    return rows


_KERNEL_FIELDS = {"_kernelname": "s", "_kernel_bin_id": "s", "_tbeKernelSize": "i"}


def _node_view(node, gates, opp_root):
    kind, name = node["type"], node["name"]
    target = name in gates
    require(not target or kind == GATE, f"{name}: expected {GATE}, got {kind}")
    require(
        kind != GATE or isinstance(node["attributes"], list), f"{name}: expected GE operator attribute entries"
    )

    def visit(value, path):
        if isinstance(value, dict):
            return {k: visit(v, path + (k,)) for k, v in value.items()}
        if not isinstance(value, list):
            return value
        top = path == ("attributes",)
        attr_list = top or (path[-1:] == ("attr",) and path[0] in ("attributes", "input_desc", "output_desc"))
        if not attr_list:
            return [visit(v, path + (i,)) for i, v in enumerate(value)]
        groups = _AttributeEntries()
        for row in value:
            require(
                isinstance(row, dict) and isinstance(row.get("key"), str) and row["key"],
                f"{name}.{path}: malformed GE attribute entry",
            )
            key = row["key"]
            item = visit(row, path + (key,))
            if top and kind == GATE and key == "custom_op_file_path":
                require(key not in groups, f"{name}: duplicate custom_op_file_path")
                item["value"] = _checked_vendor(item.get("value"), opp_root, name)
            if top and target and key in _KERNEL_FIELDS:
                # The graph-to-ELF verifier checks the actual ID, symbol, size and bytes.
                # Allow only their scalar payload to differ; the key, envelope and dtype stay bound.
                field = _KERNEL_FIELDS[key]
                payload = item.get("value")
                require(
                    key not in groups and isinstance(payload, dict) and field in payload,
                    f"{name}: missing or duplicate kernel attribute {key}",
                )
                val = payload[field]
                require(
                    (field == "s" and type(val) is str and val)
                    or (field == "i" and (type(val) is int or (type(val) is str and val.isdecimal()))),
                    f"{name}: invalid kernel attribute {key}",
                )
                item["value"] = {**payload, field: "" if type(val) is str else 0}
            groups.setdefault(key, []).append(item)
        for key, rows in groups.items():
            groups[key] = _descriptor_pair(rows, key, path, kind)
        return groups

    return visit(node, ())


def _require_equal(old, new, path):
    if type(old) is not type(new):
        raise ValueError(f"{path}: type {type(old).__name__} -> {type(new).__name__}")
    if isinstance(old, dict):
        if old.keys() != new.keys():
            missing, added = sorted(old.keys() - new.keys()), sorted(new.keys() - old.keys())
            raise ValueError(f"{path}: missing {missing[:6]}, added {added[:6]}")
        for key in old:
            _require_equal(old[key], new[key], f"{path}.{key}")
    elif isinstance(old, list):
        require(len(old) == len(new), f"{path}: length {len(old)} -> {len(new)}")
        for i, (a, b) in enumerate(zip(old, new)):
            _require_equal(a, b, f"{path}[{i}]")
    elif old != new:
        raise ValueError(f"{path}: {_short(old)} -> {_short(new)}")


def protected_graph(raw_path, new_path, gates, *, raw_opp_root, new_opp_root):
    """Compare two sealed graph dumps against their respective independently admitted OPPs."""

    def inventory(path):
        nodes = read_om_json(path)
        by_name = {node["name"]: node for node in nodes}
        require(len(by_name) == len(nodes), "duplicate compiled graph node")
        return by_name

    old, new = inventory(raw_path), inventory(new_path)
    require(old.keys() == new.keys() and set(gates) <= old.keys(), "compiled graph node inventory changed")
    for name, left in old.items():
        right = new[name]
        a = _node_view(left, gates, raw_opp_root)
        b = _node_view(right, gates, new_opp_root)
        _require_equal(a, b, f"{name} ({left['type']})")
    return len(old) - len(gates)
