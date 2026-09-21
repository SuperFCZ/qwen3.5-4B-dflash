"""Finite static OM alternatives for immutable, offline-NZ Draft weights."""
from __future__ import annotations

from .utils import contained_path, file_record
from .atc_fusion import normalized_atc_options

DYNAMIC_POLICY = "single_draft16_64_gears"
STATIC_POLICY = "static_draft16_64_oms"
ROWS = (16, 64)


def uses_static_oms(graph):
    return graph.get("metadata", {}).get("draft_compile_policy") == STATIC_POLICY


def input_shape_arg(graph, rows):
    tensors = graph["metadata"]["tensor_abi"]["inputs"]
    abi = graph["runtime_input_abi"]
    names = ([t["name"] for t in tensors]
             if abi["status"] == "NOT_APPLICABLE_EXPLICIT_TEST_DOUBLE"
             else [b["data_node_name"] for b in abi.get("bindings", [])])
    if len(names) != len(tensors):
        raise ValueError("AIR input bindings differ from tensor ABI")
    axes = graph["metadata"].get("dynamic_input_axes", {})
    shapes = []
    for tensor, name in zip(tensors, names):
        if any(char in name for char in ";:\n\r"):
            raise ValueError("AIR Data node name is not safe for ATC input_shape")
        shape = list(tensor["shape"])
        for axis in axes.get(tensor["name"], ()):
            shape[axis] = rows
        shapes.append(name + ":" + ",".join(map(str, shape)))
    return ";".join(shapes)


def om_records(graph):
    """Validate inventory before trusting either compiled gear or its files."""
    gears = graph.get("static_gear_oms")
    if not uses_static_oms(graph):
        if gears is not None:
            raise ValueError("static Draft OMs require an explicit compile policy")
        return [graph["om"]]
    if (not isinstance(gears, list) or len(gears) != 2
            or any(not isinstance(g, dict) or type(g.get("rows")) is not int
                   or not isinstance(g.get("om"), dict) for g in gears)
            or [g["rows"] for g in gears] != list(ROWS)
            or gears[0].get("om") != graph.get("om")
            or gears[0].get("atc_command") != graph.get("atc_command")
            or gears[0].get("atc_log") != graph.get("atc_log")
            or gears[0]["om"].get("path") == gears[1]["om"].get("path")):
        raise ValueError("offline NZ deployment requires distinct static M16 and M64 OMs")
    normalized = []
    for gear in gears:
        command = gear.get("atc_command")
        if (not isinstance(command, list) or not command or
                not all(isinstance(arg, str) for arg in command)):
            raise ValueError("static Draft artifact has no ATC command")
        if (sum(arg.startswith("--input_format=") for arg in command) != 1 or
                sum(arg.startswith("--input_shape=") for arg in command) != 1 or
                command.count("--input_format=ND") != 1 or
                command.count("--input_shape=" + input_shape_arg(graph, gear["rows"])) != 1 or
                any(arg.startswith(("--dynamic_", "--input_shape_range")) for arg in command)):
            raise ValueError("static Draft ATC shapes differ from its declared gear")
        normalized.append(normalized_atc_options([arg for arg in command
                          if not arg.startswith(("--input_shape=", "--output="))]))
    if normalized[0] != normalized[1]:
        raise ValueError("static Draft gears use different ATC precision/build settings")
    return [g["om"] for g in gears]


def verify_om_files(graph, root):
    for record in om_records(graph):
        if file_record(contained_path(root, record["path"]), relative_to=root) != record:
            raise ValueError(f"OM integrity check failed: {graph['name']} {record['path']}")


def om_hashes(graphs):
    result = {}
    for graph in graphs:
        records = om_records(graph)
        result[graph["name"]] = records[0]["sha256"]
        if len(records) == 2:
            result[graph["name"] + "_static64"] = records[1]["sha256"]
    return result
