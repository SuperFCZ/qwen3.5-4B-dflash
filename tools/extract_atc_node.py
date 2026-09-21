#!/usr/bin/env python3
"""Print a bounded shape-only report from existing ATC dumps; never invoke ATC.

Only the named node and the tensor ports on its immediate edges are exported.
Tensor payloads, arbitrary attributes, debug logs and full graphs stay local.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "framework/python"), str(REPO)]

from qwen35_dflash.ascend310p.atc_diagnostics import describe_tensor


def _tensor(desc):
    result = describe_tensor(desc)
    result.pop("name", None)
    return result


def _ports(descs, indices):
    return {str(i): _tensor(descs[i]) if i < len(descs) else {"status": "MISSING"}
            for i in indices if i >= 0}


def _node(node, inputs=(), outputs=(), *, constant=False):
    result = {"name": node.name, "type": node.type,
              "inputs": _ports(node.input_desc, inputs),
              "outputs": _ports(node.output_desc, outputs)}
    for key in ("_out_shape_locked", "transpose_x", "transpose_weight"):
        if key in node.attr:
            result[key] = node.attr[key].b
    for key in ("_parent_node_index", "index", "antiquant_group_size"):
        if key in node.attr:
            result[key] = node.attr[key].i
    for key in ("src_format", "dst_format"):
        if key in node.attr:
            result[key] = node.attr[key].s.decode("utf-8", errors="replace")
    if constant and node.type == "Const" and "value" in node.attr:
        result["value_descriptor"] = _tensor(node.attr["value"].t.desc)
    return result


def _edge(edge):
    name, sep, port = edge.rpartition(":")
    if sep and port.lstrip("-").isdigit():
        return name, int(port)
    return edge, 0


def node_snapshot(graph, name):
    nodes = {node.name: node for node in graph.op}
    if name not in nodes:
        return None
    target = nodes[name]
    result = {"graph": graph.name, "node": _node(target,
        range(len(target.input_desc)), range(len(target.output_desc)), constant=True),
        "producers": [], "consumers": []}
    for input_port, edge in enumerate(target.input):
        source, output_port = _edge(edge)
        if not edge or output_port < 0:
            continue  # no optional/control-edge graph traversal
        producer = nodes.get(source)
        result["producers"].append({"to_input": input_port, "from_output": output_port,
            "node": _node(producer, outputs=(output_port,), constant=True) if producer is not None
                    else {"name": source, "status": "OUTSIDE_THIS_GRAPH"}})
    for node in graph.op:
        for input_port, edge in enumerate(node.input):
            source, output_port = _edge(edge)
            if source == name and output_port >= 0:
                result["consumers"].append({"from_output": output_port, "to_input": input_port,
                    "node": _node(node, inputs=(input_port,))})
    # Protect against exporting large fan-out graphs through one named hub.
    for key in ("producers", "consumers"):
        result[key + "_omitted"] = max(0, len(result[key]) - 4)
        result[key] = result[key][:4]
    return result


def collect(directory, name, *, limit=4, model_type=None):
    if not 1 <= limit <= 16:
        raise ValueError("limit must be between 1 and 16")
    directory = Path(directory).expanduser().resolve()
    graph_dir = directory / "graphs"
    if not graph_dir.is_dir():
        raise ValueError("diagnostics directory must contain graphs/")
    if model_type is None:
        from torchair._ge_concrete_graph.ge_ir_pb2 import ModelDef
        model_type = ModelDef
    from google.protobuf import text_format

    report = {"status": "NODE_NOT_FOUND", "node": name, "files": 0, "parsed": 0,
              "errors": [], "snapshots": [], "omitted_snapshots": 0}
    seen = {}
    alternatives = set()
    for path in sorted(graph_dir.rglob("ge_proto*.txt")):
        if not path.is_file():
            continue
        report["files"] += 1
        try:
            if not path.resolve().is_relative_to(graph_dir):
                raise ValueError("outside graph directory")
            if path.stat().st_size > 64 * 1024 * 1024:
                raise ValueError("dump exceeds parser size limit")
            model = model_type()
            text_format.Parse(path.read_text(encoding="utf-8"), model, allow_unknown_field=True)
            report["parsed"] += 1
            for graph in model.graph:
                alternatives.update(op.name for op in graph.op if op.type == "TransData")
                snapshot = node_snapshot(graph, name)
                if snapshot is None:
                    continue
                key = json.dumps(snapshot, sort_keys=True)
                if key not in seen:
                    seen[key] = {"first_stage": path.name, "last_stage": path.name, **snapshot}
                else:
                    seen[key]["last_stage"] = path.name
        except Exception as error:
            # Protobuf ParseError messages can quote a raw tensor-data line.
            # Export only the exception class, never its message/traceback.
            report["errors"].append({"stage": path.name, "error_type": type(error).__name__})
    snapshots = list(seen.values())
    if snapshots:
        report["status"] = "CAPTURED_PARTIAL" if report["errors"] else "CAPTURED"
        report["omitted_snapshots"] = max(0, len(snapshots) - limit)
        first_count = (limit + 1) // 2
        if len(snapshots) <= limit:
            report["snapshots"] = snapshots
        elif limit == 1:
            report["snapshots"] = snapshots[-1:]
        else:
            report["snapshots"] = snapshots[:first_count] + snapshots[-(limit // 2):]
    elif not report["parsed"]:
        report["status"] = "NO_PARSED_DUMPS"
    if not snapshots:
        report["transdata_names"] = sorted(alternatives)[:12]
    report["omitted_errors"] = max(0, len(report["errors"]) - 4)
    report["errors"] = report["errors"][:4]
    return report


def format_report(report):
    lines = [f"{report['status']}: {report['node']} files={report['files']} parsed={report['parsed']}"]
    for record in report["snapshots"]:
        lines.append(f"\nStage: {record['first_stage']} -> {record['last_stage']} graph={record['graph']}")
        for label, node in [("target", record["node"])] + [
            (f"producer out[{edge['from_output']}] -> target in[{edge['to_input']}]", edge["node"])
            for edge in record["producers"]] + [
            (f"consumer in[{edge['to_input']}] <- target out[{edge['from_output']}]", edge["node"])
            for edge in record["consumers"]]:
            lines.append(f"  {label}: {node['name']} ({node.get('type', node.get('status'))})")
            for key in ("inputs", "outputs"):
                for port, desc in node.get(key, {}).items():
                    lines.append(f"    {key}[{port}]: {json.dumps(desc, ensure_ascii=True)}")
            for key, value in node.items():
                if key not in ("name", "type", "status", "inputs", "outputs"):
                    lines.append(f"    {key}: {json.dumps(value, ensure_ascii=True)}")
        if record["producers_omitted"] or record["consumers_omitted"]:
            lines.append(f"  Omitted neighbors: {record['producers_omitted']} producers, {record['consumers_omitted']} consumers")
    for error in report["errors"]:
        lines.append(f"Skipped: {error['stage']} ({error['error_type']}; raw details remain local)")
    if report.get("transdata_names"):
        lines.append("Available TransData names: " + json.dumps(report["transdata_names"]))
    lines.append(f"Omitted: {report['omitted_snapshots']} distinct snapshots, {report['omitted_errors']} errors")
    return "\n".join(lines)


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--diagnostics-dir", type=Path, required=True)
    cli.add_argument("--node", default="trans_TransData_1")
    cli.add_argument("--limit", type=int, choices=range(1, 17), default=4)
    args = cli.parse_args(argv)
    try:
        report = collect(args.diagnostics_dir, args.node, limit=args.limit)
    except Exception as error:
        print(f"Cannot read descriptors ({type(error).__name__}). Check diagnostics-dir/graphs and use MODEL_PYTHON with TorchAir.", file=sys.stderr)
        return 1
    print(format_report(report))
    return 0 if report["snapshots"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
