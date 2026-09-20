"""Opt-in ATC shape evidence for the synthetic WeightQuant probe.

Snapshots contain descriptors, never tensor payloads. They report the graph
at the named compiler stage; they do not emulate Const inference or certify
the final OM's numerical behavior or runtime operator count.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

from .utils import atomic_write_json, file_record, require_run_output


def describe_tensor(desc):
    enum = desc.DESCRIPTOR.fields_by_name["dtype"].enum_type
    dtype = enum.values_by_number.get(desc.dtype)
    result = {"name": desc.name, "shape": list(desc.shape.dim), "format": desc.layout,
              "dtype": dtype.name if dtype else f"UNKNOWN({desc.dtype})"}
    for key in ("origin_shape", "storage_shape"):
        if key in desc.attr:
            result[key] = list(desc.attr[key].list.i)
    for key in ("format_for_int", "origin_format_for_int", "storage_format"):
        if key in desc.attr:
            result[key] = desc.attr[key].i
    for key in ("origin_shape_initialized", "origin_format_is_set"):
        if key in desc.attr:
            result[key] = desc.attr[key].b
    return result


def weight_quant_snapshot(graph):
    """Read WeightQuant and its immediate producers without mutating protobufs."""
    nodes = {node.name: node for node in graph.op}

    def describe(node):
        item = {"name": node.name, "type": node.type, "edges": list(node.input),
                "inputs": [describe_tensor(d) for d in node.input_desc],
                "outputs": [describe_tensor(d) for d in node.output_desc]}
        if "value" in node.attr:
            item["value_descriptor"] = describe_tensor(node.attr["value"].t.desc)
        for key in ("transpose_x", "transpose_weight"):
            if key in node.attr:
                item[key] = node.attr[key].b
        if "antiquant_group_size" in node.attr:
            item["antiquant_group_size"] = node.attr["antiquant_group_size"].i
        return item

    result = []
    for node in graph.op:
        if node.type != "WeightQuantBatchMatmulV2":
            continue
        item = describe(node)
        # Include all direct operand producers, plus the weight converter's
        # producer so the ND-Const + TransData control is inspectable too.
        edges = list(node.input)
        if len(node.input) > 1:
            weight = nodes.get(node.input[1].rpartition(":")[0])
            if weight is not None and weight.type == "TransData":
                edges.extend(weight.input)
        names = list(dict.fromkeys(edge.rpartition(":")[0] for edge in edges if edge))
        item["producers"] = [describe(nodes[name]) for name in names if name in nodes]
        result.append(item)
    return result


def weight_quant_inference_lines(stdout, limit=12):
    """Keep primary shape errors even when they precede ATC's generic summary."""
    matches = []
    seen = set()
    pattern = re.compile(r"Ka\[[^\]]+\]\s*!=\s*Kb\[[^\]]+\]|"
                         r"x_shape:.*weight_shape:|"
                         r"The shape of x and weight|batch of xShape and batch of wShape")
    for line in stdout.splitlines():
        match = pattern.search(line)
        if match is None:
            continue
        # Drop timestamps and PID prefixes when deduplicating.
        text = line[match.start():].strip()[:1800]
        if text not in seen:
            seen.add(text)
            matches.append(text)
    # Put errors ahead of debug snapshots if many inference calls were made.
    matches.sort(key=lambda line: line.startswith("x_shape:"))
    return matches[:limit]


class AtcShapeDiagnostics:
    """A real ATC subprocess runner with per-case, process-local diagnostics."""
    def __init__(self, root):
        self.root = require_run_output(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.stdout_path = self.root / "atc-debug.log"
        self.executed = False

    def __call__(self, command, cwd):
        if self.executed:
            raise ValueError("shape diagnostics require a separate directory per ATC invocation")
        self.executed = True
        graphs = self.root / "graphs"
        logs = self.root / "logs"
        graphs.mkdir(); logs.mkdir()
        # Do not mutate the parent environment or let inherited collection
        # paths redirect this case into another run's evidence directory.
        env = os.environ.copy()
        for key in ("NPU_COLLECT_PATH", "ASCEND_DUMP_PATH", "ASCEND_MODULE_LOG_LEVEL"):
            env.pop(key, None)
        overrides = {"DUMP_GE_GRAPH": "2", "DUMP_GRAPH_LEVEL": "1",
                     "DUMP_GRAPH_FORMAT": "ge_proto", "DUMP_GRAPH_PATH": str(graphs),
                     "ASCEND_GLOBAL_LOG_LEVEL": "0", "ASCEND_SLOG_PRINT_TO_STDOUT": "1",
                     "ASCEND_PROCESS_LOG_PATH": str(logs)}
        env.update(overrides)
        if env.get("IGNORE_INFER_ERROR") not in (None, "", "0"):
            raise ValueError("shape diagnostics require normal inference checks; unset IGNORE_INFER_ERROR")
        args = [a for a in command if not a.startswith("--log=")] + ["--log=debug"]
        atomic_write_json(self.root / "command.json", {"command": args, "cwd": str(cwd),
                          "environment_overrides": overrides, "scope": "ATC child process only"})
        with self.stdout_path.open("w", encoding="utf-8") as output:
            result = subprocess.run(args, cwd=cwd, env=env, check=False,
                                    stdout=output, stderr=subprocess.STDOUT, text=True)
        stdout = self.stdout_path.read_text(encoding="utf-8", errors="replace")
        return subprocess.CompletedProcess(args, result.returncode, stdout, None)

    def collect(self, *, model_type=None):
        """Retain raw files even if this TorchAir cannot parse a newer GE dump."""
        stdout = (self.stdout_path.read_text(encoding="utf-8", errors="replace")
                  if self.stdout_path.is_file() else "")
        report = {"scope": "ATC stage descriptors; not OM execution or a shape emulator",
                  "status": "NOT_RUN" if not self.executed else "NO_GRAPH_DUMP",
                  "inference_lines": weight_quant_inference_lines(stdout),
                  "graph_files": [], "snapshots": []}
        if self.stdout_path.is_file():
            report["log"] = file_record(self.stdout_path, relative_to=self.root)
        try:
            if model_type is None:
                from torchair._ge_concrete_graph.ge_ir_pb2 import ModelDef
                model_type = ModelDef
            from google.protobuf import text_format
            # Only scan files produced in this case's own dump directory.
            for path in sorted((self.root / "graphs").rglob("ge_proto*.txt")):
                if not path.is_file() or not path.resolve().is_relative_to(self.root):
                    continue
                record = file_record(path, relative_to=self.root)
                report["graph_files"].append(record)
                try:
                    if path.stat().st_size > 64 * 1024 * 1024:
                        raise ValueError("descriptor dump exceeds 64 MiB parsing limit; raw file retained")
                    model = model_type()
                    text_format.Parse(path.read_text(encoding="utf-8"), model, allow_unknown_field=True)
                    found = []
                    for graph in model.graph:
                        nodes = weight_quant_snapshot(graph)
                        if nodes:
                            found.append({"graph": getattr(graph, "name", ""), "nodes": nodes})
                    record["status"] = "PARSED"
                    if found:
                        report["snapshots"].append({"path": record["path"], "graphs": found,
                                                   "failure_dump": "InferShapeBlackBox" in path.name})
                except Exception as error:
                    record.update(status="PARSE_FAILED", error=f"{type(error).__name__}: {error}")
            if report["snapshots"]:
                report["status"] = "CAPTURED"
            elif report["graph_files"]:
                report["status"] = "NO_WEIGHTQUANT_SNAPSHOT"
        except Exception as error:
            report["parser_error"] = f"{type(error).__name__}: {error}"
        self.report_path = atomic_write_json(self.root / "shape-diagnostics.json", report)
        return report


def diagnostic_summary(case):
    """Concise report that can be pasted without the full compiler debug log."""
    lines = [f"{case['name']}: {case['status']} phase={case['phase']}"]
    report = case.get("shape_diagnostics", {})
    lines.append(f"  Shape evidence: {report.get('status', 'NOT_RUN')}")
    lines.extend("  " + line for line in report.get("inference_lines", []))
    snapshots = report.get("snapshots", [])
    selected = [s for s in snapshots if s["failure_dump"]]
    if not selected and snapshots:
        # This is labeled as a stage snapshot, never as the compiled OM.
        selected = snapshots[-1:]
    for snapshot in selected[:2]:
        lines.append(f"  GE dump: {snapshot['path']}")
        for graph in snapshot["graphs"]:
            for node in graph["nodes"][:2]:
                lines.append(f"  Node: {node['name']}")
                for index, label in ((0, "x"), (1, "weight"), (2, "scale")):
                    if index < len(node["inputs"]):
                        desc = node["inputs"][index]
                        lines.append(f"    {label}: shape={desc['shape']} origin={desc.get('origin_shape', 'MISSING')} "
                                     f"format={desc['format']} dtype={desc['dtype']}")
                weight_name = node["edges"][1].rpartition(":")[0] if len(node["edges"]) > 1 else ""
                for producer in node["producers"]:
                    if producer["name"] != weight_name:
                        continue
                    for desc in producer["outputs"]:
                        lines.append(f"    producer {producer['type']}: shape={desc['shape']} "
                                     f"origin={desc.get('origin_shape', 'MISSING')} format={desc['format']}")
                    if "value_descriptor" in producer:
                        desc = producer["value_descriptor"]
                        lines.append(f"    value: shape={desc['shape']} format={desc['format']} "
                                     f"storage_shape={desc.get('storage_shape', 'MISSING')} "
                                     f"storage_format={desc.get('storage_format', 'MISSING')}")
    if not selected:
        lines.append("  No parsed WeightQuant graph; inspect the retained atc-debug.log and raw dumps.")
    lines.append(f"  Full evidence: {case.get('shape_diagnostics_path', 'NOT_RUN')}")
    return "\n".join(lines)
