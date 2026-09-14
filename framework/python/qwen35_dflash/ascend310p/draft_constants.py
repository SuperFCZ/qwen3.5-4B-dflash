"""Explicit read-only compressed Draft inputs for AIR/OM.

Weights must be graph inputs: leaving dequantization fed by Const permits an
exporter/compiler to fold a compressed checkpoint into a dense FP16 OM.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from .contracts import AirGraphSpec
from .utils import contained_path, file_record, sha256_file


class DraftConstantInputs(nn.Module):
    def __init__(self, model: nn.Module, buffer_names: tuple[str, ...]):
        super().__init__()
        self.model = model
        self.buffer_names = buffer_names

    def forward(self, input_ids, attention_mask, *weights):
        if len(weights) != len(self.buffer_names):
            raise ValueError("compressed Draft input count differs")
        return torch.func.functional_call(
            self.model, dict(zip(self.buffer_names, weights)),
            (input_ids, attention_mask), strict=False,
        )


def expose_draft_constants(spec: AirGraphSpec) -> AirGraphSpec:
    from models.dflash_v1.draft_quantization import GroupQuantLinear

    names = []
    for name, module in spec.model.named_modules():
        if isinstance(module, GroupQuantLinear):
            names.extend((name + ".qweight", name + ".scales"))
    if not names:
        return spec
    buffers = dict(spec.model.named_buffers())
    values = tuple(buffers[name] for name in names)
    # Remove the stored copies from the exported module. functional_call binds
    # each buffer to its explicit input on every invocation.
    for name in names:
        parent, attribute = name.rsplit(".", 1)
        original = buffers[name]
        setattr(spec.model.get_submodule(parent), attribute,
                torch.empty(0, dtype=original.dtype, device=original.device))
    inputs = tuple(f"draft_weight_{index:03d}" for index in range(len(names)))
    return replace(spec, model=DraftConstantInputs(spec.model, tuple(names)),
                   example_args=spec.example_args + values,
                   input_names=spec.input_names + inputs, constant_input_names=inputs)


def write_constant_inputs(spec: AirGraphSpec, graph_dir: Path, root: Path) -> dict:
    if not spec.constant_input_names:
        return {}
    directory = graph_dir / "constant-inputs"
    directory.mkdir()
    records = []
    table = ["qwen35-draft-constants-v1"]
    for index, name in enumerate(spec.constant_input_names, start=2):
        tensor = spec.example_args[index].detach().cpu().contiguous()
        dtype = str(tensor.dtype).removeprefix("torch.")
        if dtype not in ("int8", "uint8", "float16") or tensor.ndim != 2:
            raise ValueError("unsupported compressed Draft constant dtype/shape")
        path = directory / f"{name}.bin"
        tensor.numpy().tofile(path)
        record = {**file_record(path, relative_to=root), "index": index,
                  "name": name, "dtype": dtype, "shape": list(tensor.shape)}
        records.append(record)
        table.append("\t".join((name, dtype, ",".join(map(str, tensor.shape)),
                                str(path.stat().st_size), record["sha256"],
                                path.relative_to(graph_dir).as_posix())))
    table_path = graph_dir / "constant-inputs.tsv"
    table_path.write_text("\n".join(table) + "\n", encoding="utf-8")
    return {"constant_inputs": records, "constant_inputs_table": file_record(table_path, relative_to=root)}


def verify_constant_inputs(graph: dict, root: Path) -> Path | None:
    records = graph.get("constant_inputs", [])
    names = graph.get("input_names", [])
    variant = graph.get("metadata", {}).get("draft_quantization", "fp16")
    if variant not in ("fp16", "w8a16", "w4a16"):
        raise ValueError("unknown Draft quantization in graph metadata")
    if variant != "fp16" and len(records) != 72:
        raise ValueError("quantized Draft OM must expose all 36 compressed Linear pairs")
    if not records:
        if names != ["input_ids", "attention_mask"] or graph.get("constant_inputs_table"):
            raise ValueError("missing compressed Draft constant input records")
        return None
    if names != ["input_ids", "attention_mask"] + [r["name"] for r in records]:
        raise ValueError("constant input order differs from graph ABI")
    if len(set(names)) != len(names):
        raise ValueError("duplicate constant input names")
    if len(records) % 2:
        raise ValueError("compressed weights and scales must form pairs")
    for packed, scale in zip(records[::2], records[1::2]):
        bits = {"int8": 8, "uint8": 4}.get(packed["dtype"])
        if (bits is None or scale["dtype"] != "float16"
                or len(packed["shape"]) != 2 or len(scale["shape"]) != 2
                or packed["shape"][0] != scale["shape"][0]
                or packed["shape"][1] * 8 != scale["shape"][1] * 128 * bits
                or (variant == "w8a16" and bits != 8) or (variant == "w4a16" and bits != 4)):
            raise ValueError("compressed Draft weight/scale group contract differs")
    table_record = graph["constant_inputs_table"]
    table = contained_path(root, table_record["path"])
    if not table.is_file() or sha256_file(table) != table_record["sha256"]:
        raise ValueError("constant input table integrity check failed")
    expected_lines = ["qwen35-draft-constants-v1"]
    for index, record in enumerate(records, start=2):
        path = contained_path(root, record["path"])
        width = {"int8": 1, "uint8": 1, "float16": 2}.get(record["dtype"])
        shape = record["shape"]
        if (record["index"] != index or width is None or len(shape) != 2
                or any(type(v) is not int or v <= 0 for v in shape)):
            raise ValueError("invalid constant input ABI")
        size = shape[0] * shape[1] * width
        if (not path.is_file() or path.stat().st_size != size
                or record["bytes"] != size or sha256_file(path) != record["sha256"]):
            raise ValueError("constant input payload integrity check failed")
        expected_lines.append("\t".join((record["name"], record["dtype"], ",".join(map(str, shape)),
                                        str(size), record["sha256"], path.relative_to(table.parent).as_posix())))
    if table.read_text().splitlines() != expected_lines:
        raise ValueError("constant input table differs from manifest")
    return table
