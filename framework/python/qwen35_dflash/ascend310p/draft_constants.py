"""Explicit read-only compressed Draft inputs for AIR/OM.

Weights must be graph inputs: leaving dequantization fed by Const permits an
exporter/compiler to fold a compressed checkpoint into a dense FP16 OM.
"""
from __future__ import annotations

from dataclasses import replace
from math import prod
from pathlib import Path

import torch
from torch import nn

from .contracts import AirGraphSpec
from .utils import contained_path, file_record, sha256_file


class DraftConstantInputs(nn.Module):
    def __init__(self, model: nn.Module, buffer_names: tuple[str, ...], buffer_shapes: tuple,
                 dynamic_count: int):
        super().__init__()
        self.model = model
        self.buffer_names = buffer_names
        self.buffer_shapes = buffer_shapes
        self.dynamic_count = dynamic_count

    def forward(self, *inputs):
        arguments, weights = inputs[:self.dynamic_count], inputs[self.dynamic_count:]
        if len(weights) != len(self.buffer_names):
            raise ValueError("compressed Draft input count differs")
        return torch.func.functional_call(
            self.model, {name: weight.view(shape) for name, weight, shape in
                         zip(self.buffer_names, weights, self.buffer_shapes)},
            arguments, strict=False,
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
    shapes = tuple(tuple(buffers[name].shape) for name in names)
    # ACL dynamic gears concatenate the ranks of ALL inputs into 128 slots.
    # The five-layer checkpoints need 151 slots with 2-D weights, but only 99
    # with flat carriers. Views preserve bytes/storage; restore logical shapes
    # inside the graph, without an additional device allocation or conversion.
    values = tuple(buffers[name].view(-1) for name in names)
    # Remove the stored copies from the exported module. functional_call binds
    # each buffer to its explicit input on every invocation.
    for name in names:
        parent, attribute = name.rsplit(".", 1)
        original = buffers[name]
        setattr(spec.model.get_submodule(parent), attribute,
                torch.empty(0, dtype=original.dtype, device=original.device))
    inputs = tuple(f"draft_weight_{index:03d}" for index in range(len(names)))
    tensors = [{"name": n, "dtype": str(t.dtype).removeprefix("torch."), "shape": list(t.shape)}
               for n, t in zip(inputs, values)]
    metadata = dict(spec.metadata, constant_tensors=tensors,
                    constant_tensor_shapes={n: list(shape) for n, shape in zip(inputs, shapes)})
    if "tensor_abi" in metadata:
        metadata["tensor_abi"] = dict(metadata["tensor_abi"], inputs=metadata["tensor_abi"]["inputs"] + tensors)
    return replace(spec, model=DraftConstantInputs(spec.model, tuple(names), shapes, len(spec.example_args)),
                   metadata=metadata,
                   example_args=spec.example_args + values,
                   input_names=spec.input_names + inputs, constant_input_names=inputs)


def write_constant_inputs(spec: AirGraphSpec, graph_dir: Path, root: Path) -> dict:
    if not spec.constant_input_names:
        return {}
    directory = graph_dir / "constant-inputs"
    directory.mkdir()
    records = []
    table = ["qwen35-draft-constants-v1"]
    for index, name in enumerate(spec.constant_input_names, start=len(spec.example_args) - len(spec.constant_input_names)):
        tensor = spec.example_args[index].detach().cpu().contiguous()
        dtype = str(tensor.dtype).removeprefix("torch.")
        if dtype not in ("int8", "uint8", "float16") or tensor.ndim != 1:
            raise ValueError("unsupported compressed Draft constant dtype/shape")
        path = directory / f"{name}.bin"
        tensor.numpy().tofile(path)
        record = {**file_record(path, relative_to=root), "index": index,
                  "name": name, "dtype": dtype, "shape": list(tensor.shape),
                  "logical_shape": spec.metadata["constant_tensor_shapes"][name]}
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
    contract = graph.get("metadata", {}).get("incremental_contract")
    expected_count = (2 + 5 * len(contract["draft_states"])) if contract else 72
    if variant != "fp16" and len(records) != expected_count:
        raise ValueError("quantized Draft OM must expose all compressed Linear pairs")
    if not records:
        if (not contract and names != ["input_ids", "attention_mask"]) or graph.get("constant_inputs_table"):
            raise ValueError("missing compressed Draft constant input records")
        return None
    start = len(names) - len(records)
    descriptors = [{k: r[k] for k in ("name", "dtype", "shape")} for r in records]
    if contract and (descriptors != contract.get("draft_constants") or
                     descriptors != graph["metadata"].get("constant_tensors") or
                     descriptors != graph["metadata"]["tensor_abi"]["inputs"][start:]):
        raise ValueError("constant input descriptors differ from graph ABI")
    if start < 1 or names[start:] != [r["name"] for r in records]:
        raise ValueError("constant input order differs from graph ABI")
    if len(set(names)) != len(names):
        raise ValueError("duplicate constant input names")
    if len(records) % 2:
        raise ValueError("compressed weights and scales must form pairs")
    logical_shapes = graph.get("metadata", {}).get("constant_tensor_shapes", {})
    for record in records:
        shape, logical = record["shape"], record.get("logical_shape", record["shape"])
        if (len(shape) not in (1, 2) or len(logical) != 2
                or any(type(v) is not int or v <= 0 for v in (*shape, *logical))
                or prod(shape) != prod(logical)
                or (len(shape) == 1 and logical_shapes.get(record["name"]) != logical)):
            raise ValueError("invalid compressed Draft carrier/logical shape")
    for packed, scale in zip(records[::2], records[1::2]):
        bits = {"int8": 8, "uint8": 4}.get(packed["dtype"])
        packed_shape = packed.get("logical_shape", packed["shape"])
        scale_shape = scale.get("logical_shape", scale["shape"])
        if (bits is None or scale["dtype"] != "float16"
                or packed_shape[0] != scale_shape[0]
                or packed_shape[1] * 8 != scale_shape[1] * 128 * bits
                or (variant == "w8a16" and bits != 8) or (variant == "w4a16" and bits != 4)):
            raise ValueError("compressed Draft weight/scale group contract differs")
    table_record = graph["constant_inputs_table"]
    table = contained_path(root, table_record["path"])
    if not table.is_file() or sha256_file(table) != table_record["sha256"]:
        raise ValueError("constant input table integrity check failed")
    expected_lines = ["qwen35-draft-constants-v1"]
    for index, record in enumerate(records, start=start):
        path = contained_path(root, record["path"])
        width = {"int8": 1, "uint8": 1, "float16": 2}.get(record["dtype"])
        shape = record["shape"]
        if (record["index"] != index or width is None or len(shape) not in (1, 2)
                or any(type(v) is not int or v <= 0 for v in shape)):
            raise ValueError("invalid constant input ABI")
        size = prod(shape) * width
        if (not path.is_file() or path.stat().st_size != size
                or record["bytes"] != size or sha256_file(path) != record["sha256"]):
            raise ValueError("constant input payload integrity check failed")
        expected_lines.append("\t".join((record["name"], record["dtype"], ",".join(map(str, shape)),
                                        str(size), record["sha256"], path.relative_to(table.parent).as_posix())))
    if table.read_text().splitlines() != expected_lines:
        raise ValueError("constant input table differs from manifest")
    return table
