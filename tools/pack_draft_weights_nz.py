#!/usr/bin/env python3
"""Convert a W8 Draft's exported constant inputs into reusable INT8 NZ files.

Consumes an existing per-variant AIR/deployment manifest and verifies its
weight/scale binaries first. CPU only: no checkpoint loading, NPU or CANN.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "framework/python"), str(REPO)]

import torch

from qwen35_dflash.ascend310p.draft_constants import verify_constant_inputs
from qwen35_dflash.ascend310p.utils import (
    atomic_write_json, contained_path, file_record, load_json_object, require_run_output,
)
from qwen35_dflash.ascend310p.weight_prepack import PREPACK_POLICY, pack_int8_nz


def convert(manifest_path, output_dir):
    source = Path(manifest_path).expanduser().resolve()
    manifest = load_json_object(source)
    drafts = [g for g in manifest.get("graphs", []) if g.get("name") == "draft"
              and g.get("metadata", {}).get("draft_quantization") == "w8a16"]
    if len(drafts) != 1:
        raise ValueError("provide a per-variant W8 AIR/deployment manifest containing one Draft")
    graph = drafts[0]
    if not graph.get("constant_inputs"):
        raise ValueError("source must contain the original flat W8 constant inputs")
    verify_constant_inputs(graph, source.parent)
    out = require_run_output(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"offline NZ output must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for item in graph["constant_inputs"][::2]:
        shape = item.get("logical_shape", item["shape"])
        weight_path = contained_path(source.parent, item["path"])
        data = weight_path.read_bytes()
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("source W8 weight changed after manifest validation")
        matrix = torch.frombuffer(bytearray(data), dtype=torch.int8).reshape(shape)
        packed = pack_int8_nz(matrix)
        destination = out / f"weight-{len(records):03d}.nz.bin"
        packed.numpy().tofile(destination)
        records.append({**file_record(destination, relative_to=out), "dtype": "int8",
            "format": "FRACTAL_NZ", "logical_shape": list(shape), "storage_shape": list(packed.shape),
            "logical_sha256": hashlib.sha256(data).hexdigest(),
            "source_input": item["name"], "roundtrip": "BIT_EXACT", "padding": "ZERO"})
        print(f"[weight-prepack] {item['name']} {shape} -> {list(packed.shape)} BIT_EXACT", flush=True)
    return atomic_write_json(out / "manifest.json", {"schema_version": 1, "policy": PREPACK_POLICY,
        "status": "PASS", "source_manifest": {"name": source.name,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        "weight_count": len(records), "weights": records,
        "scope": "offline byte permutation; ATC, OM execution and latency NOT_RUN"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="e.g. old-bundle/air-manifest-w8a16-chunk.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(f"NZ manifest: {convert(args.manifest, args.output_dir)}")


if __name__ == "__main__":
    main()
