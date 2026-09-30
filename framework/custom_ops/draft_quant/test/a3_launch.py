"""A3 build configuration and host launch evidence; no device imports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from a2_common import sha256, write_json

HERE = Path(__file__).resolve().parents[1]
CONFIG = HERE / "op_host/d_flash_group_quant_linear_build_config.h"
POLICY = "n-tile-cyclic-v1"
PREFIX = "DFLASH_GROUP_QUANT_LAUNCH "
DEQUANT_MODES = {"legacy": 0, "batched": 1}


def core_limit(value):
    result = int(value)
    if str(result) != str(value) or not 0 <= result <= 65535:
        raise ValueError("DFLASH_CORE_LIMIT must be an integer 0..65535; 0=auto, 1=single-core control")
    return result


def configure_build(header, report, limit, mode="batched", kernel_header=None):
    limit = core_limit(limit)
    header, report = Path(header).resolve(), Path(report).resolve()
    if header == CONFIG.resolve():
        raise ValueError("configure only the generated build copy, not the tracked default")
    original = CONFIG.read_text()
    default = "#define DFLASH_GROUP_QUANT_CORE_LIMIT 0U"
    if original.count(default) != 1:
        raise ValueError("unexpected tracked core-limit declaration")
    if mode not in DEQUANT_MODES:
        raise ValueError("dequant mode must be legacy or batched")
    if original.count("#define DFLASH_GROUP_QUANT_DEQUANT_MODE 1U") != 1:
        raise ValueError("unexpected tracked dequant-mode declaration")
    if kernel_header is not None and Path(kernel_header).resolve() == (HERE / "op_kernel" / CONFIG.name).resolve():
        raise ValueError("configure only the generated kernel header copy")
    configured = original.replace(default, f"#define DFLASH_GROUP_QUANT_CORE_LIMIT {limit}U")
    configured = configured.replace("#define DFLASH_GROUP_QUANT_DEQUANT_MODE 1U",
                                    f"#define DFLASH_GROUP_QUANT_DEQUANT_MODE {DEQUANT_MODES[mode]}U")
    header.write_text(configured)
    if kernel_header is not None:
        kernel_header = Path(kernel_header).resolve()
        kernel_header.write_text(configured)
    sources = [CONFIG, HERE / "op_host/d_flash_group_quant_linear_contract.h",
               HERE / "op_host/d_flash_group_quant_linear_tiling.h",
               HERE / "op_host/d_flash_group_quant_linear.cpp",
               HERE / "op_kernel/d_flash_group_quant_linear.cpp",
               HERE / "op_kernel/d_flash_group_quant_linear_build_config.h",
               HERE / "test/main.cpp", HERE / "test/native_om.cpp", HERE / "test/runner_common.h"]
    write_json(report, {"abi": "dflash-group-quant-linear-build-v2", "policy": POLICY,
                        "dequant_mode": mode, "kernel_header": str(kernel_header) if kernel_header else None,
                        "core_limit": limit, "header": str(header), "header_sha256": sha256(header),
                        "source_sha256": {str(p.relative_to(HERE)): sha256(p) for p in sources},
                        "npu_execution": "NOT_RUN"})


def load_build_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if (config.get("abi") not in ("dflash-group-quant-linear-build-v1", "dflash-group-quant-linear-build-v2") or config.get("policy") != POLICY or
            type(config.get("core_limit")) is not int or core_limit(config["core_limit"]) != config["core_limit"]):
        raise ValueError("invalid A3 build configuration")
    header = Path(config["header"])
    if sha256(header) != config["header_sha256"] or \
            f"#define DFLASH_GROUP_QUANT_CORE_LIMIT {config['core_limit']}U" not in header.read_text():
        raise ValueError("generated core-limit header changed after configuration")
    if config["abi"].endswith("v2"):
        mode = config.get("dequant_mode")
        if mode not in DEQUANT_MODES or f"#define DFLASH_GROUP_QUANT_DEQUANT_MODE {DEQUANT_MODES[mode]}U" not in header.read_text():
            raise ValueError("generated dequantization mode differs from build configuration")
        if config.get("kernel_header") and sha256(config["kernel_header"]) != config["header_sha256"]:
            raise ValueError("host/kernel build headers differ")
    else:
        config["dequant_mode"] = "legacy"
    return config


def launch_evidence(log, shape, limit, workspace_bytes, dequant_mode=None):
    m, k, n = shape
    records = [json.loads(line.partition(PREFIX)[2]) for line in Path(log).read_text().splitlines() if PREFIX in line]
    if not records:
        raise ValueError("missing A3 host launch evidence; rebuild/use the current isolated OPP")
    first = records[0]
    if any(item != first for item in records):
        raise ValueError("host launch plan changed between repeated calls")
    mode = first.get("dequant_mode") if first.get("version") == 2 else "legacy"
    if (first.get("version") not in (1, 2) or mode not in DEQUANT_MODES or
            (dequant_mode is not None and mode != dequant_mode)):
        raise ValueError("launch dequantization mode differs from the requested build")
    expected = {"policy": POLICY, "m": m, "k": k, "n": n,
                "tile_n": 64, "tile_k": 256 if k == 256 else 128, "n_tiles": n // 64, "core_limit": limit}
    if any(first.get(key) != value for key, value in expected.items()):
        raise ValueError("launch shape/tile/core-limit differs from the requested build")
    for key in ("available_cores", "block_dim", "system_workspace_bytes"):
        if type(first.get(key)) is not int or first[key] < 0:
            raise ValueError(f"invalid launch field: {key}")
    available = first["available_cores"]
    blocks = min(available, n // 64, limit or available)
    if available == 0 or blocks == 0 or first["block_dim"] != blocks:
        raise ValueError("launch block count does not match min(available cores, N tiles, core cap)")
    if first["system_workspace_bytes"] > workspace_bytes:
        raise ValueError("ACLNN workspace is smaller than the host API workspace requirement")
    assignments = []
    for block in range(blocks):
        tiles = list(range(block, n // 64, blocks))
        assignments.append({"block": block, "tile_count": len(tiles), "first_column": tiles[0] * 64,
                            "last_tile_column": tiles[-1] * 64, "tile_stride_columns": blocks * 64})
    return {**first, "dequant_mode": mode, "evidence_source": "Host Tiling log, not hardware profiling",
            "assignments": assignments}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-limit", type=core_limit, default=0)
    parser.add_argument("--dequant-mode", choices=tuple(DEQUANT_MODES), default="batched")
    parser.add_argument("--kernel-header", type=Path)
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    configure_build(args.header, args.report, args.core_limit, args.dequant_mode, args.kernel_header)


if __name__ == "__main__":
    main()
