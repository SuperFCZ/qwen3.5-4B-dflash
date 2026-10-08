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
PIPELINE_MODES = {"serial": 0, "prefetch": 1}
KV_M80_MODES = {"baseline": 0, "a-ub": 1}
CUBE_PREFIX = "DFLASH_GROUP_QUANT_CUBE "


def pipeline_plan(k, n, mode):
    if mode not in PIPELINE_MODES:
        raise ValueError("pipeline mode must be serial or prefetch")
    enabled = mode == "prefetch" and (k in (512, 1024) or (k, n) == (9728, 2560))
    banks, tile_k = (2 if enabled else 1), (256 if k == 256 else 128)
    return {"pipeline_mode": mode, "selected_pipeline": "raw-prefetch-v1" if enabled else "serial-v1",
            "raw_banks": banks, "user_ub_bytes": 64 * tile_k * (2 + banks) + tile_k // 128 * 64 * 2 * banks}


def core_limit(value):
    result = int(value)
    if str(result) != str(value) or not 0 <= result <= 65535:
        raise ValueError("DFLASH_CORE_LIMIT must be an integer 0..65535; 0=auto, 1=single-core control")
    return result


def configure_build(header, report, limit, mode="batched", kernel_header=None, pipeline_mode="serial", kv_m80_mode="baseline"):
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
    if pipeline_mode not in PIPELINE_MODES or (pipeline_mode == "prefetch" and mode != "batched"):
        raise ValueError("pipeline must be serial or prefetch; prefetch requires batched dequantization")
    if original.count("#define DFLASH_GROUP_QUANT_PIPELINE_MODE 0U") != 1:
        raise ValueError("unexpected tracked pipeline-mode declaration")
    if kernel_header is not None and Path(kernel_header).resolve() == (HERE / "op_kernel" / CONFIG.name).resolve():
        raise ValueError("configure only the generated kernel header copy")
    configured = original.replace(default, f"#define DFLASH_GROUP_QUANT_CORE_LIMIT {limit}U")
    configured = configured.replace("#define DFLASH_GROUP_QUANT_DEQUANT_MODE 1U",
                                    f"#define DFLASH_GROUP_QUANT_DEQUANT_MODE {DEQUANT_MODES[mode]}U")
    configured = configured.replace("#define DFLASH_GROUP_QUANT_PIPELINE_MODE 0U",
                                    f"#define DFLASH_GROUP_QUANT_PIPELINE_MODE {PIPELINE_MODES[pipeline_mode]}U")
    if kv_m80_mode not in KV_M80_MODES:
        raise ValueError("KV M80 mode must be baseline or a-ub")
    if original.count("#define DFLASH_GROUP_QUANT_KV_M80_MODE 0U") != 1:
        raise ValueError("unexpected KV M80 declaration")
    configured = configured.replace("#define DFLASH_GROUP_QUANT_KV_M80_MODE 0U",
                                    f"#define DFLASH_GROUP_QUANT_KV_M80_MODE {KV_M80_MODES[kv_m80_mode]}U")
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
    sources += [HERE / "test" / name for name in ("a3_launch.py", "run_a2.py", "run_a4.py", "run_a41.py", "a4_contract.py")]
    write_json(report, {"abi": "dflash-group-quant-linear-build-v5", "policy": POLICY,
                        "kv_m80_mode": kv_m80_mode,
                        "tiling_abi": "full-m-v2", "launch_version": 5,
                        "pipeline_mode": pipeline_mode,
                        "dequant_mode": mode, "kernel_header": str(kernel_header) if kernel_header else None,
                        "core_limit": limit, "header": str(header), "header_sha256": sha256(header),
                        "source_sha256": {str(p.relative_to(HERE)): sha256(p) for p in sources},
                        "npu_execution": "NOT_RUN"})


def load_build_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if (config.get("abi") not in tuple(f"dflash-group-quant-linear-build-v{i}" for i in (1, 2, 3, 4, 5)) or config.get("policy") != POLICY or
            type(config.get("core_limit")) is not int or core_limit(config["core_limit"]) != config["core_limit"]):
        raise ValueError("invalid A3 build configuration")
    header = Path(config["header"])
    if sha256(header) != config["header_sha256"] or \
            f"#define DFLASH_GROUP_QUANT_CORE_LIMIT {config['core_limit']}U" not in header.read_text():
        raise ValueError("generated core-limit header changed after configuration")
    if not config["abi"].endswith("v1"):
        mode = config.get("dequant_mode")
        if mode not in DEQUANT_MODES or f"#define DFLASH_GROUP_QUANT_DEQUANT_MODE {DEQUANT_MODES[mode]}U" not in header.read_text():
            raise ValueError("generated dequantization mode differs from build configuration")
        if config.get("kernel_header") and sha256(config["kernel_header"]) != config["header_sha256"]:
            raise ValueError("host/kernel build headers differ")
    else:
        config["dequant_mode"] = "legacy"
    if config["abi"].endswith(("v3", "v4", "v5")):
        mode = config.get("pipeline_mode")
        if (mode not in PIPELINE_MODES or
                f"#define DFLASH_GROUP_QUANT_PIPELINE_MODE {PIPELINE_MODES[mode]}U" not in header.read_text() or
                (mode == "prefetch" and config["dequant_mode"] != "batched")):
            raise ValueError("generated pipeline mode differs from build configuration")
    else:
        config["pipeline_mode"] = "serial"
    if config["abi"].endswith("v4") and (config.get("tiling_abi") != "full-m-v2" or config.get("launch_version") != 4):
        raise ValueError("A4 build requires the full-M tiling ABI and version 4 launch evidence")
    if config["abi"].endswith("v5"):
        mode = config.get("kv_m80_mode")
        if (mode not in KV_M80_MODES or config.get("launch_version") != 5 or config.get("tiling_abi") != "full-m-v2" or
                f"#define DFLASH_GROUP_QUANT_KV_M80_MODE {KV_M80_MODES[mode]}U" not in header.read_text()):
            raise ValueError("generated KV M80 mode/launch version differs from build configuration")
    else:
        config["kv_m80_mode"] = "baseline"
    return config


def launch_evidence(log, shape, limit, workspace_bytes, dequant_mode=None, pipeline_mode=None, launch_version=None, kv_m80_mode=None):
    m, k, n = shape
    records = [json.loads(line.partition(PREFIX)[2]) for line in Path(log).read_text().splitlines() if PREFIX in line]
    if not records:
        raise ValueError("missing A3 host launch evidence; rebuild/use the current isolated OPP")
    first = records[0]
    if any(item != first for item in records):
        raise ValueError("host launch plan changed between repeated calls")
    version = first.get("version")
    if launch_version is not None and version != launch_version:
        raise ValueError("host launch version differs from the build; rebuild the isolated OPP")
    if version in (4, 5) and (first.get("tile_m") != m or first.get("weight_reuse_rows") != m):
        raise ValueError("A4 requires a single full-M output tile and full-M weight reuse")
    if m != 16 and version not in (4, 5):
        raise ValueError("M32/64/80 requires version 4/5 full-M launch evidence")
    mode = first.get("dequant_mode") if version in (2, 3, 4, 5) else "legacy"
    if (version not in (1, 2, 3, 4, 5) or mode not in DEQUANT_MODES or
            (dequant_mode is not None and mode != dequant_mode)):
        raise ValueError("launch dequantization mode differs from the requested build")
    pipeline = first.get("pipeline_mode") if version in (3, 4, 5) else "serial"
    if pipeline_mode is not None and (version not in (3, 4, 5) or pipeline != pipeline_mode):
        raise ValueError("launch pipeline mode needs version 3/4/5 evidence matching the requested build")
    plan = pipeline_plan(k, n, pipeline)
    if version == 5:
        cubes = [json.loads(line.partition(CUBE_PREFIX)[2]) for line in Path(log).read_text().splitlines() if CUBE_PREFIX in line]
        if not cubes or any(item != cubes[0] for item in cubes):
            raise ValueError("missing or inconsistent Matmul resource evidence")
        cube = cubes[0]
        selected = cube.get("kv_m80_mode")
        if selected not in KV_M80_MODES or (kv_m80_mode is not None and selected != kv_m80_mode):
            raise ValueError("KV M80 build mode differs from host evidence")
        staged = selected == "a-ub" and (m,k,n) == (80,2560,2048)
        extra = m * 128 * 2 if staged else 0
        expected_cube = dict(version=1, a_source="VECOUT" if staged else "GM", a_stage_bytes=extra,
                             a_row_stride=128 if staged else k, base_m=m, base_n=64,
                             weight_dequantizations_per_n_tile=k // (256 if k == 256 else 128))
        if any(cube.get(key) != value for key, value in expected_cube.items()):
            raise ValueError("A source/stride/full-M reuse differs from shape policy")
        if (type(cube.get("base_k")) is not int or cube["base_k"] <= 0 or cube["base_k"] % 16 or
                cube["base_k"] > (256 if k == 256 else 128) or (k != 256 and cube["base_k"] != 128)):
            raise ValueError("invalid Cube K reduction plan")
        for key in ("depth_a1", "depth_b1", "step_m", "step_n", "step_ka", "step_kb", "db_l0a", "db_l0b", "db_l0c", "trans_length"):
            if type(cube.get(key)) is not int or cube[key] < 0:
                raise ValueError(f"invalid Matmul resource: {key}")
        plan["user_ub_bytes"] += extra
        first = dict(first, cube=cube)
    elif kv_m80_mode not in (None, "baseline"):
        raise ValueError("KV M80 candidate requires version 5 evidence")
    if version in (3, 4, 5):
        if (any(first.get(key) != value for key, value in plan.items()) or
                (pipeline == "prefetch" and mode != "batched")):
            raise ValueError("launch pipeline selection/buffers differ from the shape policy")
        if (type(first.get("matmul_ub_bytes")) is not int or
                first["matmul_ub_bytes"] <= 0 or first["matmul_ub_bytes"] % 32):
            raise ValueError("invalid Matmul UB budget")
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
    return {**first, "dequant_mode": mode, **plan, "evidence_source": "Host Tiling log, not hardware profiling",
            "assignments": assignments}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-limit", type=core_limit, default=0)
    parser.add_argument("--dequant-mode", choices=tuple(DEQUANT_MODES), default="batched")
    parser.add_argument("--pipeline-mode", choices=tuple(PIPELINE_MODES), default="serial")
    parser.add_argument("--kv-m80-mode", choices=tuple(KV_M80_MODES), default="baseline")
    parser.add_argument("--kernel-header", type=Path)
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    configure_build(args.header, args.report, args.core_limit, args.dequant_mode, args.kernel_header, args.pipeline_mode, args.kv_m80_mode)


if __name__ == "__main__":
    main()
