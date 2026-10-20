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
SCALE_MODES = {"scalar": 0, "broadcast": 1}
K_TILE_POLICY = "m80-synthetic-k128-v1"
K_PREFIX = "DFLASH_GROUP_QUANT_K_TILE "
SCALE_PREFIX = "DFLASH_GROUP_QUANT_SCALE "
CUBE_PREFIX = "DFLASH_GROUP_QUANT_CUBE "


def pipeline_plan(k, n, mode, selected_tile_k=None):
    if mode not in PIPELINE_MODES:
        raise ValueError("pipeline mode must be serial or prefetch")
    enabled = mode == "prefetch" and (k in (512, 1024) or (k, n) == (9728, 2560))
    banks, tile_k = (2 if enabled else 1), (selected_tile_k if selected_tile_k is not None else (256 if k == 256 else 128))
    return {"pipeline_mode": mode, "selected_pipeline": "raw-prefetch-v1" if enabled else "serial-v1",
            "raw_banks": banks, "user_ub_bytes": 64 * tile_k * (2 + banks) + tile_k // 128 * 64 * 2 * banks}


def selected_k_tile(shape, requested=128):
    if type(requested) is not int or requested not in (128,256): raise ValueError("K tile must be 128 or 256")
    m,k,n=shape
    if m==80 and k in (512,1024) and n in (64,128,256): return 128
    return 128 if (m,k,n)==(80,2560,2048) else (256 if k==256 else requested)


def k_tile_reason(shape,requested):
    m,k,n=shape
    if (m,k,n)==(80,2560,2048): return "kv-m80-k128"
    if k==256: return "whole-k256"
    if requested==256 and m==80 and k in (512,1024) and n in (64,128,256): return "m80-synthetic-k128"
    return "configured"


def scale_plan(shape, mode):
    if mode not in SCALE_MODES: raise ValueError("unknown scale mode")
    enabled = mode == "broadcast" and tuple(shape) != (80, 2560, 2048)
    return dict(version=1, scale_mode=mode, selected_scale="brcb-mul-v1" if enabled else "scalar-muls-v1",
                broadcast_bytes=2304 if enabled else 0, sdk_reserved_bytes=8192 if enabled else 0)


def core_limit(value):
    result = int(value)
    if str(result) != str(value) or not 0 <= result <= 65535:
        raise ValueError("DFLASH_CORE_LIMIT must be an integer 0..65535; 0=auto, 1=single-core control")
    return result


def configure_build(header, report, limit, mode="batched", kernel_header=None, pipeline_mode="serial", kv_m80_mode="baseline", scale_mode="scalar", k_tile=128, gate_copy_mode="loop"):
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
    if scale_mode not in SCALE_MODES or (scale_mode == "broadcast" and mode != "batched"):
        raise ValueError("scale mode must be scalar or broadcast; broadcast requires batched")
    if original.count("#define DFLASH_GROUP_QUANT_SCALE_MODE 0U") != 1:
        raise ValueError("unexpected tracked scale-mode declaration")
    configured = configured.replace("#define DFLASH_GROUP_QUANT_SCALE_MODE 0U",
                                    f"#define DFLASH_GROUP_QUANT_SCALE_MODE {SCALE_MODES[scale_mode]}U")
    selected_k_tile((16,512,64),k_tile)
    configured = configured.replace("#define DFLASH_GROUP_QUANT_K_TILE 128U", f"#define DFLASH_GROUP_QUANT_K_TILE {k_tile}U")
    if gate_copy_mode not in ("loop","2d") or (gate_copy_mode=="2d" and (k_tile!=256 or mode!="batched" or scale_mode!="broadcast")):
        raise ValueError("Gate copy requires loop/2d and A5.1 K256/batched/broadcast")
    if original.count("#define DFLASH_GROUP_QUANT_GATE_COPY_MODE 0U")!=1:
        raise ValueError("unexpected Gate copy declaration")
    configured = configured.replace("#define DFLASH_GROUP_QUANT_GATE_COPY_MODE 0U", f"#define DFLASH_GROUP_QUANT_GATE_COPY_MODE {int(gate_copy_mode=='2d')}U")
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
    sources += [HERE / "test" / name for name in ("a3_launch.py", "run_a2.py", "run_a4.py", "run_a41.py", "a4_contract.py", "compare_a5.py", "run_a5.py", "profile_a31.py", "run_a51.py", "run_a51_pair.py", "compare_a51.py", "run_suite.py", "reference.py", "a2_common.py", "a51_fallback.py")]
    sources += [HERE / "test" / name for name in ("run_gate_copy.py", "run_gate_copy_pair.py", "gate_copy_sdk.py")]
    sources += [HERE / "run_server.sh", HERE / "run_gate_copy.sh"]
    write_json(report, {"abi": "dflash-group-quant-linear-build-v5", "policy": POLICY,
                        "gate_copy_mode": gate_copy_mode, "kv_m80_mode": kv_m80_mode, "scale_mode": scale_mode, "k_tile": k_tile, "k_tile_policy": K_TILE_POLICY,
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
    if "scale_mode" in config:
        mode = config["scale_mode"]
        if (mode not in SCALE_MODES or (mode == "broadcast" and config["dequant_mode"] != "batched") or
                f"#define DFLASH_GROUP_QUANT_SCALE_MODE {SCALE_MODES[mode]}U" not in header.read_text()):
            raise ValueError("generated scale mode differs from build configuration")
    if "k_tile" in config:
        selected_k_tile((16,512,64),config["k_tile"])
        if f"#define DFLASH_GROUP_QUANT_K_TILE {config['k_tile']}U" not in header.read_text():
            raise ValueError("generated K tile differs from build configuration")
    if config.get("k_tile_policy",K_TILE_POLICY)!=K_TILE_POLICY:
        raise ValueError("unsupported K tile policy")
    if "gate_copy_mode" in config:
        mode=config["gate_copy_mode"]
        if mode not in ("loop","2d") or f"#define DFLASH_GROUP_QUANT_GATE_COPY_MODE {int(mode=='2d')}U" not in header.read_text():
            raise ValueError("Gate copy configuration/header mismatch")
        if mode=="2d" and (config.get("k_tile")!=256 or config.get("scale_mode")!="broadcast" or config.get("dequant_mode")!="batched"):
            raise ValueError("Gate copy requires K256/batched/broadcast")
    return config


def launch_evidence(log, shape, limit, workspace_bytes, dequant_mode=None, pipeline_mode=None, launch_version=None, kv_m80_mode=None, scale_mode=None, k_tile=None, k_tile_policy=None):
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
    tile_k = selected_k_tile(shape, 128 if k_tile is None else k_tile)
    plan = pipeline_plan(k, n, pipeline, tile_k)
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
                             weight_dequantizations_per_n_tile=k // tile_k)
        if any(cube.get(key) != value for key, value in expected_cube.items()):
            raise ValueError("A source/stride/full-M reuse differs from shape policy")
        if (type(cube.get("base_k")) is not int or cube["base_k"] <= 0 or cube["base_k"] % 16 or
                cube["base_k"] > tile_k or (k != 256 and cube["base_k"] != tile_k)):
            raise ValueError("invalid Cube K reduction plan")
        for key in ("depth_a1", "depth_b1", "step_m", "step_n", "step_ka", "step_kb", "db_l0a", "db_l0b", "db_l0c", "trans_length"):
            if type(cube.get(key)) is not int or cube[key] < 0:
                raise ValueError(f"invalid Matmul resource: {key}")
        plan["user_ub_bytes"] += extra
        first = dict(first, cube=cube)
    elif kv_m80_mode not in (None, "baseline"):
        raise ValueError("KV M80 candidate requires version 5 evidence")
    if k_tile is not None:
        records = [json.loads(line.partition(K_PREFIX)[2]) for line in Path(log).read_text().splitlines() if K_PREFIX in line]
        if not records or any(item != records[0] for item in records) or any(records[0].get(key)!=value for key,value in
            dict(version=1,requested_k_tile=k_tile,selected_k_tile=tile_k).items()):
            raise ValueError("K tile evidence differs from build/shape policy")
        if k_tile_policy is not None and (k_tile_policy!=K_TILE_POLICY or
                records[0].get("selection_reason")!=k_tile_reason(shape,k_tile)):
            raise ValueError("K tile selection reason differs from explicit shape fallback policy")
        if tile_k==256 and k!=256:
            for memory in ("l1","l0a","l0b","l0c"):
                used,capacity=records[0].get(memory+"_bytes"),records[0].get(memory+"_capacity")
                if type(used) is not int or type(capacity) is not int or not 0<used<=capacity:
                    raise ValueError("K256 resource exceeds hardware capacity")
        first = dict(first, k_tiling=records[0])
    if scale_mode is not None:
        records = [json.loads(line.partition(SCALE_PREFIX)[2]) for line in Path(log).read_text().splitlines() if SCALE_PREFIX in line]
        selected = scale_plan(shape, scale_mode)
        if not records or any(item != selected for item in records):
            raise ValueError("scale broadcast plan differs from build/shape policy")
        plan["user_ub_bytes"] += selected["broadcast_bytes"] + selected["sdk_reserved_bytes"]
        first = dict(first, scale=selected)
    if version in (3, 4, 5):
        if (any(first.get(key) != value for key, value in plan.items()) or
                (pipeline == "prefetch" and mode != "batched")):
            raise ValueError("launch pipeline selection/buffers differ from the shape policy")
        if (type(first.get("matmul_ub_bytes")) is not int or
                first["matmul_ub_bytes"] <= 0 or first["matmul_ub_bytes"] % 32):
            raise ValueError("invalid Matmul UB budget")
    expected = {"policy": POLICY, "m": m, "k": k, "n": n,
                "tile_n": 64, "tile_k": tile_k, "n_tiles": n // 64, "core_limit": limit}
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
    parser.add_argument("--k-tile",type=int,choices=(128,256),default=128)
    parser.add_argument("--scale-mode", choices=tuple(SCALE_MODES), default="scalar")
    parser.add_argument("--gate-copy-mode", choices=("loop","2d"), default="loop")
    parser.add_argument("--kernel-header", type=Path)
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    configure_build(args.header, args.report, args.core_limit, args.dequant_mode, args.kernel_header, args.pipeline_mode, args.kv_m80_mode, args.scale_mode, args.k_tile, args.gate_copy_mode)


if __name__ == "__main__":
    main()
