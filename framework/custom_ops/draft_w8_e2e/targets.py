"""Recover matching 2048 OMs when present, otherwise rebuild Prefill and Verify from the W8A8 checkpoint."""

import os
from pathlib import Path
import re
import subprocess
import tarfile
from winners import c, require, HERE, OPS, REPO, UNION
from history import TARGET_SHA
from package import command

NATIVE = "319f51d0c51352c7f27f739e845ba93101f1ce8f"
HOME = Path("/home/w00949577")
RECEIVER = HOME / "dflash-gdr-chunk-om"
NEEDED = {
    "ChunkGatedDeltaRule",
    "AdnFusedInferAttention",
    "AdnRmsNorm",
    "CacheUpdate",
    "DynamicQuant",
    "QuantBatchMatmulV4444",
}


def prototypes(vendor):
    ops = set()
    for p in (vendor / "op_proto/inc").glob("*.h"):
        txt = p.read_text(errors="replace")
        for name, body in re.findall(r"REG_OP\s*\(\s*(\w+)\s*\)(.*?)\.OP_END_FACTORY_REG", txt, re.S):
            if name == "ChunkGatedDeltaRule" and ".INPUT(effective_length," not in re.sub(r"\s+", "", body):
                continue
            if name in NEEDED:
                ops.add(name)
    return ops


def receiver_vendors(cann):
    manual = os.getenv("DFLASH_TARGET_OPP_PATH")
    if manual:
        dirs = [Path(p).resolve(strict=True) for p in manual.split(os.pathsep) if p]
        require(
            all((p / "op_proto").is_dir() for p in dirs), "use colon-separated receiver vendor directories"
        )
        return dirs
    root = RECEIVER / "artifacts"
    deploy = c.read(root / "deployment-manifest.json")
    ref = deploy["air_manifest"]
    air = root / ref["path"]
    require(c.sha(air) == ref["sha256"], "older receiver AIR manifest changed")
    found = []
    preferred = []
    for p in (c.read(air).get("operator_preflight") or {}).get("ge_prototypes", []):
        if p.get("prototype_path"):
            h = Path(p["prototype_path"])
            if h.is_file() and c.sha(h) == p.get("prototype_sha256"):
                preferred.append(h.parents[2].resolve())

    def add(v):
        if v.is_dir() and v.name == "vendors":
            found.extend(sorted(p.resolve() for p in v.iterdir() if p.is_dir()))

    for v in preferred:
        add(v.parent)
    add(cann / "opp/vendors")
    # Receiver packages often live outside CANN. Inspect only installed vendor trees, never compiler caches.
    for root_, dirs, _ in os.walk(HOME):
        p = Path(root_)
        if p.name == "vendors":
            add(p)
            dirs[:] = []
            continue
        dirs[:] = [
            s
            for s in dirs
            if s
            not in {
                ".git",
                ".cache",
                "envs",
                "site-packages",
                "node_modules",
                "model_weights",
                "runs",
                "build",
                "compiler-cache",
                "__pycache__",
            }
        ]
    selected = []
    covered = set()
    for v in dict.fromkeys(preferred + found):
        names = prototypes(v)
        if not names - covered:
            continue
        if names & covered:
            continue
        selected.append(v)
        covered.update(names)
    # Some receiver releases put DynamicQuant/CacheUpdate in ops-info and an
    # ACLNN library without a REG_OP declaration in the public headers.
    for v in list(selected):
        for pkg in ("customize_dynamic", "customize_scatter"):
            sibling = v.parent / pkg
            if (
                sibling not in selected
                and (sibling / "op_api/lib").is_dir()
                and (sibling / "op_impl").is_dir()
            ):
                selected.append(sibling)
    print("Target receiver vendors", list(map(str, selected)), sorted(covered), flush=True)
    return selected


def receiver_env(cann, env, paths):
    v = [str(p) for p in paths]
    a = dict(env, ASCEND_CUSTOM_OPP_PATH=os.pathsep.join(v))
    lib = [str(p / "op_api/lib") for p in paths if (p / "op_api/lib").is_dir()]
    a["LD_LIBRARY_PATH"] = os.pathsep.join(
        dict.fromkeys(lib + a.get("LD_LIBRARY_PATH", "").split(os.pathsep))
    )
    return a


def materialize_snapshot(dest):
    pack = dest / "native.tar"
    native = dest / "native"
    native.mkdir()
    subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", "-o", str(pack), NATIVE], check=True)
    with tarfile.open(pack) as ar:
        for item in ar:
            name = Path(item.name)
            require(
                not name.is_absolute() and ".." not in name.parts and (item.isdir() or item.isfile()),
                "unexpected entry in native snapshot",
            )
            ar.extract(item, native)
    return native


def exact_oms(history, explicit=None):
    result = {}
    for name, sha in TARGET_SHA.items():
        previous = Path(history["graphs"][name]["path"])
        p = Path(explicit) / previous.name if explicit else previous
        if p.is_file() and c.sha(p) == sha:
            result[name] = dict(
                om=c.identity(p),
                abi=history["graphs"][name]["abi"],
                target_layers=list(UNION),
                origin="HISTORICAL_OM_SHA_MATCH",
            )
        elif explicit:
            raise ValueError(f"expected historical {name} OM SHA {sha} at {p}")
    return result


def target_config():
    old = c.read(RECEIVER / "factory.json")
    cfg = dict(
        old,
        max_sequence_length=2048,
        include_ordinary_decode=False,
        shared_draft_features=True,
        verify_gdr="chunk",
        draft_quantization="w8a16",
        draft_quant_matmul="weight_quant",
        draft_attention_matmul_dtype="float16",
        draft_dir=os.environ.get(
            "DFLASH_DRAFT_CHECKPOINT", str(HOME / "model_weights/qwen3.5-4b-dflash-w8a16")
        ),
        target_dir=os.environ.get("TARGET_DIR", str(HOME / "model_weights/Qwen3.5-4B")),
        quant_config=os.environ.get("DFLASH_TARGET_QUANT", old["quant_config"]),
        receiver_models_dir=os.environ.get("DFLASH_RECEIVER_MODELS", old["receiver_models_dir"]),
        device="npu:0",
    )
    cfg.pop("input_manifest", None)
    require(
        c.read(Path(cfg["draft_dir"]) / "config.json")["dflash_config"]["target_layer_ids"]
        == [1, 8, 15, 22, 29],
        "wrong W8 checkpoint",
    )
    return cfg


def export_target(d, env, cfg):
    native = materialize_snapshot(d)
    conf = d / "factory.json"
    c.write(conf, cfg)
    e = dict(env, AI_RUN_DIR=str(d), TORCH_DEVICE_BACKEND_AUTOLOAD="0", PYTHONNOUSERSITE="1")
    # Spawned TorchAir workers see the same verified source tree and receiver model imports.
    e["PYTHONPATH"] = os.pathsep.join(
        [
            str(native / "framework/python"),
            str(native),
            str(Path(cfg["receiver_models_dir"]).parent),
            e.get("PYTHONPATH", ""),
        ]
    )
    cmd = [
        "bash",
        str(HERE / "target_env.sh"),
        str(HERE / "target_worker.py"),
        "--native",
        str(native),
        "--out",
        str(d),
        "--config",
        str(conf),
    ]
    c.write(d / "command.json", cmd)
    with (d / "export.log").open("x") as log:
        with subprocess.Popen(
            cmd, cwd=d, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace"
        ) as proc:
            for line in proc.stdout:
                log.write(line)
                print(line, end="", flush=True)
            if proc.wait():
                raise RuntimeError(f"Target AIR export failed; see {d/'export.log'}")
    rows = c.read(d / "exported.json")
    require(set(rows) == set(TARGET_SHA), "wrong Target export set")
    return rows


def compile_target(d, name, source, history, env, atc):
    require(
        source["abi"] == history["graphs"][name]["abi"] and source["target_layers"] == list(UNION),
        "different documented Target interface: " + name,
    )
    c.verify_tree(source["payloads"])
    air = c.verify(source["air"])
    prefix = d / name
    cmd = [
        str(atc),
        "--mode=0",
        "--framework=1",
        "--model=" + str(air),
        "--output=" + str(prefix),
        "--soc_version=Ascend310P3",
        "--precision_mode=must_keep_origin_dtype",
    ]
    env = dict(env, ASCEND_CACHE_PATH=str(d / "cache"), ASCEND_WORK_PATH=str(d))
    Path(env["ASCEND_CACHE_PATH"]).mkdir()
    c.write(d / "command.json", cmd)
    command(cmd, d / "atc.log", env, air.parent)
    om = c.identity(prefix.with_suffix(".om"))
    return dict(
        om=om,
        abi=source["abi"],
        target_layers=source["target_layers"],
        origin="REBUILT_W8A8_TARGET",
        source_air=source["air"],
        historical_om_sha256=TARGET_SHA[name],
        historical_binary_match=om["sha256"] == TARGET_SHA[name],
        argv=cmd,
    )
