"""Build the documented Target Prefill/Verify pair without running any comparison workload."""

import argparse
import os
from pathlib import Path
import shutil
import traceback

from winners import c, require, HERE
from history import read_target_contract, SUITE, TARGET_SHA
from targets import (
    receiver_vendors,
    receiver_env,
    target_config,
    export_target,
    compile_target,
    exact_oms,
    NATIVE,
)
from experiment import Stages
from run_q import run_path
from receiver_air import receiver_specs, reconstruct
from air import protobuf_classes
from cache_backend import provider, SHARED, audit_compiled


def prepare(out, st, history, state, explicit=None):
    def stage(name, key, action):
        state["phase"] = name
        return st.run(name, key, action)

    cann = Path(os.environ["CANN_ROOT"]).resolve(strict=True)
    paths = receiver_vendors(cann)
    target = exact_oms(history, explicit)
    rebuilding = set(TARGET_SHA) - set(target)
    cache = None
    if rebuilding:
        requested = os.environ.get("DFLASH_CACHE_UPDATE_OPP")
        existing = next((p for p in paths if p.name == "customize_scatter"), SHARED)
        cache = provider(requested or existing)
        root = Path(cache["path"])
        conflicts = [
            str(p)
            for p in paths
            if p.resolve() != root and (p / "op_api/include/aclnn_cache_update.h").is_file()
        ]
        require(not conflicts, "another CacheUpdate provider is active: " + str(conflicts))
        paths = [root, *[p for p in paths if p != root]]
        print("Target CacheUpdate", root, cache["signature"], flush=True)
    env = receiver_env(cann, os.environ, paths)
    route = os.environ.get("DFLASH_TARGET_BUILD_ROUTE", "receiver-air")
    require(route in ("receiver-air", "torchair"), "unknown Target rebuild route")
    if len(target) < 2 and route == "receiver-air":
        sources = receiver_specs(history)
        desc = c.identity(out / "cann-ge.desc")
        types = protobuf_classes(c.verify(desc).read_bytes())
        atc = Path(shutil.which("atc")).resolve(strict=True)
        exported = {}
        for name, src in sources.items():
            if name in target:
                continue
            exported[name] = stage(
                "reconstruct-" + name,
                dict(
                    source=src["air"],
                    manifest=src["manifest"],
                    schema=desc["sha256"],
                    abi=src["abi"],
                    code=c.sha(HERE / "receiver_air.py"),
                ),
                lambda d: reconstruct(d, types, name, src),
            )
        for name, rewritten in exported.items():
            target[name] = stage(
                "compile-" + name,
                dict(
                    air=rewritten["air"],
                    audit=rewritten["audit"],
                    vendors=list(map(str, paths)),
                    cache_update=cache,
                    tool=c.identity(atc),
                    comparison_plan=history["plan"],
                    code=c.sha(HERE / "targets.py"),
                ),
                lambda d: compile_target(d, name, rewritten, history, env, atc),
            )
    elif len(target) < 2:
        cfg = target_config()
        py = Path(os.getenv("DFLASH_TARGET_PYTHON", "/root/anaconda3/envs/torch_npu_py3.11/bin/python"))
        keys = dict(
            config=cfg,
            native=NATIVE,
            base_python=c.identity(py),
            checkpoint=c.identity(Path(cfg["draft_dir"]) / "config.json"),
            quant_config=c.identity(cfg["quant_config"]),
            receiver=c.identity(Path(cfg["receiver_models_dir"]) / "export_model_wrapper_qwen3_5.py"),
            env=[str(p) for p in paths],
            code=[c.sha(HERE / p) for p in ("target_worker.py", "target_env.sh", "targets.py")],
        )
        exported = stage("target-export", keys, lambda d: export_target(d, env, cfg))
        atc = Path(shutil.which("atc")).resolve(strict=True)
        for name in TARGET_SHA:
            if name not in target:
                src = exported[name]
                target[name] = stage(
                    "compile-" + name,
                    dict(
                        air=src["air"],
                        export=st.refs["target-export"],
                        tool=c.identity(atc),
                        historical=history["graphs"][name]["sha256"],
                        cache_update=cache,
                        code=c.sha(HERE / "targets.py"),
                    ),
                    lambda d: compile_target(d, name, src, history, env, atc),
                )
    for name in sorted(rebuilding):
        om = target[name]["om"]
        proof = stage(
            "cache-backend-" + name,
            dict(om=om, provider=cache, atc=c.identity(atc), code=c.sha(HERE / "cache_backend.py")),
            lambda d: audit_compiled(d, name, c.verify(om), atc, env, cache["path"]),
        )
        c.verify(proof)
        target[name] = dict(target[name], cache_backend=proof)
    c.verify_tree([t["om"] for t in target.values()])
    result = dict(
        models=target, vendors=list(map(str, paths)), cache_provider=cache, historical_index=history["index"]
    )
    c.write(out / "target.json", result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resume")
    args = p.parse_args()
    out = Path(os.environ["DFLASH_OUTPUT"])
    runs = out.parent
    state = dict(status="STARTING", phase="archive")
    try:
        prior = run_path(runs, args.resume, "e2e-target-") if args.resume else None
        st = Stages(out, prior)
        h = read_target_contract(SUITE)
        prepare(out, st, h, state, os.getenv("DFLASH_E2E_TARGET_OM_DIR"))
        state.update(status="TARGET_READY", artifact=c.identity(out / "target.json"))
    except Exception as e:
        traceback.print_exc()
        state.update(status="FAILED", error=repr(e))
        (out / "traceback.txt").write_text(traceback.format_exc())
    c.write(out / "result.json", state)
    print(state["status"], "REPORT", out / "result.json", flush=True)
    return int(state["status"] == "FAILED")


if __name__ == "__main__":
    raise SystemExit(main())
