"""Build and measure two new full single-Gate OMs using only repository sources and CANN."""

import argparse
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import traceback

from air import classes
from bench import (
    ABI,
    FIRST_HASHES,
    HERE,
    K,
    M,
    N,
    MODES,
    NZ_MODE,
    TILE_MODE,
    OLD_OP,
    SHAPE_KEY,
    c,
    host,
    require,
    prepare_air,
    compile_gate,
    compile_candidate,
    paired,
    stats,
)
from fresh_dso import check_maps, seal
from fresh_inputs import materialize
from package import command, make_package
from tile_gate_graph import model_proto


def data_files(out, real_dir):
    if real_dir is None:
        directory = out / "inputs"
        materialize(directory)
        label = "SYNTHETIC_SEEDED_C16_GATE"
    else:
        directory = real_dir.resolve(strict=True)
        label = "FROZEN_LAYER0_NATIVE"
    files = {name: c.identity(directory / name) for name in ("x.bin", "w_nz.bin", "s_gn.bin", "golden.bin")}
    if real_dir is not None:
        for name, item in files.items():
            require(
                item["sha256"] == FIRST_HASHES["golden" if name == "golden.bin" else name],
                "unexpected C16 Layer0 tensor: " + str(directory / name),
            )
    return files, label


def runner_binary(out, cann, env, reference):
    # Compile the original guarded single-Gate runner. Give its generated report the actual reference name.
    dest = out / "runner-src"
    source = HERE.parent / "draft_gate_raw_atc8/runner"
    shutil.copytree(source, dest)
    cpp = dest / "runner.cpp"
    original = cpp.read_text()
    require(
        original.count("single Gate frozen Native Golden checked in process") == 1,
        "unknown Gate runner report format",
    )
    cpp.write_text(
        original.replace("single Gate frozen Native Golden checked in process", reference).replace(
            "nonfinite Native Golden", "nonfinite reference"
        )
    )
    build = out / "runner-build"
    command(
        [
            "cmake",
            "-S",
            dest,
            "-B",
            build,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCANN_ROOT=" + str(cann),
            "-DCOMMON_INCLUDE=" + str(c.QUANT / "test"),
        ],
        out / "runner-configure.log",
        env,
        out,
    )
    command(["cmake", "--build", build, "--parallel", "2"], out / "runner-build.log", env, out)
    return c.identity(build / "gate_om_runner")


def run_child(folder, runner, model, env, records, files, timed, reference, round_number):
    folder.mkdir(parents=True)
    for side, names in ABI.items():
        (folder / (side + ".tsv")).write_text(
            "".join(i["name"] + " " + i["dtype"] + " " + " ".join(map(str, i["shape"])) + "\n" for i in names)
        )
    for side, op_name, filename in (("inputs", "x", "x.bin"), ("reference", "y", "golden.bin")):
        d = folder / side
        d.mkdir()
        (d / (op_name + ".bin")).symlink_to(c.verify(files[filename]))
    args = [
        str(c.verify(runner)),
        "0",
        str(c.verify(model["om"])),
        str(folder),
        str(folder / "inputs"),
        str(folder / "reference"),
        "benchmark" if timed else "smoke",
    ]
    parent, child = socket.socketpair()
    log = folder / "runner.log"
    with parent, child, log.open("xb") as stream:
        parent.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        parent.settimeout(600)
        fd = child.fileno()
        endpoint = os.readlink("/proc/self/fd/" + str(fd))
        cmd = args + [str(fd)]
        c.write(folder / "command.json", dict(argv=cmd, runner=runner))
        with subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, pass_fds=(fd,)) as proc:
            child.close()
            try:
                size = struct.calcsize("3i")
                token, control, flags, _ = parent.recvmsg(1, socket.CMSG_SPACE(size))
                creds = [
                    struct.unpack("3i", v)
                    for level, kind, v in control
                    if level == socket.SOL_SOCKET and kind == socket.SCM_CREDENTIALS
                ]
                require(
                    token == b"?"
                    and not flags & socket.MSG_CTRUNC
                    and creds == [(proc.pid, os.getuid(), os.getgid())],
                    "runner exited before authenticated pre-launch gate",
                )
                task = Path("/proc") / str(proc.pid)
                require(
                    c.identity(task / "exe") == runner and os.readlink(task / "fd" / str(fd)) == endpoint,
                    "runner executable or gate endpoint changed",
                )
                live = host.dsos.maps_rows((task / "maps").read_text())
                pre = folder / "pre-maps.txt"
                require(live == host.dsos.maps_rows(pre.read_text()), "live pre-launch mappings differ")
                check_maps(pre, records)
                parent.sendall(b"G")
                require(proc.wait(timeout=600) == 0, "Gate runner failed")
            except BaseException:
                parent.close()
                if proc.poll() is None:
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                sys.stderr.write(log.read_text(errors="replace")[-12000:])
                raise
    check_maps(folder / "post-maps.txt", records)
    execution = c.read(folder / "execution.json")
    require(
        execution["scope"] == reference
        and execution["status"] == "PASS"
        and execution["device"] == 0
        and execution["cpu_fallback"] is False
        and all(
            execution[k] is True
            for k in (
                "guards",
                "readonly",
                "repeat",
                "poison_verified",
                "golden_match",
                "sync_completed",
                "timed_log_stable",
            )
        ),
        "incomplete output/guard validation",
    )
    for name in ["actual-0", "actual-1"] + (["benchmark-last", "postcheck"] if timed else []):
        require(
            c.sha(folder / ("y-" + name + ".bin")) == files["golden.bin"]["sha256"],
            "Gate FP16 output differs from reference: " + name,
        )
    t = execution["timing"]
    require(
        t["warmup"] == (5 if timed else 0)
        and t["repetitions"] == (30 if timed else 0)
        and execution["verification_calls"] == (4 if timed else 2)
        and t["correctness_before_calls"] == 2
        and t["correctness_after_calls"] == int(timed),
        "unexpected measurement protocol",
    )
    return dict(
        round=round_number,
        mode=model["mode"],
        folder=str(folder),
        execution=c.identity(folder / "execution.json"),
        launch_authorized=True,
        bit_mismatches=0,
        profiled=False,
        validation={"timing": stats(t["execute_sync"]["samples_ms"], 30) if timed else None},
    )


def environment(cann, tools, opp=None):
    env, _ = host.envs.make_environment(cann, opp, tools=tools)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    require(
        Path(shutil.which("python3", path=env["PATH"])).resolve() == Path(sys.executable).resolve(),
        "ATC/build child would use a different Python",
    )
    for key in ("AI_RUN_DIR", "ASCEND_WORK_PATH", "ASCEND_PROCESS_LOG_PATH", "ASCEND_CACHE_PATH"):
        env[key] = os.environ[key]
    return env


def compile_pair(out, cann, base, tools, files, reference, result):
    nz_dir = out / "nz-atc"
    nz_dir.mkdir()
    air = nz_dir / "gate.air"
    proto = model_proto(
        classes(cann, nz_dir),
        c.verify(files["w_nz.bin"]).read_bytes(),
        c.verify(files["s_gn.bin"]).read_bytes(),
    )
    air.write_bytes(proto.SerializeToString())
    del proto
    nd_dir = out / "tile-atc"
    packing = prepare_air(nd_dir, {"cann": str(cann)}, base, air, files)
    result.update(packing=packing, phase="build")
    # The wrapper executes the same argv in each private vendor/cache, with only the model/output differing.
    argv = [
        str(cann / "bin/atc"),
        "--mode=0",
        "--framework=1",
        "--model=" + str(air),
        "--output=" + str(nz_dir / "gate"),
        "--soc_version=Ascend310P3",
        "--precision_mode=must_keep_origin_dtype",
        "--deterministic=0",
    ]
    runner = runner_binary(out, cann, base, reference)
    packages = {}
    models = {}
    result.update(runner=runner, models=models, packages=packages)
    for mode in MODES:
        d = out / ("build-" + mode)
        d.mkdir()
        result["phase"] = "build " + mode
        print("Building private OPP:", mode, flush=True)
        pkg = make_package(d, None, cann, base, tiled=mode == TILE_MODE)
        env = environment(cann, tools, pkg["opp"]["install"])
        result["phase"] = "ATC " + mode
        print("Compiling single-op OM:", mode, flush=True)
        if mode == NZ_MODE:
            model = compile_gate(nz_dir, pkg, env, argv, OLD_OP, files["w_nz.bin"])
        else:
            model, _ = compile_candidate(nd_dir, models[NZ_MODE], pkg, env, argv, packing)
        models[mode] = model
        packages[mode] = pkg


def run(out, cann, real_dir, result, compile_only, reuse=None):
    tools = {}
    for name in ("msopgen", "atc", "cmake"):
        alias = shutil.which(name)
        require(alias is not None, "missing SDK/build tool on PATH: " + name)
        tools[name] = c.identity(alias)
    base = environment(cann, tools)
    result["phase"] = "inputs"
    files, label = data_files(out, real_dir)
    reference = "frozen Native Layer0 Golden" if real_dir else "seeded exact integer CPU reference"
    result.update(inputs=files, data_source=label, tools=tools)
    print(f"{label} | FP16 [{M},{K}] x W8 [{N},{K}] | 311296 output values", flush=True)
    if reuse is None:
        compile_pair(out, cann, base, tools, files, reference, result)
    else:
        from reuse_build import load_build

        result["phase"] = "verify saved Gate build"
        old, receipt = load_build(reuse, tools, files, label)
        result.update({key: old[key] for key in ("runner", "packing", "models", "packages")})
        result["reused_compile_report"] = receipt
        print("Verified and reused two Gate OMs; no rebuild:", reuse, flush=True)
    runner, packages, models = result["runner"], result["packages"], result["models"]
    print("ATC: both Gate/Up OMs use 8 blocks and identical serialized Cube tiling.", flush=True)
    if compile_only:
        return "COMPILED_NOT_EXECUTED"
    result["phase"] = "pre-launch DSO binding"
    envs = {mode: environment(cann, tools, pkg["opp"]["install"]) for mode, pkg in packages.items()}
    manifests = {mode: seal(out / ("dso-" + mode), packages[mode], runner, envs[mode]) for mode in MODES}
    result["smoke"] = []
    result["runs"] = []
    for i in (-1, 0, 1, 2):
        for mode in MODES if i in (-1, 0, 2) else MODES[::-1]:
            result["phase"] = ("smoke" if i < 0 else f"round {i+1}") + " " + mode
            print(result["phase"], flush=True)
            c.verify_opp(packages[mode]["opp"])
            folder = out / ("smoke" if i < 0 else f"round-{i+1}") / mode
            row = run_child(
                folder, runner, models[mode], envs[mode], manifests[mode], files, i >= 0, reference, i
            )
            result["smoke" if i < 0 else "runs"].append(row)
    result["timing"] = paired(result["runs"])
    c.verify_tree(files)
    return "MEASURED"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layer0-inputs", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--reuse-build", type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(exist_ok=False)
    result = {"status": "FAILED", "phase": "setup"}
    try:
        result["status"] = run(
            out,
            Path(os.environ["CANN_ROOT"]).resolve(strict=True),
            args.layer0_inputs,
            result,
            args.compile_only,
            args.reuse_build,
        )
    except Exception as error:
        result["error"] = repr(error)
        traceback.print_exc()
    c.write(out / "result.json", result)
    if "timing" in result:
        timing = result["timing"]
        print("Full single Gate/Up Execute+Sync | " + result["data_source"])
        for mode in MODES:
            t = timing["variants"][mode]
            print(f"{mode:20s} median={t['median_ms']:.6f} ms  P95={t['p95_ms']:.6f} ms  n={t['count']}")
        print(f"NZ / tile-ND = {timing['speedup']:.5f}x | latency reduction {timing['reduction_pct']:.2f}%")
    print(result["status"], "REPORT", out / "result.json", flush=True)
    return int(result["status"] == "FAILED")


if __name__ == "__main__":
    sys.exit(main())
