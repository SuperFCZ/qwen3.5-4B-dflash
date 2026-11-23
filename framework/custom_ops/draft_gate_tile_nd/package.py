"""Build one new GE operator into a private vendor tree using CANN 9.0."""

from pathlib import Path
import shutil
import subprocess
import sys

from layout import HEADER, HEADER_SHA, HERE, OLD_OP, OLD_STEM, OP, STEM, host_source, kernel_source, require

sys.path.insert(0, str(HERE.parent / "draft_gate_scale_broadcast"))
from scale_contracts import c, render
from a3_launch import configure_build
from flow_build import preset_options


def command(argv, log, env, cwd):
    with log.open("xb") as f:
        rc = subprocess.run(
            [str(arg) for arg in argv], cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT
        ).returncode
    if rc:
        sys.stderr.write(log.read_text(errors="replace"))
        raise RuntimeError(f"{argv[0]} returned {rc}, log: {log}")


def tracked_template(out, preset, cann):
    source = out / "gate-template"
    host = source / "op_host"
    dev = source / "op_kernel"
    host.mkdir(parents=True)
    dev.mkdir()
    flow = HERE.parent / "draft_a_core_scheduler_flow/op_host"
    for name in (OLD_STEM + ".cpp", "platform_diag.h", "scheduler_evidence.h"):
        shutil.copyfile(flow / name, host / name)
    for name in (OLD_STEM + "_tiling.h", OLD_STEM + "_contract.h"):
        shutil.copyfile(c.QUANT / "op_host" / name, host / name)
    (host / "scheduler_variant.h").write_text("#pragma once\n#define DFLASH_SCHEDULER_POLICY 0\n")
    configure_build(
        host / HEADER,
        source / "build-config.json",
        0,
        "batched",
        dev / HEADER,
        "prefetch",
        "a-ub",
        "broadcast",
        256,
        "loop",
        "raw",
    )
    (dev / (OLD_STEM + ".cpp")).write_bytes(
        render((c.QUANT / "op_kernel" / (OLD_STEM + ".cpp")).read_bytes())
    )
    data = c.read(preset)
    (default,) = [p for p in data["configurePresets"] if p["name"] == "default"]
    for key, value in dict(
        ASCEND_CANN_PACKAGE_PATH=str(cann),
        ASCEND_COMPUTE_UNIT="ascend310p",
        ASCEND_PYTHON_EXECUTABLE=sys.executable,
        ENABLE_BINARY_PACKAGE="True",
        ENABLE_SOURCE_PACKAGE="True",
        ENABLE_CROSS_COMPILE="False",
        ASCEND_PACK_SHARED_LIBRARY="False",
    ).items():
        default["cacheVariables"][key]["value"] = value
    c.write(source / "CMakePresets.json", data)
    return {str(p.relative_to(source)): {"file": c.identity(p)} for p in source.rglob("*") if p.is_file()}


def make_package(out, snapshot, cann, env, *, tiled=True):
    project = out / "CustomOp"
    install = out / "opp"
    op, stem = (OP, STEM) if tiled else (OLD_OP, OLD_STEM)
    schema = HERE / (op + ".json") if tiled else c.QUANT / (op + ".json")
    msopgen = shutil.which("msopgen", path=env["PATH"])
    require(msopgen, "msopgen not found in CANN PATH")
    with Path(msopgen).open("rb") as fp:
        shebang = fp.readline(256)
    generator = [sys.executable, msopgen] if shebang.startswith(b"#!") and b"python" in shebang else [msopgen]
    command(
        [
            *generator,
            "gen",
            "-i",
            schema,
            "-c",
            "ai_core-Ascend310P3",
            "-lan",
            "cpp",
            "-out",
            project,
        ],
        out / "generate.log",
        env,
        out,
    )

    if snapshot is None:
        snapshot = tracked_template(out, project / "CMakePresets.json", cann)

    def original(rel):
        return c.verify(snapshot[rel]["file"])

    dev = project / "op_kernel" / (stem + ".cpp")
    reg = project / "op_host" / (stem + ".cpp")
    require(dev.is_file() and reg.is_file(), "unexpected generated op or kernel name")
    b16 = original("op_kernel/" + OLD_STEM + ".cpp").read_bytes()
    dev.write_bytes(kernel_source(b16) if tiled else b16)
    host = original("op_host/" + OLD_STEM + ".cpp").read_text()
    reg.write_text(host_source(host) if tiled else host)
    for sub in ("op_host", "op_kernel"):
        source = original(sub + "/" + HEADER)
        require(c.sha(source) == HEADER_SHA, "not the accepted Gate B16 build header")
        shutil.copyfile(source, project / sub / HEADER)
    tiling = original("op_host/" + OLD_STEM + "_tiling.h").read_text().replace(OLD_OP, op)
    (project / "op_host" / (stem + "_tiling.h")).write_text(tiling)
    for name in (
        OLD_STEM + "_contract.h",
        "scheduler_variant.h",
        "scheduler_evidence.h",
        "platform_diag.h",
        "host_evidence.h",
    ):
        if "op_host/" + name in snapshot:
            shutil.copyfile(original("op_host/" + name), project / "op_host" / name)
    shutil.copyfile(original("CMakePresets.json"), project / "CMakePresets.json")
    build = out / "build"
    env = dict(env, ASCEND_CACHE_PATH=str(out / "build-cache"))
    Path(env["ASCEND_CACHE_PATH"]).mkdir()
    command(
        [
            "cmake",
            "-S",
            project,
            "-B",
            build,
            *preset_options(project, cann),
            "-DCMAKE_INSTALL_PREFIX=" + str(project / "build_out"),
        ],
        out / "configure.log",
        env,
        out,
    )
    for target in ("binary", "package"):
        command(
            ["cmake", "--build", build, "--target", target, "--parallel", "2"],
            out / (target + ".log"),
            env,
            out,
        )
    installers = list((project / "build_out").glob("custom_opp_*.run"))
    require(len(installers) == 1, "expected one isolated Gate operator installer")
    install.mkdir()
    command(["bash", installers[0], "--install-path=" + str(install)], out / "install.log", env, out)
    vendor = install / "vendors/customize"
    binaries = list((vendor / "op_impl/ai_core/tbe/kernel/ascend310p").rglob("*.o"))
    require(len(binaries) == 1 and binaries[0].parent.name == stem, "unexpected private kernel inventory")
    elf = c.identity(binaries[0])
    meta = c.identity(binaries[0].with_suffix(".json"))
    require(c.read(c.verify(meta))["sha256"] == elf["sha256"], "installed SDK kernel metadata mismatch")
    dyn = list(vendor.rglob(stem + ".cpp"))
    require(
        len(dyn) == 1 and c.sha(dyn[0]) == c.sha(dev) and c.sha(dyn[0].parent / HEADER) == HEADER_SHA,
        "ATC dynamic source/header differs from built candidate",
    )
    files = [
        dict(path=str(p), sha256=c.sha(p))
        for p in sorted(vendor.rglob("*"))
        if p.is_file() and (p.suffix in (".h", ".so", ".json") or ".so." in p.name)
    ]
    row = dict(
        op=op,
        binary=elf["path"],
        sha256=elf["sha256"],
        bytes=elf["bytes"],
        json=meta["path"],
        json_sha256=meta["sha256"],
        json_bytes=meta["bytes"],
    )
    opp = dict(
        kind="PRIVATE_GATE_TILE_ND" if tiled else "PRIVATE_GATE_NZ_B16",
        install=str(install),
        preflight=dict(vendor=str(vendor), files=files, kernels=[row]),
    )
    c.verify_opp(opp)
    return dict(
        opp=opp, kernel=dict(elf=elf, metadata=meta), source=c.identity(dyn[0]), host_source=c.identity(reg)
    )
