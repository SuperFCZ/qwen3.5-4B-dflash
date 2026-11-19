"""Fresh private builds derived from the accepted Raw source/package inventories."""

import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile
import traceback

from w16_common import (
    API,
    DOUBLE,
    GATE,
    GATE8,
    HEADER,
    RAW_HEADER,
    STEM,
    c,
    command,
    gate_kernel,
    host,
    one,
    patch_project,
    render_kernel,
    require,
    sdk_files,
    tree,
)
from build_gate import (
    DEVICE_CMAKE,
    check_root,
    compare_tree,
    copy_snapshot,
    install_raw_outputs,
    preset_options,
)
from gate_plan import elf_sections, sdk_layout
from opp import preflight


def compile_project(project, folder, cann, env):
    build = folder / "device-build"
    require(
        not build.exists() and DEVICE_CMAKE in (project / "CMakeLists.txt").read_text(), "invalid build target"
    )
    cache = folder / "cache"
    cache.mkdir()
    env = dict(env, ASCEND_CACHE_PATH=str(cache))
    command(
        [
            "cmake",
            "-S",
            project,
            "-B",
            build,
            *preset_options(project, cann),
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        ],
        folder / "configure.log",
        env,
    )
    command(
        [
            "cmake",
            "--build",
            build,
            "--target",
            "cust_optiling",
            "cust_op_proto",
            GATE + "_ascend310p",
            "--",
            "-j2",
            "VERBOSE=1",
        ],
        folder / "compile.log",
        env,
    )
    base = build / "op_kernel/binary/ascend310p"
    obj = one(base.rglob("*.o"), "unexpected device target inventory")
    require(obj.parent.name == STEM and elf_sections(obj.read_bytes()), "not a Gate device ELF")
    meta = obj.with_suffix(".json")
    util = project / "cmake/util"
    command([sys.executable, util / "insert_simplified_keys.py", "-p", base], folder / "keys.log", env)
    command(
        [sys.executable, util / "ascendc_ops_config.py", "-p", base, "-s", "ascend310p"],
        folder / "index.log",
        env,
    )
    require(
        c.read(meta)["sha256"] == c.sha(obj) and c.read(meta)["binFileName"] == obj.stem,
        "object metadata mismatch",
    )
    recipes = [
        c.identity(p)
        for p in sorted(build.rglob("*"))
        if p.is_file()
        and (
            p.name == "compile_commands.json" or p.suffix == ".sh" or p.name == "build.make" and GATE in str(p)
        )
    ]
    tools = {
        name: c.identity(shutil.which(name, path=env["PATH"]))
        for name in ("cmake", "c++", "make", "bisheng", "opc")
    }
    require(
        all(cann.resolve() in Path(tools[k]["path"]).parents for k in ("bisheng", "opc")),
        "compiler outside CANN",
    )
    return dict(
        binary=c.identity(obj),
        metadata=c.identity(meta),
        config=c.identity(base / (STEM + ".json")),
        index=c.identity(base / "binary_info_config.json"),
        recipes=recipes,
        tools=tools,
        commands=[c.identity(p) for p in sorted(folder.iterdir()) if p.is_file()],
    )


def build_private(folder, src, raw, variant):
    folder.mkdir(parents=True)
    prior = c.read(c.verify(src["host_source"]))["binding"]["prior"]
    cann = Path(src["cann"])
    old = Path(raw["opp"]["install"]).resolve(strict=True)
    before = raw["package_files"]
    c.verify_opp(raw["opp"])
    require(tree(old) == before, "Raw package changed")
    project = folder / "CustomOp"
    snapshot = raw["generated_sources"]
    copy_snapshot(snapshot, Path(c.verify(snapshot["CMakeLists.txt"]["file"])).parent, project)
    patch_project(project, variant)
    changed = {"op_kernel/" + STEM + ".cpp"}
    if variant in DOUBLE:
        changed.add("op_host/" + STEM + ".cpp")
    require(compare_tree(snapshot, tree(project), changed) == changed, "uncontrolled source delta")
    compile_env, _ = host.envs.for_prior(prior, None)
    require(not compile_env["ASCEND_CUSTOM_OPP_PATH"], "unexpected private build environment")
    compiled = compile_project(project, folder, cann, compile_env)
    install = folder / "opp"
    check_root(install)
    shutil.copytree(old, install, symlinks=True)
    compare_tree(before, tree(install), set())
    env, env_binding = host.envs.for_prior(prior, install)
    if variant in DOUBLE:
        allowed = install_raw_outputs(
            old, install, folder / "device-build", before, raw["kernel"], compiled, folder, env
        )
    else:
        require(variant == "down-2d", "unknown layout experiment")
        allowed = set()
        for old_key, new_key in (("elf", "binary"), ("metadata", "metadata")):
            rel = c.verify(raw["kernel"][old_key]).relative_to(old)
            target = install / rel
            require(
                target.resolve(strict=True) == target and target.name == c.verify(compiled[new_key]).name,
                "wrong Down target",
            )
            shutil.copyfile(c.verify(compiled[new_key]), target)
            allowed.add(str(rel))
    config = install / "vendors/customize/op_impl/ai_core/tbe/kernel/config/ascend310p"
    require(
        c.read(config / (STEM + ".json")) == c.read(c.verify(compiled["config"]))
        and c.read(config / "binary_info_config.json")[GATE] == c.read(c.verify(compiled["index"]))[GATE],
        "SDK dispatch/registry changed",
    )
    dynamic = []
    originals = sorted({c.verify(binding["source"]) for binding in raw["dynamic"]})
    for orig in originals:
        target = (install / orig.relative_to(old)).resolve(strict=True)
        require(
            install in target.parents and c.sha(target) == c.sha(orig), "unbound dynamic Gate implementation"
        )
        target.write_bytes(render_kernel(orig.read_bytes(), variant))
        require(
            c.sha(target) == c.sha(project / "op_kernel" / (STEM + ".cpp"))
            and c.sha(target.parent / HEADER) == RAW_HEADER,
            "packaged source differs from compiled project",
        )
        allowed.add(str(target.relative_to(install)))
        dynamic.append(c.identity(target))
    require(dynamic, "missing ATC dynamic source")
    installed = tree(install)
    delta = compare_tree(before, installed, allowed)
    require(all(str(Path(p["path"]).relative_to(install)) in delta for p in dynamic), "unchanged dynamic code")
    opp = dict(install=str(install), preflight=preflight(install), kind="W16_PRIVATE_" + variant)
    c.verify_opp(opp)
    kernel = gate_kernel(opp)
    require(
        kernel["elf"]["sha256"] == compiled["binary"]["sha256"] != raw["kernel"]["elf"]["sha256"],
        "private ELF mismatch",
    )
    api = c.identity(install / "vendors/customize/op_api/lib/libcust_opapi.so")
    require(api["sha256"] == raw["api"]["sha256"] == API, "API changed")
    manifest = folder / "manifest.json"
    standard = src["standard_manifest"]
    c.write(manifest, host.dsos.with_private(c.read(c.verify(standard)), opp, standard))
    codec = sdk_layout(
        folder / "layout", project, cann, env, c.verify(prior["current"]["evidence"]["tools"]["c++"])
    )
    # Bind the installed libraries after all package replacements. The earlier
    # environment existed solely to run ldd against the staged files.
    env, env_binding = host.envs.for_prior(prior, install)
    host.envs.verify_binding(env_binding)
    return dict(
        opp=opp,
        kernel=kernel,
        package_files=installed,
        generated_sources=tree(project),
        manifest=c.identity(manifest),
        env=env_binding,
        api=api,
        runner=raw["runner"],
        om=raw["om"],
        dynamic=dynamic,
        compiler=compiled,
        changed_files=sorted(delta),
        codec=codec,
        variant=variant,
    )


def local_check(template, parent, cann, down=False):
    out = Path(tempfile.mkdtemp(prefix="down-dma-" if down else "gate-w16-", dir=parent))
    report = dict(status="BLOCKED", variants={}, atc="NOT_RUN", npu="NOT_RUN")
    try:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        report["sdk"] = sdk_files(cann)
        for mode in ("down-2d",) if down else DOUBLE:
            folder = out / mode
            project = folder / "CustomOp"
            shutil.copytree(template, project, symlinks=True)
            patch_project(project, mode)
            built = compile_project(project, folder, cann, env)
            generated_sources = tree(project)
            # The local compiler fixture contains only Gate. The server instead
            # patches the identity-bound joint package, preserving both Head ELFs.
            command(
                ["cmake", "--build", folder / "device-build", "--target", "binary", "--parallel", "2"],
                folder / "binary-target.log",
                env,
            )
            command(
                ["cmake", "--build", folder / "device-build", "--target", "package", "--parallel", "2"],
                folder / "package.log",
                env,
            )
            installer = one((project / "build_out").glob("custom_opp_*.run"), "missing CPack installer")
            install = folder / "private-opp"
            install.mkdir()
            command(
                ["bash", installer, "--install-path=" + str(install)],
                folder / "install.log",
                {k: v for k, v in env.items() if k != "ASCEND_CUSTOM_OPP_PATH"},
            )
            packaged = one(
                (install / "vendors/customize/op_impl/ai_core/tbe/kernel/ascend310p" / STEM).glob("*.o"),
                "expected one installed Gate ELF",
            )
            require(
                c.sha(packaged) == built["binary"]["sha256"]
                and c.read(packaged.with_suffix(".json"))["sha256"] == c.sha(packaged),
                "packaged Gate code drift",
            )
            dynamic = one(install.rglob(STEM + ".cpp"), "local ATC dynamic implementation")
            require(
                c.sha(dynamic) == c.sha(project / "op_kernel" / (STEM + ".cpp"))
                and c.sha(dynamic.parent / HEADER) == RAW_HEADER,
                "local package dynamic implementation changed",
            )
            libs = {}
            for n in ("libcust_opmaster_rt2.0.so", "libcust_opsproto_rt2.0.so"):
                libs[n] = c.identity(one({p.resolve() for p in install.rglob(n)}, n))
            report["variants"][mode] = dict(
                compiler=built,
                host=libs,
                project=generated_sources,
                installer=c.identity(installer),
                private_root=str(install),
                installed_kernel=c.identity(packaged),
                dynamic=c.identity(dynamic),
            )
            from w16_static import disassembler, sections, decode, StaticBlocked

            tool = disassembler(cann)
            asm = folder / "package.disassembly"
            command([c.verify(tool), "-d", packaged], asm, env)
            rec = report["variants"][mode]
            rec.update(package_disassembly=c.identity(asm), objdump=tool)
            try:
                blob = packaged.read_bytes()
                decode(blob, sections(blob), asm.read_text())
                rec["package_decode"] = "DECODED_PACKAGE_ONLY_NOT_ATC"
            except StaticBlocked as exc:
                rec["package_decode"] = str(exc)
        if not down:
            left, right = (report["variants"][v] for v in DOUBLE)
            require(
                left["compiler"]["binary"]["sha256"] != right["compiler"]["binary"]["sha256"],
                "identical package ELFs",
            )
        report["codec"] = sdk_layout(out / "layout", project, cann, env, shutil.which("c++"))
        if down:
            from down_fallback import build_runner

            info = build_runner(out, cann, env)
            report.update(runner=info["runner"], runner_ldd=info["deps"])
        else:
            command(
                [
                    "cmake",
                    "-S",
                    GATE8 / "runner",
                    "-B",
                    out / "runner",
                    "-DCANN_ROOT=" + str(cann),
                    "-DCOMMON_INCLUDE=" + str(c.QUANT / "test"),
                ],
                out / "runner-configure.log",
                env,
            )
            command(["cmake", "--build", out / "runner"], out / "runner-compile.log", env)
            report["runner"] = c.identity(out / "runner/gate_om_runner")
            report["runner_ldd"] = c.dependencies(report["runner"], out / "runner-ldd.log", env)
        report["status"] = "LOCAL_HOST_DEVICE_COMPILED_NPU_NOT_RUN"
    except Exception as exc:
        report["error"] = repr(exc)
        (out / "traceback.txt").write_text(traceback.format_exc())
    c.write(out / "report.json", report)
    print(report["status"], report.get("error"), out / "report.json", flush=True)
    return int(report["status"] == "BLOCKED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline ARM64 compiler check only")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument(
        "--down-only", action="store_true", help="independent Down package and shape-specific Runner"
    )
    parser.add_argument("--cann", type=Path, default=Path("/usr/local/Ascend/cann-9.0.0"))
    args = parser.parse_args()
    sys.exit(local_check(args.template, args.output_parent, args.cann, args.down_only))
