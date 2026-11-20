"""Create a fresh candidate package using the frozen single-W project and host binaries."""

import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile
import traceback

from scale_contracts import (
    API,
    CANDIDATE,
    GATE,
    HEADER,
    RAW_HEADER,
    STEM,
    c,
    candidate_source,
    command,
    gate_kernel,
    host,
    one,
    read_baseline,
    render,
    require,
    resume,
    sdk_files,
    sources,
    tree,
)
from build_gate import copy_snapshot, compare_tree, check_root
from w16_build import compile_project
from opp import preflight
from gate_plan import elf_sections


def patch_project(project):
    p = project / "op_kernel" / (STEM + ".cpp")
    for side in ("op_host", "op_kernel"):
        require(c.sha(project / side / HEADER) == RAW_HEADER, "wrong Raw build header")
    p.write_bytes(render(p.read_bytes()))


def expected_package_changes(raw):
    root = Path(raw["opp"]["install"])
    return {
        str(c.verify(x).relative_to(root))
        for x in (raw["kernel"]["elf"], raw["kernel"]["metadata"], *(r["source"] for r in raw["dynamic"]))
    }


def validate_package(b, raw, src):
    root = Path(b["opp"]["install"])
    old = Path(raw["opp"]["install"])
    require(
        root != old and root not in old.parents and old not in root.parents, "private OPP overlaps control"
    )
    c.verify_opp(b["opp"])
    require(
        b["package_files"] == tree(root)
        and compare_tree(raw["package_files"], b["package_files"], expected_package_changes(raw))
        == expected_package_changes(raw),
        "unexpected candidate package differences",
    )
    generated = b["generated_sources"]
    project = c.verify(generated["CMakeLists.txt"]["file"]).parent
    name = "op_kernel/" + STEM + ".cpp"
    expected = candidate_source(raw)
    require(
        tree(project) == generated
        and compare_tree(raw["generated_sources"], generated, {name}) == {name}
        and c.verify(generated[name]["file"]).read_bytes() == expected,
        "candidate source drift",
    )
    require(
        b["variant"] == CANDIDATE
        and b["kernel"] == gate_kernel(b["opp"])
        and b["kernel"]["elf"]["sha256"]
        == b["compiler"]["binary"]["sha256"]
        != raw["kernel"]["elf"]["sha256"]
        and b["kernel"]["metadata"]["sha256"] == b["compiler"]["metadata"]["sha256"],
        "wrong device build",
    )
    control_meta = c.read(c.verify(raw["kernel"]["metadata"]))
    changed_meta = c.read(c.verify(b["kernel"]["metadata"]))
    require(
        changed_meta == dict(control_meta, sha256=b["kernel"]["elf"]["sha256"])
        and elf_sections(c.verify(raw["kernel"]["elf"]).read_bytes())
        != elf_sections(c.verify(b["kernel"]["elf"]).read_bytes()),
        "changed package ABI/resources, or no change in executable code",
    )
    require(
        b["api"]["sha256"] == raw["api"]["sha256"] == API
        and b["runner"] == raw["runner"]
        and b["om"] == raw["om"],
        "runner or ABI drift",
    )
    for original, selected in zip(raw["dynamic"], b["dynamic"]):
        for field in ("source", "header", "wrapper"):
            path = c.verify(selected[field])
            original_path = c.verify(original[field])
            require(path == root / original_path.relative_to(old), "mixed dynamic lookup: " + field)
            if field == "source":
                require(path.read_bytes() == expected, "ATC did not get the candidate source")
            else:
                require(c.sha(path) == original[field]["sha256"], "changed dispatcher/header")
        require(
            selected["wrapper"]["sha256"]
            == "2fdf2abe2a51d8cd6faf09e3c70749093d69f1c80dd9be0b1de3bd27f2f4c4c7",
            "unreviewed dynamic dispatcher",
        )
    require(len(b["dynamic"]) == len(raw["dynamic"]) > 0, "dynamic inventory differs")
    compiler = b["compiler"]
    c.verify_tree(compiler)
    cfg = root / "vendors/customize/op_impl/ai_core/tbe/kernel/config/ascend310p"
    require(
        c.read(cfg / (STEM + ".json")) == c.read(c.verify(compiler["config"]))
        and c.read(cfg / "binary_info_config.json")[GATE] == c.read(c.verify(compiler["index"]))[GATE],
        "SDK Gate dispatch/index differs",
    )
    require(host.envs.standard_identity(b["env"]) == src["standard_env"], "candidate host environment drift")
    resume.combined(b, src["cann"], src["extras"], c.read(resume.POLICY))


def build_private(out, src, raw):
    out.mkdir()
    old = Path(raw["opp"]["install"])
    project = out / "CustomOp"
    snapshot = raw["generated_sources"]
    copy_snapshot(snapshot, c.verify(snapshot["CMakeLists.txt"]["file"]).parent, project)
    patch_project(project)
    prior = c.read(c.verify(src["host_source"]))["binding"]["prior"]
    env, _ = host.envs.for_prior(prior, None)
    built = compile_project(project, out, Path(src["cann"]), env)
    install = out / "opp"
    check_root(install)
    shutil.copytree(old, install, symlinks=True)
    compare_tree(raw["package_files"], tree(install), set())
    for was, new in (("elf", "binary"), ("metadata", "metadata")):
        target = install / c.verify(raw["kernel"][was]).relative_to(old)
        require(
            target.resolve(strict=True) == target and target.name == c.verify(built[new]).name,
            "wrong device destination",
        )
        shutil.copyfile(c.verify(built[new]), target)
    dynamic = []
    for row in raw["dynamic"]:
        stage = {k: install / c.verify(row[k]).relative_to(old) for k in ("source", "wrapper", "header")}
        require(stage["source"].resolve(strict=True) == stage["source"], "redirected dynamic source")
        stage["source"].write_bytes(candidate_source(raw))
        dynamic.append({k: c.identity(p) for k, p in stage.items()})
    opp = dict(install=str(install), preflight=preflight(install), kind="PRIVATE_GATE_SCALE_TRANSPOSE")
    _, env_id = host.envs.for_prior(prior, install)
    manifest = out / "manifest.json"
    c.write(
        manifest,
        host.dsos.with_private(c.read(c.verify(src["standard_manifest"])), opp, src["standard_manifest"]),
    )
    b = dict(
        variant=CANDIDATE,
        opp=opp,
        env=env_id,
        manifest=c.identity(manifest),
        dynamic=dynamic,
        generated_sources=tree(project),
        package_files=tree(install),
        compiler=built,
        kernel=gate_kernel(opp),
        api=c.identity(install / "vendors/customize/op_api/lib/libcust_opapi.so"),
        runner=raw["runner"],
        om=raw["om"],
    )
    validate_package(b, raw, src)
    return b


def local_compile(template, parent, cann):
    out = Path(tempfile.mkdtemp(prefix="gate-scale-compile-", dir=parent)).resolve()
    report = dict(
        scope="LOCAL_GATE_PACKAGE_ONLY", atc="NOT_RUN", device_execution="NOT_RUN", status="BLOCKED"
    )
    try:
        project = out / "CustomOp"
        shutil.copytree(template, project, symlinks=True)
        patch_project(project)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", ASCEND_CUSTOM_OPP_PATH="")
        report["sdk"] = sdk_files(cann)
        report["compiler"] = compile_project(project, out, cann, env)
        for target in ("binary", "package"):
            command(
                ["cmake", "--build", out / "device-build", "--target", target, "--parallel", "2"],
                out / (target + ".log"),
                env,
                cwd=out,
            )
        installer = one((project / "build_out").glob("custom_opp_*.run"), "missing private installer")
        install = out / "private-opp"
        install.mkdir()
        command(["bash", installer, "--install-path=" + str(install)], out / "install.log", env, cwd=out)
        binary = one(install.rglob("DFlashGroupQuantLinear_*.o"), "not one local Gate image")
        header = one(install.rglob(HEADER), "ambiguous local dynamic header")
        cpp = one(install.rglob(STEM + ".cpp"), "ambiguous local dynamic source")
        require(
            c.sha(binary) == report["compiler"]["binary"]["sha256"]
            and c.read(binary.with_suffix(".json"))["sha256"] == c.sha(binary)
            and c.sha(cpp) == c.sha(project / "op_kernel" / (STEM + ".cpp"))
            and c.sha(header) == RAW_HEADER,
            "local package/source mismatch",
        )
        report.update(
            status="LOCAL_ARTIFACTS_RECORDED",
            installer=c.identity(installer),
            installed_kernel=c.identity(binary),
            dynamic_source=c.identity(cpp),
        )
    except Exception as exc:
        report["error"] = repr(exc)
        (out / "traceback.txt").write_text(traceback.format_exc())
    c.write(out / "report.json", report)
    print(report["status"], report.get("error"), out / "report.json")
    return int(report["status"] == "BLOCKED")


def worker(request):
    q = c.read(request)
    dest = Path(q["dest"])
    r = dict(status="BLOCKED")
    try:
        require(q["sources"] == sources(), "worker source drift")
        row = c.read(c.verify(q["source"]))
        binding, src, raw, _ = read_baseline(Path(row["archive"]))
        root = Path(row["root"]).resolve(strict=True)
        require(
            row["binding"] == binding
            and row["status"] == "SCALE_SOURCE_PASS"
            and row["sources"] == q["sources"]
            and request.resolve(strict=True) == root / "build/request.json"
            and dest == root / "build/package"
            and c.verify(q["source"]) == root / "source/report.json",
            "wrong source admission or private build path",
        )
        require(
            os.environ.get("ASCEND_CUSTOM_OPP_PATH") == str(dest / "opp/vendors/customize"),
            "private worker environment drift",
        )
        r["candidate"] = build_private(dest, src, raw)
        r["status"] = "SCALE_PRIVATE_BUILD_RECORDED"
    except Exception as exc:
        r["error"] = repr(exc)
        (request.parent / "worker-traceback.txt").write_text(traceback.format_exc())
    c.write(request.parent / "worker.json", r)
    return int(r["status"] == "BLOCKED")


if __name__ == "__main__":
    p = argparse.ArgumentParser(allow_abbrev=False)
    sub = p.add_subparsers(dest="mode", required=True)
    local = sub.add_parser("local")
    local.add_argument("--template", required=True, type=Path)
    local.add_argument("--parent", required=True, type=Path)
    local.add_argument("--cann", type=Path, default=Path("/usr/local/Ascend/cann-9.0.0"))
    sub.add_parser("worker").add_argument("request", type=Path)
    a = p.parse_args()
    sys.exit(worker(a.request) if a.mode == "worker" else local_compile(a.template, a.parent, a.cann))
