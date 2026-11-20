"""Prepare a compile-time-only private dispatcher for the full graph's 5/21 split."""

import inspect
import json
from pathlib import Path
import shutil
import sys

from evidence import HERE, NODES, GATE, SHAPE_KEY, c, require
from dispatch_body import record_scale_compile, select_scale_source
from scale_contracts import (
    CANDIDATE,
    CONTROL,
    HEADER,
    RAW_HEADER,
    WRAPPER,
    STEM,
    command,
    gate_kernel,
    host,
    resume,
    tree,
)
from scale_build import compare_tree, preflight


def route_rules(control):
    graph = c.read(c.verify(control["deployment"]))["graphs"][0]
    ops = [n for n in graph["runtime_input_abi"]["draft_ac"]["nodes"] if n["ge_type"] == GATE]
    keys = {}
    require(len(ops) == 26 and {n["name"] for n in ops} == set(control["atc_kernels"]), "route inventory")
    for node in ops:
        name = node["name"]
        candidates = [
            tag for tag in control["plans"] if tuple(map(int, tag.split("x")[1:])) == (node["k"], node["n"])
        ]
        require(
            candidates and ((name in NODES.values()) == (candidates == [SHAPE_KEY])), "ambiguous Gate shape"
        )
        kernel = control["atc_kernels"][name]["symbol"].split("__kernel")[0]
        keys.setdefault(kernel, set()).update(candidates)
    return dict(shapes=sorted(control["plans"]), keys={k: sorted(v) for k, v in keys.items()})


def wrapper_bytes(original, rules, package):
    from hashlib import sha256

    require(sha256(original).hexdigest() == WRAPPER, "unknown SDK wrapper")
    src = original.decode()
    target = "    src = get_kernel_source(ascendc_src_file, ascendc_src_dir, ascendc_src_dir_ex)"
    invoke = (
        "    compile_op(src, origin_func_name, op_info, options, code_channel, '{}', {'valueDepend': {}})"
    )
    require(
        src.count(target) == src.count(invoke) == src.count("def _build_args(") == 1,
        "unexpected dispatch sites",
    )
    helpers = inspect.getsource(select_scale_source) + "\n" + inspect.getsource(record_scale_compile)
    src = src.replace("def _build_args(", helpers + "\n\ndef _build_args(")
    src = src.replace(
        target,
        target
        + "\n    src, scale_receipt = select_scale_source(__inputs__, __outputs__, kernel_name, src, "
        + repr(rules)
        + ", "
        + repr(package)
        + ")",
    )
    src = src.replace(invoke, invoke + "\n    record_scale_compile(scale_receipt)")
    compile(src, "<private GE dispatcher>", "exec")
    return src.encode()


def paths(raw, install):
    base = Path(raw["opp"]["install"])
    require(len(raw["dynamic"]) == 1, "ambiguous SDK dynamic source")
    return {
        k: install / c.verify(v).relative_to(base)
        for k, v in raw["dynamic"][0].items()
        if k in ("source", "header", "wrapper")
    }


def check_package(ctx, gear, built):
    raw, pilot = ctx["builds"][CONTROL], ctx["builds"][CANDIDATE]
    install = Path(built["opp"]["install"])
    files = paths(raw, install)
    alt = files["source"].parent / "scale_b16" / (STEM + ".cpp")
    extra = {str(p.relative_to(install)) for p in (alt, alt.parent / HEADER)}
    files_now = tree(install)
    require(
        set(files_now) == set(raw["package_files"]) | extra and files_now == built["package_files"],
        "routed OPP inventory drift",
    )
    partial = {k: v for k, v in files_now.items() if k not in extra}
    allowed = {str(files["wrapper"].relative_to(install))}
    require(compare_tree(raw["package_files"], partial, allowed) == allowed, "routed package delta")
    require(
        c.sha(alt) == pilot["dynamic"][0]["source"]["sha256"] and c.sha(alt.parent / HEADER) == RAW_HEADER,
        "transpose source/header differs from Pilot",
    )
    package = dict(
        brcb=c.identity(files["source"]),
        transpose=c.identity(alt),
        header_name=HEADER,
        header_sha256=RAW_HEADER,
    )
    rules = route_rules(ctx["full"][gear])
    require(
        files["wrapper"].read_bytes()
        == wrapper_bytes(c.verify(raw["dynamic"][0]["wrapper"]).read_bytes(), rules, package)
        and built["rules"] == rules
        and built["routes"] == package,
        "unreviewed private wrapper",
    )
    require(
        built["kernel"] == gate_kernel(built["opp"])
        and built["kernel"]["elf"]["sha256"] == raw["kernel"]["elf"]["sha256"]
        and built["om"] == raw["om"]
        and built["runner"] == raw["runner"]
        and built["api"]["sha256"] == raw["api"]["sha256"]
        and built["compiler"] == pilot["compiler"],
        "private build ABI/kernel/runner/compiler drift",
    )
    c.verify_opp(built["opp"])
    require(host.envs.standard_identity(built["env"]) == ctx["src"]["standard_env"], "routed runtime drift")
    resume.combined(built, ctx["src"]["cann"], ctx["src"]["extras"], c.read(resume.POLICY))


def prepare(folder, ctx, gear):
    src = ctx["src"]
    raw, pilot = ctx["builds"][CONTROL], ctx["builds"][CANDIDATE]
    install = folder / "opp"
    folder.mkdir(parents=True)
    prior = c.read(c.verify(src["host_source"]))["binding"]["prior"]
    shutil.copytree(raw["opp"]["install"], install, symlinks=True)
    compare_tree(raw["package_files"], tree(install), set())
    files = paths(raw, install)
    require(all(p.resolve(strict=True) == p for p in files.values()), "redirected dispatcher inputs")
    alt = files["source"].parent / "scale_b16" / (STEM + ".cpp")
    alt.parent.mkdir()
    shutil.copyfile(c.verify(pilot["dynamic"][0]["source"]), alt)
    shutil.copyfile(files["header"], alt.parent / HEADER)
    rules = route_rules(ctx["full"][gear])
    package = dict(
        brcb=c.identity(files["source"]),
        transpose=c.identity(alt),
        header_name=HEADER,
        header_sha256=RAW_HEADER,
    )
    files["wrapper"].write_bytes(wrapper_bytes(files["wrapper"].read_bytes(), rules, package))
    env, env_id = host.envs.for_prior(prior, install)
    pf = folder / "preflight.json"
    command(
        [sys.executable, "-B", HERE / "routing.py", install, pf], folder / "preflight.log", env, cwd=folder
    )
    opp = dict(install=str(install), preflight=c.read(pf), kind="PRIVATE_FULL_GATE_SCALE_ROUTER")
    manifest = folder / "manifest.json"
    c.write(
        manifest,
        host.dsos.with_private(c.read(c.verify(src["standard_manifest"])), opp, src["standard_manifest"]),
    )
    built = dict(
        variant=CANDIDATE,
        opp=opp,
        env=env_id,
        manifest=c.identity(manifest),
        kernel=gate_kernel(opp),
        api=c.identity(install / "vendors/customize/op_api/lib/libcust_opapi.so"),
        runner=raw["runner"],
        om=raw["om"],
        compiler=pilot["compiler"],
        rules=rules,
        routes=package,
        package_files=tree(install),
    )
    check_package(ctx, gear, built)
    return built


def verify_dispatch(path, built, selected, targets):
    rows = [json.loads(line) for line in c.verify(path).read_text().splitlines()]
    require(rows, "missing actual ATC source-selection records")
    seen = set()
    for row in rows:
        choice = row["choice"]
        require(
            choice in ("brcb", "transpose") and row["source"] == built["routes"][choice], "unexpected source"
        )
        tags = set(row["matching_shapes"])
        require(
            bool(tags)
            and tags <= set(built["rules"]["shapes"])
            and ((tags == {SHAPE_KEY}) if choice == "transpose" else SHAPE_KEY not in tags),
            "mixed route",
        )
        key = row["kernel_name"].split("__kernel")[0]
        matching = {n for n, img in selected.items() if img["symbol"].split("__kernel")[0] == key}
        require(
            matching and ((matching <= targets) if choice == "transpose" else matching.isdisjoint(targets)),
            "compiler key not bound to the final graph: " + key,
        )
        seen.update(matching)
    require(targets <= seen, "some target ELFs have no bound Transpose compilation")
    return dict(actual_calls=len(rows), nodes_with_compile_receipts=sorted(seen), records=path)


if __name__ == "__main__":
    install, output = map(Path, sys.argv[1:])
    require(
        install.is_absolute()
        and install.is_dir()
        and output.parent == install.parent
        and not output.exists(),
        "invalid isolated preflight request",
    )
    c.write(output, preflight(install))
