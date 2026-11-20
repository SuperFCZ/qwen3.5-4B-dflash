"""Read original command journals. No compiler, ATC or model is invoked here."""

from pathlib import Path
import shlex
import shutil
from contracts import DOUBLE, FIRST, GATE_SOURCE, HEADER, RAW_HEADER, STEM, c, one, pinned, require, same
from atc_dependency_identity import plan as atc_entry
from atc_evidence import resumed_command
from split_atc import one_capture

WRAPPER_SHA = "2fdf2abe2a51d8cd6faf09e3c70749093d69f1c80dd9be0b1de3bd27f2f4c4c7"
BISHENG_SHA = "b87dd4f943eb6450d9e0f8476eb69a43af1ac3da4b581c4386b88ce76825c737"
OPC_SHA = "ea025529f10344410995dce7854e5e4ab607d9643f3fd64cda09276b3228e189"
KINDS = {
    ".ccei": "preprocessed",
    ".ii": "preprocessed",
    ".i": "preprocessed",
    ".ll": "LLVM IR",
    ".bc": "LLVM bitcode",
    ".s": "assembly",
    ".asm": "assembly",
    ".lst": "listing",
}


def intermediates(directory):
    root = directory.resolve(strict=True)
    rows = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in KINDS or not path.is_file():
            continue
        require(
            root in path.resolve(strict=True).parents, "compiler intermediate outside cache: " + str(path)
        )
        rows.append(dict(file=c.identity(path), kind=KINDS[path.suffix.lower()], relation="COLOCATED_ONLY"))
    return rows


def package_command(compiler, source):
    for item in compiler["tools"].values():
        c.verify(item)
    require(
        compiler["tools"]["bisheng"]["sha256"] == BISHENG_SHA
        and compiler["tools"]["opc"]["sha256"] == OPC_SHA,
        "unreviewed CANN compiler or OPC",
    )
    commands = {Path(i["path"]).name: i for i in compiler["commands"]}
    configure = c.read(c.verify(commands["configure.command.json"]))["argv"]
    build = c.read(c.verify(commands["compile.command.json"]))["argv"]
    require(
        "-DASCEND_COMPUTE_UNIT=ascend310p" in configure and "DFlashGroupQuantLinear_ascend310p" in build,
        "wrong device package target",
    )
    log = c.verify(commands["compile.log"])
    lines = log.read_text(errors="replace").splitlines()
    calls = []
    for n, line in enumerate(lines, 1):
        if "--op-name=DFlashGroupQuantLinear" in line and "--compute-unit=ascend310p" in line:
            args = shlex.split(line)
            require(
                "--src-file=" + source["path"] in args and "--compile-options=" in args,
                "unbound/modified device compile flags or source",
            )
            calls.append(dict(line=n, command=line, argv=args))
    require(calls, "missing actual ascendc_compile_kernel command in package build log")
    return dict(
        tools=compiler["tools"],
        configure=configure,
        build=build,
        emitted=calls,
        logs=compiler["commands"],
        recipes=compiler["recipes"],
        binary=compiler["binary"],
        metadata=compiler["metadata"],
    )


def dispatch(build, raw, env):
    install = Path(build["opp"]["install"]).resolve(strict=True)
    vendor = install / "vendors/customize"
    require(
        env["ASCEND_CUSTOM_OPP_PATH"] == str(vendor) and not env.get("BUILD_KERNEL_SRC"),
        "unexpected private OPP or device source override",
    )
    tbe = vendor / "op_impl/ai_core/tbe"
    wrapper = one(
        [p for p in tbe.rglob(STEM + ".py") if p.parent.name == "dynamic"], "ambiguous Gate compiler wrapper"
    )
    wrapper_id = pinned(wrapper, WRAPPER_SHA)
    require(
        all(x["wrapper"]["sha256"] == WRAPPER_SHA for x in raw["dynamic"]), "historical Gate wrapper drift"
    )
    # The pinned wrapper checks these concrete private paths in get_kernel_source.
    # dir_ex and dir_snake both equal STEM. BUILD_KERNEL_SRC has already been excluded.
    candidates = [
        wrapper.parent.parent / "ascendc" / STEM / (STEM + ".cpp"),
        wrapper.parent / (STEM + ".cpp"),
    ]
    found = [p for p in candidates if p.is_file()]
    installed = {c.verify(i) for i in build["dynamic"]}
    require(
        found
        and all(p.resolve(strict=True) in installed for p in found)
        and installed == {p.resolve(strict=True) for p in tbe.rglob(STEM + ".cpp")},
        "ATC source resolution is ambiguous/unbound",
    )
    cpp = found[0].resolve(strict=True)
    require(vendor in cpp.parents, "Gate implementation escapes OPP")
    header = pinned(cpp.parent / HEADER, RAW_HEADER)
    return dict(wrapper=wrapper_id, source=pinned(cpp, GATE_SOURCE[build["variant"]]), header=header)


def inspect_mode(ctx, mode):
    build = ctx.built[mode]
    model = ctx.models[mode]
    worker = c.read(c.verify(model["worker_report"]))
    argv = worker["atc_command"]
    cap = one_capture(worker["captures"], argv)
    env = cap["environment"]
    require(env == worker["child_environment"], "ATC capture/environment mismatch")
    atc_dir = c.verify(model["dump"]).parent
    compiled = c.verify(model["worker_report"]).parent
    expected_env = dict(
        build["env"]["values"],
        AI_RUN_DIR=str(atc_dir),
        ASCEND_WORK_PATH=str(compiled),
        ASCEND_CACHE_PATH=str(atc_dir / "compiler-cache"),
        DFLASH_PLATFORM_DIAG_ROUTE="ATC",
        DFLASH_PLATFORM_DIAG_PATH=str(compiled / "platform.jsonl"),
        DFLASH_SCHEDULER_EVIDENCE_PATH=str(compiled / "scheduling.jsonl"),
    )
    same(env, expected_env, "ATC compiler environment differs from isolated private build")
    frozen = ctx.src["singles"][FIRST]
    dep = c.read(c.verify(model["deployment"]))
    depdir = c.verify(model["deployment"]).parent
    air_manifest = c.linked(depdir, dep["air_manifest"])
    require(
        c.sha(air_manifest) == frozen["air"]["sha256"]
        and resumed_command(frozen, air_manifest, depdir) == argv
        and cap["atc"] == frozen["atc_binary"],
        "wrong ATC executable, flags or AIR",
    )
    for rel, identity in frozen["payloads"].items():
        p = (depdir / rel).resolve(strict=True)
        require(
            depdir in p.parents and c.sha(p) == identity["sha256"] and p.stat().st_size == identity["bytes"],
            "ATC staged AIR payload differs from frozen input",
        )
    chain = atc_entry(cap["atc"], argv[0], ctx.src["cann"], env)
    effective = dict(env, **chain["wrapper_exports"])
    same(effective, worker["elf_environment"], "wrong effective ATC environment")
    require(effective["ASCEND_OPP_PATH"] == str(Path(ctx.src["cann"]) / "opp"), "wrong standard OPP")
    cache = atc_dir / "compiler-cache"
    require(
        effective.get("ASCEND_CACHE_PATH") == str(cache) and cache.resolve(strict=True) == cache,
        "ATC cache not private to the archived invocation",
    )
    source = dispatch(build, ctx.raw, effective)
    binary = effective.get("BISHENG_REAL_PATH") or shutil.which("bisheng", path=effective["PATH"])
    require(
        binary and c.identity(binary) == build["compiler"]["tools"]["bisheng"], "ATC compiler lookup drift"
    )
    package = package_command(
        build["compiler"], build["generated_sources"]["op_kernel/" + STEM + ".cpp"]["file"]
    )
    # Inspect existing scoped directories only. A nearby .bc/.s cannot identify a linked ATC image.
    package_object = c.verify(build["compiler"]["binary"])
    build_dir = one(
        (p for p in package_object.parents if p.name == "device-build"), "package device-build root"
    )
    return dict(
        package=package,
        private_dispatch=source,
        atc_command=argv,
        atc_chain=chain,
        atc_capture=cap,
        worker=model["worker_report"],
        air=frozen["air"],
        om=model["om"],
        selected_elf=ctx.caps[mode]["elf"],
        package_intermediates=intermediates(build_dir),
        atc_intermediates=intermediates(atc_dir),
        atc_internal_frontend_argv="NOT_ATTESTED_BY_OUTER_CAPTURE",
        compiler_output_relation="Private source/GE vendor, captured ATC invocation and embedded ELF; no IR-to-ELF attestation.",
    )


def compiler_evidence(ctx):
    variants = {m: inspect_mode(ctx, m) for m in DOUBLE}
    left, right = variants.values()
    same(left["package"]["tools"], right["package"]["tools"], "compiler tools differ by mode")
    require(
        left["atc_capture"]["environment"]["ASCEND_CACHE_PATH"]
        != right["atc_capture"]["environment"]["ASCEND_CACHE_PATH"],
        "shared ATC cache",
    )
    return dict(
        variants=variants,
        capability=dict(
            compiler_sha256=BISHENG_SHA,
            local_control_source_sha256="db09d47f879f7ac840dcfb97bcff1775a5311c119310426f295054bc78eaf9c3",
            local_capability_report_sha256="93d837017d2fe9827a012184e3c10b8482e2c6956ebc11aafa318aec7d186921",
            preprocessed="-E produced .ccei text",
            optimized_bitcode="-O2 -emit-llvm -c produced LLVM bitcode",
            textual_ir="-emit-llvm -S rejected",
            assembly="-S rejected for device",
            relation="separate retained dav-m200 control; not W16 ATC provenance",
        ),
    )
