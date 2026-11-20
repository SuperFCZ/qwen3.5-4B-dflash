"""Bind the real GE node, unchanged Cube plan, and installed SDK CPU layout witness."""

from pathlib import Path
import shutil
import struct

from scale_contracts import (
    CANDIDATE,
    CONTROL,
    CUBE_SHA,
    FIRST,
    FIRST_HASHES,
    GATE,
    HEADER,
    HERE,
    RAW_HEADER,
    SHAPE,
    SHAPE_KEY,
    STEM,
    c,
    command,
    config_graph,
    flow,
    historical,
    host,
    one,
    read_om_json,
    require,
    same_plan,
    sdk_files,
    verify_compiled_graph,
)
from w16_export import compile_air
from w16_graph import protected_graph
from gate_plan import embedded, node_binding
from split_atc import one_capture
from atc_evidence import resumed_command
from atc_dependency_identity import plan as atc_entry
from w16_static import sections


def allocated_sections(model, node):
    selected = model["atc_kernels"][node]
    with c.verify(model["om"]).open("rb") as stream:
        stream.seek(selected["offset"])
        blob = stream.read(selected["bytes"])
    import hashlib

    require(hashlib.sha256(blob).hexdigest() == selected["sha256"], "selected device image drift")
    table = sections(blob)
    return dict(
        writable=[
            dict(name=s["name"], bytes=s["bytes"], alignment=s["alignment"])
            for s in table
            if (s["flags"] & 3) == 3 or s["name"] == ".bl_uninit"
        ],
        metadata=[
            dict(name=s["name"], bytes=s["bytes"], sha256=s["sha256"])
            for s in table
            if s["name"].startswith(".ascend.meta")
        ],
    )


def read_plan(worker, src, selected, control):
    q = flow.read_query(
        c.verify(worker["platform"]), c.verify(worker["scheduling"]), selected["opp"], "ATC", "A", SHAPE
    )
    proof = same_plan(q, src, control["plans"][SHAPE_KEY]["contract"])
    require(
        q["host_dso"]["sha256"] == control["plans"][SHAPE_KEY]["query"]["host_dso"]["sha256"],
        "Host tiling DSO bytes changed",
    )
    return {SHAPE_KEY: dict(query=q, contract=proof)}


def compile_candidate(folder, src, selected, control):
    frozen = src["singles"][FIRST]
    worker, dump = compile_air(folder, src, selected, frozen, "projection")
    graph = c.read(c.verify(worker["deployment"]))["graphs"][0]
    node = config_graph(graph, {FIRST: src["cases"]["16"][FIRST]}, False)[FIRST]["name"]
    gates = [node_binding(x) for x in read_om_json(dump) if x["type"] == GATE]
    require([g["node"] for g in gates] == [node], "wrong Gate constant/node mapping")
    result = dict(
        mode=CANDIDATE,
        key=FIRST,
        scope="projection",
        air=frozen["air"],
        om=worker["om"],
        deployment=worker["deployment"],
        worker_report=c.identity(folder / "compiled/report.json"),
        dump=c.identity(dump),
        abi=graph["metadata"]["tensor_abi"],
        opp_root=selected["opp"]["install"],
        private_kernel=selected["kernel"],
        gate_nodes={FIRST: node},
        atc_kernels=embedded(c.verify(worker["om"]), gates),
        plans=read_plan(worker, src, selected, control),
    )
    check_model(result, src, selected, control)
    return result


def check_model(model, src, selected, control):
    verify_compiled_graph(model, src, False)
    require(
        model["mode"] == CANDIDATE
        and model["key"] == FIRST
        and model["scope"] == "projection"
        and model["opp_root"] == selected["opp"]["install"]
        and model["private_kernel"] == selected["kernel"]
        and model["gate_nodes"] == control["gate_nodes"]
        and model["abi"] == control["abi"]
        and model["air"] == control["air"],
        "cross-wired model, constants, ABI or selected private OPP",
    )
    workers = c.read(c.verify(model["worker_report"]))
    require(
        workers["status"] == "GATE8_PROJECTION_AIR_COMPILED_NOT_EXECUTED"
        and workers["execute_calls"] == 0
        and not workers.get("error")
        and not workers.get("integrity_error")
        and (workers["om"], workers["deployment"]) == (model["om"], model["deployment"]),
        "wrong ATC worker",
    )
    records = one_capture(workers["captures"], workers["atc_command"])
    folder = c.verify(model["dump"]).parent
    compiled = c.verify(model["worker_report"]).parent
    env = dict(
        selected["env"]["values"],
        AI_RUN_DIR=str(folder),
        ASCEND_WORK_PATH=str(compiled),
        ASCEND_CACHE_PATH=str(folder / "compiler-cache"),
        DFLASH_PLATFORM_DIAG_PATH=str(compiled / "platform.jsonl"),
        DFLASH_PLATFORM_DIAG_ROUTE="ATC",
        DFLASH_SCHEDULER_EVIDENCE_PATH=str(compiled / "scheduling.jsonl"),
    )
    require(
        env == records["environment"] == workers["child_environment"] and not env.get("BUILD_KERNEL_SRC"),
        "unexpected ATC environment, cache or source override",
    )
    frozen = src["singles"][FIRST]
    dep = c.verify(model["deployment"]).parent
    manifest = c.linked(dep, c.read(c.verify(model["deployment"]))["air_manifest"])
    require(
        c.sha(manifest) == frozen["air"]["sha256"]
        and records["atc"] == frozen["atc_binary"]
        and resumed_command(frozen, manifest, dep) == records["argv"],
        "changed ATC precision/flags or AIR",
    )
    for name, item in frozen["payloads"].items():
        path = (dep / name).resolve(strict=True)
        require(
            dep in path.parents and c.sha(path) == item["sha256"] and path.stat().st_size == item["bytes"],
            "changed AIR payload",
        )
    wrapper = atc_entry(records["atc"], records["argv"][0], src["cann"], env)
    effective = dict(env, **wrapper["wrapper_exports"])
    require(workers["elf_environment"] == effective, "ATC ELF environment drift")
    binary = effective.get("BISHENG_REAL_PATH") or shutil.which("bisheng", path=effective["PATH"])
    require(
        binary and c.identity(binary) == selected["compiler"]["tools"]["bisheng"], "wrong compiler lookup"
    )
    plans = read_plan(workers, src, selected, control)
    require(
        plans == model["plans"] and plans[SHAPE_KEY]["contract"] == control["plans"][SHAPE_KEY]["contract"],
        "plan or 50 Cube fields changed",
    )
    node = one(model["gate_nodes"].values(), "expected one immutable gate")
    old = control["atc_kernels"][node]
    new = model["atc_kernels"][node]
    require(
        new["engine"] == "AIcoreEngine"
        and new["block_dim"] == old["block_dim"] == 8
        and new["sha256"] != old["sha256"]
        and new["executable_sections"] != old["executable_sections"],
        "no distinct selected Gate device code",
    )
    vendor = Path(model["opp_root"]) / "vendors/customize"
    attrs = one(
        (x for x in read_om_json(c.verify(model["dump"])) if x["name"] == node), "ambiguous Gate node"
    )["attributes"]
    require(
        one((x["value"] for x in attrs if x["key"] == "custom_op_file_path"), "Gate dispatch vendor missing")
        == {"s": str(vendor)}
        and env["ASCEND_CUSTOM_OPP_PATH"] == str(vendor),
        "candidate private vendor mismatch",
    )
    # The actual generated wrapper checks ../ascendc/<stem>/<stem>.cpp first, then dynamic/<stem>.cpp.
    for dynamic in selected["dynamic"]:
        directory = c.verify(dynamic["wrapper"]).parent
        cpp = one(
            (
                p
                for p in (directory.parent / "ascendc" / STEM / (STEM + ".cpp"), directory / (STEM + ".cpp"))
                if p.is_file()
            ),
            "ambiguous/missing ATC dynamic implementation",
        )
        require(
            c.identity(cpp) in [i["source"] for i in selected["dynamic"]]
            and c.sha(cpp.parent / HEADER) == RAW_HEADER,
            "unbound selected implementation/header",
        )
    count = protected_graph(
        c.verify(control["dump"]),
        c.verify(model["dump"]),
        {node},
        raw_opp_root=control["opp_root"],
        new_opp_root=model["opp_root"],
    )
    resource_sections = {
        CONTROL: allocated_sections(control, node),
        CANDIDATE: allocated_sections(model, node),
    }
    require(
        resource_sections[CONTROL]["writable"] == resource_sections[CANDIDATE]["writable"],
        "ATC image writable storage extents changed",
    )
    return dict(
        protected_nodes=count,
        gate_node=node,
        control_embedded=old,
        candidate_embedded=new,
        resource_sections=resource_sections,
        workspace=plans[SHAPE_KEY]["contract"]["workspace"],
        cube_sha256=CUBE_SHA,
        atc_chain=wrapper,
    )


def expected_line(n):
    return f"CPU_SCALE_WORDS {n} EXPANDED_PER_VARIANT {n * 16} PAYLOAD_GUARD_READONLY_ERRORS 0"


def validate_cpu(r, out, src, scale):
    require(
        r["scope"] == "SDK_CPU_MODEL_ONLY"
        and r["frozen_scale"] == scale
        and scale["sha256"] == FIRST_HASHES["s_gn.bin"],
        "CPU witness uses different scales",
    )
    require(c.verify(r["binary"]) == out / "build" / "scale_cpu", "redirected CPU model checker")
    require(r["sdk"] == sdk_files(src["cann"]), "CPU model/SDK identity drift")
    files = {v["file"]["sha256"] for v in r["dependencies"]}
    for rel in (
        "tools/tikicpulib/lib/Ascend310P3/libcpudebug.so",
        "aarch64-linux/simulator/Ascend310P3/lib/lib_pvmodel.so",
    ):
        require(r["sdk"][rel]["sha256"] in files, "host probe did not bind the 310P model: " + rel)
    require(
        c.verify(r["patterns"]).read_bytes() == struct.pack("<65536H", *range(65536)),
        "incomplete FP16 bit domain",
    )
    for name, item, n in (("bits", r["patterns"], 65536), ("frozen", scale, 389120)):
        run = r[name]
        log = c.verify(run["log"])
        require(
            log == out / name / "model.log"
            and c.verify(run["command"]) == log.with_suffix(".command.json")
            and c.read(c.verify(run["command"]))["argv"] == [r["binary"]["path"], item["path"]]
            and log.read_text(errors="replace").splitlines().count(expected_line(n)) == 1,
            "scale/reference/guard mismatch: " + name,
        )
    historical.verify_tree(r, {})


def run_cpu(out, src, raw):
    out.mkdir()
    prior = c.read(c.verify(src["host_source"]))["binding"]["prior"]
    env, _ = host.envs.for_prior(prior, None)
    env.update(TMPDIR=str(out), LC_ALL="C")
    require(not env["ASCEND_CUSTOM_OPP_PATH"], "private OPP leaked into independent CPU reference")
    tools = raw["compiler"]["tools"]
    command(
        [
            c.verify(tools["cmake"]),
            "-S",
            HERE / "test/sdk",
            "-B",
            out / "build",
            "-DCANN_ROOT=" + src["cann"],
            "-DCMAKE_CXX_COMPILER=" + str(c.verify(tools["c++"])),
        ],
        out / "configure.log",
        env,
        cwd=out,
    )
    command(
        [c.verify(tools["cmake"]), "--build", out / "build", "--parallel", "2"],
        out / "compile.log",
        env,
        cwd=out,
    )
    binary = c.identity(out / "build" / "scale_cpu")
    deps = c.dependencies(binary, out / "ldd.log", env)
    pattern = out / "words.bin"
    pattern.write_bytes(struct.pack("<65536H", *range(65536)))
    scale = src["cases"]["16"][FIRST]["inputs"]["s_gn.bin"]
    r = dict(
        scope="SDK_CPU_MODEL_ONLY",
        binary=binary,
        dependencies=deps,
        patterns=c.identity(pattern),
        frozen_scale=scale,
        sdk=sdk_files(src["cann"]),
        build_logs=[
            c.identity(out / n)
            for n in (
                "configure.log",
                "configure.command.json",
                "compile.log",
                "compile.command.json",
                "ldd.log",
            )
        ],
    )
    for name, item in (("bits", r["patterns"]), ("frozen", scale)):
        work = out / name
        work.mkdir()
        log = work / "model.log"
        command([c.verify(binary), c.verify(item)], log, env, cwd=work)
        r[name] = dict(log=c.identity(log), command=c.identity(log.with_suffix(".command.json")))
    validate_cpu(r, out, src, scale)
    return r
