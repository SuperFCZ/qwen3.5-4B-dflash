"""Replay sealed products and derive bounded experimental safety under the target SDK semantics."""

from dataclasses import dataclass
from pathlib import Path
import subprocess

from contracts import (
    DOUBLE,
    EXTRA_SDK,
    FIRST,
    FIRST_HASHES,
    GATE_ELF,
    GATE_TEXT,
    LABELS,
    OPS,
    OVERLAP,
    SHAPE_KEY,
    SINGLE,
    archive,
    c,
    checked_sources,
    check_trace,
    lifetime_contract,
    memory_contract,
    one,
    pinned,
    require,
    same,
    schedule,
)
from w16_common import sdk_files
from w16_export import resource_contract
from c_profile_graph import read_om_json


@dataclass
class Context:
    src: dict
    raw: dict
    built: dict
    models: dict
    caps: dict
    refs: dict
    sdk: dict
    negative: dict


def load_context(old, isa):
    refs, stages, (src, raw, old_models) = archive.load_archive(old)
    audit_file = c.identity(isa / "static-audit.json")
    report = c.read(c.verify(audit_file))
    require(
        report["root"] == str(old)
        and report["output"] == str(isa)
        and report["upstream"] == refs
        and report["process_scope"] == "CPU_OFFLINE_AUDIT"
        and report["chosen_tool"] == stages["source"]["binding"]["tool"]
        and report["status"] == "BLOCKED_STATIC_ADMISSION"
        and report["reason"] == "NO_DECODER_PASSED_INDEPENDENT_CONTROL"
        and report["errors"] == []
        and report["device_launch_authorized"] is False
        and report["atc"] == report["device_execution"] == report["benchmark"] == "NOT_RUN",
        "wrong archived ISA audit",
    )
    control = report["control"]["original"]
    require(
        pinned(c.verify(control), "47ffde8366f14aa97267a8d52b3d55fc7fec7684126c578831152ade2d5c343b")["bytes"]
        == 18632
        and report["coverage"]["DFlashDraftLmHeadTop1"]["text_bytes"] == 16244
        and report["coverage"]["DFlashDraftLmHeadTop1"]["qualified_text_bytes"] == 0,
        "wrong independent Head control",
    )
    caps = {m: one(stages["atc"]["captures"][FIRST][m].values(), "single captured node") for m in DOUBLE}
    for m, cap in caps.items():
        body, sections = archive.bound_image(cap)
        text = one((s for s in sections if s["name"] == ".text"), "executable text")
        origin = report["image_provenance"][m]
        require(
            len(body) == 32304
            and cap["elf"]["sha256"] == GATE_ELF[m]
            and text["bytes"] == 25964
            and text["sha256"] == GATE_TEXT[m]
            and origin["om"] == cap["om"]
            and origin["elf"] == cap["elf"]
            and origin["graph_binding"] == cap["node"]
            and report["coverage"][m]["qualified_text_bytes"] == 0,
            "archived OM or zero-coverage evidence is cross-wired",
        )
    sdk = sdk_files(src["cann"])
    require(sdk == stages["source"]["binding"]["sdk"], "SDK changed since Source")
    install = Path(src["cann"]).resolve(strict=True) / "aarch64-linux/asc"
    for path, sha in EXTRA_SDK.items():
        require(install in (install / path).resolve(strict=True).parents, "redirected SDK header")
        sdk[path] = pinned(install / path, sha)
    models = {m: stages["atc"]["models"][m][FIRST] for m in DOUBLE}
    models[SINGLE] = old_models["export"][FIRST]
    return Context(
        src,
        raw,
        {SINGLE: raw, **stages["build"]["variants"]},
        models,
        caps,
        refs,
        sdk,
        dict(
            archive=audit_file,
            strict_report=refs["blocked_static"],
            strict_status=report["status"],
            reason=report["reason"],
            coverage=report["coverage"],
            machine_proof="UNRESOLVED",
        ),
    )


def cpu_actions(text, mode, failure):
    actual = [("reserve", -1)]
    for line in text.splitlines():
        action, k = line.split()
        actual.append((action, int(k)))
    actual += [("release", -1), ("end", -1)]
    want = list(schedule(10, overlap=(mode == OVERLAP), fail_at=None if failure < 0 else failure))
    require(actual == want, "CPU build of the frozen W16 template produced a different order")
    require(
        check_trace(actual, 10) == (list(range(10 if failure < 0 else failure)), failure < 0),
        "CPU hazard/drain contract",
    )
    return len(actual)


def cpu_witness(folder, ctx):
    source = pinned(
        OPS / "draft_gate_w16_atc8/test/schedule_test.cpp",
        "0b1d8a3e9eec40178b6be6a87b9ac1412cd49fac701ba2003c2774cf5f9e99c4",
    )
    header = pinned(
        OPS / "draft_gate_w16_atc8/kernel/tile_schedule.h",
        "047fdbb268dd0fa5b5bd963818bc7d5f5f20d1e4a96ca5bf5bcd78464bed89b4",
    )
    cpp = ctx.built[DOUBLE[0]]["compiler"]["tools"]["c++"]
    exe = folder / "cpu-schedule"
    argv = [str(c.verify(cpp)), "-std=c++17", "-O0", str(c.verify(source)), "-o", str(exe)]
    # A plain host C++ compiler, a pinned standard-library fixture, and no ACL libraries.
    env = {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TMPDIR": str(folder)}
    log = folder / "cpu-compile.log"
    with log.open("xb") as output:
        subprocess.run(
            argv, env=env, cwd=folder, stdout=output, stderr=subprocess.STDOUT, timeout=90, check=True
        )
    binary = c.identity(exe)
    traces = {}
    for mode in DOUBLE:
        for failure in range(-1, 10):
            args = [str(c.verify(binary)), str(int(mode == OVERLAP)), str(failure)]
            output = subprocess.check_output(args, cwd=folder, env=env, timeout=15)
            f = folder / f"{mode}-k{failure}.txt"
            f.write_bytes(output)
            traces[f"{mode}/{failure}"] = dict(
                file=c.identity(f), steps=cpu_actions(output.decode(), mode, failure)
            )
    return dict(
        scope="HOST_ONLY_TEMPLATE_WITNESS",
        compiler=cpp,
        argv=argv,
        log=c.identity(log),
        exe=binary,
        fixture=source,
        header=header,
        traces=traces,
    )


def verify_cpu(w, folder, ctx):
    require(
        w["scope"] == "HOST_ONLY_TEMPLATE_WITNESS"
        and w["compiler"] == ctx.built[DOUBLE[0]]["compiler"]["tools"]["c++"]
        and c.verify(w["exe"]) == folder / "cpu-schedule"
        and c.verify(w["log"]) == folder / "cpu-compile.log",
        "wrong CPU witness",
    )
    require(
        c.verify(w["fixture"]) == OPS / "draft_gate_w16_atc8/test/schedule_test.cpp"
        and c.verify(w["header"]) == OPS / "draft_gate_w16_atc8/kernel/tile_schedule.h",
        "wrong CPU schedule/fixture",
    )
    same(
        w["argv"],
        [w["compiler"]["path"], "-std=c++17", "-O0", w["fixture"]["path"], "-o", w["exe"]["path"]],
        "CPU recipe drift",
    )
    same(
        sorted(w["traces"]),
        sorted(f"{m}/{k}" for m in DOUBLE for k in range(-1, 10)),
        "CPU control cases missing",
    )
    for mode in DOUBLE:
        for k in range(-1, 10):
            row = w["traces"][f"{mode}/{k}"]
            path = c.verify(row["file"])
            require(
                path == folder / f"{mode}-k{k}.txt"
                and row["steps"] == cpu_actions(path.read_text(), mode, k),
                "CPU witness changed",
            )


def sdk_refs(sdk, *suffixes):
    return [
        one((v for k, v in sdk.items() if k.endswith("/" + s)), "audited SDK file " + s) for s in suffixes
    ]


def safety_case(ctx, compiler):
    within = {}
    plan0 = ctx.models[SINGLE]["plans"][SHAPE_KEY]["query"]
    sdk = ctx.sdk
    install = c.identity(Path(ctx.src["cann"]) / "aarch64-linux/ascend_toolkit_install.info")
    info = dict(line.split("=", 1) for line in c.verify(install).read_text().splitlines() if "=" in line)
    require(
        info.get("package_name") == "Ascend-cann-toolkit"
        and info.get("version") == "9.0.0"
        and info.get("arch") == "aarch64",
        "unexpected toolkit/architecture",
    )
    files = ctx.src["cases"]["16"][FIRST]
    frozen = {**files["inputs"], "golden": files["golden"]}
    same(
        {k: c.identity(c.verify(v))["sha256"] for k, v in frozen.items()},
        FIRST_HASHES,
        "wrong frozen Layer0 case",
    )
    for mode in DOUBLE:
        b, model = ctx.built[mode], ctx.models[mode]
        source = checked_sources(mode, ctx.raw, b)
        private = compiler["variants"][mode]["private_dispatch"]
        require(
            private["source"] in source["dynamic"]
            and private["source"]["sha256"] == source["kernel"]["sha256"],
            "compiler source provenance does not close",
        )
        node = one(
            (x for x in read_om_json(c.verify(model["dump"])) if x["type"] == "DFlashGroupQuantLinear"),
            "Gate node ambiguity",
        )
        vendor = one(
            (a["value"] for a in node["attributes"] if a["key"] == "custom_op_file_path"),
            "missing GE compiler vendor",
        )
        same(
            vendor,
            {"s": str(Path(b["opp"]["install"]) / "vendors/customize")},
            "selected GE compiler comes from another package",
        )
        require(
            ctx.caps[mode]["node"]["node"] == node["name"] and ctx.caps[mode]["node"]["block_dim"] == 8,
            "wrong graph-selected code",
        )
        plan = model["plans"][SHAPE_KEY]
        # This verifies both original callback streams, the 224 B parent / 200 B nested payload,
        # all field offsets, the full cube hash, assignments, architecture and baseline resource delta.
        actual = resource_contract(plan["query"], plan0)
        same(actual, plan["contract"], "callback contract drift")
        within[mode] = dict(
            source=source,
            dispatch=private,
            embedded_elf=ctx.caps[mode]["elf"],
            node=ctx.caps[mode]["node"],
            om=model["om"],
            abi=model["abi"],
            memory=memory_contract(actual),
            lifetime=lifetime_contract(mode),
        )
    same(within[DOUBLE[0]]["memory"], within[DOUBLE[1]]["memory"], "B/C layout or complete Cube differs")
    # These citations are accepted only for the exact generated kernel digests and exact SDK files above.
    return dict(
        scope="C16_LAYER0_GATE_ONLY",
        trust="Audited Ascend C source/SDK semantics and sealed CANN compilation provenance",
        labels=LABELS,
        strict=ctx.negative,
        variants=within,
        sdk=sdk,
        toolkit_install=install,
        frozen=frozen,
        direct_same_core=dict(
            sdk=sdk_refs(sdk, "matmul_intf.h", "matmul.h", "tiling.h", "kfc_register_obj.h"),
            basis="Private device source rejects split-core/client macros, asserts CFG_NORM, direct MatmulImpl; ATC Arch 2002",
        ),
        issued_reader=dict(
            sdk=sdk_refs(
                sdk,
                "scheduler_norm.h",
                "scheduler_base.h",
                "copy_cube_in_using_ub.h",
                "matmul_tensor_info.h",
                "data_copy_wrapper_using_ub_nd.h",
            ),
            basis="Aligned B UB->L1 issued on same core during Iterate; MMAD consumes L0B; GetTensorC uses the accumulator",
        ),
        reuse_fence=dict(
            sdk=sdk_refs(sdk, "kernel_reg.h", "kernel_operator_block_sync_intf_impl.h"),
            basis="Original PIPE_ALL drains previously issued operations on this core before next bind or old-bank overwrite",
        ),
        ub_and_scratch=dict(
            sdk=sdk_refs(
                sdk,
                "matmul_local_workspace.h",
                "kernel_tensor_impl.h",
                "kernel_operator_vec_brcb_impl.h",
                "kernel_utils_base.h",
                "kernel_utils_constants.h",
            ),
            basis="TPipe physical base checks precede payload use; Matmul workspace exclusive; expanded rear and fixed Brcb scratch accounted",
        ),
        events=dict(
            sdk=sdk_refs(sdk, "kernel_tpipe_impl.h", "kernel_event.h", "tbuf_pool_l0_v200.h"),
            basis="MTE2_S and MTE2_V reserved before DMA; each issued pair waited before reuse/release. Immediate V_S before binding B",
        ),
        failure_cases=[
            "InitBuffer/address mismatch before payload",
            "failed raw ID preflight/partial reservation before DMA",
            "V_S Fetch exhausted before SetTensorA/B or SDK MoveNext false before Compute: wait only issued raw pair; fence; no publish; release; End",
            "device exception/failed sync: preserve error and stop; no successful drain inferred",
        ],
        numerical="k=0 partial=false, k=1..9 partial=true, one FP16 GetTensorC per N tile",
        unresolved=[
            "actual instruction order",
            "actual bank traffic or hardware overlap",
            "device observations before Smoke",
        ],
    )
