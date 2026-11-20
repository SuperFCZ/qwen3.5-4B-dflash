"""Frozen source, scale layout and numerical contracts for one W8A16 Gate."""

import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OPS = HERE.parent
sys.path.insert(0, str(OPS / "draft_gate_w16_atc8"))
from w16_common import (
    API,
    CUBE_SHA,
    FIRST,
    GATE as GATE,
    GATE8_ROOT as GATE8_ROOT,
    HEADER,
    RAW_HEADER,
    SHAPE,
    SHAPE_KEY,
    STEM,
    c,
    command as command,
    flow,
    gate_kernel,
    historical,
    host as host,
    one,
    replace_once,
    require,
    resume as resume,
    tree,
)
from gate_common import KERNEL_SOURCE, config_graph as config_graph
from w16_source import FIRST_HASHES
from gate_plan import gate_plan
from c_profile_graph import read_om_json as read_om_json
from export_gate import verify_compiled_graph
from measurement import use_specs, verify_outputs

CONTROL = "raw"
CANDIDATE = "raw-transpose-b16"
MODES = (CONTROL, CANDIDATE)
REPORTS = {
    "source": (593380, "9daf545feddefb565ad239e4315637f3e4b4ede71e31e24959deace73c90c445"),
    "build": (268571, "88f80f9c861a52e19f6a142345f5e053a490735207f8067ab1e73328f6b6f43e"),
    "export": (655387, "f48cf05081f20a6f2289090ed1a246972153f0218df006499d93c28a56bf5f4d"),
    "c16-smoke": (297205, "84f8586225e786f67e7e7edeb30bac0ce0f00dc891cd878f83d455378da74894"),
    "full-benchmark": (963117, "42ad484dde381e9f17a53915e8610760c686517fe4a5a3663a9cc288ebd8970e"),
}
RAW_OM = "938d0fa4b937f7af67f46f63d904d729957e015635c3eaab8f5d54c843c242c5"
RAW_ELF = "73e570d30e465b81aa27842f77302b1bc18aaf30a1c2794db46def5ba2a04846"
RAW_PARENT = "0bbc0ee73d673d9141065d6bb73bee22b9714cc96bfe2e03c0477dfa68eae1a3"
EXPANDED_SOURCE_SHA = "ed11b76a2a602558e0f24bd393f33d7fe4246988326e13fd57205d22284f348b"
WRAPPER = "2fdf2abe2a51d8cd6faf09e3c70749093d69f1c80dd9be0b1de3bd27f2f4c4c7"
CALL = "AscendC::Brcb(expanded, sLocal[group * kTileN], 8, {1, 8});"


def render(data):
    require(hashlib.sha256(data).hexdigest() == KERNEL_SOURCE, "wrong Gate source")
    helper = (HERE / "kernel/scale_expand.inc").read_text()
    text = replace_once(data.decode(), "}  // namespace", helper + "\n}  // namespace")
    replacement = (
        "if (DFLASH_GROUP_QUANT_GATE_PREFETCH(globalM, globalK, globalN, tileK)) {\n"
        "                        ExpandGateScale(expanded, sLocal[group * kTileN]);\n"
        "                    } else {\n                        " + CALL + "\n                    }"
    )
    result = replace_once(text, CALL, replacement).encode()
    require(
        hashlib.sha256(result).hexdigest() == EXPANDED_SOURCE_SHA,
        "candidate differs from the reviewed and SDK-compiled translation unit",
    )
    return result


def sources():
    own = [c.identity(p) for p in sorted(HERE.rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
    # Freeze the readers/builders used by this workflow, including the original Runner protocol.
    return (
        own
        + historical.sources()
        + [
            c.identity(OPS / "draft_gate_w16_atc8" / p)
            for p in (
                "w16_common.py",
                "w16_graph.py",
                "w16_source.py",
                "w16_build.py",
                "w16_runtime.py",
                "w16_export.py",
                "w16_static.py",
                "w16_cfg.py",
                "w16_operands.py",
            )
        ]
    )


def sdk_files(cann):
    root = Path(cann).resolve(strict=True)
    results = {}
    for rel, digest in c.read(HERE / "sdk-files.json").items():
        p = (root / rel).resolve(strict=True)
        require(root in p.parents and c.sha(p) == digest, "unreviewed CANN file: " + rel)
        results[rel] = c.identity(p)
    info = root / "aarch64-linux/ascend_toolkit_install.info"
    require("version=9.0.0" in info.read_text().splitlines(), "expected CANN 9.0.0")
    return results


def expansion_map():
    mapping = [None] * 1024
    for rep in range(4):
        for dst_row in range(16):
            for src_row in range(16):
                dst = (rep * 16 + dst_row) * 16 + src_row
                # All 16 input VA entries select the SAME read-only block.
                src = rep * 16 + dst_row
                require(mapping[dst] is None, "overlapping destination iterations")
                mapping[dst] = src
    require(mapping == [i // 16 for i in range(1024)], "transpose does not match Brcb")
    return mapping


def layout_contract():
    table = expansion_map()
    for bank in range(2):
        for group in range(2):
            start = 32768 + bank * 256 + group * 128
            require(start % 32 == 0 and start + 128 <= 32768 + (bank + 1) * 256, "scale-bank extent")
            for channel in range(64):
                for i in range(128):
                    # Original Mul src1 block stride=0, repeat stride=1.
                    require(table[16 * channel + (i % 16)] == channel, "Mul consumes a foreign scale")
    return dict(
        shape=SHAPE,
        groups=20,
        n_tiles=304,
        k_tiles=10,
        raw_banks=2,
        layout="expanded[16*c+lane] = source[c] (FP16 bits)",
        repeat=4,
        dst_rep_stride_blocks=16,
        src_rep_stride_blocks=1,
        input_bytes_per_group=128,
        output_bytes_per_group=2048,
        source_rows=16,
        distinct_source_rows_per_repeat=1,
        dst_rows_per_repeat=16,
        storage_bytes=dict(
            raw_codes=32768,
            raw_scale=512,
            w16=32768,
            expanded_with_rear=2304,
            sdk_scratch=185600,
            sdk_reserve=8192,
        ),
        additional_tpipe_bytes=0,
        scale_words_per_gn=389120,
        gate_output_elements=311296,
    )


def same_plan(query, src, control):
    observed = gate_plan(query, "raw", src["full"]["16"]["anchor"]["plans"][SHAPE_KEY])
    require(
        observed == control
        and observed["parent_sha256"] == RAW_PARENT
        and observed["cube_sha256"] == CUBE_SHA
        and len(observed["cube_fields"]) == 50
        and observed["resources"]["scratch"] == 185600,
        "actual 8-block Raw parent, 50 Cube fields, workspace or resources changed",
    )
    return observed


def read_baseline(root):
    reports = {}
    refs = {}
    original_sources = historical.sources()
    for stage, (length, digest) in REPORTS.items():
        ref = c.identity(root / stage / "report.json")
        require((ref["bytes"], ref["sha256"]) == (length, digest), "frozen report drift: " + stage)
        row = c.read(c.verify(ref))
        historical.stage_status(stage, row)
        require(
            row["phase"] == stage and row["root"] == str(root) and row["sources"] == original_sources,
            "wrong Raw stage/source",
        )
        for old, item in refs.items():
            require(row["predecessors"].get(old) == item, "Raw predecessor drift: " + old)
        reports[stage] = row
        refs[stage] = ref
    src = reports["source"]["binding"]
    raw = reports["build"]["variants"]["raw"]
    model = reports["export"]["models"]["raw"][FIRST]
    worker = c.read(c.verify(reports["build"]["workers"]["raw"]))
    require(
        worker["status"] == "PRIVATE_GATE_PACKAGE_BUILT" and worker["result"] == raw, "wrong Raw OPP Build"
    )
    c.verify_opp(raw["opp"])
    require(
        tree(Path(raw["opp"]["install"])) == raw["package_files"]
        and gate_kernel(raw["opp"]) == raw["kernel"],
        "Raw package drift",
    )
    cpp = raw["generated_sources"]["op_kernel/" + STEM + ".cpp"]["file"]
    require(c.sha(c.verify(cpp)) == KERNEL_SOURCE and raw["api"]["sha256"] == API, "Raw math or API changed")
    c.verify(raw["api"])
    c.verify(raw["runner"]["runner"])
    for side in ("op_host", "op_kernel"):
        require(
            c.sha(c.verify(raw["generated_sources"][side + "/" + HEADER]["file"])) == RAW_HEADER,
            "wrong actual header",
        )
    case = src["cases"]["16"][FIRST]
    files = dict(case["inputs"], golden=case["golden"])
    require(
        {k: c.sha(c.verify(i)) for k, i in files.items()} == FIRST_HASHES, "frozen Layer0 inputs/Golden drift"
    )
    require(
        (model["om"]["bytes"], model["om"]["sha256"]) == (50617371, RAW_OM)
        and model["private_kernel"] == raw["kernel"]
        and model["opp_root"] == raw["opp"]["install"],
        "bad control OM",
    )
    verify_compiled_graph(model, src, False)
    entry = one(model["atc_kernels"].values(), "not a single Gate")
    require(entry["sha256"] == RAW_ELF and entry["block_dim"] == 8, "wrong control embedded code")
    ctl = model["plans"][SHAPE_KEY]
    q = flow.read_query(
        c.verify(ctl["query"]["platform_file"]),
        c.verify(ctl["query"]["scheduling_file"]),
        raw["opp"],
        "ATC",
        "A",
        SHAPE,
    )
    require(q == ctl["query"], "archived callback drift")
    same_plan(q, src, ctl["contract"])
    smoke = one(
        (
            r
            for r in reports["c16-smoke"]["runs"]
            if (r["gear"], r["layer"], r["mode"]) == ("16", FIRST, "raw")
        ),
        "frozen Smoke inventory",
    )
    require(
        smoke["om"] == model["om"]
        and smoke["runner"] == raw["runner"]["runner"]
        and smoke["validation"] == verify_outputs(smoke, use_specs(src, "16", FIRST), False),
        "Raw Smoke not from the accepted model",
    )
    full = reports["full-benchmark"]["statistics"]["gears"]
    binding = dict(
        reports=refs,
        raw_node=entry,
        raw_om=model["om"],
        runner=raw["runner"],
        frozen=files,
        host_plan=ctl["contract"],
        generated_header=raw["generated_sources"]["op_kernel/" + HEADER]["file"],
        full_draft_historical_ms={g: full[g]["variants"]["raw"]["median_ms"] for g in ("16", "64")},
    )
    require(
        binding["full_draft_historical_ms"] == {"16": 30.56987, "64": 31.426295}, "wrong full Draft history"
    )
    return binding, src, raw, model


def candidate_source(raw):
    return render(c.verify(raw["generated_sources"]["op_kernel/" + STEM + ".cpp"]["file"]).read_bytes())
