"""Recover only the measured single-W full-K control for its direct W-bank comparison."""

from pathlib import Path

from gate_panel import (
    ABI,
    FIRST,
    HERE,
    K,
    N,
    OP,
    SHAPE_KEY,
    c,
    pack_inputs,
    panel_header,
    require,
    sha,
    validate_plan,
)
from bench import TILE_MODE, SHAPE, flow, option, read_om_json, recheck_embedded, selected_image

CONTROL = "panel-k2560"
# The original single-W source and generated header in the first full-L1 run (da1350e).
ORIGINAL = (
    "0a4c22892fb2a5e825c25dc0c892af92447600fdc458835412f5e99380255f5d",
    "4db97ba7748374fee807ac343f8883f934982d9e2cf5e0022cbd1270b626fca5",
)


def load_control(root, tools, files, label, stats_for_pair):
    experiment = root.resolve(strict=True) / "experiment"
    folder = experiment / CONTROL
    receipt = c.identity(experiment / "result.json")
    old = c.read(c.verify(receipt))
    require(
        old["status"] in ("MEASURED", "MEASURED_WITH_REJECTIONS")
        and old["data_source"] == label
        and old["tools"] == tools,
        "different measurement, dataset or compiler tools",
    )
    require(
        set(files) == set(old["inputs"])
        and all(all(old["inputs"][n][k] == row[k] for k in ("bytes", "sha256")) for n, row in files.items()),
        "new inputs or reference differ from the single-W control",
    )
    rows = [r for r in old["runs"] if r["pair"] == CONTROL]
    summary = stats_for_pair(rows, CONTROL)
    require(
        summary == old["statistics"][CONTROL] and summary["decision"] == "GO",
        "single-W control lacks its measured three-round GO",
    )
    smoke = [r for r in old["smoke"] if r["mode"] == CONTROL]
    require(
        len(smoke) == 2
        and {r["case"] for r in smoke} == {"dense", "boundaries"}
        and all(r["bit_mismatches"] == 0 and r["launch_authorized"] for r in smoke),
        "incomplete single-W correctness evidence",
    )
    c.verify_tree(rows + smoke)
    model, pkg = old["models"][CONTROL], old["packages"][CONTROL]
    runner = old["runner"]
    require(
        runner == c.identity(experiment / "runner-build/gate_om_runner")
        and model["mode"] == CONTROL
        and model["abi"] == ABI
        and model["key"] == FIRST
        and model["opp_root"] == pkg["opp"]["install"] == str(folder / "build/opp")
        and model["private_kernel"] == pkg["kernel"],
        "incorrect control binary, ABI or OPP",
    )
    c.verify_opp(pkg["opp"])
    c.verify_tree([model, pkg])
    cpp = c.verify(pkg["source"])
    generated = (c.sha(cpp), c.sha(cpp.parent / "panel_spec.h"))
    current = (c.sha(HERE / "op_kernel/full_l1.cpp"), sha(panel_header(K)))
    require(
        Path(pkg["opp"]["install"]) in cpp.parents
        and generated in (ORIGINAL, current)
        and c.sha(cpp.parent / "panel_vector.h") == c.sha(HERE / "panel_vector.h")
        and pkg["host_source"]["sha256"] == c.sha(HERE / "op_host/d_flash_gate_panel.cpp"),
        "unexpected full-L1 control implementation",
    )
    d = folder / "atc"
    require(
        all(
            model[name] == c.identity(d / file)
            for name, file in (
                ("air", "gate.air"),
                ("om", "gate.om"),
                ("dump", "om-graph.json"),
                ("deployment", "compile.json"),
            )
        ),
        "cross-wired single-W model files",
    )
    compile_record = c.read(c.verify(model["deployment"]))
    c.verify_tree(compile_record)
    require(
        option(compile_record["argv"], "--model") == model["air"]["path"]
        and option(compile_record["argv"], "--output") + ".om" == model["om"]["path"],
        "ATC model and command differ",
    )
    q = flow.read_query(d / "platform.jsonl", d / "scheduling.jsonl", pkg["opp"], "ATC", "A", SHAPE)
    require(
        q == compile_record["query"] == model["plans"][SHAPE_KEY]["query"],
        "single-W host observations differ",
    )
    proof = validate_plan(q, old["models"][TILE_MODE], K)
    require(
        all(proof[k] == v for k, v in model["plans"][SHAPE_KEY]["contract"].items()),
        "single-W full-K resource contract changed",
    )
    nodes = [
        n
        for n in read_om_json(c.verify(model["dump"]))
        if n["type"] not in ("Data", "Const", "FileConstant", "NetOutput")
    ]
    require(
        len(nodes) == 1 and nodes[0]["type"] == OP and compile_record["selected"] == model["atc_kernels"],
        "incorrect full-K embedded node inventory",
    )
    image = selected_image(nodes[0], q["platform"]["blockDim"], OP)
    recheck_embedded(c.verify(model["om"]), [image], model["atc_kernels"])
    packing = compile_record["packing"]
    require(
        packing["panel_k"] == K
        and packing["physical_shapes"] == [[N // 64, 1, K * 64], [N // 64, K // 128, 64]]
        and packing["original_nz"] == old["inputs"]["w_nz.bin"]
        and packing["original_gn"] == old["inputs"]["s_gn.bin"]
        and len(packing["constants"]) == len(compile_record["constant_offsets"]) == 2,
        "full-K constant bindings differ",
    )
    qw, sc, _ = pack_inputs(
        c.verify(files["w_nz.bin"]).read_bytes(), c.verify(files["s_gn.bin"]).read_bytes(), K
    )
    with c.verify(model["om"]).open("rb") as stream:
        for expected, bound, offset in zip(
            (qw, sc), packing["constants"], compile_record["constant_offsets"]
        ):
            stream.seek(offset)
            require(
                stream.read(len(expected)) == expected
                and bound["sha256"] == sha(expected)
                and bound["bytes"] == len(expected),
                "full-K embedded weight or scale changed",
            )
    c.verify(receipt)
    return model, pkg, runner, packing, receipt
