"""Read a completed two-OM build from an explicitly named fresh Gate run."""

from pathlib import Path

from bench import (
    ABI,
    FIRST,
    MODES,
    NZ_MODE,
    TILE_MODE,
    OLD_OP,
    OP,
    PARENT,
    SHAPE,
    SHAPE_KEY,
    allocated_sections,
    c,
    flow,
    read_om_json,
    recheck_embedded,
    require,
    selected_image,
)
from layout import OLD_STEM, kernel_source, pack, sha
from scale_contracts import render


def eligible(report):
    compiled = report.get("status") == "COMPILED_NOT_EXECUTED"
    prelaunch = (
        report.get("status") == "FAILED"
        and report.get("phase") == "smoke " + NZ_MODE
        and "unbound OM DSO:" in report.get("error", "")
        and report.get("smoke") == []
        and report.get("runs") == []
    )
    require(compiled or prelaunch, "expected a completed fresh build or a first pre-launch DSO rejection")
    require(set(report["models"]) == set(report["packages"]) == set(MODES), "incomplete Gate build")


def load_build(root, tools, files, label):
    ab = root.resolve(strict=True) / "ab"
    receipt = c.identity(ab / "result.json")
    old = c.read(c.verify(receipt))
    eligible(old)
    require(old["data_source"] == label and old["tools"] == tools, "different input protocol or SDK tools")
    c.verify_tree(old)
    require(old["runner"] == c.identity(ab / "runner-build/gate_om_runner"), "wrong Gate runner")
    require(
        set(files) == set(old["inputs"])
        and all(all(old["inputs"][n][k] == v[k] for k in ("sha256", "bytes")) for n, v in files.items()),
        "rebuilt inputs or reference differ",
    )
    b16 = render((c.QUANT / "op_kernel" / (OLD_STEM + ".cpp")).read_bytes())
    tiles, _ = pack(c.verify(files["w_nz.bin"]).read_bytes())
    tile_hash = sha(tiles)
    require(
        old["packing"]["tile_sha256"] == tile_hash
        and old["packing"]["scale_sha256"] == files["s_gn.bin"]["sha256"],
        "tile packing differs",
    )
    queries = []
    writable = []
    code = []
    for mode, op, d in ((NZ_MODE, OLD_OP, "nz-atc"), (TILE_MODE, OP, "tile-atc")):
        pkg = old["packages"][mode]
        model = old["models"][mode]
        opp = pkg["opp"]
        expected = ab / ("build-" + mode) / "opp"
        require(
            Path(opp["install"]) == expected
            and model["opp_root"] == str(expected)
            and model["private_kernel"] == pkg["kernel"]
            and model["abi"] == ABI
            and model["mode"] == mode
            and model["key"] == FIRST,
            "cross-wired Gate model/private package",
        )
        c.verify_opp(opp)
        require(
            c.verify(pkg["source"]).read_bytes() == (b16 if mode == NZ_MODE else kernel_source(b16)),
            "different built device source",
        )
        folder = ab / d
        require(
            model["om"] == c.identity(folder / "gate.om")
            and model["deployment"] == c.identity(folder / "compile.json"),
            "wrong compiled model/receipt",
        )
        record = c.read(c.verify(model["deployment"]))
        c.verify_tree(record)
        q = flow.read_query(folder / "platform.jsonl", folder / "scheduling.jsonl", opp, "ATC", "A", SHAPE)
        require(
            record["query"] == model["plans"][SHAPE_KEY]["query"] == q
            and q["platform"]["blockDim"] == 8
            and q["tiling_sha256"] == PARENT,
            "Gate launch plan changed",
        )
        nodes = [
            n
            for n in read_om_json(c.verify(model["dump"]))
            if n["type"] not in ("Const", "FileConstant", "Data", "NetOutput")
        ]
        require(len(nodes) == 1 and nodes[0]["type"] == op, "unexpected compute node in archived build")
        image = selected_image(nodes[0], q["platform"]["blockDim"], op)
        require(
            image == record["selected"]
            and record["om"] == model["om"]
            and record["air"] == model["air"]
            and record["dump"] == model["dump"],
            "compiler receipt differs",
        )
        recheck_embedded(c.verify(model["om"]), [image], model["atc_kernels"])
        weight = record["weights"]
        require(
            weight["sha256"] == (files["w_nz.bin"]["sha256"] if mode == NZ_MODE else tile_hash),
            "wrong packed weight identity",
        )
        with c.verify(model["om"]).open("rb") as f:
            f.seek(record["packed_constant_offset"])
            require(sha(f.read(weight["bytes"])) == weight["sha256"], "compiled weight bytes differ")
        code.append(model["atc_kernels"][image["node"]]["executable_sections"])
        writable.append(allocated_sections(model, image["node"])["writable"])
        queries.append(q)
    require(
        queries[0]["schedule"] == queries[1]["schedule"] and writable[0] == writable[1],
        "two builds differ in schedule or writable storage",
    )
    require(code[0] != code[1], "expected distinct NZ and tile-ND machine code")
    c.verify(receipt)
    return old, receipt
