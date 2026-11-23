"""Build the three independently tested operators in one new msopgen vendor package."""

from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
OPS = HERE.parent
sys.path.insert(0, str(OPS / "draft_down_bandwidth"))
import down_panel as down
import gate_panel as gate
from layout import OLD_OP, OLD_STEM, HEADER, HEADER_SHA
from package import c, generate, tracked_template, build_install
from scale_contracts import EXPANDED_SOURCE_SHA, require


def build(out, cann, env):
    out.mkdir()
    raw_schema = c.read(c.QUANT / (OLD_OP + ".json"))
    panel_schema = c.read(OPS / "draft_gate_tile_nd/DFlashGateTileNd.json")[0]
    specs = raw_schema + [dict(panel_schema, op=name) for name in (gate.OP, down.OP)]
    spec = out / "operators.json"
    c.write(spec, specs)
    project = generate(out, spec, env, operators=(OLD_OP, gate.OP, down.OP))
    common = tracked_template(out, project / "CMakePresets.json", cann)
    for rel, row in common.items():
        if rel != "build-config.json":
            shutil.copyfile(c.verify(row["file"]), project / rel)
    host, dev = project / "op_host", project / "op_kernel"
    # Annotation headers under the shared host source implement scheduler A for all three.
    for name in ("scheduler_evidence.h", "platform_diag.h"):
        shutil.copyfile(OPS / "draft_a_core_scheduler_flow/op_host" / name, host / name)
    for module, banks in ((gate, 2), (down, 1)):
        header = "panel_spec.h" if module is gate else "down_spec.h"
        data = gate.panel_header(gate.K, banks) if module is gate else down.header(banks)
        for folder in (host, dev):
            (folder / header).write_bytes(data)
        stem = module.STEM
        (host / (stem + "_tiling.h")).write_text(
            (host / (OLD_STEM + "_tiling.h")).read_text().replace(OLD_OP, module.OP)
        )
        if module is gate:
            shutil.copyfile(gate.HERE / "op_host" / (stem + ".cpp"), host / (stem + ".cpp"))
            shutil.copyfile(gate.HERE / "op_kernel/full_l1.cpp", dev / (stem + ".cpp"))
        else:
            (host / (stem + ".cpp")).write_text(down.host_source())
            shutil.copyfile(down.HERE / "op_kernel" / (stem + ".cpp"), dev / (stem + ".cpp"))
    shutil.copyfile(gate.HERE / "panel_vector.h", dev / "panel_vector.h")
    require(
        c.sha(dev / (OLD_STEM + ".cpp")) == EXPANDED_SOURCE_SHA and c.sha(dev / HEADER) == HEADER_SHA,
        "wrong gate-B16/Down-Brcb control source",
    )
    root = build_install(out, project, cann, env)
    vendor = root / "vendors/customize"
    files = [
        dict(path=str(p), sha256=c.sha(p))
        for p in sorted(vendor.rglob("*"))
        if p.is_file() and (p.suffix in (".h", ".so", ".json") or ".so." in p.name)
    ]
    kernels = []
    sources = {}
    for op, stem in ((OLD_OP, OLD_STEM), (gate.OP, gate.STEM), (down.OP, down.STEM)):
        (binary,) = (vendor / "op_impl/ai_core/tbe/kernel/ascend310p" / stem).glob("*.o")
        meta = binary.with_suffix(".json")
        (cpp,) = vendor.rglob(stem + ".cpp")
        require(
            c.sha(binary) == c.read(meta)["sha256"] and c.sha(cpp) == c.sha(dev / cpp.name),
            "installed source or kernel differs: " + op,
        )
        headers = (
            [HEADER]
            if op == OLD_OP
            else ["panel_vector.h", "panel_spec.h" if op == gate.OP else "down_spec.h"]
        )
        for name in headers:
            require(c.sha(cpp.parent / name) == c.sha(dev / name), "dynamic header differs: " + name)
        kernels.append(
            dict(
                op=op,
                binary=str(binary),
                bytes=binary.stat().st_size,
                sha256=c.sha(binary),
                json=str(meta),
                json_bytes=meta.stat().st_size,
                json_sha256=c.sha(meta),
            )
        )
        sources[op] = dict(device=c.identity(cpp), host=c.identity(host / cpp.name))
    opp = dict(
        kind="PRIVATE_FULL_DRAFT_PANELS",
        install=str(root),
        preflight=dict(vendor=str(vendor), files=files, kernels=kernels),
    )
    c.verify_opp(opp)
    return dict(opp=opp, sources=sources)
