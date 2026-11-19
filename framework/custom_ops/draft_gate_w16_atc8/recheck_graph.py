"""Read-only replay of the graph comparison from a retained single-Gate ATC failure."""

import argparse
import json
from pathlib import Path

from w16_common import DOUBLE, FIRST, GATE, c, one, require
from w16_graph import protected_graph
from c_profile_graph import read_om_json
from gate_plan import node_binding, embedded, recheck_embedded


def recheck(root):
    root = root.resolve(strict=True)
    stage = root / "atc"
    failed = c.read(stage / "report.json")
    require(
        failed["phase"] == "atc"
        and failed["root"] == str(root)
        and failed["folder"] == str(stage)
        and failed["status"] == "BLOCKED"
        and failed["error"]
        and not failed["integrity_error"]
        and failed["device_execution"] == failed["benchmark"] == "NOT_RUN",
        "not a retained ATC-only failure",
    )
    layer, mode = failed["active_case"]
    require(layer == FIRST and mode in DOUBLE, "not a first-layer dual-W ATC case")
    predecessors = failed["predecessors"]
    require(set(predecessors) == {"source", "build"}, "wrong ATC predecessors")
    for phase in ("source", "build"):
        require(c.verify(predecessors[phase]) == root / phase / "report.json", "crossed " + phase + " report")
    source, build = (c.read(predecessors[k]["path"]) for k in ("source", "build"))
    for phase, report in (("source", source), ("build", build)):
        require(
            report["phase"] == phase
            and report["root"] == str(root)
            and report["status"] == "W16_" + phase.upper() + "_PASS"
            and not report["error"]
            and not report["integrity_error"]
            and report["sources"] == failed["sources"],
            "wrong archived source/build chain",
        )
    require(
        not source["predecessors"] and build["predecessors"] == {"source": predecessors["source"]},
        "archived build/source mismatch",
    )
    history = source["binding"]["history"]
    archive = c.read(c.verify(history["export"]))
    require(
        archive["status"] == "GATE8_EXPORT_PASS"
        and not archive.get("error")
        and not archive.get("integrity_error")
        and archive["predecessors"]["build"] == history["build"],
        "wrong frozen Gate8 export",
    )
    baseline = archive["models"]["raw"][layer]
    raw_opp = c.read(c.verify(history["build"]))["variants"]["raw"]["opp"]
    selected = build["variants"][mode]["opp"]
    c.verify_opp(raw_opp)
    c.verify_opp(selected)
    scope = stage / mode / layer

    def retained(name):
        file = scope / name
        item = one((x for x in failed["failure_files"] if x["path"] == str(file)), "unbound " + str(file))
        c.verify(item)
        return item

    request = c.read(retained("request.json")["path"])
    child = c.read(retained("compiled/report.json")["path"])
    require(
        baseline["opp_root"] == raw_opp["install"]
        and request["opp"] == selected
        and request["frozen"]["air"] == baseline["air"]
        and request["output"] == str(scope / "compiled")
        and child["status"] == "GATE8_PROJECTION_AIR_COMPILED_NOT_EXECUTED"
        and child["execute_calls"] == 0
        and not child.get("error")
        and not child.get("integrity_error"),
        "worker/AIR/private OPP mismatch",
    )
    require(
        child["om"] in failed["failure_files"] and scope / "compiled" in Path(child["om"]["path"]).parents,
        "candidate OM outside retained worker artifacts",
    )
    gates = set(baseline["gate_nodes"].values())
    require(set(baseline["gate_nodes"]) == {layer} and len(gates) == 1, "wrong frozen Gate inventory")
    name = baseline["gate_nodes"][layer]
    new_dump = retained("om-graph.json")
    before, after = c.verify(baseline["dump"]), c.verify(new_dump)
    protected = protected_graph(
        before, after, gates, raw_opp_root=raw_opp["install"], new_opp_root=selected["install"]
    )

    def gate(path):
        n = one((n for n in read_om_json(path) if n["type"] == GATE), "expected exactly one Gate")
        require(n["name"] == name, "Gate name differs")
        return node_binding(n)

    left, right = gate(before), gate(after)
    recheck_embedded(c.verify(baseline["om"]), [left], baseline["atc_kernels"])
    embedded(c.verify(child["om"]), [right])
    require(left["sha256"] != right["sha256"], "selected the frozen single-W kernel")
    return dict(
        status="ARCHIVED_W16_GRAPH_MATCH",
        case=[layer, mode],
        protected_nodes=protected,
        raw_dump=baseline["dump"],
        candidate_dump=new_dump,
        raw_gate_elf=left["sha256"],
        candidate_gate_elf=right["sha256"],
        archived_atc_status=failed["status"],
        atc="NOT_RUN",
        device_execution="NOT_RUN",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(recheck(parser.parse_args().root), indent=2))
