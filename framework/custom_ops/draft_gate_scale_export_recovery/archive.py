"""Immutable inputs to the C16 full-export recovery. Reading them never launches ATC or a Runner."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "draft_gate_scale_expansion"))
from evidence import BASE, CONTROL, FULL_EXPORT_SHA, FULL_OMS, GATE8_ROOT, c, pinned, require

ARCHIVE = BASE / "gate-scale-expansion-dFFDXz"
FAILURE = (329533, "f26db9ca99d0910d74f7fafe78514f7fe1c4aa5810c03866d4fb6b237fbbbc08")
DECISIONS = {
    "16": "0006292ba982918bb7fc267ec34a6145d7acf22ed4a8b26b527392631c1881f9",
    "64": "90439a474e96b3a6ced084dcdc0994997bd1254760189c80df40adce674d2c1f",
}
WORKER = (130323, "ebe391e40ad80ade209a7423bb621bcc13c600c9377cbe6c33c493be0e21f180")
OM = (3090429133, "a34a7e2139a9e79082e9de8c859d221c70a19a777dc48e43680c1fd13fee8b57")


def archived(archive=ARCHIVE):
    fp = archive / "full-draft-export/report.json"
    ref = c.identity(fp)
    failed = pinned(ref, FAILURE[1], FAILURE[0])
    require(
        archive == ARCHIVE
        and failed["root"] == str(archive)
        and failed["folder"] == str(fp.parent)
        and failed["status"] == "BLOCKED"
        and failed["phase"] == "full-draft-export"
        and failed["active_case"] == ["16", "full"]
        and failed["atc"] == "ATTEMPTED"
        and failed["device_execution"] == failed["benchmark"] == "NOT_RUN"
        and failed["integrity_error"] is None
        and failed["error"]
        == "ValueError('non-target embedded implementation changed: DFlashGroupQuantLinear')",
        "different archived export failure",
    )
    for gear, digest in DECISIONS.items():
        item = failed["predecessors"][f"c{gear}-compare"]
        require(
            item["path"] == str(archive / f"five-gate-c{gear}/compare/report.json")
            and pinned(item, digest)["status"] == f"C{gear}_FIVE_GATE_GO",
            "wrong five-Gate decision",
        )
    raw = pinned(c.identity(GATE8_ROOT / "full-export/report.json"), FULL_EXPORT_SHA)["models"][CONTROL]["16"]
    cand = failed["models"]["16"]
    worker = pinned(cand["worker_report"], WORKER[1], WORKER[0])
    require(
        (cand["om"]["bytes"], cand["om"]["sha256"]) == OM
        and worker["om"] == cand["om"]
        and worker["status"] == "GATE8_FULL_AIR_COMPILED_NOT_EXECUTED"
        and worker["execute_calls"] == 0
        and raw["om"]["sha256"] == FULL_OMS["16"],
        "wrong OM or ATC worker",
    )
    for model in (raw, cand):
        c.verify(model["om"])
        c.verify(model["dump"])
    return ref, failed, raw, cand
