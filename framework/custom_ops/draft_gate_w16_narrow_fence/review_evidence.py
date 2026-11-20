"""Read-only review of the original Gate measurements and the CANN headers. CPU only."""

import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

from fence_contract import ALL, MTE3, NARROW, W_EVENT, missing_edges, simulate

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "draft_gate_w16_experimental"))
from contracts import (
    CUBE_SHA,
    DOUBLE,
    FIRST,
    FIRST_HASHES,
    GATE_ELF,
    GATE_SOURCE,
    LABELS,
    RAW_HEADER,
    c,
    memory_contract,
    render_kernel,
    require,
)
from measurement import verify_execution
from w16_runtime import decision, paired
from raw_protocol import stats


def research_signal(control, reference_rounds, candidate_rounds):
    """Preregistered numerical portion of Compare. Never grants a device launch."""
    require(control in ("single-w-raw", "double-w-overlap"), "unknown comparator")
    require(len(reference_rounds) == len(candidate_rounds) == 3, "three independent rounds required")
    per_round = []
    for ref, new in zip(reference_rounds, candidate_rounds):
        a, b = stats(ref, 30), stats(new, 30)
        per_round.append(a["median_ms"] / b["median_ms"])
    a = stats([x for r in reference_rounds for x in r], 90)
    b = stats([x for r in candidate_rounds for x in r], 90)
    ref, new = map(lambda x: Decimal(str(x)), (a["median_ms"], b["median_ms"]))
    median_ok = new <= ref * Decimal("0.97") if control == "single-w-raw" else ref >= new * Decimal("1.03")
    return dict(
        median_ok=median_ok,
        round_ratios=per_round,
        p95_regression=b["p95_ms"] > a["p95_ms"],
        signal=median_ok and min(per_round) > 1 and b["p95_ms"] <= a["p95_ms"],
    )


REPORT_HASH = {
    "compare": "7d3c15d762b9ddcb3efc67f111b5987ee430d8087b73d7c6ba28f01bc10de74a",
    "schedule-benchmark": "d8a14c40324e334b43ff71be9cfc7035e30b21e8072cb31ed095999cc2608609",
    "net-benchmark": "2814877e85df90f365cb97edbe6270b638cc7b585b655287cb578290ebce46d0",
}


def source_check():
    raw = c.QUANT / "op_kernel/d_flash_group_quant_linear.cpp"
    device = HERE.parent / "draft_gate_w16_atc8/kernel"
    for p, sha in (
        (raw, "a4861bb5f7e38aceced9e4d89fc6c2519cb4a80e0914d7dd418c50389e8f1a4f"),
        (device / "tile_schedule.h", "047fdbb268dd0fa5b5bd963818bc7d5f5f20d1e4a96ca5bf5bcd78464bed89b4"),
        (device / "double_w.inc", "a34c21ee4acac336c3c29803948ad6aa718befd1df824e211f9eae05025b2457"),
    ):
        require(c.sha(p) == sha, "source differs from archived review: " + str(p))
    for mode in DOUBLE:
        require(
            hashlib.sha256(render_kernel(raw.read_bytes(), mode)).hexdigest() == GATE_SOURCE[mode],
            "regenerated source differs: " + mode,
        )


def sdk_check(root):
    manifest = json.loads((HERE / "sdk-evidence.json").read_text())
    base = (root / manifest["root"]).resolve(strict=True)
    items = {}
    for row in manifest["files"]:
        path = (base / row["path"]).resolve(strict=True)
        require(base in path.parents and row["path"] not in items, "duplicate or out-of-tree SDK header")
        data = path.read_bytes()
        require(hashlib.sha256(data).hexdigest() == row["sha256"], "SDK SHA256 mismatch: " + str(path))
        lines = data.splitlines()
        for part in row["lines"].split(","):
            bounds = [int(n) for n in part.split("-")]
            require(1 <= bounds[0] <= bounds[-1] <= len(lines), "SDK citation out of range: " + str(path))
        items[row["path"]] = row["sha256"]
    return items


def experimental_results(root, sdk):
    final_path = root / "compare/report.json"
    require(c.sha(final_path) == REPORT_HASH["compare"], "different historical comparison")
    final = c.read(final_path)
    require(
        final["status"] == "W16_EXPERIMENTAL_NO_GO"
        and final["root"] == str(root)
        and final["labels"] == LABELS
        and not final["error"]
        and not final["integrity_error"],
        "invalid archived outcome",
    )
    stages = {}
    for phase, item in final["predecessors"].items():
        path = c.verify(item)
        require(
            path == root / phase / "report.json" and item["sha256"] == REPORT_HASH.get(phase, item["sha256"]),
            "archived predecessor drift: " + phase,
        )
        stages[phase] = c.read(path)
    safety = stages["admit"]["safety"]
    require(
        all(safety["frozen"][key]["sha256"] == digest for key, digest in FIRST_HASHES.items()),
        "Layer0 weights/input/Golden drift",
    )
    for item in safety["frozen"].values():
        c.verify(item)
    for rel, item in safety["sdk"].items():
        require(sdk[rel] == item["sha256"], "different admitted SDK: " + rel)
        c.verify(item)
    for mode in DOUBLE:
        row = safety["variants"][mode]
        require(
            c.identity(c.verify(row["source"]["kernel"]))["sha256"] == GATE_SOURCE[mode]
            and c.identity(c.verify(row["source"]["generated_header"]))["sha256"] == RAW_HEADER
            and c.identity(c.verify(row["embedded_elf"]))["sha256"] == GATE_ELF[mode],
            "wrong baseline image",
        )
        require(memory_contract(row["memory"]["actual_atc"]) == row["memory"], "dual-W plan changed")
    summaries = {}
    for tag in ("schedule", "net"):
        phase = tag + "-benchmark"
        report = stages[phase]
        require(
            report["status"] == "W16_EXPERIMENTAL_" + phase.upper().replace("-", "_") + "_PASS"
            and not report["error"]
            and not report["integrity_error"],
            "invalid benchmark: " + phase,
        )
        for row in report["runs"]:
            plan = row["atc_plan"]
            resources = plan["resources"]
            single = row["mode"] == "single-w-raw"
            require(
                plan["block_dim"] == 8
                and resources["tile_k"] == 256
                and resources["raw_banks"] == 2
                and resources["weight_bytes"] == (32768 if single else 65536)
                and resources["scratch"] == (185600 if single else 152832)
                and resources["l0b_used"] == 65536
                and resources["workspace"] == 2097152,
                "changed launch or Gate storage",
            )
            require(
                row["labels"] == LABELS
                and row["atc_plan"]["cube_sha256"] == CUBE_SHA
                and row["golden"]["y"] == safety["frozen"]["golden"],
                "different Gate benchmark",
            )
            execution = c.read(c.verify(row["execution"]))
            require(
                verify_execution(execution, True) == row["validation"], "raw samples or validation changed"
            )
        summary = paired(report["runs"], "16", FIRST, tag)
        require(summary == report["statistics"] == final["comparison"][tag], "paired statistics drift")
        summaries[tag] = summary
    require(decision(summaries["schedule"], summaries["net"]) == final["comparison"], "decision drift")
    return {
        p: dict(
            variants={m: [v["median_ms"], v["p95_ms"], v["count"]] for m, v in x["variants"].items()},
            ratios=x["round_ratios"],
            median_ratio=x["ratio"],
            p95_regression=x["p95_regression"],
            decision=x["decision"],
        )
        for p, x in summaries.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--sdk-root", required=True, type=Path)
    parser.add_argument("--experiment", type=Path, help="optionally recheck the measured YHsAKu archive")
    args = parser.parse_args()
    source_check()
    hashes = sdk_check(args.sdk_root)
    result = experimental_results(args.experiment.resolve(strict=True), hashes) if args.experiment else None
    comparisons = {}
    for policy in (MTE3, W_EVENT, ALL, NARROW):
        t = simulate(fence=policy, n_tiles=2)
        comparisons[policy] = dict(Counter(kind for kind, _, _ in missing_edges(t, 2)))
    trace = simulate(NARROW, NARROW, n_tiles=2)
    require(
        not missing_edges(trace, 2), "proposed steady-state constraints do not cover modelled memory accesses"
    )
    print(
        json.dumps(
            dict(
                sdk_files_verified=len(hashes),
                historical=result,
                source_model="REVIEWED",
                missing_storage_edges_in_two_tile_model=comparisons,
                unproven_common_raw_rearms=[
                    {"wait": a, "following_set": b}
                    for key, a, b, ok in trace.queues.rearms
                    if key == ("MTE2", "V", "raw") and not ok
                ],
                blockers=[
                    "Unchanged raw prologue needs same-ID consume/rearm evidence",
                    "No candidate ATC ELF or independently decoded CFG, operands and event generations",
                ],
                stages={
                    "Source": "DESIGN_REVIEWED_ONLY",
                    "Build": "NOT_RUN",
                    "ATC": "NOT_RUN",
                    "Static Admission": "BLOCKED",
                    "Smoke": "NOT_RUN",
                    "Benchmark": "NOT_RUN",
                    "Compare": "NOT_RUN",
                },
                candidate=NARROW,
                device_admission="BLOCKED_MACHINE_ADMISSION",
                build="NOT_RUN",
                atc="NOT_RUN",
                smoke="NOT_RUN",
                benchmark="NOT_RUN",
                device_launch_authorized=False,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
