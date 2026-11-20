"""Use the admitted ACL model Runner, its original live DSO gate, and frozen Native bytes."""

from decimal import Decimal

from evidence import CANDIDATE, CONTROL, FIRST, GEARS, LAYERS, MODES, fresh_layers, full_abi, pilot, require
from w16_runtime import case_run, verify_runtime_row


def layer_statistics(rows, scope, provenance):
    stat = pilot.statistics(rows)
    stat.update(scope=scope, sample_origin=provenance, historical_samples_used=provenance == "FROZEN_PILOT")
    return stat


def five_decision(gear, stats):
    require(set(stats) == set(LAYERS), "expected five independent measurement sets")
    sums = {m: sum(Decimal(str(stats[k]["variants"][m]["median_ms"])) for k in LAYERS) for m in MODES}
    for k, v in stats.items():
        require(
            v["scope"] == f"C{gear}/{k}"
            and len(v["round_ratios"]) == 3
            and all(v["variants"][m]["count"] == 90 for m in MODES)
            and v["sample_origin"] == ("FROZEN_PILOT" if (gear, k) == ("16", FIRST) else "FRESH"),
            "incorrect per-layer sample scope",
        )
    direction = all(all(r > 1 for r in v["round_ratios"]) for v in stats.values())
    tail = [k for k, v in stats.items() if v["p95_regression"]]
    passed = direction and not tail and sums[CANDIDATE] <= Decimal("0.97") * sums[CONTROL]
    return dict(
        gear=gear,
        scope="FIVE_ISOLATED_PROJECTIONS",
        layers=stats,
        no_p95_regression=not tail,
        layers_with_p95_regression=tail,
        all_layer_rounds_improved=direction,
        sum_of_layer_medians_ms={k: float(v) for k, v in sums.items()},
        weighted_latency_reduction=float(1 - sums[CANDIDATE] / sums[CONTROL]),
        decision="GO" if passed else "NO_GO",
    )


def groups(gear):
    return [(gear, k) for k in fresh_layers(gear)] if gear else [(g, "full") for g in GEARS]


def get_assets(ctx, chain, gear, layer):
    if layer == "full":
        export = chain["full-draft-export"]
        return (
            {CONTROL: ctx["builds"][CONTROL], CANDIDATE: export["packages"][gear]},
            {CONTROL: ctx["full"][gear], CANDIDATE: export["models"][gear]},
        )
    atc = chain[f"c{gear}-atc"]
    return ctx["builds"], {CONTROL: ctx["controls"][layer], CANDIDATE: atc["models"][layer]}


def runtime(out, ctx, chain, report, gear, timed, *, launch):
    src = ctx["src"]
    label = "scale-expansion-pair" if timed else "scale-expansion-smoke"
    prior = {}
    if timed:
        smoke = chain[f"c{gear}-smoke" if gear else "full-draft-smoke"]
        for row in smoke["runs"]:
            prior[row["gear"], row["layer"], row["mode"]] = row
    if launch:
        report["runs"] = []
    tasks = [(g, k, i, m) for g, k in groups(gear) for i, m in pilot.schedule(timed)]
    if not launch:
        require(
            [(r["gear"], r["layer"], r["round"], r["mode"]) for r in report["runs"]] == tasks,
            "extra, reused, missing or reordered Runner process",
        )
    by_case, memory = {}, {}
    for index, (g, k, i, m) in enumerate(tasks):
        builds, models = get_assets(ctx, chain, g, k)
        ref = prior.get((g, k, m))
        require(not timed or ref is not None, "missing matching Smoke")
        folder = out / f"c{g}" / k / (str(i) + "-" + m)
        constants = None
        if k == "full":
            full_abi(src["full"][g]["io"])
            constants = chain[f"c{g}-source"]["binding"]["raw_air"]["payloads"]
        if launch:
            report.update(active_case=[g, k, i, m], device_execution="ATTEMPTED")
            if timed:
                report["benchmark"] = "ATTEMPTED"
            case_run(
                folder, src, builds[m], models[m], g, k, m, i, label, timed, report, ref, constants=constants
            )
        row = report["runs"][index]
        require(
            (row["gear"], row["layer"], row["round"], row["mode"], row["pair"]) == (g, k, i, m, label),
            "incorrect run or data selection",
        )
        verify_runtime_row(folder, row, src, builds[m], models[m], timed, ref, constants=constants)
        require(
            all(
                all(
                    (v["sha256"], v["bytes"]) == (row["golden"][n]["sha256"], row["golden"][n]["bytes"])
                    for v in copies
                )
                for n, copies in row["outputs"].items()
            ),
            "repeat or cross-variant bits differ",
        )
        sizes = (row["validation"]["workspace_bytes"], row["validation"]["model_weight_bytes"])
        require(memory.setdefault((g, k), sizes) == sizes, "per-case runtime allocations changed")
        by_case.setdefault((g, k), []).append(row)
    if launch:
        report.update(
            device_execution="COMPLETED", benchmark="MEASURED" if timed else "NOT_RUN", active_case=None
        )
    if not timed:
        return None
    stats = {
        ((k if gear else g)): layer_statistics(rows, f"C{g}/{k}", "FRESH") for (g, k), rows in by_case.items()
    }
    if gear == "16":
        stats[FIRST] = layer_statistics(ctx["pilot"]["benchmark"]["runs"], f"C16/{FIRST}", "FROZEN_PILOT")
    return stats
