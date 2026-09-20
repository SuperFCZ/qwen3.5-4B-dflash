"""Observed differences between measured generations; no device execution."""
from __future__ import annotations


def representative_output(report):
    """Use measured repetition 0, including reports predating observation fields."""
    runs = report.get("measurements", [])
    if runs:
        return runs[0].get("generated_token_ids"), runs[0].get("stop_reason")
    return (report.get("representative_generated_token_ids", report.get("stable_generated_token_ids")),
            report.get("representative_stop_reason", report.get("stable_stop_reason")))


def repeatability_observation(report):
    runs = report.get("measurements", [])
    if not runs:
        raise ValueError("repeatability requires raw measurements")
    expected, stop = representative_output(report)
    differences = []
    for repetition, run in enumerate(runs[1:], 1):
        actual = run["generated_token_ids"]
        indices = [i for i in range(max(len(expected), len(actual)))
                   if i >= len(expected) or i >= len(actual) or expected[i] != actual[i]]
        stop_changed = run["stop_reason"] != stop
        if not indices and not stop_changed:
            continue
        index = indices[0] if indices else None
        differences.append({
            "repetition": repetition, "token_id_mismatches": len(indices),
            "reference_token_count": len(expected), "token_count": len(actual),
            "stop_reason_changed": stop_changed,
            "reference_stop_reason": stop, "stop_reason": run["stop_reason"],
            "first_difference": None if index is None else {
                "index": index,
                "reference_token_id": expected[index] if index < len(expected) else None,
                "token_id": actual[index] if index < len(actual) else None,
            },
        })
    return {
        "status": "DRIFT_OBSERVED" if differences else "STABLE",
        "reference_repetition": 0, "compared_repetitions": len(runs) - 1,
        "differences": differences,
    }
