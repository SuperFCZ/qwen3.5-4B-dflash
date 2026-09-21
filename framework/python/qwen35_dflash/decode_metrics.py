"""Dependency-free decode ratios; missing phases never use model totals."""
from __future__ import annotations

import math

SPEEDUP_SCOPE = "decode_loop"
SPEEDUP_NOTE = ("Decode speedup = sum ordinary decode-loop time / sum DFlash decode-loop time; "
                "prefill and pre-decode Draft context construction excluded.")
THROUGHPUT_NOTE = ("Generation tok/s and throughput ratios use actual output tokens / (prefill + decode).")


def measured_decode_ms(benchmark):
    values = [m.get("latency_ms", {}).get("decode") for m in benchmark.get("measurements", [])]
    return complete_total(values)


def complete_total(values):
    """No partial denominator when any admitted measurement lacks timing."""
    values = list(values)
    for value in values:
        if value is not None and (type(value) not in (int, float)
                                  or not math.isfinite(value) or value < 0):
            raise ValueError("invalid decode timing")
    return None if not values or any(v is None for v in values) else math.fsum(values)


def time_ratio(baseline_ms, candidate_ms):
    if baseline_ms is None or candidate_ms is None or baseline_ms <= 0 or candidate_ms <= 0:
        return None
    return baseline_ms / candidate_ms


def row_decode_ms(row, mode):
    field = mode + "_decode_measured_ms"
    if field in row:
        return complete_total([row[field]])
    # Saved summaries may predate the explicit decode denominator fields.
    phase = row.get("phase_timings", {}).get(mode + "_decode", {})
    return complete_total([phase.get("total_ms")]) if phase.get("available") else None


def aggregate_decode_ms(rows, mode):
    return complete_total(row_decode_ms(row, mode) for row in rows)
