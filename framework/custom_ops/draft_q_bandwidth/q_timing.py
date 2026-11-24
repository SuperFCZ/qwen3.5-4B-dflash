"""Fresh, uninstrumented single-Q pairs. No profiled samples enter selection."""

from q_panel import CONTROL, PANEL, PREFETCH, MODES, SHAPE, BYTES, FLOPS
from run_gate import boundary_case, pair_statistics

PAIRS = ((CONTROL, PANEL), (CONTROL, PREFETCH), (PANEL, PREFETCH))


def smoke(folder, call, files, stress):
    # Independent two-term reference with distinct rows, Group128, baseK and full-K endpoints.
    sparse = boundary_case(
        folder / "boundaries",
        files,
        shape=SHAPE,
        pairs=(
            (0, 2559),
            (15, 16),
            (31, 32),
            (127, 128),
            (255, 256),
            (1279, 1280),
            (2047, 2048),
            (2431, 2432),
        ),
    )
    rows = []
    for mode in MODES:
        for label, data in (
            ("measured-tokens", files),
            ("dense-token-stress", stress),
            ("boundaries", sparse),
        ):
            rows.append(dict(call(folder / mode / label, mode, data, False, -1), case=label))
        print(
            f"Bitwise + guards PASS {mode} (65,536 FP16 values, three inputs, two poisons each)", flush=True
        )
    return dict(boundaries=sparse, runs=rows)


def timing(folder, call, files):
    rows = []
    for i in range(3):
        for pair in PAIRS[i:] + PAIRS[:i]:
            order = pair[::-1] if (i + PAIRS.index(pair)) % 2 else pair
            print(f"Round {i+1}: " + " → ".join(order), flush=True)
            key = "__".join(pair)
            for mode in order:
                row = call(folder / f"round-{i+1}" / key / mode, mode, files, True, i)
                rows.append(dict(row, pair=key))
    stats = {}
    for control, candidate in PAIRS:
        key = control + "__" + candidate
        stats[key] = pair_statistics(
            [r for r in rows if r["pair"] == key], candidate, control, flops=FLOPS, logical_bytes=BYTES
        )
    return dict(runs=rows, statistics=stats, selected=select(stats))


def select(stats):
    def go(a, b):
        return stats[a + "__" + b]["decision"] == "GO"

    if go(CONTROL, PREFETCH) and (not go(CONTROL, PANEL) or go(PANEL, PREFETCH)):
        return PREFETCH
    return PANEL if go(CONTROL, PANEL) else CONTROL


def display(measured):
    print("C16 Layer0 Q | single OM Execute+Sync | Median/P95 ms, 90 fresh samples per side/pair")
    for a, b in PAIRS:
        row = measured["statistics"][a + "__" + b]
        x, y = (row["variants"][m] for m in (a, b))
        print(
            f"{a} → {b}: {x['median_ms']:.6f}/{x['p95_ms']:.6f} → {y['median_ms']:.6f}/{y['p95_ms']:.6f} ms  "
            f"{row['speedup']:.4f}x  {row['logical_gbps']:.2f} logical GB/s  {row['decision']}",
            flush=True,
        )
    print("Selected:", measured["selected"], flush=True)
