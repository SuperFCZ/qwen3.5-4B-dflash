# Target W8A8 Gate/Up on Ascend310P3

Each layer consumes one normalized FP16 activation with 2,560 channels and produces two
separate FP16 projections of width 9,216. The experiment preserves the receiver's
DynamicQuant and uses its actual weights, two scalar weight scales and per-row scales.
Prefill uses 64 physical rows and Verify uses 16.

Run from the checkout on the receiver.

```bash
bash framework/custom_ops/target_gate_up/run.sh --from-run target-conv-window-qyWq56
```

`--references-only` captures the native layer-0 inputs and outputs. Pass
`--resume target-gate-up-RUN` to reuse them. Artifacts and weight packs stay under the new
server run directory. The shared-DQ control and custom candidates expose the same two outputs
as the original pair. Replays check every FP16 bit on the real activation, sparse basis rows
and a mixed-scale stress input. Three timing rounds bracket each candidate sweep with native
controls. Selection requires lower medians in all three rounds and a non-increasing pooled P95.

The custom operator partitions output columns across eight AI Cores. Each core keeps A in L1.
Offline-reordered signed INT8 weight panels feed the Cube. The variants compare internal and
external packing of A, 128 and 256 output columns per tile, next-panel prefetch, and vector
versus on-transfer dequantization. Both output paths round the weight-scaled accumulator to
FP16 before multiplying by the FP16 row scale. The device checks decide numerical equivalence.

msprof records matrix-task time, the span including DynamicQuant and activation layout, and
pipeline counters. Host Execute+Sync is measured separately without profiling. For two weights
read once, the conditional lower bound is 47,185,920 bytes / 204.8 GB/s = **230.4 µs**.
Logical bandwidth counts each weight byte once. The full graph timing also charges activation
preparation, output writes and launches. This reference uses the Atlas 300I Pro published
specification. [Hardware specifications](https://e.huawei.com/cn/products/computing/ascend/atlas-300-ai)
