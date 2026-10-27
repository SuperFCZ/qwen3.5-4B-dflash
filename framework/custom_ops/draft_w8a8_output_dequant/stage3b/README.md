# Stage 3B-1: ordered NPU reduction of accepted Stage 3A partials

This is a new `DFlashW8A8PartialReduce` operator, private OPP and ACLNN runner.
It consumes the **saved real NPU outputs** of Stage 3A, not regenerated K128
outputs. Stages 1/2A/2B/2C/3A, their OPPs and Golden files remain unchanged.
Local CPU checks do not establish CANN compilation or NPU correctness: both
are **NOT_RUN** until the server executes the commands below.

## ABI and arithmetic

- Input: contiguous FP16 ND `[20,16,64]`, group order 0 through 19.
- Output: contiguous FP16 ND `[16,64]`.
- Start with FP32 positive zero. Cast each group's FP16 values to FP32, add to
  the FP32 accumulator in group order, then perform one final FP32-to-FP16
  `Cast(CAST_NONE)`. There is no tree reduction, FP16 accumulator, clamp or
  tolerance. The CPU reference uses sequential float32 addition and final
  nearest-even float16 conversion; hardware agreement is a smoke-test gate.
- One block, no Cube/Matmul, no timing, no earlier operator execution.

| Resource | Bytes |
| --- | ---: |
| One input VECIN queue buffer | 2,048 |
| FP32 conversion buffer | 4,096 |
| FP32 accumulator | 4,096 |
| One output VECOUT queue buffer | 2,048 |
| Explicit UB total | 12,288 |
| Conservative UB headroom | 8,192 |
| Minimum actual platform UB required by Host | 20,480 |
| User L1 / L0A / L0B / L0C | 0 |
| Operator-requested GM workspace | 0 |
| GM input / output | 40,960 / 2,048 |

The Host queries `PlatformAscendC(c->GetPlatformInfo())`; the headroom is a
conservative admission threshold, not a claim that the SDK allocates 8 KiB.
ACLNN's actual workspace size is separately recorded by the runner.
Queue depth is one. VECIN EnQue/DeQue establishes MTE2-to-Vector readiness;
PIPE_V barriers order Cast/Add and scratch reuse. The output queue establishes
Vector-to-MTE3 readiness and the kernel ends with PIPE_ALL. Complete
elementwise overlap in `Add(sum,sum,temp)` does not introduce dependencies
between vector repeats. Device synchronization and subnormal/signed-zero
behavior still require the strict NPU comparison, not the CPU emulator.

## Evidence gates

The driver requires `STAGE3A_20_GROUP_MATCH`, all 20 ordered group records,
two accepted repeats per group, each recorded execution/guard/readonly/poison
check, source/tool/SDK hashes and all candidate input/reference hashes. It
rechecks the saved K128 outputs against their existing candidate references.
Both saved repeats must produce exactly the same packed input.

It then independently sums the saved partials and demands byte identity with
Stage 3A's saved CPU `partial_f32` and `partial_f16` references and both saved
host aggregations. These references are copied into the new directory without
changing their hashes. Altered, missing, reordered or duplicate evidence fails
before building or running the reduction. Native Full-K semantics are not a
dependency; no Native output is substituted for this Golden.

The runner executes twice with different poison patterns, verifies poison
readback, all guard regions and the entire readonly input, and saves both
outputs. Success requires exact FP16 bits, stable repeats and finite outputs.
No warmup or timed calls are enabled. Failure leaves logs and a failed report.

## Server commands

Run in the already accepted Ascend310P3/CANN 9.0.0 environment. Each destination
must be new. `set -e` stops before Smoke if Build fails.

```bash
set -euo pipefail
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
S3A=/home/w00949577/qwen35-dflash-run/w8a8-stage3a-20261027-011523
S3B=/home/w00949577/qwen35-dflash-run/w8a8-stage3b-$(date +%Y%m%d-%H%M%S)
ENTRY=framework/custom_ops/draft_w8a8_output_dequant/stage3b/run.sh
bash "$ENTRY" prepare --stage3a-root "$S3A" --device 0 --output-dir "$S3B/prepare"
bash "$ENTRY" build --stage3a-root "$S3A" --device 0 --output-dir "$S3B/build"
bash "$ENTRY" smoke --stage3a-root "$S3A" --device 0 \
  --build-report "$S3B/build/report.json" --output-dir "$S3B/smoke"
printf 'Evidence: %s\n' "$S3B"
```

The driver accepts exactly one of `$S3A/npu/report.json` and `$S3A/report.json`.
If both exist, explicitly provide `--stage3a-report /path/to/accepted/report.json`
on every invocation; it never guesses between reports.

Expected success states are `STAGE3B1_CPU_PREPARED_NPU_NOT_RUN`,
`STAGE3B1_BUILT_NPU_NOT_RUN`, then **only after device execution**
`STAGE3B1_REDUCTION_MATCH`. This says nothing about a fused 20-group kernel,
N19456, full Draft or performance.

Return `build/report.json`, `smoke/report.json`, and
`smoke/execute/{runner.log,execution.json}`. On compile failure also return
`build/{msopgen.log,build.log,cmake.log,runner-build.log}` if present. On a
numerical failure retain `smoke/data/case.json`, references and both
`smoke/execute/actual-*.bin`; do not change Golden or rerun earlier stages.

## Bounded VDEQ source inspection

Build also collects **four explicitly named files**, with source SHA256,
selected line ranges and snapshots, into `build/vdeq-static/`. This does not
scan the SDK tree or declare hardware support. It distinguishes the accepted
INT32 L0C → enhanced DataCopy(VDEQ16) → FP16 route from later NZ/ND layout
handling. Actual low-level intrinsic/condition review remains UNRESOLVED until
these server excerpts are read. Missing source is recorded; it is not a
reduction prerequisite.

To collect only these files, without compiling or executing any kernel:

```bash
bash "$ENTRY" inspect-vdeq --stage3a-root "$S3A" --device 0 \
  --output-dir "$S3B/static-only"
```

Return `vdeq-static/vdeq-static.json` and `vdeq-static/vdeq-excerpts.txt` for
source-level review. SDK snapshots and excerpts are server evidence only;
they must not be committed. See [DESIGN_3B2.md](DESIGN_3B2.md) for the conditional
fusion design and outstanding LocalTensor output constraints.

## Local CPU checks

```bash
python3 -B -m pytest -q framework/custom_ops/draft_w8a8_output_dequant/stage3b/test
bash -n framework/custom_ops/draft_w8a8_output_dequant/stage3b/run.sh
```

The test-only scalar AscendC API emulator compiles the actual new kernel body
and compares ordered arithmetic with CPU references, checks fixed buffer
bounds and queue operations, and counts allocations/operations. It does **not**
model asynchronous hardware events, actual CANN overloads or device rounding.
Additional tests reject evidence tampering, lost/reordered groups, changed
Golden, guards/poison/readonly/repeat failure and accidental timing.
