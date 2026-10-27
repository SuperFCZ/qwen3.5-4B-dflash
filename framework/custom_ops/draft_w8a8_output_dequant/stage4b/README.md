# Stage 4B: full Layer-0 Gate/Up correctness, 304 N64 tiles

Independent `DFlashW8A8FullGate`, M16/K2560/N19456, Group128. Previous stages,
OPPs, source locks and frozen evidence are unchanged. The development host
has no CANN/NPU: new device compilation/Tiling/Smoke are **NOT_RUN**. No
performance test, full Draft, Target/Verify or Stage4C work is started.

## Fixed ABI and unchanged per-tile path

| Tensor | Packed shape | Bytes |
| --- | --- | ---: |
| A INT8 ND | `[20,16,128]`, group,row,k | 40960 |
| B INT8 ND | `[304,20,128,64]`, tile,group,k,n | 49807360 |
| Scale UINT64 ND | `[304,20,64]`, tile,group,n | 3112960 |
| Output FP16 ND | `[16,19456]`, row,tile*64+n | 622592 |

The Stage4A Kernel sequence is retained: independent INT8 Cube per K128
group, SetQuantVector and VDEQ16 to Local NZ FP16, Cast to FP32, ordered Add
g=0..19, final FP16 conversion and NZ→ND GM output. No independent Vector
Scale Mul, integer cross-group accumulation, FP16 accumulator, tree/FMA,
prefetch, overlap or double buffering is added.

Core Policy remains dav_2002, nonzero aic/aiv/all, aic==aiv, all<=aic,
selected=GetCoreNum(). There is no hard-coded 7. Auto launch is
min(304,selected); explicit limits must not exceed selected. Each block owns
`tile=GetBlockIdx()+i*blockDim`, resets its FP32 sum for each tile, and writes
only `output[row*19456 + tile*64 : row*19456 + (tile+1)*64]`.

Single-core allocations are unchanged: SDK scratch 239104 B, user UB 12288 B,
guards 2560 B, headroom 8192 B. Arena sizing, offsets, system reservation,
guard protocol and Matmul lifecycle are copied from Stage4A. Host reruns the
same K128 per-core Tiling/resource checks. Larger GM inputs do not cause one
UB allocation per tile. Mode21 scheduler witnesses still run all groups
before emitting their tag. Normal mode0 is the only full numerical result.

## CPU preparation before Build

Prepare verifies the accepted Stage4A Build/Host-only/Smoke, numerical outputs,
safety flags and Core policy, and preserves original file identities. It
reads Stage3A's saved full INT8 Activation and saved activation-scale metadata;
**Stage4B does not recalculate or choose an Activation Scale**. Historical
read-only validators may recheck their own frozen quantization contracts;
their recomputation is not used as a new input/scale choice. New packed A
must be byte-identical to the accepted Stage4A A input.

For every tile 0..303 and group 0..19, NZ inverse bytes must match original
signed NK weights at the exact N/K ranges. Original FP16 GN scales combine
with the saved full-X scale under the unchanged FP64-product→FP32 rule and
existing uint64 codec. A bounded tile calculation uses INT64 dot, FP32 scaled
product, FP16 partial RNE, sequential FP32 group additions and final FP16 RNE.
No large full-N INT64/FP32 intermediate is constructed. B, words and references
are streamed into memory-mapped files; temporary dot/weight data is one tile
or group at a time. The final output and diagnostic matrices are small.

The historical Stage4A tiles `[0,1,15,16,151,152,302,303]` must match its saved
B, Scale words, CPU partials, final Golden and accepted final NPU outputs by
identity and bits. Other 296 tiles have independent CPU references. This
includes both sides of the Gate/Up boundary and the final output column.

`data/progress.json` is INCOMPLETE until all 304 tiles / 6080 NZ group checks
finish. `case.json` records every tile's range, input/word/Golden digests,
history match and encoding-boundary observations. Nonfinite candidate data
raises a structured error with tile/group/row/channel (and K index when
applicable). No Golden is clamped or repaired. Valid coefficient underflow
or subnormal observations are recorded with exact channel indices and remain
subject to strict NPU comparison.

Build requires a successful `--prepare-report` bound to the same source,
SDK and evidence. Smoke reuses that exact preparation through the Build
report; it does not regenerate a looser Golden. Candidate arithmetic is
rechecked tile-wise on admission, and file/source identities are checked
again on exit. Partial CPU preparation cannot authorize Build.

## Independent precision diagnostics

Prepare also computes an explicitly labeled W8A16 **CPU** reference:
`RN16(q * original FP16 scale_W)`, original FP16 X, separate FP32 products and
sequential k=0..2559 FP32 additions, then FP16 RNE. It uses no saturation and
is not asserted to match Native OM's reduction or conversion bits.

`accuracy-report.json` reports RMSE, max_abs, relative L2 and cosine over:

- all N19456 channels;
- Gate `[0:9728]`;
- Up `[9728:19456]`.

CPU candidate and actual NPU output are compared independently with this
reference. W8A16 differences are diagnostic only and never relax candidate
bitwise acceptance. An unavailable/nonfinite W8A16 CPU reference is explicitly
SKIPPED for affected scopes; candidate nonfinite data still fails preparation.
Native Full-K is SKIPPED because its semantics are not re-audited here. No CPU
reference is labeled Native Golden or used to alter historical evidence.

## Device sequence and acceptance

A fresh Stage4B Host-only run is required for the new Build. Normal Smoke
binds that proof and rechecks its Core tuple before each operator submission,
along with actual workspace allocation and disjoint arenas.

Normal correctness order is:

1. Automatic Core limit, mode21 then mode0, full 304 tiles.
2. Two-core control, mode21 then mode0; if auto is exactly two, use one core
   as the independent control. A scheduling limit below two blocks admission.

Every mode runs two poisoned repeats. Default is four runs / eight device
executions, all full shape. Optional `--diagnostic-groups 0 19` adds only the
first/last group modes to each schedule; it does not repeat all 21 modes.
All numerical outputs must agree between schedules, and every tile must have
zero bit mismatches with finite expected/actual output. The eight historical
tile comparisons are also reported separately. Workspace Admission, UB/GM
guards, Readonly, Poison, Repeat and all 311296 output elements remain gates.

Only complete auto and independent-control success produces
`STAGE4B_FULL_GATE_MATCH`. Any resource, execution, layout or numerical error
stops immediately, retaining failed outputs/logs. No fallback or resource
failure can produce a full PASS. Mode21 proves logical ownership, not measured
physical concurrency or performance.

## Server commands

Run in the accepted environment; all output directories must be new:

```bash
set -euo pipefail
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
S4B=/home/w00949577/qwen35-dflash-run/w8a8-full-gate-$(date +%Y%m%d-%H%M%S)
ENTRY=framework/custom_ops/draft_w8a8_output_dequant/stage4b/run.sh
bash "$ENTRY" prepare --device 0 --output-dir "$S4B/prepare"
bash "$ENTRY" build --device 0 --prepare-report "$S4B/prepare/report.json" \
  --output-dir "$S4B/build"
bash "$ENTRY" smoke --host-only --device 0 \
  --multi-build-report "$S4B/build/report.json" --output-dir "$S4B/host-only"
printf 'Evidence: %s\n' "$S4B"
```

After inspecting successful Host-only resource admission, in the same shell:

```bash
bash "$ENTRY" smoke --device 0 \
  --multi-build-report "$S4B/build/report.json" \
  --host-probe-report "$S4B/host-only/report.json" --output-dir "$S4B/smoke"
```

Default Stage4A evidence is
`/home/w00949577/qwen35-dflash-run/w8a8-core-policy-20261027-083223`.
Expected states are STAGE4B_PREPARED_NPU_NOT_RUN,
STAGE4B_BUILT_NPU_NOT_RUN, STAGE4B_HOST_TILING_ONLY (NPU NOT_RUN), then only
after actual full-device success STAGE4B_FULL_GATE_MATCH. Stage4C is NOT_RUN.

Return each `report.json`, `accuracy-report.json`, and Prepare's
`data/case.json`. On failure, include `error_detail`/`data/progress.json` for
reference failures, compiler logs for build failures, or failed
`execute/limit-*/mode-*/{runner.log,execution.json,resource-admission.json,actual-*.bin}`.
Kernel output diagnostics identify the tile and global channel; no original
evidence needs to be overwritten to investigate.

## CPU regression limits

```bash
python3 -B -m pytest -q framework/custom_ops/draft_w8a8_output_dequant/stage4b/test
bash -n framework/custom_ops/draft_w8a8_output_dequant/stage4b/run.sh
```

CPU tests generate all 304 tiles, check 6080 NZ mappings and historical slices,
error coordinates, incomplete/tampered preparations, all-element coverage,
schedule ownership and full-output failure handling. The actual new Kernel
body runs with a cached-independent-partial Matmul double to test 304-tile
addressing, state reuse, guards and late GM output. It does not emulate MMAD,
VDEQ16 or hardware events. A normalized source comparison also enforces that
the accepted Stage4A numeric/workspace sequence has only dimension/name changes.
