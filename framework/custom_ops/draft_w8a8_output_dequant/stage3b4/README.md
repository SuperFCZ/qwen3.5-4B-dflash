# Stage 3B-4: M16/K2560/N64, twenty-group fused correctness

Independent `DFlashW8A8TwentyGroup`, based on the accepted Stage3B-3 serial
implementation. All previous stages, source locks, OPPs and evidence remain
unchanged. This workstation has no CANN/NPU: new compilation, Tiling and
device correctness are **NOT_RUN**. There is no performance stage, full
N19456 Gate, Draft, Simulator, prefetch or asynchronous overlap.

## ABI and exact computation

- A: INT8 ND `[20,16,128]` (40960 B).
- B: INT8 ND `[20,128,64]` (163840 B), each group non-transposed.
- Scale: UINT64 ND `[20,64]` (10240 B).
- O: FP16 ND `[16,64]` (2048 B).
- Optional integer `output_mode`: 0 is the full result; 1..20 select group
  0..19 respectively. No other mode is accepted.

One Matmul object is registered once. For each g=0..19, rebind local workspace,
A, B and Scale; call Iterate(false), GetTensorC(Local NZ), PIPE_ALL, End,
PIPE_ALL; cast the scaled FP16 partial to FP32 and Add to a persistent FP32
sum. The sum starts at +0 and follows group order exactly. Mode zero converts
the sum to FP16 only after all groups finish. The existing INT32 L0C → enhanced
DataCopy(VDEQ16) → FP16 Local NZ path performs group scaling. No separate
Vector Scale Mul, integer cross-group accumulation, FP16 accumulator, tree
reduction or FMA is introduced.

Two FP16 buffers are reused. The selected diagnostic group's GetTensorC
destination is the retained buffer; all other groups use the temporary
buffer. In normal mode all groups use temporary storage and the retained
buffer receives the final Cast. Every mode still executes all twenty groups,
including later groups after the selected partial. Only then does the kernel
perform one final output phase: the accepted 64 aligned 32-byte NZ→ND copies.
There is no GM partial traffic between groups. Diagnostic invocations observe
partials separately, not as simultaneous extra outputs of the normal launch.

## Unchanged resource envelope, newly checked Tiling

| Allocation | Bytes |
| --- | ---: |
| SDK scratch | 239104 |
| Two FP16 buffers plus FP32 temporary and sum | 12288 |
| Five buffer prefix/suffix guards | 2560 |
| Conservative headroom | 8192 |
| Required actual platform UB | 262144 |

There is no per-group UB allocation. The Host queries actual UB/L1/L0
capacities, reruns MatmulApiTiling for each independent K128 operation, and
checks M16/N64/K128 single/base tiling, scratch, Scale-at-transLength boundary,
alignment and L1/L0 depth/double-buffer capacities. Platform Matmul system
workspace and actual ACLNN workspace remain separately recorded. The new ABI
does not turn this into one K2560 integer Matmul with a single coefficient.

The Stage3B-3 guard initialization, each-group checks, final checks, V_S events
and PIPE_ALL boundaries are preserved. An invalid tiling, failed Iterate or
guard check returns before the output, leaving Runner poison. Finite strict
Golden comparison and distinct poisons prevent such a return from passing.
Guard checks are boundary diagnostics, not a formal proof of every SDK access.
Twenty iterations and retained-buffer lifetimes must still pass real NPU
Smoke; the historical two-group PASS is not reused as a twenty-group result.

## Provenance and immutable real Golden

Prepare validates the accepted Stage3B-3 Build/Smoke, all three cases, all
three modes, both repeats, safety checks, partial-to-sum consistency and Scale
swap. It reuses the existing offline Stage3B-2/3B-1/3A identity validators;
none of their runners execute.

The twenty A/B/Scale groups are copied from the accepted Stage3A cases in
their recorded order. The real UINT64 words are concatenated byte-for-byte;
the encoder is used only to verify them. There is **no activation
requantization and no Scale adjustment**. Each real partial Golden is the
saved Stage3A NPU output, checked against both original repeats and original
candidate Reference.

For the final Golden, follow the accepted Stage3B-1 reduction report's `case`
identity, then that case's reference identities. Verify both saved reduction
NPU outputs equal that bound FP16 reference. Copy those bytes into the fresh
Stage3B-4 directory and retain the original SHA256, rather than inventing a
Golden path or replacing it with a newly generated value. Independently
recomputed ordered CPU sums must also match the bound FP32/FP16 files.

Every comparison records `reference_binding`: the original Stage3B1 identity
for real mode 0, or the two Stage3A NPU output identities for a selected real
partial. Synthetic references are separately labeled. Native Full-K semantic
uncertainty is unrelated to this gate.

## Mandatory Smoke matrix

All four cases run modes **1 through 20, then 0**, with two differently
poisoned repeats per mode: **84 modes, 168 executions**. There is no mode
filter that can produce a partial-suite success.

| Case | Purpose |
| --- | --- |
| real_20_groups | Original frozen data, all Stage3A partials, Stage3B1 final Golden |
| synthetic_distinct | Different per-group/per-channel coefficients and A/B; detect stale Scale, stale A/B and INT32 L0C carry at every group 1..19 |
| synthetic_rotated_scale | Same synthetic A/B with cyclically shifted group coefficients; final outputs must change |
| synthetic_order_rounding | Normal finite powers-of-two and cancellation distinguish sequential FP32 from FP16 accumulation, FP32 pairwise tree and reversed group order |

Synthetic fixtures are new diagnostic inputs, not replacements for frozen
data. Their coefficient values and rounding boundaries are explicitly part
of the new NPU test scope. A failure there is reported separately, not assumed
to prove a particular lifecycle bug or repaired by changing references.

All partial and final comparisons require zero bit mismatches; ULP is a
reported diagnostic, not a tolerance. Full input Readonly, GM guards,
workspace guards, UB guard protocol, poison readback and repeat checks remain
mandatory. After the 21 modes, separately sum this new kernel's observed
partials on CPU and require agreement with its final output as an additional
consistency check. This never replaces the original Golden comparison.

Failure stops before the next mode/case. Pending modes remain NOT_RUN. Reports
retain failed comparisons, output identities and failure artifact paths;
there is no fallback. `summary` reports expected and completed case/mode
counts. Only the complete matrix may return `STAGE3B4_20_GROUP_FUSED_MATCH`.

## Server commands

Run in the already validated Ascend310P3/CANN9 environment. Each destination
must be new. The build contains only this independent operator and installs
into its own private OPP.

```bash
set -euo pipefail
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
S3B4=/home/w00949577/qwen35-dflash-run/w8a8-twenty-group-$(date +%Y%m%d-%H%M%S)
ENTRY=framework/custom_ops/draft_w8a8_output_dequant/stage3b4/run.sh
bash "$ENTRY" prepare --device 0 --output-dir "$S3B4/prepare"
bash "$ENTRY" build --device 0 --output-dir "$S3B4/build"
bash "$ENTRY" smoke --device 0 --fused-build-report "$S3B4/build/report.json" \
  --output-dir "$S3B4/smoke"
printf 'Evidence: %s\n' "$S3B4"
```

Defaults bind Stage3B-3 `w8a8-two-group-20261027-042044`, Stage3B-2
`w8a8-local-nz-20261027-031856`, Stage3B-1
`w8a8-stage3b-20261027-024844`, Stage3A
`w8a8-stage3a-20261027-011523` and their existing Stage2B/2C dependencies.
`--build-report` still means the historical Stage2B report; the new report is
passed with **--fused-build-report**.

Expected stages: `STAGE3B4_PREPARED_NPU_NOT_RUN`,
`STAGE3B4_BUILT_NPU_NOT_RUN`, and only after all device gates pass,
`STAGE3B4_20_GROUP_FUSED_MATCH`. No success automatically starts full Gate or
performance work.

Return Prepare/Build/Smoke `report.json`. For failure, retain and return the
failed `execute/<case>/mode-<mode>/{runner.log,execution.json,actual-*.bin}`
plus `data/<case>/case.json` and references. On build failure also return
`msopgen.log`, `build.log` or `runner-build.log` for the failed phase. Host logs
record output mode and actual Tiling/resources. Old evidence is never edited.

## Local validation limits

```bash
python3 -B -m pytest -q framework/custom_ops/draft_w8a8_output_dequant/stage3b4/test
bash -n framework/custom_ops/draft_w8a8_output_dequant/stage3b4/run.sh
```

Tests check original Golden identity, all partial modes, middle/late stale
state witnesses, strict rounding order, tampering, all-mode orchestration and
stop-on-failure. The actual new Kernel body runs against CPU test doubles for
all 21 modes, including faults for stale Scale, overwritten retained output
and UB guards. Its UB allocation stays constant and no modeled GM output is
allowed before twenty iterations/Ends/Adds. This does not emulate CANN's
asynchronous events or prove actual Matmul/Vector instruction behavior.
