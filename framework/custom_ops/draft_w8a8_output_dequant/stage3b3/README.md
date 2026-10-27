# Stage 3B-3: serial two-group fused correctness experiment

Independent `DFlashW8A8TwoGroup`. Stage 3B-2's accepted Local NZ path is reused
in new source; no historical Kernel, Host, ABI, OPP, Golden or report changes.
This workstation has no CANN/NPU. New compilation, Tiling execution and NPU
checks are **NOT_RUN**. There is no timing, K2560, N19456 or Draft execution.

## ABI and diagnostic selection

Inputs are INT8 ND A `[2,16,128]`, INT8 ND B `[2,128,64]`, and UINT64 ND Scale
`[2,64]`. The sole output is FP16 ND `[16,64]`. Optional integer attribute
`output_mode` defaults to zero:

| Mode | Output, written only after both groups finish |
| --- | --- |
| 0 | Ordered FP32 sum, converted once to FP16 |
| 1 | Group 0's exact FP16 partial |
| 2 | Group 1's exact FP16 partial |

All modes execute the same two Matmuls, Scale updates, output destinations,
FP32 Cast/Add operations and synchronization. Only the final cast/output
selection differs. No GM partial is written between the groups. Group 0's
FP16 buffer doubles as the final output buffer after its partial is consumed;
Group 1 has a separate FP16 buffer. This avoids an extra diagnostic allocation.

Smoke runs modes 1, 2, 0, with two poisoned repeats each. Thus each partial is
independently observable after the second calculation, even if two errors
could cancel in the fused sum. These are separate diagnostic invocations, not
simultaneous observations of one launch. The final output must also match a
CPU sum of this new kernel's separately observed partial outputs.

The new ACLNN attribute follows the generated header's input/attribute/output
ordering; compilation checks its signature. Host validates and logs the mode.
The registration uses an optional integer default as described by the
[msopgen prototype format](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/others/tbeaicpudevg/atlasopdev_10_0024.html).
This documentation is syntax guidance, not device validation.

## Numerical contract and state transition under test

For g=0,1:

1. Rebind the same SDK scratch, A at `g*2048`, B at `g*8192`, and Scale at
   `g*64` elements. Do not reuse the prior coefficient pointer implicitly.
2. `Iterate(false)` starts a fresh INT32 output, never an integer partial sum
   across differently scaled groups.
3. `GetTensorC(local_nz,0,true)`, PIPE_ALL, End, PIPE_ALL.
4. Cast the FP16 partial to FP32, PIPE_V, then Add to a persistent FP32 sum
   initialized to positive zero. PIPE_ALL completes consumption before reuse.

Mode zero then performs the sole final FP32-to-FP16 CAST_NONE. NZ→ND uses the
accepted aligned 32-byte DataCopy mapping. No standalone Vector Scale Mul,
weight dequantization, FP16 accumulator, FMA, tree reduction or asynchronous
pipeline is introduced. The enhanced DataCopy VDEQ16 performs group scaling.

The reuse of one Matmul object after End is **the new lifecycle experiment**.
Stage 3B-2 proved only one calculation. The driver saves bounded excerpts for
End/SetTensorA/SetTensorB/SetQuantVector/CopyQuantTensorImpl from already-bound
SDK snapshots, with source SHA256 and line numbers, in `lifecycle.json` and
`lifecycle-excerpts.txt`. It does not scan the SDK or grant SOURCE_REVIEWED
from symbols. Whether End and the setters reset all iteration/Scale state is
confirmed only by the two-group device tests. A failure stops the experiment;
there is no hidden alternative Matmul object, GM partial or Vector fallback.

The target is ordered `FP16(+0_FP32 + FP32(P0) + FP32(P1))`, where Pg is the
already-rounded VDEQ16 FP16 partial. For only two FP16 terms the final bits
usually cannot distinguish FP16 addition from FP32 addition followed by FP16;
source and actual-kernel CPU type checks enforce FP32 here. Twenty-group
Stage 3B-4 must still independently test the longer accumulation sequence.

## Resource budget and guards

| Allocation | Bytes |
| --- | ---: |
| Group 0 partial / final FP16 | 2048 |
| Group 1 FP16 partial | 2048 |
| FP32 cast temporary | 4096 |
| FP32 persistent sum | 4096 |
| User payload total | 12288 |
| Five buffer prefix/suffix guards, 256 B each side | 2560 |
| SDK scratch | **239104** |
| Remaining conservative SDK headroom | 8192 |
| Total expected UB | 262144 |

Scratch is below the requested 241664-byte upper bound. The additional 2560
bytes fund redzones around SDK scratch and each user buffer while retaining
the full 8 KiB headroom. Slices passed to SDK/Vector operations are 256-byte
aligned and have explicit sizes excluding guards. Guards are initialized
before use, checked after each group and after the final conversion, with
PIPE_ALL and V_S synchronization before scalar inspection. Guard failure
returns before GM output, leaving the two distinct runner poisons observable.
These checks cover buffer boundaries; they are not a formal proof of all
possible SDK memory accesses.

Host queries real UB/L1/L0 capacities and independently reruns MatmulApiTiling
with M16/N64/K128, Local NZ half C and tensor dequant. It verifies the returned
single/core/base shapes, transLength alignment plus 512-byte Scale storage,
conservative conversion space, L1 depths and L0 double-buffer capacities.
Scratch is an allocation cap, not a claimed minimum used by the SDK.
System workspace is the actual platform library request; ACLNN workspace is
recorded separately. There is no GM workspace for partials.

## Frozen provenance and numerical witnesses

The driver validates accepted Stage3B-2, its six Local/GM comparisons, and its
Stage3B-1/2B/2C identity chain offline. It then revalidates Stage3A and selects
only groups 0 and 1. Original A/B/Scale chunks are concatenated byte-for-byte.
Activation is **never requantized** on K256; full-X quantization is inherited
from the accepted Stage3A files. Both saved partial repeats must agree with
their original CPU candidate references before preparing a new two-group
Golden. The full twenty-group sum is not used as a two-group Golden.

Three cases run:

- `real_g0_g1`: original Stage3A inputs and saved FP16 partials;
- `synthetic_distinct`: different A, B and per-channel Scale in both groups;
- `synthetic_swapped`: same A/B, but swap the two complete Scale vectors.

CPU preparation verifies that the synthetic fixtures distinguish stale group-0
Scale, stale A, stale B, carried INT32 L0C and swapped scales. Each partial and
final output must match exactly, and the two synthetic final NPU results must
differ. No ideal-scale tolerance replaces candidate bits. Guard, full-input
Readonly, poison readback and repeat equality remain mandatory for all modes.
Any failure stops before the next mode/case. Native Full-K semantics are not
a gate for this experiment.

## Server commands

Use the already accepted server environment. Every destination must be new.
Build installs only this new operator into its private OPP. No prior Runner
or OPP is rebuilt or executed.

```bash
set -euo pipefail
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
S3B3=/home/w00949577/qwen35-dflash-run/w8a8-two-group-$(date +%Y%m%d-%H%M%S)
ENTRY=framework/custom_ops/draft_w8a8_output_dequant/stage3b3/run.sh
bash "$ENTRY" prepare --device 0 --output-dir "$S3B3/prepare"
bash "$ENTRY" build --device 0 --output-dir "$S3B3/build"
bash "$ENTRY" smoke --device 0 --fused-build-report "$S3B3/build/report.json" \
  --output-dir "$S3B3/smoke"
printf 'Evidence: %s\n' "$S3B3"
```

Defaults bind Stage3A `w8a8-stage3a-20261027-011523`, Stage3B-1
`w8a8-stage3b-20261027-024844` and Stage3B-2
`w8a8-local-nz-20261027-031856`. The old `--build-report` option refers to
Stage2B; use **--fused-build-report** for the new build.

Expected stages are `STAGE3B3_PREPARED_NPU_NOT_RUN`,
`STAGE3B3_BUILT_NPU_NOT_RUN`, and, only after strict real-device success,
`STAGE3B3_TWO_GROUP_MATCH`. No status automatically enables Stage3B-4.

Return `build/report.json`, `smoke/report.json`, and
`smoke/execute/*/mode-*/{runner.log,execution.json}`. On build failure also
return `msopgen.log`, `build.log` or `runner-build.log`, whichever phase failed.
On numerical failure preserve both `actual-*.bin`, `data/*/case.json` and its
references. Return `prepare/lifecycle-excerpts.txt` for precise state-reset
analysis if a partial, Scale or iteration failure occurs. Poisoned output can
mean an Iterate/guard/invalid-tiling early return; do not infer a specific cause
without these logs. Historical evidence stays untouched.

## Local tests

```bash
python3 -B -m pytest -q framework/custom_ops/draft_w8a8_output_dequant/stage3b3/test
bash -n framework/custom_ops/draft_w8a8_output_dequant/stage3b3/run.sh
```

The actual new Kernel body is compiled with test-only scalar Matmul/Vector
doubles to check state transitions, group pointers, FP32 types, array bounds,
late GM writes and injected stale-Scale/UB-guard failures. Runner syntax is
also checked with declarations. Neither double implements actual SDK event
semantics or proves the 310P3 ABI, rounding, CANN compilation or NPU safety.
