# Stage4V — Full-K INT8 / per-channel dequant

Independent **new numerical scheme**, not a Group128 optimization preserving its
Golden. No Stage4B–4U files, SDK, OPPs or historical reports are changed.
Quality GO is explicitly **UNRESOLVED**, per user instruction; performance and
quality decisions remain separate. No production/Draft integration.

## Contract

Original source is the frozen signed INT8 NK + FP16 GN Group128 Scale, not an
unquantized checkpoint. Reconstruct `W_eff=RN16(q_old*s_old[k//128,n])`, and compare
every reconstructed element against the accepted Stage4R FP16 KN weight identity.
For each output channel, independent of X:

- `scale_w = max(abs(W_eff.astype(FP64)), K)/127` in FP64;
- zero channel: `scale_w=1`, all quantized values zero;
- `q_w=clip(rint_RNE_FP64(W_eff/scale_w),-127,127)`;
- `combined=RN32(FP64(saved_scale_A)*scale_w)`;
- UINT64 word: low FP32 bits masked with`0xffffe000`, plus bit46;
- effective Scale is the masked FP32 value. Nonfinite, positive-to-zero encoding,
  combined underflow and FP16 output overflow are rejected, never repaired/clipped.

The encoded format was tested in earlier K128 paths; **new Full-K rounding and
output must still pass device validation**. It is not a universal codec proof.
Activation is copied from accepted `[20,16,128]`, transposed to `[16,2560]`, checked
against the saved full-X identity and inverse mapping. No new Activation Scale.
Original NZ→NK is checked for all1520 historical tile/group slices before deriving
new weights. New B is ND `[76,2560,256]`; each logical row is256 contiguous channels.

Independent new Golden C:

```
dot = INT64(A_i8 @ B_i8)
Y_C = FP16(FP32(dot) * effective_combined_scale)
```

All outputs must be **bitwise equal** to C. Hardware rounding disagreement stops the
experiment and retains first row/channel/Tile, INT32 dot, Scale word and actual /
reference bits. No tolerance adaptation. For any K-prefix, absolute INT32 sum is
bounded by`2560*127*127=41290240 < 2147483647`; extrema/zero/signed cases are CPU tested.
FP32 conversion may lose integer low bits above2^24; that rounding is explicitly
part of C, not an assumption of exact integer representation in FP32.

A is the original Stage4B Group128 Golden (diagnostic only). B is independently
recomputed FP64 `X @ W_eff`, byte-checked against Stage4R's reference. Prepare emits
`data/quality.json`: new/old versus B, new versus A, encoding effect, effective Scale
error, Activation reconstruction error, activation-only error, weight requantization
effect given quantized A, Full/Gate/Up metrics,19456 per-channel weight-error/Scale
records and worst channels. One frozen capture only; no calibration or X-based tuning.
Neither old Golden nor Native output is used as the new Kernel's Golden.

## SDK and Full-K lifecycle

`sdk-contract.json` holds file identities and line ranges only. The server Host phase
rechecks all reviewed bytes inside selected CANN_ROOT, in addition to the existing
CANN9.0.0 version/shared-UB/dependency identity gates. Local-only identities are
explicitly labeled; a mismatch blocks rather than requesting a broad SDK scan.

Key local source chain (full SHA256 in that manifest):

- `scheduler/base/scheduler_norm.h:93–122,164–206`: one ScheduleOnce allocates C,
  ComputeMultiIter loads K slices, updates compute flags and calls CubeCompute until
  KLoop.OuterNext finishes. **One logical Iterate is not one MMAD instruction.**
- `scheduler/base/scheduler_norm_base.h:180–188`: initializes L0C on the first K
  block when `enPartialSum=false`; later blocks accumulate within the logical result.
- `matmul_impl_base.h:601–646` and `scheduler/base/scheduler_base.h:363–386`:
  Local GetTensorC follows computation; Local output and CO1 queue release precede End.
- `copy_cube_out/quant/quant_processor_datacopy.h:49–54,113–160` and
  `copy_cube_out_datacopy.h:245–294`: one per-channel Scale descriptor and enhanced
  VDEQ16 Local NZ output. Output block copies are not20 separately scaled K groups.
- `utils/matmul_config_utils.h:338–345`: the Kernel statically requires non-Basic
  CFG_NORM, so the reviewed multi-K scheduler path is selected.

All paths above are under `aarch64-linux/asc/impl/adv_api/detail/matmul/`.
Actual server GetTiling must return singleCore16/256/2560 and base16/256/128.
Host uses GM ND INT8 A/B, B non-transposed, VECIN NZ FP16 C, TENSOR dequant,
INT32 disabled Bias, fixed Scratch202240. No BaseN128 fallback is implemented.

The standalone probe uses real selected device + no-argument Platform manager.
Its count8 cannot authorize8 Blocks. Build embeds its exact Tiling fields/resource
capacities into the new Op Host, which then queries actual TilingContext, applies
the accepted coupled Core policy and checks the expected historical7-Core policy
**before launch**. Actual Workspace size and per-core Arena partitions are rechecked.
Any changed Tiling, BaseN, resources, SDK identity or Core policy blocks.

## New Kernel and Guard protocol

One Matmul object per core, static `tile=blockIdx+i*blockDim`,76 Tiles. For each Tile:

1. Check all UB redzones at pre-Matmul boundary.
2. Rebind Scratch, full A, current B and256 Scale words.
3. `Iterate(false)` → `GetTensorC` → ALL → End → ALL.
4. Check all redzones, then GM Arena, then all redzones again.
5. Write Local NZ result to original ND stride19456, then ALL before buffer reuse.

No outer20-group loop, intermediate FP16 partial rounding, FP32 Cast/Add, async
pipeline, Workspace hoist, uint64 Guard or Stage4S strided output experiment.
Original32B fragment output copies are retained. New arithmetic, fewer Matmul
lifecycles **and different Guard boundaries** all contribute to implementation timing.

|Per-core resource|Bytes|
|---|---:|
|SDK Scratch ceiling|202240|
|FP16 Local NZ result|8192|
|Arena-copy temporary|512|
|Three prefix/suffix redzone pairs|1536|
|Reserve, never allocated|8192|
|Accounted UB|220672|
|Unallocated, not assigned to an optimization|41472|

Actual explicit TPipe allocation is212480 B. L0A/B/C, L1 depths, steps, transLength,
Scale storage and raw/effective shared UB come from **new** GetTiling and conservative
capacity checks. L0B must fit65536 B, with no additional allocation. The old K128
resource result is not proof that K2560 is feasible.

Protocol: `FULL_K_TILE_384_WORD_THREE_BOUNDARIES`.
Three buffers × two256B Guards =384 uint32 words at each of three boundaries.
This does **not** inherit the old Group128640-word/20-boundary safety label. A transient
fault entirely inside Matmul's internal K loop and restored before its return may
escape the new boundary checks; no internal-MMAD fault coverage is claimed.

Normal and diagnostic Fault OPPs are separate private installations with distinct
binaries. Source audit proves the Fault build differs only in its dedicated compile
define, with source/evidence identities and a baseline→new-arithmetic diff retained.
Fault mode2 sweeps every one of384 words ×32 bits at each of3 actual boundaries,
restoring each word:36864 injections/run, two repeats =73728. It emits an exact
32B completion witness containing the checked injection counter, with Poison
everywhere else; it is not counted as a numerical Golden result. Modes3/4/5 inject a
persistent fault at each boundary and require the exact32B abort witness and Poison
elsewhere. Fault OPPs cannot enter timing. CPU source tests check zero/one Matmul
execution before the respective abort. GM Guards, Arena, Readonly and Poison remain.

## Gates and timing

Historical N/O/P and R identities, accepted P Fault/full proofs and R independent
FP64 admission are rechecked. The accepted P normal OPP and accepted R FP16 Runner
are reused in **separate processes**; neither is rebuilt or reinstalled. New Full-K
normal/Fault OPPs alone are installed under the new experiment directory.

|Stage|Success status|
|---|---|
|prepare|STAGE4V_CPU_PREPARED_QUALITY_UNRESOLVED|
|host|STAGE4V_FULL_K_HOST_FEASIBLE|
|build, including actual Op Host-only admission|STAGE4V_BUILT_NPU_NOT_RUN|
|bounded Tiles0/37/38/75|STAGE4V_BOUNDED_MATCH|
|full Auto/2-Core × numeric/scheduler witness|STAGE4V_FULL_MATCH|
|fault sweep +3 abort witnesses|STAGE4V_TILE_FAULT_COVERAGE_MATCH|
|benchmark|STAGE4V_PAIRED_MEASURED|
|profile|STAGE4V_PROFILE_COMPLETED|

Bounded inactive columns must retain each run's Poison; they are never counted as
numerical coverage. Full requires all311296 FP16 elements. Every phase binds the
previous report, source set, SDK, device, Prepare, Host Tiling and installed files.
Failure saves reports/logs/raw outputs and stops; never overwrite/retry in place.

Before formal timing, both historical schemes get fresh untimed numerical checks.
FP16 is independently checked using its unchanged0.002+0.002*abs(reference), relative
L2≤0.001 contract, twice, before freezing the new native output. All timed native
prechecks/tail/postcheck still run that independent reference and strict repetition.

Three cyclic interleaved rounds:

- Group128 → FullK → FP16
- FullK → FP16 → Group128
- FP16 → Group128 → FullK

Each process:2 Poison prechecks,5 warmups,30 continuous Execute+Sync samples, Timed
Tail and independent Poison Postcheck.90 fresh samples per implementation. Prepare
latency is separate. Timed logging, Workspace drift, numerics or Guard failure
invalidates the round. Original2.775050/0.786980 ms are context, never denominators.
Performance GO needs Group128/FullK≥1.03 overall and all3 rounds>1. Quality remains
UNRESOLVED regardless of speed. Profiles are separate for all3; normalized operator-name substring
filter failures remain UNAVAILABLE, with raw CSV/type/Task/Stream/units retained.
Pipe times are not summed or interpreted as removable wall time.

`preprocessing-inventory.json` distinguishes resident payload sizes and excluded
costs: offline effective-weight reconstruction/requantization/packing, dynamic
Activation quantization, combined Scale/encoding, H2D/D2H, whole Draft/Decode.
Reported payload sizes are not peak allocation or measured DDR traffic. Comparing
these implementations is not a pure INT8-versus-FP16 hardware throughput experiment.

## Local validation versus server evidence

```bash
PYTHONPATH=framework/custom_ops/draft_quant/test python3 -B -m pytest -q -p no:cacheprovider \
 framework/custom_ops/draft_w8a8_output_dequant/stage4v/test
bash framework/custom_ops/draft_w8a8_output_dequant/stage4v/local_check/check.sh NEW_LOCAL_DIR
```

Local checks use synthetic captures and real actual-source CPU models, not a CANN
simulator or NPU. The local script compiles/links the real Host probe, compiles the
Runner with actual CANN libraries plus a private API stub that always fails, and
runs actual BiSheng SDK syntax checks of normal/Fault source using a local-only
Tiling-decode adapter. It does not prove generated OPP ABI or device behavior.
No SDK originals or local build artifacts are committed. A separate manual object
probe produced identical1200-byte normal/Fault objects with132-byte text sections;
that is **not accepted as proof that the compute body was generated**. It is not
used by the experiment or promoted into official OPP compilation evidence. No
further disassembler investigation is started.

- `SOURCE_VERIFIED_FOR_SELECTED_SDK_IDENTITIES_ONLY`: server gate for reviewed files;
  does not imply runtime correctness.
- Local Host/Runner compilation and Kernel syntax checks are `LOCAL_COMPILED` in
  the limited scopes above; local libraries remain LOCAL_ONLY_NOT_SERVER_VERIFIED.
- Real frozen quality/Prepare, real K2560 Host query, official OPP Build, NPU,
  fault coverage, Benchmark and Profile are **NOT_RUN locally**. No numerical
  quality or performance values for the actual new scheme have been invented.

## Server commands

After `git pull --ff-only`, choose a new root. Defaults bind the supplied Stage4P
and Stage4R roots. Pass consistent `--stage4n-root/--stage4o-root/--stage4p-root/
--stage4r-root` overrides only when their actual reports reside elsewhere.

```bash
set -euo pipefail
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
V=framework/custom_ops/draft_w8a8_output_dequant/stage4v/run.sh
RUN=/home/w00949577/qwen35-dflash-run/w8a8-full-k-channel-$(date +%Y%m%d-%H%M%S)
bash "$V" prepare --device 0 --output-dir "$RUN/prepare"
bash "$V" host --device 0 --prepare-report "$RUN/prepare/report.json" --output-dir "$RUN/host"
bash "$V" build --device 0 --prepare-report "$RUN/prepare/report.json" --host-report "$RUN/host/report.json" --output-dir "$RUN/build"
COMMON=(--device 0 --prepare-report "$RUN/prepare/report.json" --host-report "$RUN/host/report.json" --build-report "$RUN/build/report.json")
bash "$V" bounded "${COMMON[@]}" --output-dir "$RUN/bounded"
bash "$V" full "${COMMON[@]}" --prior-report "$RUN/bounded/report.json" --output-dir "$RUN/full"
bash "$V" fault "${COMMON[@]}" --prior-report "$RUN/full/report.json" --output-dir "$RUN/fault"
bash "$V" benchmark "${COMMON[@]}" --prior-report "$RUN/fault/report.json" --output-dir "$RUN/benchmark"
bash "$V" profile "${COMMON[@]}" --prior-report "$RUN/benchmark/report.json" --output-dir "$RUN/profile"
```

Run each only after the preceding exact state. `server.sh NEW_ROOT [options]` is an
optional explicit-user-invoked sequence that enforces those statuses and stops on
failure; it never changes inputs/parameters or retries automatically.

Return the failing stage's `report.json`; for Host include `tiling-report.json`,
`resource-report.json`, `sdk-binding.json` and configure/compile/query logs. For
OPP Build include normal/fault build and Runner logs. For device failures include
`validation.json` / `failure-diagnostics.json`, `execution.json`, `resource-admission.json`, Runner log and retain
all actual binaries. First mismatch includes dot/Scale bits. Successful Prepare's
`data/quality.json`, Benchmark `comparison.json` and Profile `report.json` answer
the quality/performance questions independently.

If quality is worse, inspect channel Scale span, outliers, activation-only and
encoding diagnostics before proposing independent Group256/512 studies. Those
would require new references and gates; no such Kernel, N512, overlap or checkpoint
quality claim is part of Stage4V. If Host resources or new semantics fail, stop this
candidate without falling back to Group128 and calling it Full-K success.

Local delivery validation (2026-11-06): macOS35/35 and Docker ARM64 35/35 CPU tests
passed, including full-shape synthetic Prepare, source-executing normal/Fault models,
Runner Golden/Guard/Readonly/log failures and evidence gates. Host/Runner compilation
and normal/Fault SDK syntax checks returned0 within the limited scopes above.
`local-validation.json` explicitly leaves actual frozen-data quality, real Host
resources, official OPP Build and every device/performance stage NOT_RUN.
