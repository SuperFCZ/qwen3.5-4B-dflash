# Fresh B-NZ W8A8 vs Native FP16 resident-input pairing

Baseline971d73e. Independent Driver only: no new mathematical Kernel, OPP,
Runner source, reference, tolerance or historical report modification. It reuses
Stage4R's accepted `aclnnMm` Runner and the normal B-NZ Runner/OPP from
`w8a8-fullk-bnz-20261110-085944`.

The installed custom Op name is derived from its frozen API header and operator
JSON, cross-checked with the source audit and execution records. Current code
spells it **DFlashFullKBNz**, not a name inferred from prose abbreviations.
Both actual ACLNN symbols and Kernel/API/Host/Runner hashes are recorded. Fault
OPPs are never selected. Processes load either the normal custom OPP or native
CANN, never both custom variants in one process.

## Evidence gates

Source revalidates all B-NZ Source/Prepare/Hostplan/Build/Formal Host/Tile0/Full/
Fault/Benchmark reports, links, source hashes, all packed-B inverse identities,
Tiling, private artifacts, raw outputs and the prior paired summary. It also
validates the Stage4R Prepare/Build/Smoke/Benchmark chain discovered through the
bound original reference identity. No legacy missing Profile directory is used.

Both paths descend from the same frozen original X and RN16 Group128 effective
weights. Their actual arithmetic inputs differ; matching shape is not used as
provenance proof. The Native reference must be the exact reference linked in the
Full-K case. Header aliases, reviewed SHA, libraries and recorded machine version
files remain mandatory.

Build is a **read-only binary/dependency revalidation stage**, not compilation:
ldd/nm, exact native library identities, private custom API/Host identities,
separate Driver root/SONAME/hash/version and system library classification. It
never rebuilds or installs historical artifacts. Header/API mismatch or unknown
runtime dependency blocks. Native Core count is not exposed by the reused Runner
and remains **UNAVAILABLE_INTERNAL_ACLNN**, rather than being guessed as7.
W8A8-NZ still verifies its actual formal7-Block/Core policy and Workspace.

## Independent numerical and safety contracts

| Item | Native FP16 | W8A8-NZ |
|---|---|---|
| A |original FP16 X[16,2560]|frozen uniformly quantized INT8 A[16,2560]|
| B |RN16 effective FP16, ND[K2560,N19456]|Full-K per-channel requantized INT8, transposed-storage NZ[76,80,16,16,32]|
| B payload |99,614,720B|49,807,360B|
| Scale |incorporated into effective FP16 weights offline|frozen UINT64 combined/encoded Scale[76,256],155648B, SDK VDEQ16|
| Input payload/upload |99,696,640B|50,003,968B|
| Output |FP16[16,19456],622592B|same shape/bytes|
| Numeric admission |original independent FP64 reference and fixed policy|frozen W8 baseline bitwise, original201 one-ULP CPU differences remain|
| Guard scope |Runner GM guards, readonly, poison, repeat; internal builtin guard implementation not exposed|same external checks plus frozen UB/Arena checks inside custom Kernel|
| Cores |internal ACLNN choice, not exposed|actual verified7-Block policy|

FP16 policy remains per-element `0.002 + 0.002*abs(reference)` and relative L2
<=0.001, finite outputs mandatory, no adaptive tolerance. The reused Runner and
Python checker both validate all311296 elements. A fresh FP16 repeat anchor is
saved only after two independent poisoned passes; timed checks still apply the
FP64 rule and also require that admitted output bytewise. The anchor is not a
replacement independent reference and no historical Golden is overwritten.

W8A8-NZ must preserve frozen output bytes, Guard/Readonly/Poison/Repeat/sync,
full coverage, Tiling/Workspace and the existing CPU difference reports. It stays
**NUMERIC_MISMATCH_UNRESOLVED**. The two variants are never required to have the
same output bits. The comparison reports Full/Gate/Up errors against the common
FP64 diagnostic reference separately; that is not an equality or quality claim.
No CPU fallback or profiler is permitted.

## New pairing, never historical samples

Exactly:

* Round0 FP16 → W8A8-NZ
* Round1 W8A8-NZ → FP16
* Round2 FP16 → W8A8-NZ

Each independent process: two untimed Poison checks,5 warmups,30 continuous
Execute+Sync samples, Timed Tail, separate Poison Postcheck. Both variants pass
a fresh Smoke stage before any formal pairing. Each timed process repeats its
own prechecks. Inputs are resident; H2D/D2H, Host comparison/Guard scans/file IO
are outside timed callbacks. Log/Workspace/output drift rejects the round.
Kernel-internal work, including W8 Guard checks, remains included.

Reports preserve all180 samples, per-round and aggregate median/P95/min/max,
Prepare separately, actual Workspace and Core-information availability. Ratios
are unambiguous: `fp16_over_w8a8_nz > 1` means W8A8-NZ is faster;
`w8a8_nz_over_fp16 < 1` expresses the reciprocal latency ratio. Only new same-run
samples populate these fields. No device Kernel time is inferred without device
timestamps. Equal performance protocol does not mean equal numerical semantics,
weights, internal Guard implementation or raw hardware throughput.

## Prepack and reuse cost

The exact earlier server Prepack record, file SHA and per-Tile identities are
retained. It is **not remeasured** or silently replaced by local CPU timings.
Report permutation/read/write/verification/full-prepare costs separately. B NZ
is49,807,360B with zero padding; retained original plus NZ files occupy99,614,720B.
The current call uploads only its selected representation, not both.

For positive fresh `(FP16_ms - W8A8_NZ_ms)`, report
`ceil(B_NZ_cost_ms / saving_ms)` using permutation-only and full-prepare costs.
No positive saving means no finite amortization. This is an **incremental B-layout
cost model**: Full-K weight requantization, FP16 effective-weight materialization,
online Activation quantization, combined Scale generation/encoding, upload,
model scheduling and Draft/Decode are excluded. FP16 materialization cost was not
newly measured and is not called free. It is not an end-to-end or model-quality
claim. Static weight reuse assumptions must be stated when using the model.

## Server commands: fresh directories and exact states

```bash
set -eo pipefail
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
source "$CANN_ROOT/set_env.sh"
set -u
D=framework/custom_ops/draft_w8a8_output_dequant/stage4v_bnz_fp16_paired
BNZ=/home/w00949577/qwen35-dflash-run/w8a8-fullk-bnz-20261110-085944
OUT=/home/w00949577/qwen35-dflash-run/w8a8-bnz-fp16-paired-$(date +%Y%m%d-%H%M%S)
COMMON=(--bnz-root "$BNZ" --cann-root "$CANN_ROOT" --device 0)
python3 -B "$D/run_pair.py" source "${COMMON[@]}" --output-dir "$OUT/source"
python3 -B "$D/run_pair.py" build "${COMMON[@]}" --source-report "$OUT/source/report.json" --output-dir "$OUT/build"
GATES=(--source-report "$OUT/source/report.json" --build-report "$OUT/build/report.json")
python3 -B "$D/run_pair.py" smoke "${COMMON[@]}" "${GATES[@]}" --output-dir "$OUT/smoke"
```

After both independent fresh admissions succeed:

```bash
python3 -B "$D/run_pair.py" benchmark "${COMMON[@]}" "${GATES[@]}" \
 --smoke-report "$OUT/smoke/report.json" --output-dir "$OUT/benchmark"
```

| Stage | Exact successful state | Device work |
|---|---|---|
| Source |BNZ_FP16_SOURCE_REVERIFIED|read-only history/environment|
| Build |BNZ_FP16_RUNNERS_REVERIFIED|read-only ldd/nm, no compilation/Kernel|
| Smoke |BNZ_FP16_FRESH_SMOKE_ADMITTED|two fresh executions per scheme|
| Benchmark |BNZ_FP16_FRESH_PAIRED_MEASURED|90 fresh timed samples per scheme|

Every stage writes report.json and preprocessing-inventory.json. Benchmark adds
benchmark-report.json and comparison.json, including raw samples, ratios,
per-round/Core/Workspace records, separate numerical diagnostics and amortization.
On failure the stage is BLOCKED, comparison publication is removed, and no later
stage is allowed. Do not reuse the failed directory or regenerate a looser anchor.
Return report.json plus the failing ldd/symbol log or Runner log/execution.json;
keep all raw outputs, numeric reports, admission JSON and hashes on the server.

## Local checks

```bash
PYTHONPATH=framework/custom_ops/draft_quant/test python3 -B -m pytest -q -p no:cacheprovider "$D/test"
```

No new C++/Kernel needs compilation. The reused Native source has existing CPU
contract tests; these can run locally without an NPU and remain labeled mock
runtime tests. The current Docker SDK lacks public aclnn_mm.h and libopapi.so,
so a real local Native CANN link/runtime check is **BLOCKED_MISSING_NATIVE_SDK**.
No new version, fake header or stub is installed to bypass this. The server
already has the accepted Native binary; the Build gate rechecks its real files.
All new real Source/Build/Smoke/Benchmark stages remain NOT_RUN locally.
