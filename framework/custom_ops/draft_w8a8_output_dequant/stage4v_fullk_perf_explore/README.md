# Stage4V Full-K + SDK VDEQ16 exploratory performance

Independent Runner/driver only. Reuses the **normal** OPP bound by the frozen
Stage4V Build; no Kernel compilation, msopgen, installation, math/Tiling/event
change or historical report rewrite. No External CO1 consumer is developed here.
Source identities, generated normal source audit, private API/Host/ELF inventory,
historical Runner/link libraries, SDK/Driver/system dependencies and original
A/B/Scale/CPU Golden are reverified. Fault OPP may be read by the historical Build
validator but is never selected or executed. Separate processes use only the
normal OPP's environment.

## Separate numerical and performance contracts

Any mismatch remains **NUMERIC_MISMATCH_UNRESOLVED**. A zero-mismatch observation
is labeled BITWISE_REFERENCE_OBSERVATION, not production approval. Performance
can independently become **EXPLORATORY_PERFORMANCE_MEASURED** only after the new
safety and data-completeness gates pass. No threshold is widened and no new
numerical Golden is constructed.

Source admission reads the historical bound failure output, full-stride expected
file, execution.json, resource-admission.json and runner.log. It requires exactly
the six recorded coordinates,1 ULP each toward zero, no inactive-region changes,
and finite active data. It verifies the original Runner's exact source identity:
that source checks sync, GM guards, Readonly and Workspace **before** throwing
its exact strict-Golden failure. Only that failure is admitted; other/multiple
FAIL reasons, missing/corrupt guard logs or altered upstream evidence block.
These are historical control-flow/log inferences; fresh Smoke is still mandatory.

## Safety checks and coverage limits

* Host-only invokes GetWorkspaceSize, no Kernel. Exact original T1 diagnostics,
  actual coupled Core policy8/8/7/selected7, per-Core arena formula and exact
  frozen16784384B Workspace are checked; no assumed8-block launch or fallback.
* Tile0 runs mode0/core_limit0/tile_id0 twice, with full FP16[16,19456] ABI.
  All4096 active values must be finite and repeat bitwise. Every inactive element
  remains0x7f7f on run0 /0xffff on run1. Whole files intentionally differ in the
  inactive poison region; repeat equality is scoped to active elements.
* Full runs mode0/core_limit0/tile_id=-1 twice. All311296 outputs must be finite,
  covered and bitwise repeat-equal. Other75 Tiles are never assumed correct from
  Tile0 evidence. All mismatches are saved, even if more than six.
* Each independent process checks input readonly, external GM input/output/
  Workspace Guards, poison readback, resource admission and synchronization.
  Failure preserves available raw output/logs and stops further phases.
* Original Kernel UB/Arena checks remain **unchanged**: three384-word Guard
  boundaries per Tile plus Arena checks. Failure aborts before that Tile's output
  write. Active-region poison/finite/coverage checks therefore verify this
  protocol indirectly. There is no new direct UB readback, new fault coverage,
  Witness output or stronger safety claim. This is explicitly reported as
  `POISON_ON_FAILURE_AND_COMPLETE_FINITE_OUTPUT`.

Every output gets `.numeric.json`: all mismatch row/channel, actual/expected
FP16 bits, ULP, absolute error, counts/maxima. Python independently recomputes
and checks the C++ report. Original Full-K CPU Golden stays authoritative for
these observations; it is never replaced by actual output. Signed-zero bit
mismatch is recorded even when numerical ULP distance is0.

## Continuous timing

Exactly3 fresh processes/rounds; each performs two untimed Poison prechecks,
5 warmups,30 measured calls, Timed Tail, then separate Poison Postcheck.
The Full Smoke output is a **repeat-only anchor**, checked before warmup and
again at Tail/Postcheck. It is not numerical validation or a replacement Golden.
An inter-phase or inter-round output drift rejects timing.

Inputs stay resident. Each sample measures Execute + StreamSynchronize;
GetWorkspaceSize/Prepare is a separate series. No Host H2D/D2H, Guard scans,
numeric comparisons, file IO or Host diagnostic parsing occurs inside measured
callbacks. SDK/Host log growth during the continuous interval rejects the round;
diagnostics are not muted. Workspace size is checked every prepare. No profiler.

**Kernel-internal Guard scans still execute and are included in latency.** Removing
them would violate reuse of the unchanged normal OPP. The no-scan timing rule is
applied to Host checks; the report makes this distinction explicit. There is no
Witness cost in this original Stage4V ABI. Setup/upload/checks remain excluded.

All90 original Execute+Sync and Prepare samples, per-round/aggregate median,
nearest-rank P95/min/max are retained. Device Kernel time is
UNAVAILABLE_NO_DEVICE_TIMESTAMPS; never infer it by subtraction or relabel Host
Execute+Sync as pure device time.

Historical Stage4R full-shape FP16 median0.78698ms/P950.79286ms and Group128 N256
median2.77505ms/P952.78094ms are reference-only, not fresh paired measurements.
Ratios are explicitly named historical_ratios_not_paired. Full-K uses different
weight quantization/Scale and one Full-K Matmul per Tile, versus20 independently
scaled Group128 partials/reduction. Guard lifetimes also differ. No pure INT8/FP16
hardware throughput claim; no single-Tile denominator or layer extrapolation.

## Server commands: fresh root, stop on any failed gate

```bash
set -euo pipefail
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
D=framework/custom_ops/draft_w8a8_output_dequant/stage4v_fullk_perf_explore
FROZEN=/home/w00949577/qwen35-dflash-run/w8a8-full-k-l1fix-20261106-073850
OUT=/home/w00949577/qwen35-dflash-run/w8a8-fullk-perf-explore-$(date +%Y%m%d-%H%M%S)
COMMON=(--frozen-root "$FROZEN" --cann-root "$CANN_ROOT" --device 0)
python3 -B "$D/run_explore.py" source "${COMMON[@]}" --output-dir "$OUT/source"
python3 -B "$D/run_explore.py" build "${COMMON[@]}" --source-report "$OUT/source/report.json" --output-dir "$OUT/build"
python3 -B "$D/run_explore.py" host "${COMMON[@]}" --source-report "$OUT/source/report.json" --build-report "$OUT/build/report.json" --output-dir "$OUT/host"
```

Inspect successful Host, then explicitly run untimed Smoke:

```bash
GATES=(--source-report "$OUT/source/report.json" --build-report "$OUT/build/report.json" --host-report "$OUT/host/report.json")
python3 -B "$D/run_explore.py" tile0 "${COMMON[@]}" "${GATES[@]}" --output-dir "$OUT/tile0"
python3 -B "$D/run_explore.py" full "${COMMON[@]}" "${GATES[@]}" --tile-report "$OUT/tile0/report.json" --output-dir "$OUT/full"
```

Inspect Full safety/completeness and all numeric differences. If only stable,
recorded numerical mismatches remain, explicitly start exploration:

```bash
python3 -B "$D/run_explore.py" benchmark "${COMMON[@]}" "${GATES[@]}" \
 --tile-report "$OUT/tile0/report.json" --full-report "$OUT/full/report.json" \
 --output-dir "$OUT/benchmark"
```

| Stage | Exact successful stage status | NPU scope |
|---|---|---|
| source |FULLK_EXPLORE_SOURCE_REVIEWED|NOT_RUN; read-only historical diagnosis|
| build |FULLK_EXPLORE_RUNNER_BUILT|NOT_RUN; new Runner only|
| host |FULLK_EXPLORE_HOST_TILING_ONLY|NOT_RUN; GetWorkspaceSize only|
| tile0 |FULLK_EXPLORE_TILE0_SAFE|two fresh Tile0 executions|
| full |FULLK_EXPLORE_FULL_SAFE|two fresh full-shape executions|
| benchmark |EXPLORATORY_PERFORMANCE_MEASURED|three rounds,90 new timed samples|

All phases produce report.json; Benchmark additionally produces
benchmark-report.json and comparison.json. No old report/path is overwritten.
Numeric status remains independent. Stage success is not NUMERICS_PASS,
GOLDEN_MATCH or PRODUCTION_APPROVED.

On failure return report.json plus the failing runner.log/execution.json and
resource-admission.json, or build cmake.log/compile.log/*ldd.log. Preserve raw
actual/benchmark-last/postcheck binaries and their numeric JSON locally; do not
rerun in the failed directory. The driver retains file identities even when a
Runner exits nonzero. Source failures need historical-mismatches.json if written
and the stated missing/changed identity; do not alter the frozen evidence.

## Local verification and limitations

```bash
PYTHONPATH=framework/custom_ops/draft_quant/test python3 -B -m pytest -q -p no:cacheprovider "$D/test"
```

Tests compile the same C++ numerical checker used by the Runner, compare every
active element with Python, reject nonfinite/Poison/repeat/safety failures and
exercise the actual shared continuous-v1 call scheduler with CPU-only stubs.
No mocked result is NPU evidence. Local Docker builds/links the actual new Runner
against real CANN and a previously generated Full-K private API. This is
LOCAL_ONLY_NOT_SERVER_VERIFIED; local HAL resolves to the SDK devlib stub, not
an actual Driver. Server runtime binding explicitly rejects that substitution
and requires the independently bound Driver root for libascend_hal.so.

Source/Host/Smoke/Benchmark on the real frozen server data remain NOT_RUN locally.
The six1-ULP cause and global quality remain UNRESOLVED. Online activation
quantization, Scale generation/encoding, input preparation/transfers, full Draft,
end-to-end Decode and Profile are not measured.
