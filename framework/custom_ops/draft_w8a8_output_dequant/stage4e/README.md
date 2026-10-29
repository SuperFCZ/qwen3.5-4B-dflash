# Stage 4E: remaining Scalar guard contrasts

Independent performance diagnostics for Ascend310P3 / CANN 9.0.0. No historical
Kernel, Host, OPP, Runner, Golden, source lock or report is modified. Local CPU
tests do not establish CANN compilation, NPU correctness or performance.

## Controlled variants

All variants start from **the accepted Stage4D B source**, verified against its
recorded SHA256 and the exact historical Stage4B→4D transformation. Each gets a
separate generated project, installed private OPP, Kernel identity and Runner.
The operator name/ABI remain `DFlashW8A8FullGate`; each invocation is a fresh
process with only its chosen private OPP in the custom-OPP/library environment.
The Runner's variant argument labels evidence; it does not select branches in a
shared Kernel. Build rejects identical variant Kernel binaries.

| Coverage | B0 | B1 | B2 |
|---|---|---|---|
| Per-group five UB scans | absent, as in Stage4D B | absent | absent |
| Per-tile five UB scans | retained | removed | retained |
| Per-tile internal Arena Scalar comparisons | retained | retained | removed |
| Arena DataCopy, PIPE_ALL and MTE2_S Set/Wait | retained | retained | retained |
| Initial pre-loop Arena scan | retained | retained | retained |
| Final extra p0/cast two UB scans | retained | retained | retained |
| Guard initialization; per-group V_S; all Matmul/sync/math | retained | retained | retained |
| Runner Golden, external GM guards, readonly, poison, repeat | retained | retained | retained |

**B2 removes the per-tile Arena predicate, not its transport or synchronization.**
Its inlined transfer block matches the original CheckArena body except for the
GetValue comparison loop and failure return. The original helper remains
unchanged for the initial Arena check. This intentionally measures the predicate
contrast; it does not measure the cost of eliminating Arena DMA or waits.

All three protocols are diagnostic-only and weaker than Stage4B. In particular:

- All can miss group-level transient corruption restored before the tile checks.
- B1 can miss persistent scratch/p1/sum guard damage; final p0/cast scans remain.
- B2 can miss internal Workspace Arena guard damage after the initial check.
  Runner external allocation guards do not replace internal per-core guards.
- A correct numerical result does not prove the omitted checks unnecessary.

The exact source diffs are recorded in `build/sources/B{0,1,2}-only.diff` (B0's is
empty), with a bound `source-audit.json`. Successful-path source call counts
removed relative to B0: B1 **194560** and B2 **77824** GetValue calls. These are
not measured device instruction counts. Compiler scheduling consequences remain
part of the measured contrast. No new final-p0/cast variant is included.

## Evidence and execution gates

Default prerequisite: the complete accepted Stage4D root
`/home/w00949577/qwen35-dflash-run/w8a8-guard-ab-20261029-012635`.
Build/Smoke/Benchmark/Profile reports and their files are checked read-only with
the original Stage4D validators; the existing Stage4C/4B evidence chain remains
bound. Missing or changed evidence fails closed. A new output directory is
required for every command. Historical timings are not fresh Stage4E samples.

The stage is fixed to M16/K2560/N19456, 304 tiles, 20 K128 groups and automatic
Core selection from the accepted Host policy (currently 7, never hardcoded).
Host source, UB 239104B scratch, workspace arena formulas, output stride, ABI,
Scale encoding, VDEQ16 and FP32 ordered additions are unchanged. Returned full
Host configuration, plan and actual ACLNN workspace must agree across variants.

1. **Build** creates three independently audited private OPPs and Runners.
2. **Smoke** runs all three with two poisoned, strict complete-Golden checks.
   Benchmark is blocked until every variant passes its declared protocol.
3. **Benchmark** uses rotations B0/B1/B2, B1/B2/B0, B2/B0/B1. Per run:
   two untimed correctness calls → warmup 5 → continuous 30 → timed-tail check
   → poisoned postcheck. No application readback/poison/file I/O in timing.
   Raw 90 execute+sync samples and separate prepare samples per variant are kept.
   Timed Host logging is rejected, not silently suppressed.
4. **Profile** requires the accepted benchmark, then runs msprof PipeUtilization
   separately for each variant. All CSVs/timelines and hashes are retained.
   Profile durations never enter formal latency statistics.

New success statuses are scoped diagnostics:
`STAGE4E_BUILT_NPU_NOT_RUN`, `STAGE4E_SCOPED_CORRECTNESS_MATCH`,
`STAGE4E_CONTRASTS_MEASURED_DIAGNOSTIC_ONLY`,
`STAGE4E_PROFILES_COLLECTED_DIAGNOSTIC_ONLY`.
Missing counters remain UNAVAILABLE; incomplete/raw-incomparable columns remain
UNRESOLVED. No production adoption or new Stage4B safety certification occurs.

## Server commands

Run from the repository after `git pull --ff-only`:

```bash
set -euo pipefail
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
ENTRY=framework/custom_ops/draft_w8a8_output_dequant/stage4e/run.sh
S4E=/home/w00949577/qwen35-dflash-run/w8a8-tile-guard-$(date +%Y%m%d-%H%M%S)
bash "$ENTRY" build --device 0 --output-dir "$S4E/build"
bash "$ENTRY" smoke --device 0 \
  --pair-build-report "$S4E/build/build-report.json" --output-dir "$S4E/smoke"
bash "$ENTRY" benchmark --device 0 \
  --pair-build-report "$S4E/build/build-report.json" \
  --smoke-report "$S4E/smoke/smoke-report.json" --output-dir "$S4E/benchmark"
bash "$ENTRY" profile --device 0 \
  --benchmark-report "$S4E/benchmark/benchmark-report.json" --output-dir "$S4E/profile"
printf 'Stage4E evidence: %s\n' "$S4E"
```

Override `--stage4d-root` on every command if the accepted evidence was moved
without changing its internally bound paths. `--msprof /path/to/msprof` selects
the profiling executable. Do not resume into an existing output folder or edit
failed reports; preserve outputs/logs and retry in a fresh directory.

Return `build/build-report.json`, `build/sources/source-audit.json`, the three
source diffs, `smoke/smoke-report.json`, `benchmark/benchmark-report.json`,
`profile/profile-report.json` and `guard-coverage.json`. For a failure also return
the failing variant's `build.log`/`runner-build.log` or `case/runner.log`,
`execution.json`, `resource-admission.json`; retain partial outputs in place.

## Attribution limits / deferred Stage 4F

Report medians/p95 and B0-minus-B1/B2 latency differences independently. Raw
profiler columns report Task Duration, Scalar Time/Ratio, MTE1/2/3, Vector,
MAC and Cube metrics when present. Do not sum overlapping pipe times, subtract
Scalar time from wall time, add both independent deltas as a measured combined
saving, or call any variant guard-free. Even B1 retains Arena/final scans; B2
retains UB/final scans. Non-guard Scalar critical-path cost remains UNRESOLVED.

Stage4F is NOT_RUN pending real Stage4E evidence. Only then review bounded CANN
9.0.0 implementation evidence for Iterate/GetTensorC/End, Scale-buffer updates,
mandatory PIPE_ALL and cross-group dependencies, with a real UB/L1/L0 budget.
No asynchronous overlap, prefetch, changed group size, buffer reuse before
completion, removed synchronization or new fused numerical semantics is implemented.

## CPU checks

```bash
PYTHONPATH=framework/custom_ops/draft_quant/test python3 -B -m pytest -q \
  framework/custom_ops/draft_w8a8_output_dequant/stage4e/test
```

Tests compile the actual materialized Kernel bodies against CPU doubles and
compare clean-path event/barrier/DMA counts, exact GetValue deltas, and injected
scratch/p1/Arena corruption detection. They also compile the real new Runner
against an ACL test double and check continuous-v1, precheck failure, timed-log
rejection and protocol separation. CPU doubles are test-only, never a server
fallback or evidence of device event safety.
