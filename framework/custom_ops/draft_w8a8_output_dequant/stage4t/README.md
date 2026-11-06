# Stage4T — N256 GroupDelta cost diagnostic

**WEAKER_GUARD_DIAGNOSTIC_ONLY — never a production candidate or safety equivalent.**
Stage4P N256 XOR/OR remains the complete per-group Guard baseline. Stage4S is closed:
the user-reported 1.003300855 ratio missed the predeclared1.03 threshold. The separate
`stage4s-closeout.json` records that report without modifying S or pretending its
remote artifacts were read here. No further DataCopy changes are made.

## Single variable and limits

`source_t.py` independently re-derives the committed Kernel from Stage4P and removes
exactly the **one** `if(GroupDelta(...)!=0)return;` statement inside the20-group loop.
The now-unused helper definitions deliberately remain. Host/ABI/operator definition
are byte-identical to P. All InitBuffer, Guard initialization, End, PIPE_ALL/PIPE_V,
V_S, SetLocalWorkspace, input/Scale updates, Cast/Add, Tile-end five Guard scans,
extra p0/cast scans, Arena checks and original256-copy final store remain unchanged.
No Stage4S strided store, uint64 Guard, extra UB or Reserve use.

This removes972800 source-level uint32 Guard reads per full launch. It does **not**
prove any particular machine instruction count or wall-time contribution. The
compiler may change surrounding code. Each new build must have a distinct binary
identity; installed ELF/API/Host/OPP and generated sources are independently bound.

Safety difference:

| Check | P baseline | T diagnostic |
|---|---|---|
|640 uint32 Guard words after every Group|Present; abort before next Group|Removed|
|Transient corruption restored before Tile-end|Detected at affected Group boundary|Can escape detection|
|Persistent Guard corruption|Detected at affected Group boundary|Detected only at Tile-end; later Groups may already have run|
|Tile-end UB / extra p0,cast / Arena|Retained|Retained|
|Runner GM Guard, Readonly, two Poison patterns, Repeat, full Golden|Retained|Retained|

A numerical PASS under the T protocol is **not** full Guard fault certification.
No fault-injection OPP is built for timing. Prior P fault evidence is verified but
never transferred as T fault coverage. Reports use
`fault_coverage=NOT_EQUIVALENT_NO_FULL_GUARD_CERTIFICATION`.

## Evidence and timing

The driver reuses read-only Stage4N/O/P/Q admission, including frozen input packing,
original Stage4B Golden, actual Core policy, T1, compiler/SDK identities and installed
OPPs. Only T gets a new private OPP. The P normal OPP is not rebuilt or reinstalled.
Each Runner process has exactly one private OPP. Independent benchmark Runner
binaries are compiled with mandatory `STAGE4T_DIAGNOSTIC=0/1`; execution JSON and
validation check the exact coverage label. Common external safety validators are
reused only after that explicit check; persisted records never relabel weak as full.

Host-only → bounded0/37/38/75 (Group0/19/final) → full Auto/2-Core mode21/0.
Full also executes the **accepted P Runner** for an independent untimed baseline
precheck. Every result retains the real plan, Workspace admission, logs and outputs.
Benchmark revalidates the whole successful chain. Failure stops progression.

Three fresh orders: baseline/candidate, candidate/baseline, baseline/candidate.
Each process uses continuous-v1: two Poison prechecks,5 warmups,30 Execute+Sync
samples with resident inputs, Timed Tail and independent Poison Postcheck.
No transfers, checks, file writes or Host logging inside formal timed execution.
Every version contributes90 samples. Statistics include raw samples, median/p95,
min/max, per-round ratios, Prepare timing, Core/Workspace and file identities.
Comparison decision is always `DIAGNOSTIC_MEASUREMENT_ONLY_NOT_PRODUCTION`;
report baseline−candidate as **net implementation difference**, not pure Guard time.
No Stage4L/S historical samples enter these statistics. Profile is separate, retains
CSV/type/Task/Stream/units/duplicate provenance via the existing row-aware collector;
no mixed-row medians or addition of Pipe times.

## Local checks and their limits

```bash
PYTHONPATH=framework/custom_ops/draft_quant/test python3 -B -m pytest -q \
  framework/custom_ops/draft_w8a8_output_dequant/stage4t/test
bash framework/custom_ops/draft_w8a8_output_dequant/stage4t/local_check/check.sh NEW_LOCAL_OUTPUT
```

CPU tests execute the actual baseline/candidate C++ source with cached-partial SDK
test doubles: full7/2-Core coverage, bounded/partial outputs, exact source delta,
source tampering, group0/9/18/19 transient/persistent fault timing, all640 words ×32
bits ×20 boundaries in a CPU boundary model, actual Runner5/30 protocol, mislabeled
coverage, bad Golden, timed logs, CLI and fail-closed phase gates.
This is **not** a device fault sweep or a model of CANN instruction/event timing.

Local CANN check: real SDK Kernel syntax with a **local-only tiling decode adapter**
(no OPP ABI claim), and real ARM64 CANN-linked Runners with a private API stub that
always fails. The stub never executes and is not an installed production/diagnostic
OPP. Actual private OPP generation, compiler-produced Tiling ABI, Host Tiling,
Kernel execution, numerical safety, Benchmark and Profile remain **NOT_RUN** until
server evidence is produced. Local libraries are `LOCAL_ONLY_NOT_SERVER_VERIFIED`.

Local validation on2026-11-06: macOS **28/28** and Docker ARM64 **28/28** CPU
tests passed; pyflakes and shell syntax checks passed. CANN syntax / Runner link
checks above completed with their stated adapters; real OPP/NPU/timing remain NOT_RUN.

## Server stages

From the repository after `git pull --ff-only`, set `CANN_ROOT` and a fresh root.
Default historical N/O/P/Q roots are the accepted dated experiments; override the
corresponding `--stage4*-root` if necessary, consistently in every phase.

```bash
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
T=framework/custom_ops/draft_w8a8_output_dequant/stage4t/run.sh
RUN=/home/w00949577/qwen35-dflash-run/w8a8-n256-guard-diagnostic-$(date +%Y%m%d-%H%M%S)
bash "$T" build --output-dir "$RUN/build" --device 0
bash "$T" host-only --build-report "$RUN/build/report.json" --output-dir "$RUN/host-only" --device 0
bash "$T" bounded --build-report "$RUN/build/report.json" --prior-report "$RUN/host-only/report.json" --output-dir "$RUN/bounded" --device 0
bash "$T" full --build-report "$RUN/build/report.json" --prior-report "$RUN/bounded/report.json" --output-dir "$RUN/full" --device 0
bash "$T" benchmark --build-report "$RUN/build/report.json" --prior-report "$RUN/full/report.json" --output-dir "$RUN/benchmark" --device 0
bash "$T" profile --build-report "$RUN/build/report.json" --prior-report "$RUN/benchmark/report.json" --output-dir "$RUN/profile" --device 0
```

Run each stage only after its predecessor's exact status. Alternatively the opt-in
`server.sh NEW_ROOT [--device 0 ...]` checks these statuses and executes the whole
sequence when explicitly invoked by the user; it never repairs/overwrites failures.

| Phase | Exact successful status |
|---|---|
|build|STAGE4T_BUILT_NPU_NOT_RUN|
|host-only|STAGE4T_HOST_ONLY|
|bounded|STAGE4T_DIAGNOSTIC_BOUNDED_MATCH|
|full|STAGE4T_DIAGNOSTIC_FULL_MATCH|
|benchmark|STAGE4T_PAIRED_MEASURED|
|profile|STAGE4T_PROFILE_COMPLETED|

Return the failed phase's `report.json`, relevant build/Runner log,
`execution.json`, `resource-admission.json` and first numerical difference from its
validation. Keep raw actual outputs, installed artifacts, source audit and all
historical reports intact. Benchmark summary is `comparison.json`. No complete
Draft, online preprocessing, production adoption or FP16 remeasurement occurs.
