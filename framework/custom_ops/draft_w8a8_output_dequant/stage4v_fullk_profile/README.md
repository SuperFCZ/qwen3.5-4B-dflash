# Stage4V Full-K normal OPP profiling

Baseline46f9f80861f6f35da7ebcf9bc84c975fb9675256. This directory adds a Profile
Driver, evidence binding, parser and analysis/report generation. It does not
change/reinstall/rebuild a Kernel, Host Tiling, Scale, Golden, old Runner or OPP.
No External CO1 investigation is performed. Numerical state stays
**NUMERIC_MISMATCH_UNRESOLVED** (six Tile0,201 Full, max1 ULP); no production claim.

The user-reported uninstrumented Full-K median1.641665ms/P951.64961ms with90
samples is a performance observation. Historical Group1282.77505ms and native
FP160.78698ms are not fresh paired measurements. Their shape is the same but
quantization, Scale and Guard/control paths differ. The driver revalidates actual
Benchmark samples and summary; it never substitutes these text values for files.

## What runs

`prepare` revalidates Source, Runner Build, Host-only, Tile0, Full and Benchmark
reports, links and file hashes. It verifies the normal OPP source audit, actual
private API/Host/Kernel ELF, accepted Runner and its Toolkit/Driver/system
bindings, frozen inputs and unmodified CPU reference. Benchmark summary is
recomputed from its90 samples. New machine OS/version evidence must match.

It then runs **only `msprof --help`**, records executable/help SHA256, and checks
support for all options. Local CANN9.0.0 confirms these options; the server must
confirm its own installed tool. No downloaded components or guessed switches.

`collect` invokes the **same accepted exploration Runner binary**, without
`--benchmark`, in a new process and private normal OPP environment:

```
RUNNER DEVICE NEW_CASE_DIR 0 0 -1
```

That existing non-benchmark path performs exactly two Poison/full-output/Guard/
Readonly/Repeat/Workspace/sync checks, with no warmups or timed loop. Thus only
**two full Auto executions** are collected, not90 (nor38 calls from a timed round).
No new device Runner is needed and no ABI is changed. The launch command is:

```
msprof --ai-core=on --aic-mode=task-based --aic-metrics=PipeUtilization \
 --task-time=on --task-trace=on --type=text \
 --output=NEW_PROFILE_DIR --application='bash NEW_LAUNCHER'
```

Actual paths/arguments, launcher SHA, profiler help/tool identities and runtime
logs are saved. After profiling, the original validator checks every safety and
numeric field again. Both622592B outputs must equal the accepted Full Smoke
output byte-for-byte and reproduce its complete201-difference report. This
comparison is repeat evidence; the CPU Golden is unchanged. If msprof returns0
but the Runner fails or outputs are missing, the Profile fails.

No Profile times enter Benchmark samples. Two traced passes may have cold/warm
cache differences and are not steady-state latency estimates. No full new
Benchmark is automatically run. Kernel Guard and Arena overhead remains included.

## Server commands

Use a new root, and retain both phase directories:

```bash
set -eo pipefail
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
source "$CANN_ROOT/set_env.sh"
set -u
D=framework/custom_ops/draft_w8a8_output_dequant/stage4v_fullk_profile
EXP=/home/w00949577/qwen35-dflash-run/w8a8-fullk-perf-explore-20261110-022343
OUT=/home/w00949577/qwen35-dflash-run/w8a8-fullk-profile-$(date +%Y%m%d-%H%M%S)
COMMON=(--experiment-root "$EXP" --cann-root "$CANN_ROOT" --device 0)
python3 -B "$D/run_profile.py" prepare "${COMMON[@]}" --output-dir "$OUT/prepare"
# Requires FULLK_PROFILE_READY; no Kernel has run in the prepare stage.
python3 -B "$D/run_profile.py" collect "${COMMON[@]}" \
 --prepare-report "$OUT/prepare/profile-report.json" --output-dir "$OUT/collect"
```

`FULLK_PROFILE_COMPLETED` means two safe outputs and two identified primary
AI Core tasks with no parser skips. It does NOT mean phase attribution is solved.
`FULLK_PROFILE_COMPLETED_METRICS_PARTIAL` means safe execution completed but some
artifact/identity/type coverage is unresolved; inspect availability/skipped rows.
Both statuses retain UNRESOLVED numerical and causal conclusions. `BLOCKED`
means failed evidence/tool/execution/output checks; stop and keep all files.

Collection writes:

* `profile-report.json`: six-stage history and runtime/source/tool identities,
  launch arguments, output safety and numeric validation, original file hashes.
* `metrics-report.json`: raw rows, types, units, identities, duplicate flags,
  independent per-file groups, timeline records, unbound rows and unavailable fields.
* `analysis-report.json` / `BOTTLENECK_REPORT.md`: actual supported counters,
  source counts, limitations, provisional single-variable investigation choices.
* `profiler/`: complete raw profiler tree, CSV, timeline and binary data preserved.
* `run/`: raw outputs, all differences, execution/resource admission and runner.log.

An optional **offline-only** reparse writes a separate new directory and requires
all sealed artifacts to remain unchanged:

```bash
python3 -B "$D/analyze_profile.py" --report "$OUT/collect/profile-report.json" \
 --output-dir "$OUT/reanalysis"
```

No automatic re-export, overwritten reports or silent schema fallback. If the
tool produces only raw data and no usable CSV, retain it and report UNAVAILABLE;
do not guess counters or run more workloads to manufacture a bottleneck claim.
Return profile-report.json, metrics-report.json, analysis-report.json and
BOTTLENECK_REPORT.md. On failure additionally return profile.log and
run/runner.log/execution.json/resource-admission.json, or prepare/msprof-help.txt.
Keep raw binaries and the entire profiler tree on the server with their hashes.

## Parser boundaries

* Kernel/Op names and Task Type are retained. Named records are matched to the
  Full-K operator; unnamed task_time rows join only one unambiguous named
  op_summary task with matching session/device/stream/task/context/PID identity.
* Group keys include source file, CSV kind, row type, Task Type, device, stream,
  Core, header schema and units. Core/task/aggregate/CPU/unknown rows do not share
  statistics. Exact duplicate rows are flagged and excluded from group samples;
  cross-table rows are not pooled or added.
* `TASK_TAGGED` means a Task ID exists, not proof that every counter is per-Core
  rather than an SDK-produced aggregate. Original strings and schemas remain.
* Only finite nonnegative counter values with explicit units are summarized;
  duration/time requires recognized time units. Unknown ratio units remain
  UNAVAILABLE. Missing Memory counters stay UNAVAILABLE under PipeUtilization.
* Timeline Task/Stream/Core fields and raw timestamps are preserved separately.
  A displayTimeUnit preference is not a timestamp encoding contract. Without an
  explicit time unit, timeline durations are not converted or used as wall time.
* MAC/Cube, Vector, Scalar and MTE activity can overlap. No pipe sum, no ratio→
  wall-time allocation, and no Host Execute+Sync subtraction to invent phases.
* ND→NZ, VDEQ16 CopyOut, final NZ→ND, End/PIPE_ALL and Guard/Arena phase durations
  remain UNAVAILABLE without PC/source mapping or controlled equivalent evidence.
* 76 Tiles use static ownership11/11/11/11/11/11/10 on the admitted7 Blocks.
  Those are source-derived assignments, not measured per-Core duration/long tail.

## Initial optimization decision

Actual bottleneck ranking awaits the real report. The **first bounded
investigation candidate is B-only offline NZ packing**, because it targets the
GM INT8 ND input conversion path and can be isolated without combining K groups.
This is not an implementation GO or a predicted speedup. SDK type/layout/Tiling,
resource and numerical equivalence must be established first. The report gives
principle/risk/minimal A/B for all five requested areas.

Small writeback calls are a valid hypothesis, not an established bottleneck.
Historical Stage4S changed comparable N256 output geometry in the grouped-K
path and found only1.0033x (below1.03). That weakens an automatic “writeback first”
choice but does not establish Full-K performance. No old samples enter a new A/B.

No new optimization, weak Guard, removed End/barrier/event, Workspace hoist,
N512 or full Draft integration is implemented. Only after real evidence should
one independently validated variable be selected for fresh paired measurement.

## Local validation

```bash
PYTHONPATH=framework/custom_ops/draft_quant/test python3 -B -m pytest -q -p no:cacheprovider "$D/test"
# Existing Docker, fresh output and existing local private API:
bash "$D/local_build_check.sh" /work/build/NEW_PROFILE_RUNNER_CHECK \
 /work/build/stage4v-release-20261106/api
```

The build check compiles/links the **unchanged** original exploration Runner
source into a new local directory. It does not replace the server Runner or
compile/install a Kernel. Local CANN HAL is a development stub, not a real
Driver. Help checks and CPU fixtures do not constitute real NPU/Profile data.
Actual server prepare/collection and bottleneck conclusions remain NOT_RUN /
UNRESOLVED until the provided commands complete on310P3.
