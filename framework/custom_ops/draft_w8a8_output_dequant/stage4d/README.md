# Stage 4D: Scalar group-guard attribution, not a production optimization

This independent A/B experiment changes exactly one source region of the
accepted Stage4B Kernel. Historical Kernel, Host, OPP, Runner, Golden and
Stage4C evidence are read-only. No Vector guard replacement, quantization,
Matmul End, K group, PIPE_ALL, prefetch or scheduling change is made.
Local CPU tests are not CANN/NPU evidence; new device stages are **NOT_RUN**.

## Two deliberately different safety protocols

| Variant | Kernel source | Safety label |
| --- | --- | --- |
| A | Byte-identical Stage4B Kernel, including five group-tail scans | BASELINE_GROUP_GUARDS |
| B | Deletes only the group-tail five-CheckGuard conditional and its error return | DIAGNOSTIC_ONLY_WEAKER_GUARD_COVERAGE |

Both retain every group Matmul call, End, Scale update, Cast/Add, PIPE_ALL,
FetchEventID and V_S SetFlag/WaitFlag. Both retain guard initialization, all
five final tile UB scans, subsequent final checks, and per-tile GM arena
checks. There are no partial GM writes or new math instructions.

For a clean execution, the removed source block performs
`304 * 20 * 5 * 64 * 2 = 3,891,200` GetValue calls. This is a source-level
success-path count, not an assertion of generated instruction counts or a
conversion to milliseconds. B can miss transient guard corruption restored
before the tile ends. Its numerical match **does not** upgrade it to the
original Stage4B safety acceptance, even if it is much faster.

CPU fault injection demonstrates this distinction: A detects a group-0
scratch-guard corruption; B can miss it if group 1 restores it. Persistent
corruption reaching the tile end is rejected by both. No safety equivalence
or production adoption is reported.

## Source and build isolation

Build copies the verified Stage4B operator JSON and Host source byte-for-byte
into two new source bundles. A's Kernel is also an exact copy. B is produced
by one exact, context-checked deletion; missing, duplicate or changed context
fails instead of adapting the patch. `sources/source-audit.json` and
`sources/group-guard-only.diff` bind original/generated hashes and the removed
span. Generated CustomOp sources are checked after compilation too.

The two private projects, packages, installed OPPs and linked Runners have
separate identities. Identical A/B Kernel binary hashes block measurement
because a compiled contrast would not be established. A differing hash is
necessary for identity, not proof that compiler scheduling outside the region
is identical. The net effect may include code-layout/compiler consequences.

The operator name and ABI intentionally stay `DFlashW8A8FullGate` to avoid a
second source change. **Never co-load the variants.** Every invocation uses a
fresh C++ process and exactly its own private OPP/library environment. The
shared Stage4D Runner source has identical numerical/timing behavior for A/B;
the required A/B CLI label only records coverage, it does not switch guards
inside one binary. Driver admission binds that label to the audited binary.

## Correctness and timing

Input is the same Stage4B frozen M16/K2560/N19456 data and full FP16 Golden.
Only auto Core scheduling is allowed; the returned Core policy must equal the
accepted Stage4C context (currently seven Blocks, never hard-coded). Complete
Host Matmul resource lines, Core/Arena plans and actual ACLNN workspace bytes
must agree between variants and across rounds. UB allocation and workspace
offsets are unchanged.

Smoke requires two poisoned full-output repeats for A and B. Benchmark starts
only after both pass under their declared protocols. Each invocation then
uses the Stage4C continuous-v1 implementation: two untimed Golden/readonly/
guard checks, warmup=5, repetitions=30 with resident inputs and no callback
Poison/readback/file IO, timed-tail verification, then a poisoned postcheck.
Workspace drift aborts before launch. The timed-log gate is retained; logs
are never silently muted to obtain good numbers.

Fresh rounds are **A/B, B/A, A/B**: 90 formal samples per variant. Reports keep
all samples, round/position, median/p95/min/max, per-round differences and
pooled A-minus-B latency and A/B ratio. Stage4C historical measurements are
prerequisite identity evidence only, never new samples. This remains
preprocessed/resident-input latency; online preprocessing and full Draft are
NOT_RUN.

## Independent profiling and attribution limits

After the complete unprofiled A/B benchmark, profile both variants separately
with msprof PipeUtilization. Each still checks Golden, input readonly, GM
guards, permitted UB checks, repeat, timed-tail and postcheck. Profiled timing
cannot enter the formal sample pool.

All raw CSV/JSON file identities are retained through the Stage4C parser.
Task Duration, Scalar, MAC, Vector, MTE1/2/3 and Cube fields keep original
headers/units. Same-column comparisons group by CSV kind, require equal
matched-row counts, and report A/B medians and deltas. These rows may include
precheck/warmup/postcheck tasks. Missing or unequal evidence stays UNRESOLVED;
unavailable counters are not invented. Pipe activity is never added as total
time, and Scalar delta is never labeled wall-time savings.

The experimental conclusion is the measured net effect of removing the
group scan region under weaker guard coverage. Critical-path wait attribution
and an equivalent every-group Vector Compare/Reduce guard are NOT_RUN. Only
after reviewing real A/B results should a separate implementation prove that
every 256-byte prefix/suffix element and corruption case is covered.

## Server commands

New output roots only. Build both independent OPPs and Runners:

```bash
set -euo pipefail
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
S4D=/home/w00949577/qwen35-dflash-run/w8a8-guard-ab-$(date +%Y%m%d-%H%M%S)
ENTRY=framework/custom_ops/draft_w8a8_output_dequant/stage4d/run.sh
bash "$ENTRY" build --device 0 --output-dir "$S4D/build"
bash "$ENTRY" smoke --device 0 --pair-build-report "$S4D/build/build-report.json" \
  --output-dir "$S4D/smoke"
bash "$ENTRY" benchmark --device 0 --pair-build-report "$S4D/build/build-report.json" \
  --smoke-report "$S4D/smoke/smoke-report.json" --output-dir "$S4D/benchmark"
printf 'Evidence: %s\n' "$S4D"
```

Only after the paired benchmark succeeds:

```bash
bash "$ENTRY" profile --device 0 \
  --benchmark-report "$S4D/benchmark/benchmark-report.json" --output-dir "$S4D/profile"
```

Default prior Stage4C root:
`/home/w00949577/qwen35-dflash-run/w8a8-baseline-20261027-111748`.
Build, Smoke, Benchmark and Profile write their respective `*-report.json`
plus `guard-coverage.json`. Success labels are STAGE4D_BUILT_NPU_NOT_RUN,
STAGE4D_SCOPED_CORRECTNESS_MATCH, STAGE4D_AB_MEASURED_DIAGNOSTIC_ONLY and
STAGE4D_PROFILES_COLLECTED_DIAGNOSTIC_ONLY. None means B has baseline safety.

Return the reports and source audit/diff. Failures retain compiler logs,
per-variant `execution.json`, `runner.log`, all available output files and raw
samples. Profile failures retain the profiler directories too. No default
production operator, historical report or Golden is replaced.

## Local regression

```bash
python3 -B -m pytest -q framework/custom_ops/draft_w8a8_output_dequant/stage4d/test
bash -n framework/custom_ops/draft_w8a8_output_dequant/stage4d/run.sh
```

Tests cover the exact source deletion, unchanged synchronization, fail-closed
source drift, both Runner coverage labels, continuous timing, actual Kernel
source-level read counts, transient versus persistent guard faults, profile
exclusion and raw-column comparison. CPU doubles are not hardware evidence.
