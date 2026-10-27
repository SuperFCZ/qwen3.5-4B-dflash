# Stage 4C first baseline: frozen W8A8 full Gate, no Kernel optimization

This directory builds only a new host Benchmark Runner. It links the accepted
Stage4B private OPP and reuses its exact Kernel, Host, inputs and full Golden.
No OPP is rebuilt/installed, and no frozen file is changed. Local CPU tests
and syntax/runtime doubles are not CANN/NPU evidence. Server Runner build,
Smoke, benchmark and profiling remain **NOT_RUN** until executed.

## Measurement contract

M16/K2560/N19456, 304 tiles, 20 groups/tile. Measure core limits 1, 2 and 0;
0 resolves through the actual accepted Core policy, not hard-coded seven.
The new Runner uses the unchanged `runner_common.h::RunMeasuredCalls`:

1. Allocate/copy inputs once and load the SHA-bound Stage4B Golden.
2. Two poisoned untimed calls, each checked against Golden **inside the
   Runner before timing can begin**, plus complete input readonly and guards.
3. Five warmup calls and thirty continuous measured calls.
4. Check/read back the last timed output, then one new poisoned postcheck.

There are 38 calls per benchmark invocation, four checked output files, and
30 execute+stream-sync samples plus 30 GetWorkspaceSize/prepare samples.
Prepare measures GetWorkspaceSize and fixed size/pointer bookkeeping,
separately from execute+sync. The first allocation is outside measured calls.
Workspace size drift aborts before the corresponding launch. Core/workspace
admission and guard/D2H/file checks run at the untimed boundaries; they are
not repeated inside the continuous warmup/sample interval. No check is
removed from the frozen device Kernel. Its UB/GM canaries, PIPE_ALL and scalar
guard loops remain part of measured execution cost.

The log size is captured after the second precheck and inspected before
timed-tail verification. If Host/SDK diagnostics append during the continuous
interval, the run fails formal timing rather than muting logs or silently
ignoring the IO. Samples and the timed output remain available for diagnosis.
This checks the Runner log and our callback IO boundaries, not every internal
SDK/driver filesystem operation. SDK-internal logging can therefore block
this first baseline and must be reported, not hidden by changing Stage4B.

Three rotating rounds are fixed: `[1,2,0]`, `[2,0,1]`, `[0,1,2]`. Each run is
an independent 5/30 process with its own prechecks/tail/postcheck. Reports
preserve round/position, all raw samples, median, p95 (nearest-rank), min and
max. Pooled summaries contain exactly 90 samples per Core limit. No old Smoke
calls or historical latency values are added to these samples.

The driver records actual launch/workspace plans, driver version files when
available, raw npu-smi output, CANN identity, executable/link identities,
frozen OPP/Kernel hashes and input/Golden hashes. Missing device-version
metadata is explicit UNAVAILABLE. Source, OPP and version-file integrity are
rechecked after the stage. There is no CPU fallback or new numerical Golden.

## Scope of the first release

This first release completes W8A8 Runner/measurement/profiling entrypoints.
**Fresh W8A16 paired execution is NOT_RUN**, not COMPARISON_COMPLETED. A
read-only readiness check examines the documented A3.1 batched suite,
Capture and Native OM manifests and compares all original X/q_nk/w_nz/s_gn
hashes with this Stage4B workload. Missing or mismatched artifacts are listed
in `comparison-report.json`; matching inputs/OM alone do not authorize a
historical custom Runner/OPP or a latency claim. No historical A3.1 number is
used as a new sample or speedup denominator.

The next comparison phase must separately admit the actual batched Runner,
OPP, Native Runner and Golden, then perform at least three fresh rotating
rounds of W8A8-auto / W8A16-custom / Native W8A16 under 5/30 continuous-v1.
Each scheme keeps its own strict Golden. W8A8 output must not be forced to
equal W8A16. Until that phase is implemented and executed, the comparison
status stays NOT_RUN even if this release's readiness check finds matching
Capture/OM files. `--a31-suite` can select the actual accepted historical suite
without guessing a random build directory.

## Online preprocessing is not measured

`preprocessing-inventory.json` separates these components:

| Component | Current state | Included in latency? |
| --- | --- | --- |
| INT8 weight tile pack, 49,807,360 B | Offline/static CPU layout validated | No |
| FP16 X 81,920 B → INT8 A 40,960 B | Saved full-X CPU quantization | No; online NPU NOT_RUN |
| scale_A × original GN Scale, 1,556,480 B FP32 | CPU, input-dependent | No; online NPU NOT_RUN |
| uint64 encoding/layout, 3,112,960 B | CPU, input-dependent | No; online NPU NOT_RUN |
| Input upload, 52,961,280 B | Before measured calls | No |
| Frozen Cube/VDEQ16/FP32 reduction and guards | Stage4B Kernel | Yes |
| Final FP16 Cast and 622,592 B GM output | Stage4B Kernel | Yes |
| D2H and correctness checks | Outside timing | No |

The overall label is **ONLINE_PREPROCESSING_NOT_RUN**. These are resident,
preprocessed-input Kernel latencies, not complete FP16-input forward latency
or model/Decode performance. A future online quantization/Scale kernel must
independently reproduce the full-X binary64-scale/RNE/zero/encoding contract
before claiming equivalence. None is developed here. The accepted cross-scheme
error does not establish whole-model quality.

## Separate profiling

`profile` requires a complete unprofiled Stage4C baseline with the same Runner,
OPP, input and device identities. It invokes only the auto-Core Runner, in a
new directory, with msprof flags reused from `draft_quant/test/profile_a31.py`:
`--ai-core=on --aic-mode=task-based --aic-metrics=PipeUtilization`.
Its timed samples are labeled profiled and never pooled with formal latency.
Golden, readonly, guards, repeats, timed-tail and postcheck are still checked.

The parser retains matched operator CSV columns, values, units, file hashes,
line numbers and device timeline events. Unavailable/N/A fields remain
UNAVAILABLE. It does not sum overlapping Pipe activity, convert ratios to
elapsed time, or infer a per-core tail from an aggregate Task Duration. Static
tile ownership is reported separately from measured core-tagged rows. Matmul,
VDEQ16, Cast/Add and PIPE_ALL instruction cost attribution is UNAVAILABLE
without actual PC/source/instruction evidence. Missing data is not fabricated.
No bottleneck or optimization benefit is declared before inspecting real
counters. Tool success without usable operator metrics is reported separately
from STAGE4C_PROFILE_COMPLETED.

## Server commands

Build the **host Runner only**, Smoke it, then run the unprofiled baseline:

```bash
set -euo pipefail
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
S4C=/home/w00949577/qwen35-dflash-run/w8a8-baseline-$(date +%Y%m%d-%H%M%S)
ENTRY=framework/custom_ops/draft_w8a8_output_dequant/stage4c/run.sh
bash "$ENTRY" build --device 0 --output-dir "$S4C/build"
bash "$ENTRY" smoke --device 0 \
  --runner-build-report "$S4C/build/benchmark-report.json" --output-dir "$S4C/smoke"
bash "$ENTRY" benchmark --device 0 \
  --runner-build-report "$S4C/build/benchmark-report.json" \
  --smoke-report "$S4C/smoke/benchmark-report.json" --output-dir "$S4C/benchmark"
printf 'Evidence: %s\n' "$S4C"
```

Only after successful unprofiled measurement, collect msprof separately:

```bash
bash "$ENTRY" profile --device 0 \
  --benchmark-report "$S4C/benchmark/benchmark-report.json" \
  --output-dir "$S4C/profile"
```

All destinations must be new. The default accepted Stage4B root is
`/home/w00949577/qwen35-dflash-run/w8a8-full-gate-20261027-093617`.
No GE/TorchAir import, model export, OPP installation, complete Draft or
Simulator is launched. Each stage writes the four requested JSON reports:
`benchmark-report.json`, `comparison-report.json`, `profile-report.json`,
`preprocessing-inventory.json`.

Return those reports and `npu-smi.log`. On failure retain the failing
`execution.json`, `runner.log`, actual outputs, timed-tail/postcheck if present,
and build logs. Failed/partial sample arrays remain in execution.json and are
never summarized as accepted formal latency. For profiling return `profile.log`
and the original profiler directory; the report binds raw file identities.

## CPU tests

```bash
python3 -B -m pytest -q framework/custom_ops/draft_w8a8_output_dequant/stage4c/test
bash -n framework/custom_ops/draft_w8a8_output_dequant/stage4c/run.sh
```

The actual Runner is compiled against CPU ACL doubles that abort on transfers,
memsets or allocations during continuous warmup/samples. Tests exercise 38
calls, all four Golden checks, failure before timing, timed-log rejection,
sample validation, profile exclusion and missing-counter handling. These
fixture timings and mocked device labels are not CANN or NPU measurements.
