# Stage 4F: one SetLocalWorkspace per N64 tile

Isolated Ascend310P3 / CANN9.0.0 experiment. The only computation-source change
is relocation of `mm.SetLocalWorkspace(scratch)` from the K-group loop to tile
entry, after the existing Sum initialization/PIPE_ALL and before group 0. Its
comment is updated accordingly. Nothing is installed into the production OPP.
No historical source, SDK, Golden, input or evidence is changed.

## Four independent identities

| Variant | Source baseline | Workspace binds | Guard protocol |
|---|---|---:|---|
| A0 | exact accepted Stage4B | 6080 | complete Stage4B checks |
| A1 | A0 + statement move | 304 | same complete checks |
| B0 | exact accepted Stage4D B | 6080 | permanently weaker diagnostic checks |
| B1 | B0 + statement move | 304 | same weaker diagnostic checks |

A0→A1 and B0→B1 have separate byte-checked diffs. All Host/ABI files remain
identical; no End, Iterate/GetTensorC, Scale update, barrier, event, guard,
Core policy, workspace arena formula, FP16 partial or FP32 Add changes.
Scratch 239104 B + user UB 12288 B + guards 2560 B + reserve 8192 B = 262144 B.
No additional local buffer or event is allocated.

Each variant has a separate generated project, private OPP and linked Runner.
The original operator name is retained to avoid a second Kernel source variable;
each execution is a fresh process whose environment selects only that OPP.
The Runner variant argument records identity, not a runtime Kernel switch.

`source-audit.json` binds A0 to accepted Stage4B identities and B0 to accepted
Stage4D B. Materialized and actually compiled source files are checked. Binary
SHA256 comparisons are independent for A and B. Identical binaries are permitted
and explicitly reported as `IDENTICAL_BINARY_NO_COMPILED_CHANGE`; they cannot
establish a compiled optimization. Different ELF hashes can include debug/path
changes and are not proof of instruction reduction or performance improvement.

## Approved SDK scope

`sdk-contract.json` contains only already reviewed relative paths, SHA256,
line ranges and brief conclusions. No SDK text, snapshots or server logs are
included. The receiver only verifies these fixed identities; it does not scan
or copy SDK sources or reopen the review. Changed identities fail closed.

The user-reviewed GO is admission to this experiment, not a new universal
SOURCE_REVIEWED certificate or device PASS. Workspace binding/partition remains
constant across each tile; reviewed Reset/Destroy paths do not clear that
binding. Every End and all synchronization remain in place. No inference is
made that queue event release is free, that End is redundant, or that asynchronous
input/Scale rebinding is safe. Runtime correctness still requires NPU evidence.

## Gates and protocol

Prerequisites are the accepted Stage4B/4C/4D evidence chain and Stage4E at
`/home/w00949577/qwen35-dflash-run/w8a8-tile-guard-20261029-020949`.
Existing validators recheck their identities/correctness without executing old
experiments. All new outputs use fresh directories; old reports are read-only.

- Build: four OPPs and benchmark-capable ACLNN Runners, with SDK/tool/library,
  generated-source, Kernel, Host, input and Golden identities.
- Smoke: A0/A1 auto, then A0/A1 two-core; then B0/B1 auto and two-core. Each
  has two poisoned calls and strict 311296-element frozen FP16 Golden matching,
  Readonly, outer GM guards, its declared Kernel guards and Workspace Admission.
  All eight runs must pass before **any** benchmark. Auto uses the real accepted
  Core policy (currently seven), never a hardcoded launch count.
- Benchmark: A0/A1, A1/A0, A0/A1; separately B0/B1, B1/B0, B0/B1. Every run
  uses continuous-v1: two untimed checks, warmup 5, repetitions 30, timed-tail
  check, then Poison+correctness postcheck. Inputs stay resident; no application
  readback, poison, file I/O or allocation in measured callbacks. Host logging
  during the continuous interval rejects formal timing. Each variant retains
  90 raw execute+sync samples and separate prepare samples, median/p95/min/max.
- Profile: only after accepted unprofiled benchmark; four separate msprof
  PipeUtilization runs. Raw CSV/timeline identities are retained, missing
  counters are UNAVAILABLE and unmatched column coverage UNRESOLVED. Compare
  A0/A1 and B0/B1 only. Never add overlapping pipes or treat Scalar as wall time.

Returned full tiling/configuration and ACLNN workspace must agree for equal
Core limits, including across Smoke, Benchmark and Profile. A1's full-guard
correctness and B1's scoped diagnostic correctness are reported separately.
B success never upgrades it to Stage4B safety equivalence or production use.

Current runtime output mode is final sum (0). Group 0/19 diagnostic retention
is exercised in the CPU model; **device partial diagnostics are NOT_RUN** in
this first entry. They are not falsely reported as tested. Final frozen-Golden
comparison and exact per-group operation preservation remain mandatory.

## Server commands

Run in the already accepted device environment:

```bash
set -euo pipefail
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
ENTRY=framework/custom_ops/draft_w8a8_output_dequant/stage4f/run.sh
S4F=/home/w00949577/qwen35-dflash-run/w8a8-workspace-tile-$(date +%Y%m%d-%H%M%S)
bash "$ENTRY" build --device 0 --output-dir "$S4F/build"
bash "$ENTRY" smoke --device 0 \
  --pair-build-report "$S4F/build/build-report.json" --output-dir "$S4F/smoke"
bash "$ENTRY" benchmark --device 0 \
  --pair-build-report "$S4F/build/build-report.json" \
  --smoke-report "$S4F/smoke/smoke-report.json" --output-dir "$S4F/benchmark"
bash "$ENTRY" profile --device 0 \
  --benchmark-report "$S4F/benchmark/benchmark-report.json" --output-dir "$S4F/profile"
printf 'Stage4F evidence: %s\n' "$S4F"
```

Use `--stage4e-root` on each invocation only when selecting a different intact
accepted Stage4E root. Other prerequisite roots have the historical defaults.
`--msprof /path/to/msprof` can select the profiler explicitly. Every output
folder must be new; failures retain logs and numerical outputs, never fallback.

Reports: `build/build-report.json`, `smoke/smoke-report.json`,
`benchmark/benchmark-report.json`, `profile/profile-report.json`, plus each
stage's `guard-coverage.json`. Build includes `sources/source-audit.json`,
`A-workspace-only.diff`, `B-workspace-only.diff`, SDK metadata binding and
binary_comparison. Benchmark's `comparison.pairs.A` and `.B` contain independent
contrasts. No automatic speedup/adoption verdict is generated. Lack of a
reproducible benefit leads to codegen analysis, not automatic End/barrier edits.

On failure return the stage report, relevant variant's `build.log`,
`runner-build.log` or `case/runner.log`, `execution.json`, and
`resource-admission.json`. Keep all actual binary outputs and old evidence.

## Local verification

```bash
PYTHONPATH=framework/custom_ops/draft_quant/test python3 -B -m pytest -q \
  framework/custom_ops/draft_w8a8_output_dequant/stage4f/test
```

CPU tests compile the actual materialized kernels against test-only lifecycle
models: 304 tile resets/bindings, 20 Ends per tile, independent A/B/Scale updates,
1/2/7-core coverage, Group 0/19 retention, unchanged guard/event/barrier counts
within each pair, and a counterfactual End-clears-binding failure. This is a
model of the reviewed contract, not simulation proof of SDK/NPU behavior.
The actual new Runner is tested with ACL doubles for timing boundaries, auto/2
Core admission, strict prechecks and protocol labels. No CPU fallback exists in
the server flow. CANN compilation, NPU and new performance are NOT_RUN locally.
