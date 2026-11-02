# Stage4L: Tile width × group Guard coverage

Independent 2×2 diagnostic experiment. Historical Stage4B–4K source, OPPs,
Golden and reports remain read-only. Server compilation, Smoke, Benchmark
and Profile are NOT_RUN at development delivery. No production adoption,
Workspace hoist, synchronization changes, N256 or Vector Guard implementation.

| Variant | Kernel origin | Coverage |
|---|---|---|
| N64Full | accepted Stage4B OPP, reused | complete original group/tile/Arena Guards |
| N128Full | accepted Stage4J OPP, reused | complete original group/tile/Arena Guards |
| N64Weak | accepted Stage4D B OPP, reused | diagnostic; no five group-tail UB scans |
| N128Weak | new private OPP derived from Stage4J | diagnostic; same five group-tail scans removed |

Both Weak variants permanently carry
`DIAGNOSTIC_ONLY_WEAKER_GUARD_COVERAGE`. Golden matching under the remaining
checks does not prove equivalent safety. A transient overwrite restored before
Tile-end may be missed. All Guard initialization, final Tile UB scans, final
extra p0/cast checks, Arena checks, synchronization/events (including V_S),
Matmul End, workspace binding, A/B/Scale updates and arithmetic are unchanged.
No extra UB buffer or use of the8192B reserve is introduced.

## Admission and source audit

Stage4K's current historical Benchmark and Profile proofs are revalidated.
Stage4D B must have the unchanged exact source audit, compiled source hashes,
private OPP and accepted measurements/Golden checks. Its old samples are
never reused for the new comparison. Stage4K's admitted N64/N128 inputs,
Stage4I SDK contract and Stage4J Full proof remain the provenance chain.

Build creates only the N128 Weak compute OPP and four standalone Benchmark
Runners. Full OPPs and N64 Weak are never regenerated or reinstalled. Every
process uses one private OPP environment. Reusing an operator name across
isolated processes does not register two experimental OPPs together.

N128 Weak source materialization removes exactly the single five-condition
CheckGuard/return block at the Group-loop tail, anchored immediately after
V_S. `source-audit.json` binds all original/generated files and the exact
`group-guard-only.diff`. The compiled source inventory, unchanged OpDef and
Stage4J T1-generated Host contract are checked again. Kernel binaries retain
ELF/SHA256 identities; identical Full/Weak binary hashes block the contrast.
Different hashes alone are not machine-instruction attribution.

## Fresh balanced measurements

Four Williams-balanced rounds place each version in every position once and
each ordered adjacent pair once:

1. N64Full → N128Full → N128Weak → N64Weak
2. N128Full → N64Weak → N64Full → N128Weak
3. N64Weak → N128Weak → N128Full → N64Full
4. N128Weak → N64Full → N64Weak → N128Full

Each process uses continuous-v1: two poisoned strict prechecks, warmup5,
30 measured execute+stream-sync samples, checked timed tail, then a fresh
poisoned postcheck. This gives120 fresh samples per variant. Prepare latency
is separate. The runners reject timed logging, changed inputs, external
Guard failure, Repeat drift or any of311296 Golden bit mismatches. No timed
readback/Poison/file IO is added. Every run uses the admitted seven-Block
policy. Within each width, Full/Weak Workspace and resource plans must match.

`comparison.json` reports each variant's raw samples and distribution, each
round's contrasts, and aggregate:

- N64Full − N64Weak;
- N128Full − N128Weak;
- N64Full / N128Full;
- N64Weak / N128Weak;
- interaction = (N64Full − N64Weak) − (N128Full − N128Weak).

These are net effects of the compiled variants, including code generation
and scheduling consequences. They are not pure Guard instruction time.
All measurements are `PREPACKED_RESIDENT_W8A8_KERNEL_ONLY`; dynamic online
activation/combined-scale/encoding, full Draft and Decode remain NOT_RUN.

## Profile provenance first

Before building, the driver rereads Stage4K's hashed CSV records and records
file kind, CSV record number, full original row, Task/Core/Stream/type fields,
column units, and duplicate-looking records. Missing type/unit stays
UNRESOLVED. It does not mix task summaries, per-Core rows, duplicate exports
or different units into a causal median. Server reports are not accessible
on the development host, so this review occurs on the server and remains
available as `historical_csv_review` even in later failures.

After successful fresh Benchmark, all four variants are independently
profiled. Raw CSV/timelines and hashes are kept. Each new profile receives the
same provenance review. No automated cross-row Profile pooling or Pipe-time
sum is produced; unknown counters remain UNAVAILABLE. Device Task Duration,
Scalar, MTE1/2/3, Vector, MAC and Cube columns retain their original names and
units. Profile timing is never a formal latency sample.

## Server stages

Use fresh output subdirectories; any nonzero exit stops the sequence. Do not
edit historical reports to satisfy a gate.

```bash
set -euo pipefail
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
S4L=/home/w00949577/qwen35-dflash-run/w8a8-width-guard-cross-$(date +%Y%m%d-%H%M%S)
ENTRY=framework/custom_ops/draft_w8a8_output_dequant/stage4l/run.sh
bash "$ENTRY" build --device 0 --output-dir "$S4L/build"
```

After STAGE4L_RUNNERS_BUILT, inspect `source_audit`, `weak_opp`, the four Runner
identities and `historical_csv_review` in build-report.json:

```bash
bash "$ENTRY" smoke --device 0 \
  --runner-build-report "$S4L/build/build-report.json" --output-dir "$S4L/smoke"
```

After STAGE4L_SMOKE_MATCH (scoped correctness, not safety equivalence):

```bash
bash "$ENTRY" benchmark --device 0 \
  --runner-build-report "$S4L/build/build-report.json" \
  --smoke-report "$S4L/smoke/smoke-report.json" --output-dir "$S4L/benchmark"
```

After STAGE4L_PAIRED_MEASURED:

```bash
bash "$ENTRY" profile --device 0 \
  --runner-build-report "$S4L/build/build-report.json" \
  --benchmark-report "$S4L/benchmark/benchmark-report.json" --output-dir "$S4L/profile"
```

Return four phase reports, benchmark/comparison.json and
preprocessing-inventory.json. Preserve failing variant logs/outputs and
weak-opp/{build.log,install.log,host-ldd.log,build-report.json}. Nothing is
installed in production. Original SDK text/logs/binaries must not be uploaded
to public GitHub.

## CPU regression

```bash
PYTHONPATH=framework/custom_ops/draft_quant/test python3 -B -m pytest -q \
  framework/custom_ops/draft_w8a8_output_dequant/stage4l/test
```

Four actual Runner sources execute against CPU ABI doubles that forbid
copies/allocations during the continuous interval. Tests reject bad Golden,
timed logging, mislabeled Guard coverage, incomplete/profiled samples and
unapproved source edits. The actual N128 Kernel control flow is compiled with
cached-partial Matmul doubles: a Group3 Scratch Guard overwrite repaired in
Group4 stops Full but can escape Weak; a persistent Group19 overwrite stops
Weak at Tile end. This is a CPU safety-coverage counterexample, not a device
MMAD/VDEQ16/event correctness result.

Only after real results should a separate experiment consider lifecycle work
or full-coverage Vector Guard checks. No such optimization is enabled here.
