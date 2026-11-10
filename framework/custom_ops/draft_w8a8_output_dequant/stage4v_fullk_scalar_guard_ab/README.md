# Full-K Scalar Guard A/B — Arena XOR/OR only

Baseline f225ec79eeac2f05f593025e8ed1cd5c91d2f01e. Independent candidate normal
`DFlashFullKGuardAggregate` and diagnostic `DFlashFullKGuardFault` private OPPs.
Baseline uses the frozen original Stage4V normal OPP and accepted exploration
Runner. No historical source/OPP/Golden/Scale/report is changed. The candidate
keeps the original Kernel mathematics, VDEQ16, all barriers/events/End, Tiling,
Workspace formula, UB allocation,76 Tiles and formal7-Core policy.

## Selected single variable

Vector implementation is **not admitted for the reviewed routes**. Actual local
CANN9.0.0 confirms that the relevant dav_2002 CompareScalar accepts half/float
input and uint8 bitmap output, and reviewed Reduce/BlockReduce interfaces accept
half/float, not uint32. Previous bitmap/GetCmpMask write-footprint contracts
remain unresolved. We do not bypass those restrictions, invent an intrinsic or
claim every possible vector algorithm is impossible. See SDK_REVIEW.md.

The existing `Delta`/`AllDelta` already use exact uint32 XOR/OR. Rewriting them
again is not the candidate. Only the per-side detection loop in `CheckArena`
changes:

```
for i=0..127: if GetValue(i) != guard: return false
```

becomes:

```
delta=0
for i=0..127: delta |= GetValue(i) XOR guard
if delta != 0: return false
```

The same128 words are read in the same index order after the same MTE2_S wait.
The failure decision remains inside the same side/checkpoint before advancing
to the next side or computation. A corrupt side may be read further than the old
first-error early return, but all reads stay within the already-loaded512B
payload. It is not a Tile-end-only or weakened check.

OR cannot cancel differences: delta=0 iff every word equals the original
0x5a39c671. No floating conversion, uint64 paired access, bitmap, new UB or Reserve
use. Normal GetValue count remains108800. This tests compiled control/condition
cost, not a claimed reduction in loads or a prediction from Scalar Time.

`sg_transform.py` verifies exact source transformation and identifier renaming.
Checked-in Kernel/Host/Runner files must equal those transformations. The normal
Host code is identical after Op/type/file-name normalization. Diagnostic Host
alone permits output modes6/7 for Arena abort witnesses; Tiling/resource setters
remain unchanged. Compiled sources and the generated frozen T1 contract are
reverified against the audit before any execution. Normal and Fault cannot be
substituted into each other's phase or performance selection.

## Coverage, timing and resource audit

| Item | Normal baseline and candidate |
|---|---|
| UB redzones |Scratch202240, FP16 partial8192, Arena temp512; each prefix/suffix256B|
| UB words per check |384 uint32 words, six complete256B regions|
| UB boundaries per Tile |before Iterate; after GetTensorC/End; before output write|
| UB GetValue/full Auto |384×3×76=87552|
| Arena sites |initial per-Core check7 + post-Matmul per-Tile check76|
| Arena words/site |128 prefix +128 suffix;512B each|
| Arena GetValue/full Auto |256×83=21248|
| Arena data-dependent checks |baseline21248; candidate166 (loop control still exists)|
| Normal Scalar SetValue |0 in these Guard routines; Duplicate initialization unchanged|
| Explicit UB |212480B, same three allocations|
| Budget including Reserve |220672B; Reserve8192 unchanged|
| Scratch |202240B, unchanged|
| L1/L0/Workspace/Core |all fields checked against actual frozen Full-K T1/Formal policy|

Counts are source-level normal-path counts, not machine instructions, critical
path time or removable work. Profiler Task1578.919us / Scalar1339.887us and other
pipes overlap. We do not sum or allocate their times. Current1.641665ms is a
historical reference, never the denominator of a new paired result.

## Independent Fault OPP

No fault instrumentation reaches the normal candidate. No extra UB is allocated.

* Mode2 preserves original384×32×3=36864 UB injections, and adds actual **GM
  Arena** injections at both Arena sites:2×2×128×32=16384. Each corruption is
  written into a GM redzone, read/detected by the candidate CheckArena, restored,
  and rechecked. Both runs must report53248 injections each,106496 total.
* Modes3/4/5 preserve exact UB checkpoint abort witnesses.
* Modes6/7 corrupt the Arena suffix's last word at initial/post-Matmul site,
  require detection, restore the redzone, publish a distinct abort witness and
  return before later math/output. Full output after the first32 witness bytes
  must remain the repeat-specific Poison.
* Fault-only V_S/S_MTE3/MTE3_MTE2/MTE3_S edges order vector initialization, Scalar
  corruption, GM publication, reload and temporary-buffer reuse. Normal events
  and buffers are unchanged. MTE3_S explicitly closes source-buffer DMA before
  reuse. No application event reservation or SDK state changes are introduced.
* Fault runs are Tile0,1 Block, two repeats for all six modes. This proves the
  stated detection sites/words and abort behavior, not injections on every
  physical Core or all76 Tiles, nor an exhaustive device bit sweep of the Runner
  input/output outer canaries. Those unchanged checks have CPU receiver fault
  tests. Normal Full Smoke covers full output mapping.

Fault witness match is **not** a numerical Golden PASS. Intentional UB damage is
reported as diagnostic coverage, not healthy normal Guard state. GM input/output
outer Guards and readonly remain required. Mode2 must finish every sweep to
publish its count witness. Fault mode cannot enter Benchmark, in both CLI and
Runner. Any timeout, incomplete count, wrong witness or unexpected output blocks.

## Gates and exact states

1. Source: `GUARD_AB_SOURCE_REVIEWED` — bind all accepted exploration phases and
   the real Profile; SDK hashes; source/Runner difference audit.
2. Build: `GUARD_AB_BUILT_NPU_NOT_RUN` — two new isolated OPPs and Runners only.
3. Host: `GUARD_AB_FORMAL_HOST_MATCH` — GetWorkspaceSize-only for both private
   OPPs, actual TilingContext, original T1/capacity/Core/Workspace contract.
4. Tile0: `GUARD_AB_TILE0_MATCH` — two finite complete active outputs; inactive
   regions retain per-repeat Poison; all active bits match frozen baseline.
5. Full: `GUARD_AB_FULL_MATCH` — two full311296-element outputs equal frozen
   baseline; original201 one-ULP CPU differences preserved exactly.
6. Fault: `GUARD_AB_FAULT_COVERAGE_MATCH` — all six modes twice, complete scope.
7. Benchmark: `GUARD_AB_PAIRED_MEASURED` — only after all preceding gates.

Build precedes Formal Host because the real registered callback requires the new
OPP/API. Local compilation is not Formal Host or NPU admission. The server generates
its T1 contract from actual frozen evidence, never the local compile-only fixture.
No fallback, changed Scratch/Base/Core choice or tolerance change is allowed.

Candidate numerical output must match the frozen baseline **bitwise**, while
CPU differences remain `NUMERIC_MISMATCH_UNRESOLVED`. This is not production
approval. Kernel-internal Guard checks remain included in timing. GM guards,
input readonly, two Poisons, Repeat, sync, Workspace admission and output coverage
are retained. Normal Kernel UB/Arena detection is observed through its original
Poison-on-failure protocol; there is no invented direct Host UB readback.

## Server commands, new experiment only

```bash
set -eo pipefail
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
source "$CANN_ROOT/set_env.sh"
set -u
D=framework/custom_ops/draft_w8a8_output_dequant/stage4v_fullk_scalar_guard_ab
OUT=/home/w00949577/qwen35-dflash-run/w8a8-fullk-arena-ab-$(date +%Y%m%d-%H%M%S)
COMMON=(--cann-root "$CANN_ROOT" --device 0 \
 --experiment-root /home/w00949577/qwen35-dflash-run/w8a8-fullk-perf-explore-20261110-022343 \
 --profile-report /home/w00949577/qwen35-dflash-run/w8a8-fullk-profile-20261110-030853/collect/profile-report.json)
python3 -B "$D/run_guard_ab.py" source "${COMMON[@]}" --output-dir "$OUT/source"
python3 -B "$D/run_guard_ab.py" build "${COMMON[@]}" --source-report "$OUT/source/report.json" --output-dir "$OUT/build"
GATES=(--source-report "$OUT/source/report.json" --build-report "$OUT/build/report.json")
python3 -B "$D/run_guard_ab.py" host "${COMMON[@]}" "${GATES[@]}" --output-dir "$OUT/host"
GATES+=(--host-report "$OUT/host/report.json")
python3 -B "$D/run_guard_ab.py" tile0 "${COMMON[@]}" "${GATES[@]}" --output-dir "$OUT/tile0"
GATES+=(--tile-report "$OUT/tile0/report.json")
python3 -B "$D/run_guard_ab.py" full "${COMMON[@]}" "${GATES[@]}" --output-dir "$OUT/full"
GATES+=(--full-report "$OUT/full/report.json")
python3 -B "$D/run_guard_ab.py" fault "${COMMON[@]}" "${GATES[@]}" --output-dir "$OUT/fault"
```

Inspect completed safety/fault evidence before explicitly starting timing:

```bash
python3 -B "$D/run_guard_ab.py" benchmark "${COMMON[@]}" "${GATES[@]}" \
 --fault-report "$OUT/fault/report.json" --output-dir "$OUT/benchmark"
```

A/B order is baseline→candidate, candidate→baseline, baseline→candidate. Each
process uses exactly its own private normal OPP. Both receive fresh untimed full
prechecks before ANY formal sample. Then each of3 rounds has2 Poisons/prechecks,
5 warmups,30 Execute+Sync samples, Timed Tail and fresh Poison Postcheck. Inputs
are resident; no Host H2D/D2H/check/log/file IO in measured callbacks. Kernel
checks remain. Reject log changes, Workspace drift, output drift or any safety
failure. Candidate/baseline outputs must remain equal to the frozen Full Smoke
repeat anchor; it is not a replacement CPU Golden.

Reports retain90 samples per variant (180 total), Execute+Sync and Prepare
separately, median/P95/min/max, per-round ratios, actualWorkspace, identities and
all numeric differences. Device Kernel time is UNAVAILABLE without timestamps.
Signal: median speedup>=1.03 AND all three rounds>1 → `GO_FURTHER_REVIEW` only.
Otherwise `NO_GO_NO_MATERIAL_IMPROVEMENT`; do not keep rewriting this same Arena
expression. B-only ND→NZ packing remains a separate possible future experiment.

On failure return the phase report.json and failing Runner/compile log. Preserve
execution.json, resource-admission.json, all actual/witness/numeric files and
build source audits. Never reuse a failed output directory. No automatic repair,
weakened Guard, new Golden or benchmark admission after a failed Fault stage.

## Local checks

```bash
PYTHONPATH=framework/custom_ops/draft_quant/test python3 -B -m pytest -q -p no:cacheprovider "$D/test"
# Existing local Docker, fresh directory; real compiler, synthetic T1 only for build:
bash "$D/local_check.sh" /work/build/NEW_FULLK_ARENA_AB
```

Tests exhaust single bits mathematically and execute the actual candidate/fault
C++ source with the prior bounded CPU Matmul/buffer model (entry-name normalization
only). They test all source injection loops, exact witnesses, CPU numerical
mapping, allocation bounds and abort timing. They do not emulate real event or
VDEQ16 hardware behavior. Local CANN builds both real Kernel/Host/private API and
Runner packages, without installation, Formal query or NPU execution. Server
Host/Smoke/Fault/Benchmark remain NOT_RUN until real evidence is returned.

**Did Scalar Guard optimization actually lower Full-K Execute+Sync? UNRESOLVED.**
Only the new paired server samples can answer that question; no prediction is
made from Scalar Time or Cube Utilization.
