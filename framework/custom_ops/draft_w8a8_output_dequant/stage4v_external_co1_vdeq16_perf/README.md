# Single-Tile VDEQ16 performance exploration — normal-path admission blocked

Baseline cc1ad00. This independent directory prioritizes the proposed single
M16/K2560/N256 experiment, with Scheme B diagnostics, original A/B/Scale, Raw
Echo and real VDEQ16. It does not modify Phase1 Echo, Stage4V, Raw Probe, Phase2.1,
Golden, OPP or any historical report.

**No executable experiment is delivered yet.** Current status is
`BLOCKED_NORMAL_PATH_EVENT_BUDGET`. The user explicitly requires stopping device
execution when normal-path event ownership/lifetime is not proved. Production
fault recovery and the six1-ULP cause are deferred; they are NOT the reason this
normal-path gate stays closed. Host/Build/Smoke/Benchmark are NOT_RUN. There is
no Kernel, private OPP or ACLNN Runner in this directory. The CLI cannot turn
source/CPU/syntax checks into a device admission.

## What this round established

See NORMAL_PATH_REVIEW.md for exact source identities/ranges and the remaining
finite gap. The normal input-queue End path has actual WaitFlag calls through
`FreeAllEvent`, so the earlier lack of evidence is narrowed. It is not just an
unconditional clear. The L0 ping/pong implementation mixes allocated event IDs
with fixed0/1 use in its base class: application reservation must not displace
those SDK IDs. GM ND→L1 queue events differ from GM→UB events on2002.

The SDK input ND→NZ helper chain also uses transient Fetch on pools shared with
proposed application fences. We have not completed its selected-call-graph
per-HardEvent peak and reuse ledger, including helpers called by the conversion
wrapper. Pre-reserving app IDs without that proof is not admitted. Nor is late
Alloc followed by early return while work is pending. This is a basic normal
execution requirement, not a production-grade recovery feature.

`compile_check/interfaces.cpp` checks actual CANN public signatures for External
CO1 Iterate, CO1 allocation, direct INT32→half VDEQ16 and the relevant queue-event
constants. It has no `__global__` entry or `main`, and is compiled with
`-fsyntax-only`. Success proves accepted interfaces/constants only; no device
object, executable or OPP is created, installed or run.

## Minimum safe implementation delta, after closing the normal event ledger

* Keep Scheme B's independent diagnostic payload: Arena512 + Witness256,
  prefix/suffix256 each, total1280. Existing UB end230912 becomes232192;
  Reserve8192 remains untouched, total240384. External CO116384, Scratch202240
  and mathematical inputs remain fixed. Actual TPipe addresses/new Formal Host
  still need verification; old239104B Host PASS is not reused as new admission.
* Allocate only the proven-safe application completion tickets before user
  asynchronous work. Do not allocate SDK-owned M_MTE1/MTE1_M IDs. No app S_V in
  the sixteen-conversion window. A valid free SDK S_V slot is mandatory.
* Preserve all existing barriers and End. Observe M_V→V_S before consuming CO1,
  MTE2_V before Scale reads, and V_S after each conversion. Retain V_M plus the
  M_V→V_S reverse-completion observation before End. Close GM outputs with
  MTE3_S. Release events only after their last observed completion.
* For ordinary data-check failure, record failure and converge on an already
  provisioned completed terminal path. Do not replace that path with return,
  End or PIPE_ALL. Hardware timeout is a failed run with no trusted success
  Witness; comprehensive recovery may remain outside this experiment.

This is a proposed delta, not implemented device code or a granted launch.

## Prepared offline Smoke contract

`perf_contract.py` provides CPU-only postprocessing for two supplied complete
outputs. Raw must match all4096 INT32 values and Repeat exactly. FP16 must be
finite and bitwise Repeat-consistent, with full coverage; NaN Poisons/nonfinite
outputs reject for this frozen finite case. All specified Guard, readonly,
Workspace, Witness, sync and no-fallback prerequisites remain mandatory.

All FP16 differences against BOTH original actual and original CPU reference are
listed, with bits, ULP and max absolute error, and the six coordinates highlighted.
Historical output extraction is `reshape(16,19456)[:,:256]`, not a flat4096-word
prefix. One-ULP mismatches are not tolerated away: they remain mismatches and
`numeric_cpu_bitwise=false`. The user's exploration policy permits reporting
latency after basic device safety/integrity checks despite those mismatches; it
does not create numerical acceptance or a new Golden. This offline function
cannot prove that real Runner/OPP/SDK evidence exists or open the execution gate.

## Prepared timing protocol (not measured)

Three rounds, each in an isolated process, warmup5 and30 measured Execute+Sync
samples. Two untimed Poison prechecks, Timed Tail and separate Poison Postcheck
must pass every round. Inputs resident. Upload, D2H, correctness checks and file
IO stay outside the timed interval. Reject timed Host logging, instrumentation,
Workspace drift, missing identities or any safety failure. Keep90 raw samples.
Prepare has its own sample series; never add it silently to Execute+Sync.

`benchmark_summary()` implements median, nearest-rank P95, min/max and per-round
statistics on synthetic/test data only. It cannot produce a Benchmark PASS.
Device task time is `UNAVAILABLE` until separately measured; Execute+Sync must
not be relabeled device time. No Profile timing is mixed into the samples.

The proposed measured scope includes Full-K Matmul, Raw Echo, VDEQ16, two
numerical GM outputs, five UB Guard pairs640 uint32 words per check, Arena and
Witness handling, and per-conversion completion joins. Report these overheads;
do not subtract guessed costs to call the result pure VDEQ throughput.

## Baseline comparability

| Available history | Use / limitation |
|---|---|
| Phase1 External CO1 Echo, same Frozen A/B | Input/Raw proof; old Smoke timings are not benchmark samples. New paired receiver would be needed; extra FP16/diagnostics make the difference a net implementation delta, not pure instruction cost |
| Stage4V original Full-K FP16 Tile0 | Actual/CPU comparison; same frozen data can support a future paired workload, but original numerical BLOCKED state remains and its different CType/output protocol must be disclosed |
| Stage4Q/P grouped N256 or Stage4R FP16 full layer | N19456 and/or different weight/Scale arithmetic; not a same-M/K/N paired denominator |

No suitable new paired latency samples exist. Do not use the historical2.77ms
full-layer W8A8 or0.787ms FP16 as single-Tile denominators. Never multiply a
single-Tile result by76 to claim layer performance. N19456 planning follows only
measured single-Tile exploration and a separate scheduling/resource design.

## Commands that are actually available

From repository root, fresh paths only:

```bash
D=framework/custom_ops/draft_w8a8_output_dequant/stage4v_external_co1_vdeq16_perf
PYTHONPATH=framework/custom_ops/draft_quant/test python3 -B -m pytest -q -p no:cacheprovider "$D/test"
# Existing local Docker: syntax check only, not device compilation/OPP build:
bash "$D/compile_check/check.sh" /work/build/NEW_CO1_PERF_INTERFACE_CHECK
```

Optional server read-only identity review (no new source excerpts required):

```bash
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
D=framework/custom_ops/draft_w8a8_output_dequant/stage4v_external_co1_vdeq16_perf
OUT=/home/w00949577/qwen35-dflash-run/w8a8-co1-perf-admission-$(date +%Y%m%d-%H%M%S)
python3 -B "$D/run_perf.py" review --cann-root "$CANN_ROOT" --output-dir "$OUT" \
 --echo-root /home/w00949577/qwen35-dflash-run/w8a8-external-co1-echo-hostfix-20261109-102407 \
 --co1-root /home/w00949577/qwen35-dflash-run/w8a8-co1-nestedfix-20261109-074724 \
 --raw-root /home/w00949577/qwen35-dflash-run/w8a8-independent-raw-int32-driverfix-20261109-022500 \
 --frozen-root /home/w00949577/qwen35-dflash-run/w8a8-full-k-l1fix-20261106-073850 \
 --review-root /home/w00949577/qwen35-dflash-run/w8a8-vdeq16-review-20261109-031709
```

Exit2 is expected: review may be complete, but execution remains BLOCKED. Exit1
means evidence checking failed. Return only `report.json` for review failures,
or `compile.log` and `status.txt` for syntax failures. Do not rerun old experiments.
`host`, `build`, `smoke`, `benchmark` are explicit fail-closed CLI gates; they
write a BLOCKED report without invoking SDK commands. There are deliberately no
runnable Host/Build/Smoke/Benchmark commands to present as a finished experiment.

Current unresolved numerical questions remain six original1-ULP differences and
hidden VDEQ precision. They do not block future exploratory timing once basic
normal-path safety and actual Smoke pass. Local CPU/syntax successes prove neither.
