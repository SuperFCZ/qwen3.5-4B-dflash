# Stage4P: independent N256/BaseN256 correctness

No Benchmark/Profile entry point. Stage4B–4O source, OPPs, SDK and Golden are
read-only. New CANN operator compilation and device results are NOT_RUN at
development delivery; CPU doubles do not establish MMAD/VDEQ16 safety.

## Source and input contract

The normal Kernel is an exact dimensional derivation of Stage4N's accepted
N128 XOR/OR source, reconstructed from its unchanged audited transformation.
Prepare additionally binds the actual generated Stage4N XOR Kernel file and
requires byte equality with that derivation baseline. The source diff is
against the XOR baseline, not a weaker or original short-circuit Guard Kernel.

A INT8[20,16,128] is copied from the saved full-X quantization. B becomes
INT8[76,20,128,256]; Scale becomes UINT64[76,20,256]. Output stays FP16
[16,19456]. Every one of1520 groups is checked against original frozen NK/NZ/GN
and both corresponding N128 packings. Whole-file B/Scale hashes must equal the
accepted Stage4O layout digests. Independent INT64 dot references verify every
FP16 group partial against the accepted N128 halves. Ordered FP32 summation
must equal the original Stage4B final Golden, whose original file identity is
retained: no replacement final Golden is written.

Local NZ has16 N16 blocks: `(n//16)*256 + row*16 + n%16`. GM uses
`row*19456 + tile*256 + n`. Gate/Up boundary is Tile38; final Tile75 ends at
channel19455. B uses32768B per group; Scale uses256 uint64 words per group.

Every group retains all640 uint32 GetValue checks through the same XOR/OR
aggregation, then one failure decision before advancing. All20 independent
K128 Cube/VDEQ16/FP16 partials, casts and ordered FP32 additions remain.
Iterate/GetTensorC/End, per-group SetLocalWorkspace, input/Scale rebinding,
all original barriers/events, Tile-end Guard and Arena protocol are retained.
No prefetch, vector bitmap, uint64 Guard, FMA, K merging or alternate BaseN.

## Resource admission

Per Core: Scratch202240B + two FP16 buffers8192B each + FP32 temporary16384B
+ FP32 Sum16384B + Guards2560B + Reserve8192B =262144B. Explicit allocations
consume253952B; the reserve is not allocated. Buffers are reused across groups.

Prepare binds the actual Stage4O reports, current SDK source/library identities
and the live historical Stage4N freeze. Build generates an exact Tiling
contract header from that accepted report. The new formal Op Host queries
TilingContext and rejects ANY mismatch in recorded Tiling fields, capacities,
Scratch or system Workspace. In particular BaseN256, transLength34816,
shareL1Size34816, shareL0CSize16384, raw shareUbSize2048 and dbL0B2 are bound.
Effective shared UB reservation remains zero under the identity-bound2002
consumer contract. Coefficient scratch range is[34816,36864).

The conservative L0B requirement65536B must equal the accepted actual capacity;
there is no assumed spare bank or extra L0B allocation. No resource fallback.
The formal Core tuple must equal Stage4N's accepted TilingContext tuple.
Standalone Manager8 is not scheduling permission: formal selected7 is checked
before execution, and auto/2-Core runs are separate. Arena and system Workspace
address/size formulas are unchanged.

## Separate phases

- Prepare: complete frozen layout/reference/hash checks and source audit.
- Build: new DFlashW8A8N256Gate private OPP and Runner, plus an isolated fault
  OPP/Runner. Both use new directories. Generated source/Host contract, ABI,
  installed ELF/metadata and libraries are bound; normal/fault binaries must
  have distinct identities. `fault-only.diff` records diagnostic additions.
- Host-only: actual new Op Host Tiling/workspace query; no Kernel submission.
- Bounded: Tiles0,37,38,75 individually; modes1/20/0 for Group0, Group19 and
  final output. Every mode executes all20 groups. Two poisoned repeats;
  unselected columns must remain the corresponding Poison and are not claimed
  as numerical PASS.
- Full: auto and2-Core, scheduling mode21 and numerical mode0, each twice.
  All311296 elements must match the unchanged Golden, with all remaining safety
  gates. Full numerical success alone is NOT sufficient for Stage4Q admission.
- Fault: requires the accepted Full report from this exact Build/input/source.
  A separate instrumented OPP uses Tile0/core1. Mode0 performs640 words×32 bits
  at each of20 actual group boundaries (409600 injections per sweep), restores
  Guards, and must reach the correct bounded final output. Two repeats give
 819200 injections. Modes1..20 verify exact target-group abort, including the
  last group, using a32B witness and unchanged Poison everywhere else.

Fault instrumentation retains the candidate's GroupDelta implementation with
N256-sized offsets. Its additional Scalar writes, S→MTE3 witness event and GM
witness write exist ONLY in the diagnostic OPP. It cannot be timed or adopted
as the normal candidate. A wrong/absent witness, unrelated early return,
missing case, timeout, resource failure or mismatch blocks completion. No
automatic timeout increase, tolerance change or skipped case.

## Server commands

Use fresh directories; execute each next block only after the stated status.
Defaults bind the accepted Stage4O
`w8a8-n256-host-20261102-081401` and Stage4N
`w8a8-n128-xor-guard-20261102-063409` under the existing run root.

```bash
set -euo pipefail
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
S4P=/home/w00949577/qwen35-dflash-run/w8a8-n256-correctness-$(date +%Y%m%d-%H%M%S)
ENTRY=framework/custom_ops/draft_w8a8_output_dequant/stage4p/run.sh
bash "$ENTRY" prepare --device 0 --output-dir "$S4P/prepare"
```

After STAGE4P_PREPARED_NPU_NOT_RUN:

```bash
bash "$ENTRY" build --device 0 \
 --prepare-report "$S4P/prepare/report.json" --output-dir "$S4P/build"
```

After STAGE4P_BUILT_NPU_NOT_RUN:

```bash
bash "$ENTRY" host-only --device 0 \
 --build-report "$S4P/build/report.json" --output-dir "$S4P/host-only"
```

After STAGE4P_HOST_TILING_ONLY:

```bash
bash "$ENTRY" bounded --device 0 \
 --build-report "$S4P/build/report.json" \
 --host-report "$S4P/host-only/report.json" --output-dir "$S4P/bounded"
```

After STAGE4P_BOUNDED_MATCH:

```bash
bash "$ENTRY" full --device 0 \
 --build-report "$S4P/build/report.json" \
 --host-report "$S4P/host-only/report.json" \
 --bounded-report "$S4P/bounded/report.json" --output-dir "$S4P/full"
```

Only after STAGE4P_FULL_GATE_MATCH:

```bash
bash "$ENTRY" faults --device 0 \
 --build-report "$S4P/build/report.json" \
 --full-report "$S4P/full/report.json" --output-dir "$S4P/faults"
```

Final fault gate: STAGE4P_DEVICE_FAULT_COVERAGE_MATCH. It binds the preceding
Full report; Stage4Q is still NOT_RUN and requires a separate experiment.
Historical4.299590ms is never a new paired denominator. FP16 historical review
remains separate/unresolved and is neither repaired nor required here.

Return every phase's report.json. Preserve source-audit.json/source.diff,
build/fault-only.diff, generated t1_contract.h, compiler/install/Runner logs,
and failed execute/limit-*/tile-*/mode-* or faults/mode-* folders containing
runner.log, execution.json, resource-admission.json and actual-*.bin. Numerical
failures report first row/global-channel/Tile, bits and bound partial/Scale
reference values; a final-only mismatch cannot identify an actual failing
device group and remains UNRESOLVED_FINAL_SUM until diagnostic evidence exists.
Never upload SDK originals or server binaries/logs to public GitHub.

CPU regression:

```bash
PYTHONPATH=framework/custom_ops/draft_quant/test python3 -B -m pytest -q \
 framework/custom_ops/draft_w8a8_output_dequant/stage4p/test
```

CPU tests check full1520-group preparation, original Golden identity, all76
Tile indices, N16 blocks, lower/upper-half comparison coverage, inactive Poison,
source operation invariance, UB/Arena bounds, Matmul pointer/lifetime reuse and
all20 fault abort witnesses. Actual transformed C++ runs with cached-partial
Matmul doubles, including the expanded-buffer exhaustive sweep. These are not
CANN compilation or NPU numerical/safety results.
