# Independent Full-K Raw INT32 Tile0 probe

Purpose: obtain an independent `INT8 A[16,2560] @ INT8 B[2560,256] -> INT32
C[16,256]` device result, to narrow the six Stage4V FP16 discrepancies. It does
not change Stage4V's Golden, tolerance, failed report or benchmark admission.
Unique operator: `DFlashFullKInt32Probe`. No Scale input, bias, DequantType,
SetQuantVector, FP16 conversion, performance mode or device fault mode exists.

## Gates and evidence

`host -> build -> bounded` are separate commands, each with a new output directory.
A formal TilingContext exists only after the private Op Host/API is built. Thus
**host first compiles/packages the independent OPP, installs it only under that
new experiment's `host/private/opp`, links the independent Runner, and calls
GetWorkspaceSize only**. It never calls the execute API. `build` then revalidates
and seals these exact artifacts; it does not rebuild, reinstall or query the
historical operators. `bounded` requires both successful reports and is the only
command allowed to execute the compute Kernel.

| Gate | Exact success status | Meaning |
|---|---|---|
| Source admission (inside host) | `source_review=SOURCE_REVIEWED` | Audited selected SDK identities match the accepted Raw Host and current files; independent Raw lifecycle/layout reviewed |
| host | `FORMAL_HOST_FEASIBLE` | Private build completed, exact real TilingContext/ABI/Core/Workspace query accepted; NPU NOT_RUN |
| build | `BUILT_NPU_NOT_RUN` | Immutable formal Host, generated source/ABI, private OPP/Kernel ELF, Runner and SDK evidence rechecked |
| bounded | `RAW_INT32_NPU_MATCH` | Both untimed device executions match all4096 INT32 values and safety checks |

Any error produces `BLOCKED` and a nonzero exit. A compiled Kernel or standalone
Manager query never establishes formal Host or NPU correctness. No stage edits a
predecessor report. No fallback, auto-correction of Tiling, or tolerance exists.

The driver rereads the accepted Raw report and original frozen failed experiment,
verifies their source/SDK/dependency/library/file identities, and regenerates an
**independent INT64 dot from the same frozen INT8 input bytes**. A is copied byte
for byte; B is exactly bytes `[0,655360)` of the frozen Stage4V packed B. It also
checks the standalone report's B slice SHA256 and the existing independent dot.
There is no re-quantization or new weight reconstruction. Each stage saves its
own A/B, INT64 Oracle and range-checked INT32 Oracle with identities.

All27 `TCubeTiling` fields from the accepted Raw report, all five capacities,
system Workspace and scratch are embedded into generated `t1_contract.h` and
compared again by the actual Op Host. This includes usedCoreNum=1, transLength=
174080, depthA1/B1=20/10, stepKa/Kb=20/5, base16/256/128 and fullK2560. There is no
BaseN or scratch fallback. The formal scheduling policy is read from the frozen
formal evidence (currently8/8/7/7, arch2002), not copied from the Manager's8 Core
result. Available Core count, launchBlocks=1 and usedCoreNum=1 are reported
separately. The actual ACLNN Workspace request is bound by the Host report and
must be identical before either device execution.

SDK sources/line numbers/hashes are reused from
`stage4v_diagnostics/sdk-review.json` and the accepted Raw report. No SDK source
snapshot is copied here. The server command checks that the selected SDK files
match those identities; local matching alone is not server runtime validation.
The generated Op Host exact object and dependency lists bind the compiled
platform header to the reviewed file, including alias resolution and SHA256.
Missing, changed, escaped or conflicting dependencies block admission.

## Resource and dataflow review

| Resource | Bytes / contract |
|---|---|
| User Scratch | 202240, explicit nonoverlapping view |
| INT32 NZ Partial | 16384 (4096 elements) |
| Arena-check temporary | 512 |
| Three UB prefix/suffix pairs | 1536 (256B on each side of each buffer) |
| Actual explicit UB allocation | 220672 |
| Untouched Reserve | 8192 |
| Accounted UB | 228864 <= actual262144 |
| L1 | 2048×20 +32768×10 =368640; equals reported shareL1Size, not added twice |
| L0A/L0B | 2048×2 /32768×2; L0B fills65536, no spare allocation |
| L0C | conservatively16384×dbL0C +shareL0CSize, bound by actual capacity |
| SDK Raw scratch | transLength174080 <=202240; no Scale view; sharedUB0 |
| GM arena | platform-sized arena, 512B prefix/suffix, plus unchanged framework system reservation |

The Normal Matmul scheduler remains compile-time required on dav_2002. Kernel
sequence: initialize and verify guards/arena, bind scratch/A/B, `Iterate(false)`,
`GetTensorC(partial,0,true)`, PIPE_ALL, End, PIPE_ALL, verify all UB guards and
arena, then output. No float operation or quant processor configuration is added.
V_S protects scalar UB guard reads; MTE2_S protects arena readbacks. The existing
registration pointer convention `workspace+512` is retained for the single arena.

Writeback consists of256 aligned64B copies (16 INT32 each). Source index
`nb*256 + m*16` and destination `m*256 + nb*16` cover4096 elements exactly once.
Both offsets are32B aligned; no strided/padded-copy optimization is introduced.
All computation and guard checks finish before the first output write. The
Runner guards A/B/output/full Workspace, verifies A/B readonly, pre-fills and
reads back two distinct Poison patterns (`0x7f7f7f7f` and `0x80808080`, both outside the complete mathematical dot range), synchronizes execution, saves actual
bytes **before** checking numerical/safety results, and demands exact Repeat.
A failure stops immediately; a first-run failure does not launch the second run.

`integer-comparison.json` contains every repeat's bitwise match count, mismatch
count, first coordinate/value, max absolute integer delta, delta histogram,
CPU/NPU sign/parity distributions and these six coordinates:
`(0,84), (1,53), (1,201), (5,138), (7,235), (15,210)`.
One available failed output is analyzed, but cannot count as a two-repeat PASS.
Raw output identities are in `report.json.outputs`. GM output is compact16384B;
external guards delimit Tile0, so no other output tile is allocated or written.

## Local checks

```bash
D=framework/custom_ops/draft_w8a8_output_dequant/stage4v_raw_int32_probe
PYTHONPATH=framework/custom_ops/draft_quant/test python3 -B -m pytest -q -p no:cacheprovider "$D/test"
bash "$D/local_check/check.sh" /work/build/NEW_RAW_LOCAL_CHECK
bash "$D/local_check/compile_opp.sh" /work/build/NEW_RAW_LOCAL_CHECK /work/build/NEW_RAW_LOCAL_OPP
```

CPU tests execute the actual Kernel source against bounded CPU test doubles,
check all4096 integer outputs and unique writes, full-K lifecycle, UB/Arena
bounds, input readonly, injected guard corruption and fail-before-output paths.
They do not emulate hardware events or prove NPU safety. The local scripts use
synthetic Tiling fixtures strictly for compilation; neither executes the Host
library nor installs/launches the OPP. The real generated API Runner links to
CPack's local private API without installation. Local libraries remain
`LOCAL_ONLY_NOT_SERVER_VERIFIED`. See `local-validation.json` for recorded scope.

## Server commands (each stage must pass before the next)

Run from the repository root after reviewing/pulling this commit. Keep the same
`RAW` variable for all three commands; each subdirectory must not already exist.

```bash
set -euo pipefail
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
D=framework/custom_ops/draft_w8a8_output_dequant/stage4v_raw_int32_probe
RAW=/home/w00949577/qwen35-dflash-run/w8a8-independent-raw-int32-$(date +%Y%m%d-%H%M%S)
FROZEN=/home/w00949577/qwen35-dflash-run/w8a8-full-k-l1fix-20261106-073850
RAW_HOST=/home/w00949577/qwen35-dflash-run/w8a8-raw-host-importfix-20261109-011705/raw-host-report.json

# Private OPP compile/install + formal GetWorkspaceSize only; no execute.
bash "$D/run.sh" host --frozen-root "$FROZEN" --raw-host-report "$RAW_HOST" \
  --device 0 --output-dir "$RAW/host"
python3 - "$RAW/host/report.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));assert r['status']=='FORMAL_HOST_FEASIBLE' and r['npu']=='NOT_RUN',r
print(r['formal_plan'],r['workspace_bytes'])
PY
```

```bash
# Revalidate and seal the exact private artifacts. Still no execute.
bash "$D/run.sh" build --frozen-root "$FROZEN" --raw-host-report "$RAW_HOST" \
  --host-report "$RAW/host/report.json" --device 0 --output-dir "$RAW/build"
python3 - "$RAW/build/report.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));assert r['status']=='BUILT_NPU_NOT_RUN' and r['npu']=='NOT_RUN',r
PY
```

```bash
# First compute execution: two Poison/Readonly/Guard/Repeat INT32 checks.
bash "$D/run.sh" bounded --frozen-root "$FROZEN" --raw-host-report "$RAW_HOST" \
  --host-report "$RAW/host/report.json" --build-report "$RAW/build/report.json" \
  --device 0 --output-dir "$RAW/bounded"
python3 - "$RAW/bounded/report.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));print(r['status']);print(r.get('integer_comparison'))
assert r['status']=='RAW_INT32_NPU_MATCH',r
PY
```

Do not proceed after a failed command. Return the failed stage's `report.json`
and its failing log. Host compile failures: `host/private/msopgen.log`,
`build.log`, `install.log`, `cmake.log`, `runner-build.log` or `ldd.log` as
applicable. Host/query/runtime failures: `execution/runner.log`, `execution.json`,
`resource-admission.json`, and any `actual-*.bin` identities plus
`integer-comparison.json`. Preserve all original bytes locally even when only
small report excerpts are sent back. No server logs or SDK excerpts belong in Git.

## Interpretation boundary

An all-element Raw match proves only this independent Raw CType/output path on
these frozen inputs. It does **not** prove that the original VDEQ16 FP16 Kernel
has identical internal INT32 values. A Raw mismatch requires input, layout,
Tiling, lifetime and writeback investigation before any claim about MMAD hardware.
The six FP16 differences remain UNRESOLVED. Original FP16 Golden/tolerance and
Stage4V Benchmark admission remain unchanged. Benchmark, Profile and device Fault
are NOT_RUN and unavailable in this experiment.
