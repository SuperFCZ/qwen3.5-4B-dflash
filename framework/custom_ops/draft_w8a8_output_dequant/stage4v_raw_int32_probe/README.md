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

## Driver dependency identity bugfix (after279c306)

The original Host `ldd` check incorrectly treated every `/usr/local/Ascend/`
resolved path as a Toolkit file. `libascend_hal.so` is instead a separate Driver
runtime dependency at the confirmed root `/usr/local/Ascend/driver`. The fix
changes dependency validation only; Kernel, Host Tiling, workspace/UB/L1/L0,
Oracle, numerical rules and execution gates are unchanged.

`host/private/runtime-dependencies.json` now records separate `sdk_libraries`,
`driver_libraries`, `system_libraries` and `driver_binding`. The same categories
are in `report.json.artifacts` and the compiled Host binding. Toolkit paths and
resolved targets must stay inside the selected CANN_ROOT. Driver admission allows
only the requested name **and ELF SONAME** `libascend_hal.so`, with ELF64 LE
AArch64 ET_DYN validation; other Driver names/SONAMEs are BLOCKED. All libraries
have real-path/size/SHA256 identities. Every directory/file symlink is recorded;
Driver/Toolkit escapes and changed aliases are rejected, even if bytes match.
The confirmed Driver root itself cannot redirect elsewhere. System dependencies
are limited to `/lib`, `/lib64`, `/usr/lib`, `/usr/lib64` and their canonical
locations; unknown Ascend installations and arbitrary external paths remain
BLOCKED. A Toolkit copy of HAL is not accepted as a Driver identity.

Driver version inspection is limited to `version.info` and `driver/version.info`
under that root, with identity and parsed version fields. Missing/unparseable
version evidence is explicitly `UNRESOLVED`; no version is inferred from a path
or SONAME. The driver compares identities against the already-bound original
Preflight and frozen Stage4V `machine` evidence, including `{file, text}` version
wrappers. It does not scan server logs or follow arbitrary historical reports.
Historical version matches do **not** establish a historical HAL binary match.
Without such a binary identity, the report records
`UNRESOLVED_NO_HISTORICAL_DRIVER_BINARY_IDENTITY` and creates a new independent
`driver_binding`, never a CANN SDK or NPU acceptance claim. Any available
historical identity disagreement blocks admission. SHA256 and alias/version
identities are rechecked before formal Host query, at stage completion and when
Build Seal / Bounded validate prior artifacts.

Regression uses synthetic AArch64 ELF and filesystem fixtures on macOS and
Docker ARM64, without loading or executing a Driver. It covers Toolkit/Driver/OS
classification, HAL SONAME, file changes, file/directory symlink escape and
retargeting, unknown Ascend paths, missing version information, historical
machine version/binary comparison, and existing SDK-header/Host-library tamper
rejection. No calculation Kernel rebuild or NPU launch is needed for this fix.

For the server rerun, retain the commands above but use a **new** root, e.g.:

```bash
RAW=/home/w00949577/qwen35-dflash-run/w8a8-independent-raw-driverfix-$(date +%Y%m%d-%H%M%S)
bash "$D/run.sh" host --frozen-root "$FROZEN" --raw-host-report "$RAW_HOST" \
  --device 0 --output-dir "$RAW/host"
```

Do not reuse or edit `w8a8-independent-raw-int32-20261109-020324`. Target status
remains `FORMAL_HOST_FEASIBLE`; Build Seal and Bounded remain gated on that actual
server result. Return `host/report.json`, `host/private/host-ldd.log` and
`host/private/runtime-dependencies.json` (if created) if still BLOCKED.

Bugfix regression on2026-11-09: macOS39/39 and Docker ARM64 39/39 CPU tests
completed successfully, plus Python static checks. The earlier19-test local
compile record remains historical and unchanged. This bugfix performed no new
CANN compute build, Driver loading, formal Host query or NPU execution; server
`FORMAL_HOST_FEASIBLE` and Raw NPU result remain NOT_RUN pending the fresh rerun.
