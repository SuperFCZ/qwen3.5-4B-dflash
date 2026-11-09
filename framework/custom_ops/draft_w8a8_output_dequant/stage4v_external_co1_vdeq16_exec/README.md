# External CO1 execution diagnosis — Phase1 Echo only

This delivery implements the complete **ECHO_ONLY** path with a new operator
`DFlashExternalCo1EchoProbe` and a separate private OPP. It obeys the explicit
requirement that VDEQ16 may be developed/executed only **after real Echo PASS**.
Thus no VDEQ16 conversion branch, SetQuantVector or replacement FP32 multiplication
is included. `vdeq` is a closed gate, not a runnable conversion. After server
`ECHO_NPU_MATCH`, a separate Phase2 implementation/review must be developed without
editing this accepted Echo evidence. This is not a claim that both execution
phases have already been implemented or that VDEQ16 is unsupported.

All historical Stage4V, Raw Probe, Formal CO1 Host, Golden and reports are read
only. No Benchmark/Profile/production integration exists. Local compile or CPU
models never establish NPU correctness.

## Evidence and actual Host admission

`source` requires the completed Formal CO1 report from
`w8a8-co1-nestedfix-20261109-074724/host/formal-host-report.json`, its successful
source/build stages, exact private Host/API/placeholder/SDK identities, complete
parent serialization and200B slice,50 getters, schema SHA and Workspace4195328.
The existing nested-serialization verifier is reused read-only for that history.
Raw NPU evidence from `w8a8-independent-raw-int32-driverfix-20261109-022500` must
have two4096-element exact matches, correct safety/witness metadata and unchanged
A/B and independent INT64 Oracle. The original frozen FP16 experiment is also
verified, including correct Tile0 extraction as `full[0:16,0:256]` from the
16×19456 output, **not** its first4096 contiguous elements.

Each phase copies exact frozen Raw A[16,2560], Tile0 B[2560,256], original256
UINT64 words, and the historical Raw NPU reference. `expected.bin` is a bytewise
copy of that accepted reference, already cross-checked against INT64; it is not
a new Golden. File sizes and SHA256 remain bound.

The new registered Host is separately compiled and queried. It retains the
existing M16/N256/K2560, Base16/256/128, CO1 NZ INT32, no-Bias/no-dequant Matmul
configuration. Its generated contract compares **all50 actual fields**, capacities,
Core policy and system Workspace against the accepted formal report. Before
GetWorkspaceSize returns, C++ also compares all actual serialized fields and the
outer48B against getters/configuration. Python independently validates the captured
parent/slice and identities. No standalone values populate the callback's result.
The additional GM witness output is UINT64[32]; actual returned Workspace must
still equal4195328. Any changed plan/request blocks without fallback.

## SDK path and lifetime

Source identities for Matmul, scheduler, external buffer, events, TPipe and
L0C DataCopy are inherited from the bound CO1/source audit. Two additional exact
files are checked at the new source gate:

* `aarch64-linux/asc/impl/basic_api/kernel_tensor_impl.h`, SHA256
  `1e5d6669d91e5687db76fcace1cb1b2a9af0dfde92c4b5ee65dd2b23b99b9e0a`,
  lines646–668 and864–866: device physical address and logical position.
* `aarch64-linux/asc/impl/basic_api/kernel_tbuf_impl.h`, SHA256
  `e838500d23ab3c5269fbcc41c64de1fb674782657727a4a0dbb4d1bf8706abf0`,
  lines20–65 and68–81: TBuf view stores the logical position, allocated address
  and byte extent. Local identities are not silently called server-verified.

Kernel allocates `TBuf<CO1>`16384B **first**, before registration. Actual address0,
1024B alignment,4096-element extent and CO1 logical position are checked. Normal
Full-K Matmul is compile-time required on coupled dav_2002. The caller tensor is
bound by `Iterate(false,co1)`. GetTensorC is not used because the reviewed CO1
GetTensorC branch returns without copying; Echo instead uses public enhanced
DataCopy with matrix mode, no dequantization and no ReLU, one16×16 INT32 block
at a time. Source and destination extents are1024B per block.

After MMAD issuance, explicit allocated M_V SetFlag/WaitFlag completes Cube→Vector
before Echo reads L0C. PIPE_V and V_S complete the consumer/Guard read, V_M closes
the reverse dependency before End, and existing PIPE_ALL barriers are retained.
End remains after Echo consumption, not a replacement for events. TPipe and the
caller buffer live until Kernel return. No next Matmul or L0C overwrite exists.
The public event allocator is used with range checks and matched wait/release;
`QUE_MAX_EVENT==8` is compile-time asserted against the actual SDK. Actual IDs
and visit counts are written to the diagnostic witness.

This independent path's successful Echo would not itself establish the original
FP16 Kernel's internal L0C or VDEQ16 rounding behavior. It is the prerequisite for
an isolated consumer experiment. The original six1-ULP issue remains UNRESOLVED.

## Resources, aliases and coverage

| UB allocation | Payload | Prefix + suffix | Actual start |
|---|---:|---:|---:|
| Scratch |202240|512|0|
| Echo |16384|512|202752|
| Scale |2048|512|219648|
| Reserved FP16 |8192|512|222208|

Explicit user UB ends at230912; adding untouched8192 Reserve gives239104.
Every user buffer's actual physical start is checked. Witness field5 records the
**explicit user-buffer end**, not an unread private SDK pool high-water counter.
`TPipe::GetQueueEndAddress` is private in this SDK and is not used or bypassed.
Hidden SDK allocation assumptions remain governed by the exact admitted Tiling
and reviewed zero shared-UB/external-output consumers; this is not a claim to
have inspected an inaccessible allocator field at runtime.

The FP16 payload is unused numerically in ECHO_ONLY. It contains two disjoint
phase1 diagnostic views: arena temporary `[0,512)` and witness `[512,768)`.
Neither uses Reserve or extra UB. These aliases cannot silently survive into a
future VDEQ16 consumer; Phase2 requires new lifetime review.

All four UB prefix/suffix pairs (512 uint32 words) use exact XOR/OR checking.
GM input/output/witness/whole-workspace guards, guarded Matmul arena, readonly
A/B/Scale, two distinct read-back Poisons, strict Raw reference and Repeat remain
mandatory. Scale is copied unchanged into its guarded UB view and never decoded
or used arithmetically in this phase. Arena checking uses only the phase1 dead
FP16 view, so it cannot overwrite Echo. Witness updates are placed before S_MTE3
publication fencing. Guard failures after an active Matmul synchronize and call
End before aborting, with failure witness where storage/event safety permits.

**L0C coverage is range/alignment/typed-allocation proof plus full Echo, not
Canary protection.** No unsupported UB→L0C injection is used to initialize L0C
canaries. The report permanently labels `RANGE_ONLY_NO_CANARIES`; a matching Echo
must not be described as complete L0C out-of-range fault coverage.

Echo ND writeback is256 aligned64B transfers, source `nb*256+m*16`, destination
`m*256+nb*16`. CPU tests check all4096 addresses exactly once. No FP16 GM output
or conversion is present.

## Stages and commands

| Phase | Success |
|---|---|
| source | SOURCE_REVIEWED |
| cpu | CPU_TESTS_PASSED |
| build | PRIVATE_OPP_BUILT |
| host | FORMAL_HOST_FEASIBLE (GetWorkspaceSize only) |
| echo | ECHO_NPU_MATCH |
| vdeq | BLOCKED: development/execution awaits Echo PASS and subsequent implementation |

Every phase creates a fresh directory and checks predecessor source/evidence
hashes. Failed reports/output/logs remain. No automatic repair, changed reference
or tolerance exists. The source/build stages bind exact OPP/API/ELF/Runner and
SDK/runtime identities. The private receiver uses a separate process and OPP.
C++ GetWorkspaceSize admission rejects a different50-field/Workspace contract
before every execute call; Python revalidates all captured callback evidence.

```bash
set -euo pipefail
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
D=framework/custom_ops/draft_w8a8_output_dequant/stage4v_external_co1_vdeq16_exec
OUT=/home/w00949577/qwen35-dflash-run/w8a8-external-co1-echo-$(date +%Y%m%d-%H%M%S)
CO1=/home/w00949577/qwen35-dflash-run/w8a8-co1-nestedfix-20261109-074724
RAW=/home/w00949577/qwen35-dflash-run/w8a8-independent-raw-int32-driverfix-20261109-022500
FROZEN=/home/w00949577/qwen35-dflash-run/w8a8-full-k-l1fix-20261106-073850
COMMON=(--co1-root "$CO1" --raw-root "$RAW" --frozen-root "$FROZEN" --device 0)

bash "$D/run.sh" source "${COMMON[@]}" --output-dir "$OUT/source"
bash "$D/run.sh" cpu "${COMMON[@]}" --source-report "$OUT/source/report.json" --output-dir "$OUT/cpu"
bash "$D/run.sh" build "${COMMON[@]}" --source-report "$OUT/source/report.json" \
 --cpu-report "$OUT/cpu/report.json" --output-dir "$OUT/build"
bash "$D/run.sh" host "${COMMON[@]}" --source-report "$OUT/source/report.json" \
 --cpu-report "$OUT/cpu/report.json" --build-report "$OUT/build/report.json" --output-dir "$OUT/host"
# Stop and inspect the successful Formal Host before explicitly issuing Echo:
bash "$D/run.sh" echo "${COMMON[@]}" --source-report "$OUT/source/report.json" \
 --cpu-report "$OUT/cpu/report.json" --build-report "$OUT/build/report.json" \
 --host-report "$OUT/host/report.json" --output-dir "$OUT/echo"
```

No VDEQ16 command is provided to run a Kernel. Even a future Echo report cannot
turn an absent implementation into VDEQ16_NPU_MATCH. Submit the Echo result first.

## Reports and failure diagnostics

Return the failed/current phase `report.json`; build errors additionally need the
failing log under `build/private`. Query/runtime errors need `run/runner.log`,
`execution.json`, actual callback JSON and retained parent/payload binaries.
Echo saves `actual-0/1.bin` and `witness-0/1.bin` before numerical/safety rejection
when device synchronization permits readback. `echo-comparison.json` covers every
integer, first mismatch, max absolute delta, histogram, sign/parity and six key
coordinates. A first failure stops repeat2. An unavailable readback is not filled
with synthetic device data.

Witness32 uint64 words: magic/version/error at0–2, L0C address/bytes at3–4,
explicit user UB end at5, guard-complete flag6, L0C-canary flag7(always0), event
IDs8–15, matched-fence visit counts16–23, reserved24–31. The event order is M_V,
V_M,V_S,MTE2_V,V_MTE3,MTE3_MTE2,MTE2_S,S_MTE3. Error10 is user-buffer range,11
UB/arena checks,12 Scale fence,13 Iterate,14 M_V,15 V_M,16 post-End address,
17 publication fence,18 witness fence. Missing magic means failure before a
valid witness or damaged witness, not a successful allocation. All successful
records require valid IDs/counts and exact integer output; no witness substitutes
for numerical validation.

Local regression executes the actual Kernel source through a bounded CPU test
double, including integer dot, NZ→ND, allocation/view bounds, event sequencing,
input readonly, guard/arena failures and numerical corruption. It also flips all
512 Guard words'32 bits (16384 CPU injections). These are not hardware fault
injection results. Real CANN compilation uses a clearly marked local Formal
contract fixture only for compilation, never device admission. See
`local-validation.json`. Server Host/Echo, VDEQ16, Benchmark and Profile remain
NOT_RUN until the corresponding real authorized test occurs.

## Host execution-status contract fix (after0257084)

The initial C++ Host branch emitted `HOST_ONLY`, while both Python Host gates
required `HOST_TILING_ONLY`. This deterministically rejected an otherwise
successful GetWorkspaceSize. `HOST_TILING_ONLY` is now the only accepted spelling;
old reports are not rewritten or admitted through an alias.

The exact Host `execution.json` contains four fields:

| Field | Required value/type |
|---|---|
| status | string `HOST_TILING_ONLY` |
| npu | string `NOT_RUN` |
| workspace_bytes | integer, equal to the admitted Workspace |
| device | integer, equal to the selected device |

The production C++ JSON producers live in `runner/execution_report.h` and are
used by the actual Runner. CPU regression compiles those same producers, runs
them without CANN/device APIs, parses their JSON and passes every field through
the real Python validator. The tests also exercise both `execute(host=True)` and
`verify_host()` while mocking only the unrelated SDK/resource boundaries, checking
that those boundary checks are still invoked. Missing, changed, extra or mistyped
fields reject. The old spelling and RUNNING/ATTEMPTED states reject. Report write
failures throw instead of returning Host success.

The phase audit confirms these intentionally separate levels:

| Phase | Runner completion | Phase report |
|---|---|---|
| Source | no Runner invocation | SOURCE_REVIEWED / NPU NOT_RUN |
| CPU | no NPU Runner invocation | CPU_TESTS_PASSED / NPU NOT_RUN |
| Build | no Runner execution | PRIVATE_OPP_BUILT / NPU NOT_RUN |
| Formal Host | HOST_TILING_ONLY / NOT_RUN | FORMAL_HOST_FEASIBLE / NOT_RUN |
| Echo | ECHO_EXECUTION_MATCH / COMPLETED | ECHO_NPU_MATCH / COMPLETED |

Echo completion still requires exactly two repeats, all existing safety flags,
no CPU fallback, the explicit L0C coverage limitation and the closed VDEQ16
label. Full numerical, witness, callback, serialization, artifact and SHA256
checks remain additional mandatory gates. No Kernel, Tiling, allocation budget,
event, data, reference or tolerance is changed by this fix.

Use **all five commands above with a fresh root**, for example
`w8a8-external-co1-echo-statusfix-$(date +%Y%m%d-%H%M%S)`. Source identities changed,
so old Source/CPU/Build/Host reports cannot be reused. The local check for this
fix recompiles only the changed Runner against real CANN headers/libraries and
its previously generated actual private API in a new directory. It does not
install an OPP or run a device process; existing compute Kernel artifacts are
not rebuilt for this report-only change. Historical validation reports stay
unchanged. See `status-fix-validation.json` for this bugfix's local results.
