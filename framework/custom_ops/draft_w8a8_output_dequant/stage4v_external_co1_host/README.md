# External CO1 Formal Host-only validation

This independent operator creates a real registered CANN Host callback for
External CO1 resource validation. It does **not** calculate, upload test inputs,
execute a Kernel, run events or generate a Golden. All historical stages remain
unchanged. The prior completed source audit can admit this new Host investigation
without changing that audit's historical BLOCKED decision.

Unique operator: `DFlashExternalCo1HostProbe`. Private OPP is installed only under
the new experiment's `build/private/opp`. The registration ABI has INT8 ND
A[16,2560], B[2560,256], UINT64 ND Scale[256], and an INT32 ND descriptor O[16,256].
O is a registration/possible future Echo descriptor, not an output computed by
this experiment. Scale contents are not read: Host-only validates its type,
shape and2048B future consumer budget, not device dequantization semantics.

The internal Matmul configuration is **GM ND INT8 A/B, non-transposed B, CO1 NZ
INT32 C, no bias, no Matmul dequant setter**, M16/N256/K2560, Base16/256/128,
SetBufferSpace(-1,-1,202240). No type/base/scratch fallback exists.

## Gates

| Command | Required success status | Scope |
|---|---|---|
| source | SOURCE_REVIEWED | Accepted source audit, current SDK/release/header identities, prior device/Core evidence and50-field schema bound |
| build | PRIVATE_OPP_COMPILED | Real CANN Host and empty COMPILE_ONLY registration placeholder packaged, privately installed and independently linked Host-only receiver verified |
| host | FORMAL_CO1_HOST_FEASIBLE | Real callback GetTiling and all resource checks accepted, actual50-field bytes and ACLNN Workspace request cross-checked |

Any failure is BLOCKED with a nonzero exit. Build/Host reports do not grant NPU
admission. The source audit reference is
`w8a8-external-co1-review-20261109-054004/review-report.json`; it must remain
`audit_completed=true`, source identities matched, with no error/integrity_error.
The historical code and report hashes are rechecked at stage boundaries/finally.

## Actual callback capture

The new Host source calls PlatformAscendC on `c->GetPlatformInfo()`, never
PlatformAscendCManager or ContextFaker. It records actual capacities and
GetCoreNumAic/Aiv/GetCoreNum/architecture, applies the reviewed coupled policy,
and uses one diagnostic Block. Actual available/selected cores and Matmul
usedCoreNum are separate. The receiver requires Ascend310P3; the Python gate
requires the policy to equal the prior accepted formal device policy (currently
8/8/7/7,2002). Those counts are not hard-coded scheduling choices.

Inside this callback, **after real GetTiling succeeds**, all50 public getters are
recorded, TCubeTiling::GetDataSize must equal200, and SDK SaveToBuffer writes
`callback-N-cube.bin`. All values are SDK-returned, including unused/default
fields; no standalone values are copied into them. The callback also writes
its complete registered Op serialization as `callback-N-op.bin`. Python hashes
both and decodes the200B file using the identity-bound SDK serializer schema,
requiring exact agreement with the50 logged fields. Files are exclusive-created
and cannot overwrite prior captures. Contradictory repeated callbacks block.

`callback-N.json` retains setup return values (null for void setters), actual
GetTiling rc, actual platform fields, checks and rejection reason. If a setter or
GetTiling fails, no complete TCube result is fabricated. If resource checks fail
after GetTiling, the actual50 fields and binary remain available for diagnosis.
An absent callback cannot be replaced by a successful GetWorkspaceSize alone.
Historical FP16/Raw23 missing formal fields remain **UNKNOWN**; the new CO1 fields
are not used to fill their reports.

## Resource contract

* External CO1: planned first/only allocation `[0,16384)`, conservative1024B
  alignment, within actual L0C capacity. Scheduler's CO1 specialization has
  effective shared-L0C reservation0, while the raw returned field is preserved.
  Host success validates the plan, **not a device allocation address**.
* UB: scratch202240 + Echo16384 + Scale2048 + FP168192 + four Guard pairs2048 +
  untouched Reserve8192 = **239104B**, against actual UB262144. Effective shared
  UB is0 on dav_2002. Scale is separate caller-owned storage; this INT32 Matmul
  has no coefficient consumer. No additional scratch is silently assigned.
* L1: input tile bytes2048/32768 times actual depths; steps must divide20 and have
  the reviewed1/2 batch relationship. Reported shareL1Size must equal that same
  allocation, not be added twice. Unknown extra consumers block.
* L0A/B: actual db factors1/2 and capacities; L0B must be65536, no extra space
  assumed. L0C db factor is recorded/checked but does not create another caller
  output allocation in the external specialization.
* transLength must be nonnegative,32B aligned and no greater than202240; the
  reviewed two B conversion views fit that scratch. Additional UB cache depths,
  MX parameters and auxiliary ND layout metadata must be0. Non-batched counters
  may be0(unset) or1; values describing multiple batches reject. All50 fields are
  covered by either shape/resource checks or these explicit auxiliary rules.
* GM Workspace: actual platform system value plus one full rounded arena and
  512B prefix/suffix; total <=512MiB. Actual ACLNN returned size must be at least
  the callback's request and <=512MiB. Any framework overhead is preserved as
  a separate actual size, never mistaken for extra UB/L1 capacity.

Event design is recorded, not run: dynamic AllocEventID/ReleaseEventID with
runtime range/collision checks, M_V before Echo/VDEQ16, MTE2_V for Scale, V_S for
Guard visibility, V_MTE3 for output, V_M before a future Cube overwrite, and End
retained after consumers. No Host-only result proves event allocation, buffer
lifetime or numerical/device safety. Those remain future execution gates.

## Non-execution assurance

CANN packaging requires an Op Kernel entry; this one is empty, accesses no data,
and is labeled **COMPILE_ONLY**. It contains no Matmul, DataCopy or events. Its
ELF is a registration identity only, never evidence of compute correctness.

The receiver creates device-backed tensor descriptors, invokes exactly the new
GetWorkspaceSize API, saves its rc/returned Workspace, and exits. It creates no
stream, performs no H2D/D2H or Workspace allocation, and never calls the execute
API. The linked executable is checked with `nm -u` for the expected preparation
symbol and against execute/launch symbols. There is no execution CLI option.

Toolkit, Driver and OS runtime identities retain the reviewed separate binding,
including HAL SONAME/symlink/SHA checks. Compiled Host dependency records bind the
actual platform header. The private generated API/Header, Host library, placeholder
ELF, package, generated sources and receiver are all identity-bound.

## Server commands

Run from repository root. Each stage writes a new directory. Stop after any
nonzero exit; inspect the report before starting the next stage.

```bash
set -euo pipefail
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
D=framework/custom_ops/draft_w8a8_output_dequant/stage4v_external_co1_host
OUT=/home/w00949577/qwen35-dflash-run/w8a8-co1-formal-host-$(date +%Y%m%d-%H%M%S)
REVIEW=/home/w00949577/qwen35-dflash-run/w8a8-external-co1-review-20261109-054004/review-report.json

bash "$D/run.sh" source --review-report "$REVIEW" --device 0 --output-dir "$OUT/source"
# Must return SOURCE_REVIEWED.

bash "$D/run.sh" build --review-report "$REVIEW" \
  --source-report "$OUT/source/formal-host-report.json" \
  --device 0 --output-dir "$OUT/build"
# Must return PRIVATE_OPP_COMPILED; still no device query or execution.

bash "$D/run.sh" host --review-report "$REVIEW" \
  --source-report "$OUT/source/formal-host-report.json" \
  --build-report "$OUT/build/formal-host-report.json" \
  --device 0 --output-dir "$OUT/host"
# Only FORMAL_CO1_HOST_FEASIBLE means the new formal resources were accepted.
```

Return `host/formal-host-report.json` plus `host/callback/callback-*.json`,
`host/callback/execution.json` and `host/callback/runner.log`. Keep both serialized
binary files and their hashes. For a build failure return
`build/formal-host-report.json` and the failing log under `build/private`:
msopgen.log/build.log/install.log/host-ldd.log/cmake.log/runner-build.log/ldd.log.
No log, generated package or SDK source belongs in the public repository.

## Local verification

```bash
PYTHONPATH=framework/custom_ops/draft_quant/test python3 -B -m pytest -q -p no:cacheprovider \
 framework/custom_ops/draft_w8a8_output_dequant/stage4v_external_co1_host/test
# Existing ARM64 CANN Docker, fresh output only; no install/query:
bash framework/custom_ops/draft_w8a8_output_dequant/stage4v_external_co1_host/local_check/check.sh \
 /work/build/NEW_CO1_FORMAL_HOST_CHECK
```

Local tests use synthetic callback records, never a fake Context represented as
Formal evidence. The local compile check packages the real Host callback and
empty placeholder, and links the receiver to the actual generated private API
inside CPack without installation. opbuild uses an older C++ language mode; void
versus integer setter return recording uses compatible SFINAE. Local compilation
is `LOCAL_ONLY_NOT_SERVER_VERIFIED`. Real CO1 GetTiling, formal capacities and
Workspace remain NOT_RUN until the server executes the host command.

Local validation on2026-11-09: macOS30/30 and Docker ARM64 30/30 CPU regressions
completed; real CANN package and generated-API receiver compiled/linked without
installation or execution. The receiver imports GetWorkspaceSize and no operator
execute/launch symbol. See `local-validation.json` for artifact/source hashes.
No additional UB Arena temporary is admitted by this239104B plan. Any future
extra consumer requires a separate resource proof; reuse of a dead region in an
already charged buffer also needs explicit completion/lifetime validation and
must not silently consume Reserve.
