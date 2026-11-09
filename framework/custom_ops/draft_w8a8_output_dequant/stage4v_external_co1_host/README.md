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
recorded and TCubeTiling::GetDataSize must equal200. All values are SDK-returned,
including unused/default fields; no standalone values are copied into them.
The sole authoritative bytes are the complete248B registered Op serialization
in `raw->GetData()`, saved as `callback-N-op.bin`. The callback extracts its
SDK-proven `[48,248)` slice as `callback-N-cube-payload.bin`. Python validates
all48 outer bytes and all50 decoded inner fields against the actual config and
Getter JSON. `callback-N-cube.bin` remains the old child SaveToBuffer diagnostic
and is never an admission source. See the nested-serialization bugfix below. Files are exclusive-created
and cannot overwrite prior captures. Contradictory repeated callbacks block.

`callback-N.json` retains setup return values (null for void setters), actual
GetTiling rc, actual platform fields, checks and rejection reason. If a setter or
GetTiling fails, no complete TCube result is fabricated. If resource checks fail
after GetTiling, actual50 Getter fields and the child diagnostic remain available; an accepted parent serialization is not fabricated for a rejected callback.
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
`host/callback/execution.json` and `host/callback/runner.log`. Keep all three binary files and their hashes; only the parent and its extracted payload are authoritative. For a build failure return
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


## Nested TCube serialization bugfix (after0d53887)

The server run `w8a8-co1-formal-host-20261109-063148` established that its248B
parent contains the50 correct fields at48–247, while the separately saved200B
child target remained zero. That old BLOCKED report and its files are untouched.
This fix does not accept them retroactively and does not change any Matmul,
resource, Workspace or event configuration.

Audited installed header:
`/usr/local/Ascend/cann-9.0.0/aarch64-linux/include/register/tilingdata_base.h`
SHA256 `242c15b887d92f3c1d36bb927bb7540d1a9c4e626fdd3c5db1ca202590da57ab`.
This same header identity is enforced by the existing CANN9 source binding and
is now cross-checked against the actual Host compile dependency list.

* Lines75–103 declare SaveToBuffer/GetDataSize/InitData and expose the distinction
  between storage pointer, declared sizes, nested pointer registrations and the
  `inited_data_ptr` ownership flag. The method bodies are not in this header.
* Lines178–186 register a nested object's pointer and its serialization offset,
  applying8-byte alignment before adding the nested structure size.
* Lines190–204 distinguish a normal constructor calling InitData from the
  explicit pointer constructor that returns early for nullptr.
* Lines206–216 show setters updating both the scalar cache and bound backing
  storage; getters return the scalar cache. Getter correctness alone does not
  prove that a child SaveToBuffer wrote its output argument.
* Lines240–249 instantiate the nested child with nullptr and register it with
  the parent. This is a parent-managed view, not an independently default-created
  standalone TCubeTiling allocation.

The actual out-of-line symbols on the local SDK are exported/delegated by
`libregister.so`; `TilingDefImpl::{InitData,SetDataPtr,GetDataSize,SaveToBuffer,
CheckAlignAndGenPlaceHolder}` are exported by its `libopp_registry.so` dependency.
Their implementation source bodies are not supplied in the inspected installed
headers. We do not invent source line numbers or an undocumented internal branch.
Local library identities are recorded in `serialization-fix-validation.json`;
the real server libraries are separately SHA-bound by each new build's runtime
binding. Same version labels alone are not binary equivalence evidence.

A real CPU experiment on these local libraries prefills the child destination
with0x00,0xA5 and0xFF. Each target remains unchanged after the nested child's
SaveToBuffer, whereas parent SaveToBuffer contains the values. Thus the local
zero result is an unchanged pre-zeroed target, not a serialized all-zero Tiling.
The new real callback also records whether the0xA5 self-test target stayed
unchanged. This observation and the visible macros explain this specific
parent/child case; they do not establish identical behavior for every SDK version,
standalone TCube object or arbitrary nested type.

### Offset and source proof

No `offsetof` is taken on the non-wire C++ TilingData object. A standard-layout
wire-prefix mirror has explicit offsetof/size assertions, but it is **not** the
sole proof. `Co1ProveLayout<the actual Parent type>` creates separate calibration
objects, sets unique markers in all50 child fields, serializes a standalone
TCube with the same markers, and finds exactly one complete match inside the
SDK-serialized parent. It requires offset48, child200, parent248, and exact
agreement of every outer marker/padding byte with the48B mirror. A mismatch
blocks the real callback. Calibration never calls GetTiling and never supplies
values to the actual Formal object.

The actual callback still obtains its Tiling from GetTiling, sets the unchanged
outer fields, calls parent SaveToBuffer on `raw->GetData()`, then copies that
buffer and its proven slice. The actual payload is not packed from Getter JSON.
Python requires the outer layout:

| Offset | Type | Actual configuration |
|---|---|---|
|0|u64|localMemSize202240|
|8|u32|outputMode0|
|12|i32|tileId0|
|16|u32|launchBlocks1|
|20–23|padding|zero|
|24|u64|actual systemBytes|
|32|u64|actual arenaBytes|
|40|u64|actual arenaStride=arenaBytes+1024|
|48–247|200B|actual TCube payload, all50 fields checked|

The reviewed Schema SHA256 is pinned to
`f1688b61034f91c7518f65dcf73382ee9b007ab2c461fc8244cb6b883bd1d944`.
Both its file identity and parsed contents must match the bound prior SDK schema.
The test fixture contains that exact layout metadata, not SDK source code.

Raw callback JSON records offset/length/source, schema hash, layout proof and
child diagnostic role. The Python report adds SHA256 directly from immutable
file bytes without rewriting callback JSON. The top-level report and each
callback's `serialization` record expose:
`cube_payload_offset`, `cube_payload_bytes`, `cube_payload_source`,
`cube_payload_sha256`, `op_serialization_sha256`, `schema_sha256`.
The per-callback record also binds op/payload/JSON file identities, compiled Host
binary/sources, SDK header sources and actual runtime libraries. Repeated
callbacks must have identical Getter/configuration data and parent/payload hashes.
Any mismatch is BLOCKED; the original50-field equality check remains mandatory.

### Fresh server rerun

Use the three source/build/host commands above with a new root, for example:

```bash
OUT=/home/w00949577/qwen35-dflash-run/w8a8-co1-nestedfix-$(date +%Y%m%d-%H%M%S)
```

Rebuild the new private Host library; do not reuse the old source/build reports.
Target status remains only `FORMAL_CO1_HOST_FEASIBLE` after a new real capture
passes all checks. Return `host/formal-host-report.json`, raw callback JSON,
`execution.json`, `runner.log`, and retain op/cube-payload/diagnostic cube files.
NPU, event execution, Benchmark and Profile remain NOT_RUN.

Bugfix validation: macOS149/149 and Docker ARM64 149/149 CPU tests passed.
The real SDK CPU serializer reproduced unchanged child targets for0x00/0xA5/0xFF,
and proved the parent layout/markers. New Host/placeholder/package and generated
API receiver compilation completed locally without installation or device query.
These are local compilation/serialization observations, not a new Formal Host PASS.
