# Stage 4A: eight real N64 tiles, serial and multicore correctness

Independent `DFlashW8A8MultiTile`. No previous implementation or evidence is
changed, no previous Runner is executed. New CANN compilation, platform Tiling
and NPU validation are **NOT_RUN** on the development workstation. This is not
full N19456, Draft, Simulator or performance work.

## Fixed compact ABI and provenance

| Tensor | Layout | Bytes |
| --- | --- | ---: |
| A INT8 | `[20,16,128]`, group,row,k; shared by all slots | 40960 |
| B INT8 | `[8,20,128,64]`, slot,group,k,n | 1310720 |
| Scale UINT64 | `[8,20,64]`, slot,group,n | 81920 |
| O FP16 | `[16,512]`, row,slot*64+n | 16384 |

Slots 0..7 map exactly to original N64 tiles **[0,1,15,16,151,152,302,303]**.
`case.json` records original channel ranges, NZ N16 block ranges, compact
columns, byte offsets and per-slot B/Scale/Golden hashes. Tile151 ends at 9728;
Tile152 starts at 9728. Tile303 ends at 19456. The row stride is **512 elements**,
not 64. Each core writes only its assigned slots, using the accepted NZ→ND
32-byte copies with destination `row*512 + slot*64 + nb*16`.

Prepare validates the complete accepted Stage3B-4 proof (all four cases and
84 modes) and its earlier identity chain without rerunning any device test.
It follows Stage3A's bound CPU manifest to find the full frozen X, signed NK
weights, NZ weights, GN Scale and saved full INT8 Activation. It independently
verifies the saved Activation against the original full-X quantization once
per preparation/check, then packs it by K128 group. There is no per-tile or
partial-X activation scale calculation.

Every selected tile/group compares the NZ inverse with the corresponding
original NK bytes. Original FP16 Scale `[g,n0:n0+64]` and full-X scale_A form
the same FP32 combined coefficient as Stage3A; the existing candidate encoder
is unchanged. References use exact INT64 dot, per-group FP16 partial rounding,
ordered FP32 g=0..19 accumulation and one final FP16 conversion. All references
must be finite. Tile0 inputs/words must equal the accepted Stage3B-4 bytes;
its Golden is copied by identity from that accepted case after equivalence
checks, including both saved NPU outputs. Other tiles have independent CPU
references. Native Golden is neither modified nor used as a substitute.

## Scheduling and lifecycle

Optional integer attributes:

- `core_limit=1` by default; `0` requests min(8, selected scheduling limit).
  Eight is the number of tile slots, not an assumed hardware Core count.
  Positive limits must be <=8 and <=the selected limit; they are not
  silently clipped.
- `output_mode=0` gives the numerical result. Modes 1..20 select a partial
  group, after every tile has still completed its twenty groups. Mode21 is a
  scheduler witness described below.

Host logs all three installed SDK getter results and requires:
`arch==2002`, nonzero aic/aiv/all, `aic==aiv`, and `all<=aic`. Only then is
`selected=all` admitted. The fixed log is
`STAGE4A_CORE_POLICY aic=N aiv=N all=N selected=N`. An invalid or unavailable
architecture/count returns selected=0 and rejects Tiling. There is no max()
or alternative-source fallback.

The CANN9 [GetCoreNum contract and Host example](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_1028.html)
use its result for SetBlockDim. The [Cube view](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_1030.html)
and [Vector view](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_1031.html)
provide independent bounds. The supplied installed-SDK/TilingContext evidence
at `w8a8-core-probe-20261027-073730` reports aic=8, aiv=8, all=7, soc=2,
arch=2002. Under this policy the scheduling limit is 7, not a hard-coded
physical-Core diagnosis. These values do **not** establish that a Core is
broken, reserved, or that exactly seven physical Cores exist. Why the SDK
views differ remains UNRESOLVED; no out-of-line implementation is invented.

The real Runner still requires Ascend310P3, the Host requires dav_2002 and the
unchanged Kernel has its 2002 compile-time assertion. Thus this policy does
not apply a separated Cube/Vector launch rule to the coupled Matmul path.
Build records compiler-resolved installed platform header identities/API lines,
Host library identity and actual linked SDK libraries in `core-sdk-binding.json`.
Missing header resolution is explicit; declarations are not presented as the
implementation explanation for 8/8/7.

Each block handles `slot=GetBlockIdx()+i*blockDim`. Its TPipe, UB, Matmul object
and FP32 sum are core-local. Sum is cleared for **each slot**. Within a slot,
the accepted Stage3B-4 SetLocalWorkspace/A/B/Scale → Iterate(false) →
GetTensorC → PIPE_ALL → End → PIPE_ALL → Cast/Add sequence is unchanged. No
atomic reduction, overlap or prefetch is added. PIPE_ALL completes output
before reusing buffers for the next slot.

The actual MatmulApiTiling remains a per-core M16/N64/K128 operation. External
slot scheduling does not change its shape to an automatically split M/N
problem. Returned single/base dimensions, depths and transLength are checked
again for this operator and logged; historical Tiling is not copied.

## Workspace is explicitly partitioned

Per-core UB remains scratch 239104 B + user payload 12288 B + guards 2560 B +
headroom 8192 B. No per-slot copy of these buffers is allocated.

GM system space does **not** blindly reuse a blockDim=1 reservation. Host
queries S=`GetLibApiWorkSpaceSize()` on the actual platform. The
[CANN9 API](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_1036.html)
returns the system workspace size; it does not expose a documented per-core
fraction. This correctness experiment conservatively reserves a **full S-sized
arena for every launched core**, retaining any SDK-internal core-index offset
within a full platform-sized arena:

- arena size A = ceil512(S);
- stride = A + 1024, including 512-byte prefix and suffix guards;
- user space = blockDim * stride;
- total request = framework system space S + user space.

`REGIST_MATMUL_OBJ` receives the current core's explicit arena pointer, as
per its [system-workspace pointer interface](https://www.hiascend.com/doc_center/source/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_0628.html).
The ordinary framework system reservation is retained for framework/SDK uses.
The generated operator supplies the user-workspace argument; it is not passed
through GetUserWorkspace a second time. No documented global system size is
divided by block count or assumed to be per-core. Total requested space is
bounded at 512 MiB; larger/zero queries fail with the real values in logs.
This intentionally over-reserves memory and is not a minimum-memory or
performance result. The SDK may not consume this arena on every 2002 path;
reserved bytes are not reported as measured used bytes.

Only the retained output buffer is allocated first to initialize the GM arena
guards **before** registration. This order is restricted to the coupled 2002
target; the documented registration-before-InitBuffer constraint applies to
separated mode, which is rejected. Guards are checked immediately after
registration and before every slot output, using the existing FP32 temporary
as a 512-byte readback buffer plus MTE2_S synchronization. All five UB guard
pairs remain checked. A failure leaves unwritten output poison and cannot pass.

The installed compiler's REGIST_MATMUL_OBJ macro expansion is emitted into
`build.log` and bound in `workspace_registration` in the build report for
source inspection; it is not converted into an automatic hardware-support
claim. Actual Host sizes, disjoint arena intervals and logical slot ownership
are recorded per run. After GetWorkspaceSize and **before any operator
execution**, Runner reads the flushed Host plan from its redirected
`runner.log`, checks core scope/disjoint sizing and requires the returned
ACLNN allocation to cover the declared total. Missing, drifting or
undersized plans stop before device execution and are never guessed.
`resource-admission.json` records this gate and is verified again by the
Python driver. Direct manual Runner calls must also redirect stderr/stdout
to that file path; the supplied driver handles it automatically. These
source/API assumptions and new initialization order still require the actual
server compilation and numerical/guard tests; CPU doubles do not certify SDK
initialization or asynchronous memory safety.

## Required NPU gates

Default Smoke performs scheduler mode21 and numerical mode0 for:

1. core_limit=1 (all eight slots sequentially);
2. core_limit=2 (four slots/core, requiring >=2 real available cores);
3. core_limit=0 (automatic upper bound), unless that is already two cores.

Only one available core yields a blocked/failing report after preserving the
serial results; it cannot yield Stage4A PASS. Mode21 executes all arithmetic,
then fills each slot with exact FP16 `blockIdx+1+slot/16`. Its expected value
uses the actual logged blockDim. This tests **logical block ownership**, not
physical core IDs or temporal overlap. It cannot replace numerical mode0.

Every run has two distinct poison/readback repeats, full-input Readonly,
workspace/GM/UB guards, and strict comparison of all 8192 output elements,
reported separately for each original tile. Serial and parallel numerical
outputs must also agree. A missing tile leaves NaN poison and fails. No ULP
tolerance or fallback is permitted. Optional partial diagnostics select only
groups 0,1,9,10,18,19; this does not repeat the historical 84-mode suite.

## Server commands

### First: Build and Host-only admission

Always create a fresh Build and Host-only directory after this policy change.
Host-only performs mode21/core_limit1 GetWorkspaceSize and resource admission,
without submitting the operator. Expected status is STAGE4A_HOST_TILING_ONLY,
NPU NOT_RUN, with selected=7 for the supplied 8/8/7 context. Workspace totals
and arena ranges must still pass. The raw SoC identifier is diagnostic; an
unavailable architecture is now a rejection, not an optional admission.

```bash
set -euo pipefail
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
S4A=/home/w00949577/qwen35-dflash-run/w8a8-core-policy-$(date +%Y%m%d-%H%M%S)
ENTRY=framework/custom_ops/draft_w8a8_output_dequant/stage4a/run.sh
bash "$ENTRY" build --device 0 --output-dir "$S4A/build"
bash "$ENTRY" smoke --host-only --device 0 \
  --multi-build-report "$S4A/build/report.json" --output-dir "$S4A/host-only"
printf 'Evidence: %s\n' "$S4A"
```

Inspect `host-only/report.json`, its `core_policy` and the single run's plan
and `resource-admission.json`. They must agree on aic/aiv/all/selected/arch.
No new numerical or multicore PASS is claimed at this point.

### Then: normal Smoke in another new directory

Only after Host-only succeeds, with the same Build and unchanged evidence:

```bash
bash "$ENTRY" smoke --device 0 \
  --multi-build-report "$S4A/build/report.json" \
  --host-probe-report "$S4A/host-only/report.json" \
  --output-dir "$S4A/smoke"
```

Normal Smoke refuses missing, stale or differently bound Host-only proof. The
validated count tuple is passed to each Runner; its pre-execution admission
rechecks actual Core probe/policy/architecture against that tuple before any
operator call. Policy drift stops before device execution. Workspace pointer
offsets, arena formula, per-core UB, Kernel mathematics and Golden remain
unchanged. Normal execution order is limit1 mode21/0, limit2 mode21/0, then
auto mode21/0 (7 blocks for the observed context). No hard-coded 7 is used.

Macro evidence remains independent: bounded generated-project log collection
can report EXPANSION_NOT_EMITTED without proving a wrong Workspace pointer.
Neither the Core policy nor a Host-only PASS upgrades that macro evidence.

To include partial diagnostics on the first run, append
`--diagnostic-groups 0 1 9 10 18 19` to Smoke. Output directories must be new.
Defaults bind the supplied Stage3B-4 path
`w8a8-twenty-group-20261027-055348` and the unchanged earlier evidence.
`--build-report` refers to old Stage2B; use **--multi-build-report** for this
operator. Expected final status is `STAGE4A_MULTI_TILE_MATCH` only after all
serial and parallel gates pass. Stage4B and performance remain NOT_RUN.

Return Prepare/Build/Smoke `report.json`, `data/case.json` (contains tile/hash
mapping), and failed `execute/limit-*/mode-*/{runner.log,execution.json,actual-*.bin}`.
For compile failures include `build.log` or `runner-build.log`. Workspace/core
plans and macro evidence must be retained; do not override a failure by
switching pointers, changing Golden or pretending a serial run is multicore.

## CPU regression

`python3 -B -m pytest -q framework/custom_ops/draft_w8a8_output_dequant/stage4a/test`

Tests cover eight original boundary tiles, NZ inverse, full-X reuse, Tile0
identity, Scale broadcast, wrong B offsets, Gate/Up boundary swaps, missing
slots, wrong compact stride, missing sum reset, workspace partition guards,
actual Kernel control flow for 1/2/3/7/8 logical blocks, and refusal to grant
multicore PASS on a one-core fixture. CPU models are not NPU evidence.
