# Stage4S: final N256 NZ-to-ND writeback coalescing

Independent single-variable experiment. Stage4B–4R, SDK, Golden and installed
OPPs remain read-only. No uint64 Guard, Weak Guard, Workspace hoist, deleted
sync, N512, changed Scale/math or reserve use. Full candidate/diagnostic OPP
build and NPU execution are NOT_RUN at local delivery.

## Read-only diagnosis

Supplied Stage4Q N256 observations: task duration2708.5805us, Scalar2232.3995us,
MTE2 443.161us, MTE3 92.813us, Vector541.5495us, MAC15.294us. These are profiler
observations, not a decomposition of wall time. Stage4R's fresh2.775050ms W8
and0.786980ms FP16 are context only and not a Stage4S paired denominator.
FP16 and W8A8 have distinct input/numerical semantics.

Build revalidates Stage4Q's actual hashed Profile records and runs the existing
row-provenance review. It preserves CSV source/type, Task/Stream/Core identity,
units and duplicates; op_summary and task_time are not independent populations.
No pooled counter median or Pipe-time sum is computed.38 profiler tasks include
checks/warmup/timing/postcheck and are not substituted for formal90 samples.

Static whole-launch counts for the normal success path:1520 Matmul groups,
972800 group Guard uint32 reads,19456 final32B DataCopy calls. Additional
Tile/Arena/final Guard reads and Matmul lifecycle/control remain. Scalar
activity could include these and SDK bookkeeping; no instruction-level
attribution or removable critical-path fraction is claimed. MTE3 active time
is not a bound on savings from reducing Scalar issue/loop overhead.

## Bounded SDK contract

Local root `/usr/local/Ascend/cann-9.0.0/aarch64-linux/asc`.
Local source review admits this exact geometry for isolated device validation.
At server Build, every listed SHA must match before generating any source;
local equality is not assumed. All new source identities are initially
LOCAL_ONLY_NOT_SERVER_VERIFIED. No SDK original text is committed.

| Relative file | Lines | SHA256 |
|---|---|---|
| impl/basic_api/kernel_operator_data_copy_intf_impl.h | 243–272 (MTE3/UB→GM),699–704 (32B count units) | 0a432937742d2625565d3eb1ef3caa930a190271faadb0344345d2382235737b |
| impl/basic_api/dav_m200/kernel_operator_data_copy_impl.h | 25–29 (count/length),69–79 (32B source alignment, direct ordinary intrinsic) | 8a890aa80c39380519bcba461bfd19a67060e650253622d8b49b4b2c01ea3c8f |
| include/basic_api/kernel_struct_data_copy.h | 44–68 (uint16 fields, stride/gap aliases) | 8402ed99bc1be3c05f688fb92e9262531ebce6199ba953c827dcef2a47bf5343 |
| impl/basic_api/utils/kernel_check_data_copy_util.h | 246–311 (unit lookup),315–353 (maximum-offset formula) | 63b0874089739c09f37c390608bd0a66d99bb7c928e10c94f46b7bf823051c05 |
| impl/basic_api/utils/kernel_utils_mode_cpu.h | 161–164,177–224,234–240 (UB→OUT, no unit override, VECIN→UB) | 794ec64d4e7e871ca79704c87770d920946544bcefe155f22b4dd7607f4d1f40 |
| impl/basic_api/utils/kernel_utils_constants.h | 32,52,188–207 (32B,4095) | 3ce93de8c25d198386e87656481f6bc9dfbeae86fe66a248f7e3d36c86cd32da |

The public GlobalTensor<half> destination / LocalTensor<half> source overload
selects ordinary UB→GM, not enhanced quantization or an NZ-format-converting
API. Source and destination burst lengths and gaps for UBOUT use32B. The
max-offset formula is count*length*unit+(count-1)*gap*unit. blockCount range
1..4095, blockLen1..65535; gaps are uint16 fields (0..65535 representable).
For this actual8192B source, the last row limits source gap to15. No claim is
made that arbitrary representable gaps fit arbitrary tensors. Both actual
source and destination starts are32B aligned; destination starts also inherit
512B alignment from the unchanged output allocation/row/tile offsets.

Proposed and audited call: DataCopyParams{16,1,15,0}. For row m and block nb:

- source half index:16*m + nb*(1+15)*16 =16*m+256*nb;
- destination half index:row*19456+tile*256+16*nb;
- source relative extent:16*32+15*15*32=7712B;
- last row start480B + extent7712B =8192B exactly;
- destination extent512B per row, no gap, no overlap with adjacent Tile.

The only normal-source change replaces nb×m copy loops with16 row calls and
one constant parameter object. Total payload remains622592B. Source-level
API calls become76×16=1216, not a claimed machine-instruction count or16×
speedup. All other source bytes are checked, including Host/OpDef.

Dependencies: the original final Cast/ALL, V_S, Guard/Arena checks precede
output consumption; both old and new copy overloads use MTE3. The original
post-output ALL remains before the next Tile reuses p0. No producer/consumer
buffer, event or UB allocation is added or moved.

## Local compile and CPU limits

An exact half DataCopyParams device-function probe was compiled with installed
Bisheng targeting dav-m200, architecture2002: returncode0, nonempty84B function
in the object. This is LOCAL_ONLY_NOT_SERVER_VERIFIED, not a complete OPP build
or NPU layout proof. Compiler `-emit-llvm` was unavailable, so no machine/IR
instruction reduction is claimed. Initial umbrella-header include-path
failures were retained; the successful probe uses the direct public basic API
headers. `compile_probe/check.sh NEW_DIR` reproduces compile-only checking and
never executes its object. Server numeric/Poison/Guard acceptance remains
mandatory to detect any target/compiler behavior inconsistent with the API.

CPU tests prove all source/destination addresses, every one of311296 output
positions, inactive Poison, boundary tiles, actual transformed C++ control
flow and retained fault paths. Cached Matmul partials are test doubles, not
MMAD or DMA emulation.

## Server stages

Each directory must be new. No automatic repair, fallback or old report rewrite.

```bash
set -euo pipefail
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
S4S=/home/w00949577/qwen35-dflash-run/w8a8-strided-store-$(date +%Y%m%d-%H%M%S)
S=framework/custom_ops/draft_w8a8_output_dequant/stage4s/run.sh
bash "$S" build --device 0 --output-dir "$S4S/build"
# Require STAGE4S_BUILT_NPU_NOT_RUN
bash "$S" host-only --device 0 --build-report "$S4S/build/report.json" --output-dir "$S4S/host-only"
# Require STAGE4S_HOST_ONLY; no Kernel submission
bash "$S" bounded --device 0 --build-report "$S4S/build/report.json" --prior-report "$S4S/host-only/report.json" --output-dir "$S4S/bounded"
# Require STAGE4S_BOUNDED_MATCH: Tiles0/37/38/75, Groups0/19/final, full20 groups each
bash "$S" full --device 0 --build-report "$S4S/build/report.json" --prior-report "$S4S/bounded/report.json" --output-dir "$S4S/full"
# Require STAGE4S_FULL_MATCH: Auto7 and2-Core, full Golden
bash "$S" faults --device 0 --build-report "$S4S/build/report.json" --prior-report "$S4S/full/report.json" --output-dir "$S4S/faults"
# Require STAGE4S_FAULT_COVERAGE_MATCH: separate instrumented OPP; all word/bit/group sweeps and20 abort witnesses, twice
bash "$S" benchmark --device 0 --build-report "$S4S/build/report.json" --prior-report "$S4S/faults/report.json" --output-dir "$S4S/benchmark"
# Require STAGE4S_PAIRED_MEASURED
bash "$S" profile --device 0 --build-report "$S4S/build/report.json" --prior-report "$S4S/benchmark/report.json" --output-dir "$S4S/profile"
# Require STAGE4S_PROFILE_COMPLETED; counter availability separately reported
```

Normal candidate and diagnostic OPPs have independent private installs. The
original Stage4P OPP is reused. Two new Benchmark Runners link baseline or
candidate separately. Diagnostic Runner/OPP never enters measured orders.
All fault instrumentation and its witness transfer remain separately audited.

Formal order: baseline→candidate, candidate→baseline, baseline→candidate;
continuous-v1, two poisoned prechecks, warmup5/reps30, timed tail and poisoned
postcheck. Each variant supplies90 NEW samples. All checks remain. Materiality
requires aggregate baseline/candidate median>=1.03 and each round>1, else
NO_GO_NO_MATERIAL_IMPROVEMENT. Independent Profile never supplies those samples.

Return all phase report.json, comparison.json and source audits/diffs. Preserve
build/install/Runner logs, execution.json, resource-admission.json and raw
actual/tail/postcheck files on failure. Any failure stops the next phase; no
partial or diagnostic output is promoted to full correctness.

## If this change is insufficient — design only

Do not immediately change Guard width, End or synchronization. Future work
could separately examine SDK-supported grouping/queue lifecycle and B/Scale
transfer overlap, but only after resource/event ownership proofs. L0B is
already65536B at N256; two Matmul scratch allocations do not fit current UB.
No reserve may be treated as spare buffering. INT32 partial→FP32 Scale/Add
would remove the current per-group FP16 rounding boundary and is a distinct
numerical proposal requiring new references and quality analysis. Combining
20 differently scaled groups into one ordinary K2560 INT8 Matmul is invalid.
Model integration must account for online quantization/encoding and accuracy;
current paired latency is not end-to-end Decode. No such follow-up is coded.
