# Stage4X — real Draft Full-K W8A8 Down

For the Runner Scale byte-count bug in baseline35797f7, see
[SCALE_BYTES_FIX.md](SCALE_BYTES_FIX.md). Scale is20480B; A remains155648B.
It includes confirmed failure evidence and fresh Source→Host rerun commands.

Baseline4e433eb. Independent `DFlashDownFullKBNz` and diagnostic
`DFlashDownFullKBNzFault`. No Stage4W, accepted Gate, Golden, checkpoint or old
report is edited. The private normal/fault OPPs and Runner processes are separate.
No full Draft/Target/Verify integration, weak guards, N512, new quantizer,
Workspace hoisting, overlap, or synchronization deletion.

## Exact data and arithmetic

M16/K9728/N2560. A INT8[16,9728]; B opaque ND carrier of transposed-storage NZ
[10,304,16,16,32],24,903,680B with no padding; Scale UINT64[2560],20480B;
output FP16[16,2560],81920B. Matmul sees B=GM NZ INT8, transpose=true, exactly as
the accepted Gate implementation. Logical B is KN; checkpoint codes are NK.

`NZ[t,k//32,n//16,n%16,k%32] = q_fullk_nk[t*256+n,k]`.
Each tile inverse restores every INT8 element. Local output stays NZ16 with16 N
blocks; writeback index is `row*2560 + tile*256 + channel`. All10 tiles cover40960
elements exactly once, with static tile=core+i*blocks ownership.

Prepare exclusively reads Stage4W's real NPU
`cases[].boundaries.MIXED_BNZ_GATE_W8A16_DOWN[0].multiply`; verifies the second
repeat, shape, source/hash chain, four distinct inputs and C16/C64×2 coverage.
It verifies mixed Gate bytes against accepted Gate Smoke. It does not substitute
original `down_x.bin` or CPU SwiGLU. Frozen q_fullk_nk and weight_scale_f64 are
reused and independently checked against the checkpoint's effective FP16 weights;
they are not requantized or tuned to produce a pass.

One activation scale over all16×9728 FP16 inputs: FP64 maxabs/127, RNE and
clip[-127,127], zero-input scale1. Combined scale is FP32(FP64(sA)*sW); the exact
existing UINT64 encoding is reused. No rowwise scale is claimed. Reference dot
uses INT64, with9728×127×127=156902912 <2^31; the stored INT32 conversion is exact.
Candidate Golden is RN16(FP32(INT32 dot)*effective FP32 coefficient). Original
201 Gate differences are not applied as a Down tolerance or expected count.

Each input has exact integer and encoded-reference files, independent FP64 dot
against the same effective FP16 weight, same-input native W8A16 Down output, and
original W8A16 MLP output from Stage4W. Reports retain all row/channel bit
differences, ULP, max absolute error, RMSE, relativeL2 and cosine. Bit comparisons
to FP64 are explicitly against RN16(reference) for diagnostic reporting only;
the original FP64 errors are also retained. Quality stays UNRESOLVED.

## Resource and source gates

The Down Host probe is compiled independently and queries the selected real
device using no-argument PlatformAscendCManager. It requests16/256/9728,
Base16/256/128, A ND INT8, B NZ INT8 transposed, C Local NZ FP16, Tensor dequant,
no bias. It records setter/GetTiling results, actual fields, serialized Tiling,
SDK/compiler/library identities and capacities. Failure stops before private
compute Build; no alternate tile, K split or scratch increase occurs.

The formal Op Host then queries its own real TilingContext, and must exactly
match the admitted Down fields/capacities/workspace contract. Manager core count
does not substitute for formal policy. Formal Auto must match the Stage4W-bound
policy (currently8/8/7,selected7), while2-core Smoke uses the same available-core
proof. usedCoreNum remains1 for each logical Matmul. Ten logical tiles are not
ten physical cores.

Fixed per-core plan: Scratch202240B; Partial8192B; Arena temporary512B;
three prefix/suffix guard pairs1536B; Reserve8192B. Total220672B <= queried UB.
Unused UB is not allocated or borrowed. Base L0B32768B×actual double-buffer factor
must fit65536B. L1 uses2048×depthA1 +32768×depthB1 and must equal actual shareL1Size
(not added again); fixed no-bias family requires legal step/depth cache counts,
divisible K steps within76 basic K blocks, and compatible stepKa/stepKb. A
different returned family is BLOCKED, not silently admitted. L0A/L0C, Scale at
transLength..transLength+2048, SDK shared UB dav_2002 consumption and conservative
scratch staging bounds are checked afresh. The old Gate resource result is not
accepted as Down proof.

Full platform-sized system arena per launched core plus redzones and framework
system workspace is recalculated from actual SDK queries. The existing address
formula, REGIST_MATMUL_OBJ, SetLocalWorkspace, input/scale updates, Iterate(false),
GetTensorC, End, all barriers/events, three384-word UB checks and Arena XOR/OR
checks are retained. `x_transform.py` regenerates and audits exact shape/name/ABI
changes; added transfer timing is confined to untimed Runner CopyIn/CopyOut.

`hostplan` is real Standalone SDK resource evidence; `host` is formal
GetWorkspaceSize-only evidence. Neither is a numerical NPU pass. Local CANN
compile checks use explicitly synthetic T1 fields solely to compile signatures
and source. Those fields can never enter server Build: its generated t1_contract.h
comes only from the freshly revalidated successful hostplan report.

## Server execution

Use the existing CANN9.0.0/ARM64 server Python environment (NumPy; no torch import
is needed for the new Down runner). Start a new experiment; never overwrite any
old directory. Each command runs just one stage and refuses missing/failed or
identity-drifted predecessors.

```bash
set -eo pipefail
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
source "$CANN_ROOT/set_env.sh"
set -u
export DEVICE_ID=0
S=framework/custom_ops/draft_w8a8_output_dequant/stage4x_down
W=/home/w00949577/qwen35-dflash-run/real-draft-mlp-mountfix-pqbjQq
X=$(mktemp -d /home/w00949577/qwen35-dflash-run/w8a8-down-fullk-XXXXXX)

bash "$S/server_stage.sh" source "$X" "$W"
bash "$S/server_stage.sh" prepare "$X" "$W"
bash "$S/server_stage.sh" hostplan "$X" "$W"
# Do not proceed if actual GetTiling or any resource check is blocked.
bash "$S/server_stage.sh" build "$X" "$W"
bash "$S/server_stage.sh" host "$X" "$W"
# No Kernel execution has occurred through host.
bash "$S/server_stage.sh" tile0 "$X" "$W"
bash "$S/server_stage.sh" smoke "$X" "$W"
bash "$S/server_stage.sh" fault "$X" "$W"
# Can assemble numerical evidence even when CPU bit mismatches are unresolved.
bash "$S/server_stage.sh" mlp "$X" "$W"
# Requires ZERO CPU bit mismatches in every full Auto/2-Core result plus Fault.
bash "$S/server_stage.sh" benchmark "$X" "$W"
```

Every stage writes `report.json`. Hostplan additionally writes host-query.json,
tiling-report.json, resource-report.json and sdk-binding.json. Build retains
normal/fault build reports, precise source audit/diffs, private OPP/ELF/API/Host
identities and dependency bindings. Each device invocation keeps runner.log,
execution.json, admission, raw outputs and complete numeric differences.

| Stage | Successful state | Device interpretation |
|---|---|---|
| source |STAGE4X_SOURCE_REVIEWED|read-only evidence recheck|
| prepare |STAGE4X_CPU_PREPARED|CPU references/packing only|
| hostplan |STAGE4X_HOST_RESOURCE_FEASIBLE|actual Standalone GetTiling; no compute|
| build |STAGE4X_BUILT_NPU_NOT_RUN|private builds; no compute|
| host |STAGE4X_FORMAL_HOST_ADMITTED|formal GetWorkspaceSize only|
| tile0 |STAGE4X_TILE0_SAFETY_OBSERVED|four inputs, two Poison repeats each|
| smoke |STAGE4X_FULL_SAFETY_OBSERVED|four inputs×Auto/2 Core×two repeats|
| fault |STAGE4X_GUARD_COVERAGE_MATCH|isolated fault OPP; no timing|
| mlp |STAGE4X_FROZEN_BOUNDARY_MLP_RECORDED|exact frozen NPU boundary composition|
| benchmark |STAGE4X_DOWN_RESIDENT_MEASURED|strictly admitted new Down samples|

Safety errors abort immediately. Finite/repeatable numeric discrepancies are
preserved for all cases in that diagnostic stage, remain
NUMERIC_MISMATCH_UNRESOLVED, and never become a Golden match. Tile0 checks4096
active elements with the remaining36864 Poison untouched; Full checks40960.
Both are required; a Tile0 result never stands for full coverage. Strict formal
Benchmark is blocked on any full CPU-Golden difference. There is no tolerance
switch or exploratory timing override. A separate numerical investigation/admission
would be needed if VDEQ16 rounding differs again.

Fault modes2..7 retain the accepted full384-word×32-bit×3-boundary sweep,
three abort-boundary witnesses, and both Arena-side cases. Fault's extra writes
and synchronization are only in its independent OPP. Its outputs cannot become
timing anchors or numerical Golden.

## Cost and complete MLP interpretation

Per sample:2 poisoned checks,5 warmups,30 continuous Execute+Sync calls, timed
tail and new poisoned postcheck. Three rounds reverse sample order in round1;
90 fresh samples per real input. Report all samples, median/p95/min/max,
GetWorkspaceSize prepare time and actual Workspace/Core ownership. No profiler,
Host scan/readback/log/file IO is added to the measured callbacks. Fault cannot
be timed. Internal full Guards remain included in execution time.

Separate ledgers: frozen weight conversion identity, CPU B permutation and
verified prepare wall time; CPU online quantization and Scale-encoding one-shot
observations; initial CopyIn wall time and25,079,808 payload bytes; per-check
CopyOut wall time and81920 output bytes. Transfer wrapper/host-allocation overhead
is included in these wall observations, not mislabeled pure DMA device time.
Readonly/Guard validation traffic and all transfer/preprocessing are outside
resident Execute+Sync. Online NPU preprocessing and device-only timestamps are
NOT_RUN/UNAVAILABLE.

`mlp` joins the accepted real Stage4W Gate→NPU SiLU→NPU Multiply bytes to fresh
Down NPU results. This yields the **complete W8A8 MLP numerical boundary chain**
before residual addition. It is an independent, file-bound execution experiment,
not a freshly integrated single-process/resident full-layer timing. The report
explicitly marks integrated latency NOT_RUN and never adds Gate/Down medians.
Historical Gate≈0.476ms and native Down≈1.98ms are context only, not fresh paired
denominators. Fresh integrated orchestration and model-quality admission remain
separate work; no Draft OM is modified.

## Local tests and failure return

```bash
PYTHONPATH=framework/custom_ops/draft_quant/test python3 -B -m pytest -q -p no:cacheprovider "$S/test"
python3 -B -m pyflakes "$S"
# Within the existing ARM64 CANN Docker; new output only, no OPP install/query/NPU:
bash "$S/local_check.sh" /work/build/stage4x-down-NEW
```

Return the failing phase report and relevant configure/compile/query/build or
runner.log, resource/admission report, execution.json and numeric JSON. Keep raw
actual output, dot_i64/dot_i32, encoded scales and all identities on the server.
CPU fixtures and compile-only packages cannot prove resources or NPU correctness.
Unknown Tiling, resource or ABI differences remain BLOCKED without fallback.
