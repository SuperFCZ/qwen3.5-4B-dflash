# Stage 4I — N128 Host-only Tiling/resource feasibility

No N128 compute Kernel, OpDef, device compilation, OPP installation, stream,
device compute buffer, launch, benchmark or profiler is provided. The only
executable is a standalone C++ Host program linked to `tiling_api`, `platform`
and `ascendcl`. Local CPU tests use test-only platform/SDK doubles; they are not
server measurements or a fallback in the production probe.

## Real platform route and SDK identity

The selected device is initialized with ACL, its actual SoC must be
Ascend310P3, then the program calls **PlatformAscendCManager::GetInstance()**
with no arguments. A null result blocks the experiment. GetCurNpuArch() is
called on the returned PlatformAscendC object and must report 2002. UB/L1/L0,
core counts and system workspace come from that object. No customSocVersion,
manual PlatformInfo, virtual SoC or hardcoded resource capacity is constructed.
Requested limits are experimental budgets, checked against measured capacities.

The user's installed header has canonical location
`aarch64-linux/asc/include/utils/tiling/platform/platform_ascendc.h`, SHA256
`44cf3f35292f15a8735b0d43b7e88198ca15d7b4a3bf44719eb3f099697f7f2a`:
GetCurNpuArch at line 141, no-argument GetInstance at 151–161. The public include
alias and actual compiler dependency must resolve to this identity. Other
already-reviewed fixed SDK headers and CANN version records are hashed, not
copied. All actually compiled SDK dependencies and linked libraries are also
recorded in the private output. No SDK original text is committed.

Official API context (the installed SDK and real run remain authoritative):
[PlatformAscendCManager](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_1039.html),
[Matmul Tiling](https://www.hiascend.com/doc_center/source/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_0671.html).
MatmulApiTiling is the single-core API; the probe does not substitute the
MultiCoreMatmulTiling class or guess a SetDim method. Returned usedCoreNum and
singleCore sizes must confirm the requested one-core scope.

## Three fresh independent SDK queries

| Case | Shape / OrgShape M,N,K | FixSplit M,N,K | Scratch requested | User UB |
|---|---|---|---:|---:|
| N64 control | 16,64,128 | 16,64,128 | 239104 B | 12288 B |
| T1 | 16,128,128 | 16,128,128 | 226816 B | 24576 B |
| T2 | 16,128,128 | 16,64,128 | 226816 B | 24576 B |

Each constructs a new MatmulApiTiling using the same real platform. A/B are GM
INT8 ND, B nontransposed; C is VECIN NZ FP16; BiasType INT32 but bias disabled;
DequantType is TENSOR. There is no precision/configuration fallback.
All setter return values are preserved (null means a void API, not a success
code). A negative setter return stops that candidate before GetTiling. T2 is
only queried if its own settings are accepted; GetTiling and actual returned
base/singleCore shapes determine legality. It is an explicitly requested
candidate, not an automatic fallback or a changed interpretation of T1.

Raw TCubeTiling serialization, size and SHA256 are saved for every successful
GetTiling. Extracted fields include capacities, shapes, depths, db factors,
transLength, steps, sharing fields, and bias. Missing required getters are null
and block resource admission instead of being assumed zero. CANN compilation
or SDK failures keep their actual logs/return codes; no Kernel is generated.

## Resource policy

N128 budget: 226816 Scratch + 24576 user buffers + 2560 guards + 8192 reserve
= 262144 B. The 8192 B reserve is never repurposed. N64 is actually queried
with its original budget, not populated from old expected Tiling.

Checks require real SDK GetTiling success **and**:

- exact requested base and singleCore dimensions, one core and no bias;
- transLength aligned to32, and transLength + logical-N uint64 scale storage
  within Scratch (1024 B for both N128 candidates, conservatively including T2);
- normal nontransposed INT8 ND two-B-layout workspace and full C staging bound
  within Scratch; user-owned Local NZ Partial/Final storage is already budgeted;
- M/N aligned to16 and INT8 K to32 for physical payload bounds;
- actual depths, db and step factors applied to L1/L0 bounds; returned shared
  L1/L0C reservations are additionally charged conservatively;
- **nondefault shareMode blocks admission**. Stage4I.1 binds the dav_2002
  device accessor: GetShareUbSize returns zero, independently of the preserved
  raw Host field. The coefficient storage is a view inside caller Scratch,
  not an additional allocation. See SHARED_UB_REVIEW.md and the hash-bound
  shared-ub-contract.json; a changed consumption header blocks the probe.
  Capacity bounds are conservative and can reject a layout
  the SDK accepted; such a rejection is not a hardware UNSUPPORTED verdict.

This is Host resource feasibility, not observed live high-water memory or
proof of buffer/event safety in a future Kernel. The real system workspace
size is recorded; ACLNN workspace allocation and execution remain NOT_RUN.
No existing per-core Arena formula is changed.

T2 needs two N64 base outputs for one logical N128 group. CPU indexing proves
where those subtiles would fit, but does not certify the future Iterate/
GetTensorC sequence. Logical 20-group Matmul count is 3040 for N128; that is
not a claim that T2 has only3040 base-tile computations or instructions.

## Read-only frozen-data layout check

The accepted Stage4B report, unchanged source identities and case file hashes
are checked using existing readers. Saved full-X INT8 A is reused without
requantization. Every one of 152×20 groups is checked:

- original INT8 NZ `[80,1216,16,32]` inversely maps to the exact original Q_NK
  slice for 128 consecutive channels and K128;
- new in-memory B `[152,20,128,128]` equals the channel concatenation of the
  corresponding two accepted N64 B tiles;
- Scale words `[152,20,128]` concatenate the two accepted encoded N64 vectors;
  each half is checked through the unchanged 64-channel codec against the
  original GN Scale and saved full-X activation scale;
- Local NZ indexing is `(n//16)*256 + row*16 + n%16`, now eight N16 blocks;
- GM output index remains `row*19456 + tile*128 + n`, with complete bijective
  coverage; column9728 aligns exactly at tile76;
- group order remains0..19. No numerical Golden is regenerated or rewritten.

Only streaming hashes of the prospective packed B/Scale are emitted; no N128
execution input binary is written. Full FP16 output and its original Golden
remain untouched. This phase checks mapping/encoding, not N128 computation.

## Decision and server command

`STAGE4I_N128_HOST_TILING_FEASIBLE` requires the live N64 control, CPU frozen
layout and **at least one** of T1/T2 to pass all checks. Each candidate keeps its
own result; a failed T1 is never called successful because T2 passed. Otherwise
`STAGE4I_N128_BLOCKED` reports exact failed checks or the compile/query error.
Either status leaves N128 Kernel compilation/execution, Benchmark and Profile
NOT_RUN. A subsequent N128 Kernel requires explicit admission.

```bash
set -euo pipefail
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
S4I=/home/w00949577/qwen35-dflash-run/w8a8-n128-host-$(date +%Y%m%d-%H%M%S)
bash framework/custom_ops/draft_w8a8_output_dequant/stage4i/run.sh \
  --device 0 --cann-root "$CANN_ROOT" --output-dir "$S4I"
```

This runs CPU layout validation, CMake/c++ Host-only compilation, then the
Host resource query. It does not call msopgen, a device compiler, Runner or
kernel launch. Return `tiling-report.json`, `resource-report.json`,
`layout-report.json`, `sdk-binding.json`, `comparison.md`; on failure also
`configure.log`, `compile.log`, `ldd.log` or `host-query.log` as applicable.
Raw `host-query.json` and per-candidate Tiling binaries stay in the new output.
Never overwrite an old output folder or edit its status.

Local checks:
`PYTHONPATH=framework/custom_ops/draft_quant/test python3 -B -m pytest -q framework/custom_ops/draft_w8a8_output_dequant/stage4i/test`.
Tests include actual probe.cpp compilation against CPU doubles, null/incorrect
platform rejection, SDK setter/GetTiling refusal, resource overflow/missing
fields, codec/layout corruption, Gate/Up alignment and Local NZ/T2 indexing.
They do not count as CANN Host execution or real platform feasibility.
