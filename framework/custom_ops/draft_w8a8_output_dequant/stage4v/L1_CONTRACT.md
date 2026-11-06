# Stage4V L1 accounting correction

The received310P3 query returned GetTiling=0, single16/256/2560,
base16/256/128, depthA1=20, depthB1=10, stepM/N=1, stepKa=20,
stepKb=5, shareL1Size=368640, capacity=1048576 bytes. The old Python
2826240-byte expression counted step dimensions again and added the shared
pool to its own contents. It was wrong; no existing BLOCKED report is edited.

Correct fixed-scope result:

|Input|INT8 basic block|depth|L1 bytes|
|---|---:|---:|---:|
|A1|16×128 =2048|20|40960|
|B1|256×128 =32768|10|327680|
|Total / reported pool| | |**368640**|

`40960+327680 == shareL1Size <= actual L1 capacity` is mandatory. Both a
smaller and a larger unexplained pool report are BLOCKED. We do not use
`max(calculated,reported)` to conceal an unknown consumer.

## Actual installed SDK evidence

All paths below are relative to
`/usr/local/Ascend/cann-9.0.0/aarch64-linux/asc/`. Full SHA256 values and line
ranges are in `sdk-contract.json`, and the existing Host entry checks these
identities before querying the device. Previously server-reviewed hashes remain
identified separately from new `LOCAL_ONLY_NOT_SERVER_VERIFIED` files. New files
must match on the next server run; no SDK checking is removed and no raw SDK
source is committed.

|Source|Lines|Relevant contract|
|---|---|---|
|`impl/adv_api/detail/matmul/policy/matmul_private_modules.h`|55–60|A/B input parameter types and unchanged INT8 transfer types|
|`impl/adv_api/detail/matmul/stage/copy_cube_in/base/copy_cube_in_params_base.h`|121–134,247–265,288–299;396–407,532–558|GM input buffer size is an aligned **basic block**; cache count is GetDepthA1/B1|
|`impl/adv_api/detail/matmul/stage/copy_cube_in/base/copy_cube_in_base.h`|38–42|Passes that basic-block size and depth into CubeInBuffer.Init|
|`impl/adv_api/detail/matmul/resource/cube_in_buffer/cube_in_buffer_utils.h`|95–146|CFG_NORM, GM and no special input sharing select **NORMAL**, not the MDL DOUBLE_BUFFER specialization|
|`impl/adv_api/detail/matmul/resource/cube_in_buffer/cube_in_buffer_normal.h`|29–71|Full cache: depth blocks. Partial cache: (depth−2) cached blocks +2 queue blocks. Small-depth branches also total depth blocks. No second step multiplier|
|`impl/adv_api/detail/matmul/scheduler/base/scheduler_base.h`|61–138|Starts shared pools, initializes A/B, then CO1 and quant handling, then ends shared allocation; Norm2002 excludes the optional BiasScheduler.Init branch|
|`impl/basic_api/kernel_tpipe_base.h`|220–246|Initial shared pool records start and start+shareLens; it does not advance the allocation cursor by shareLens before children allocate|
|`impl/basic_api/kernel_tpipe_impl.h`|211–295,552–593|Queue lengths align to32 bytes, each buffer advances the same pool cursor; InitShareBufEnd restores the recorded pool end. Pool and contents are not additive|
|`include/adv_api/matmul/tiling.h`|36–95,400–403|CFG_NORM; L1-UB cache, bank-conflict optimization, vector ND2NZ padding and IB-sharing routes disabled|
|`impl/adv_api/detail/matmul/stage/copy_cube_out/quant/quant_processor_datacopy.h`|38–54,113–160|Quant Init has no L1 allocation; VDEQ16 coefficient storage is the existing UB LocalWorkspace view|
|`impl/adv_api/tiling/matmul/matmul_constant_tiling_impl.h`|396–431|Installed static tiler illustrates depthA=stepKa and depthB=2×stepKb; step is already represented in depth|
|`impl/adv_api/detail/matmul/param/matmul_shape_tiling.h`|40–63,237–265|SDK debug depth/step checks have architecture/MDL guards; **they are not claimed to execute for dav_2002 Normal**|

The dynamic Host library's internal choice algorithm is not reconstructed from a
newer public document. The decisive allocation proof is the current NORMAL
consumer and TPipe shared-pool behavior, checked against the actual returned
shareL1Size. Static-tiler formulas corroborate units only; that static overload is
not claimed to have produced this dynamic Host result.

The earlier investigation of DOUBLE_BUFFER describes another specialization.
It must not be mistaken for the selected NORMAL path. In particular, B depth10
with stepKb5 does not mean multiplying10 by5 again. The ratio2 is a Host batching
relationship, not a claim that NORMAL physically creates two5-block queues.

## Fixed-scope admission

The correction remains restricted to the existing no-Bias GM INT8 A/B,
Local NZ FP16, base16/256/128, single16/256/2560 and shareMode0 configuration.
A/B basic block byte sizes are already multiples of32; no padding is omitted.
Output and coefficient buffers use L0C/UB, not additional L1. User buffers are UB.
No Kernel configuration or allocation is changed.

Host checks explicitly require:

- positive, bounded integer depths/steps;
- stepM=stepN=1 for the single M/N basic tile;
- stepK values divide the20 basic K blocks and are mutually compatible;
- depth is divisible by its `stepM/N*stepK`, yielding one or two step batches,
  and does not exceed the fixed20-block tensor span;
- L0 double-buffer factors are1 or2, checked separately from these L1 ratios;
- exact L1 accounting equality and actual capacity compliance.

This is a deliberately bounded admission family. It does not assert that every
excluded partial-step configuration is impossible in all SDK modes. Current
values yield A ratio1 and B ratio2 while L0A/L0B may both use factor2; these are
different resources and must not be equated. Unknown/tail-step configurations
remain BLOCKED until independently explained.

Extra Bias, MX Scale tensors, L1 cache/padding paths or unexplained reported bytes
are outside this proof. They are not silently assigned zero cost.
`local_check/l1_contract.cpp` uses actual CANN types and static assertions to confirm
the current NORMAL selector, INT8 B type, no MX tensors and the relevant flags.
It is compile-only, with no Kernel entry, platform simulation or device launch.

## Compatibility with formal Op Host

`op_host/host_common.h` is unchanged. Its L1 check already uses
`2048*depthA1 + 32768*depthB1 <= l1`. Before that check, the generated
`S4VAcceptedT1` contract compares every returned depth, step, sharing field,
capacity and Workspace value to the accepted standalone Host report.
Consequently the new strict Python checks cannot be bypassed by changing a step
or share size in the formal TilingContext: exact-match admission rejects drift.
CPU tests compile the real generated contract and verify these rejections.

UB, Scratch202240, Reserve8192, L0B65536 and all other existing capacity checks
remain. The old L0C conservative accounting is not changed in this fix. No Kernel,
Tiling setter, ABI, Golden, numerical tolerance or Core policy is modified.

## Validation / rerun

Tests include the supplied L1 subset (other integration-fixture values are marked
synthetic), true capacity overflow, unexplained positive/negative sharing deltas,
invalid depths/steps, invalid L0 buffering, exact C++ Host contract drift, and all
existing Stage4V CPU regressions. New SDK files receive strict SHA256 checks through
the existing `host_v.py` loop. Local syntax acceptance is not server/NPU acceptance.

Use a new root and rerun Prepare because Stage4V source identities changed:

```bash
set -euo pipefail
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
V=framework/custom_ops/draft_w8a8_output_dequant/stage4v/run.sh
RUN=/home/w00949577/qwen35-dflash-run/w8a8-full-k-l1fix-$(date +%Y%m%d-%H%M%S)
bash "$V" prepare --device 0 --output-dir "$RUN/prepare"
bash "$V" host --device 0 --prepare-report "$RUN/prepare/report.json" --output-dir "$RUN/host"
```

Expected next gates: `STAGE4V_CPU_PREPARED_QUALITY_UNRESOLVED`, then
`STAGE4V_FULL_K_HOST_FEASIBLE` only if **all** actual SDK/resource checks pass.
Inspect `host/resource-report.json` (`details.l1_allocation`), `tiling-report.json`
and `sdk-binding.json`. This patch does not execute official Build or NPU and
does not retroactively promote any prior BLOCKED report.

Local validation on2026-11-06: **62/62 macOS**, **62/62 Linux ARM64 Docker**
CPU tests passed. The CANN resource-selection static assertion check returned0;
all22 fixed SDK identities/ranges matched locally. Newly added files still require
the server's existing strict hash gate. No real Host re-query, OPP Build or NPU
execution was performed during this fix. The original `local-validation.json`
remains the unmodified initial Stage4V delivery record.
