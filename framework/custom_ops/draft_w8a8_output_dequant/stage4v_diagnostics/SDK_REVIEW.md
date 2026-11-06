# Independent numerical diagnosis: source boundaries

This review does not change Stage4V's Golden, tolerance, failed Bounded report or
Benchmark gates. Its conclusion is **UNRESOLVED** until independent device evidence
is collected. Local source/type review is not a hardware arithmetic observation.

## Documented fields versus software-visible behavior

The official [CANN900 enhanced DataCopy documentation, Table6](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0104.html)
defines the following fields:

|Bits|Meaning|
|---|---|
|0–31|M, interpreted as FP32 multiplier|
|32–35|N encoding: raw0 means1, raw15 means16|
|36|MCB;0 selects direct INT32-to-float conversion;1 selects the documented shift/INT16 path|
|37–45|Offset, used for8-bit output modes|
|46|Sign, used for DEQ8/VDEQ8, not documented as MCB or an FP16 sign switch|
|47|ReLU; vector-transfer behavior also depends on enhancedParams.isRelu|
|48–63|Reserved|

For the existing mask13/bit46 codec, expected MCB=0, raw N=0, Offset=0,
Sign=1, ReLU=0. Raw N=0 must not be described as a zero shift; its decoded value
is1, but the MCB-controlled shift is not enabled by that word. The analyzer decodes
all256 actual words and rejects a mismatch with the frozen codec. This documentation
does not expose the internal multiplier precision or every rounding boundary.

The [CANN900 Cast documentation](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0073.html)
describes precision-conversion modes. Its Atlas inference AI Core float-to-half
entry lists CAST_NONE and CAST_ODD; integer-to-float lists CAST_NONE. These are
**Cast** contracts, not a guarantee that DataCopy VDEQ16 uses the same pipeline.
Our odd-rounding comparison uses the explicit inexact-jamming mathematical
convention. Directed product/INT32 conversion models are hypotheses, not claims
that a matching VDEQ mode can be selected on this device.

## Bound CANN9.0.0 sources

Full relative paths, SHA256 and line ranges are in `sdk-review.json`. The root is
`/usr/local/Ascend/cann-9.0.0/aarch64-linux/asc`. The new server Host tool checks
these exact bytes. New files remain LOCAL_ONLY_NOT_SERVER_VERIFIED until that
check; original frozen SDK identities are independently rechecked as well.
Only metadata and analysis are committed, not SDK source text.

|Source ID / relative path|Lines|What is established|
|---|---|---|
|scheduler — `impl/adv_api/detail/matmul/scheduler/base/scheduler_norm.h`|93–122,164–206,242–257|One Iterate executes the internal K loop before the Local output request|
|scheduler_reset — `impl/adv_api/detail/matmul/scheduler/base/scheduler_norm_base.h`|180–188|First K block initializes C when enPartialSum=false|
|compute — `impl/adv_api/detail/matmul/stage/compute/mmad_compute.h`|26–69|Typed L0 operands and MmadParams dispatch to Mmad|
|mad_device — `impl/basic_api/dav_m200/kernel_operator_mm_impl.h`|368–391|The signed INT8/INT8→INT32 tuple is supported and calls the `mad` intrinsic; no hidden integer arithmetic implementation is exposed|
|base — `impl/adv_api/detail/matmul/matmul_impl_base.h`|601–646|Iterate and Local GetTensorC delegate to the scheduler|
|copyout — `impl/adv_api/detail/matmul/stage/copy_cube_out/copy_cube_out_datacopy.h`|107–136,245–294|Local NZ output; per-N16 VDEQF16 copies versus plain fallback copy|
|quant — `impl/adv_api/detail/matmul/stage/copy_cube_out/quant/quant_processor_datacopy.h`|49–54,113–160|SetQuantVector stores the GM descriptor; coefficient data is copied to UB; the output is configured as VDEQ16|
|quant_scenario — `impl/adv_api/detail/matmul/stage/copy_cube_out/quant/quant_processor_utils.h`|23–46|INT8→half is a quantization scenario; INT8→INT32 is not|
|raw_quant — `impl/adv_api/detail/matmul/stage/copy_cube_out/quant/quant_processor_intf.h`|37–75,124–133|Generic NoQuant processor; parameter update is empty|
|params — `include/basic_api/kernel_struct_data_copy.h`|118–135|Defaults include DEQ_NONE and isRelu=false|
|datacopy_device — `impl/basic_api/dav_m200/kernel_operator_data_copy_impl.h`|487–505|INT32→INT32 with ReLU disabled uses CRMODE_NONE|
|same file|542–572|INT32→half VDEQ16 selects `copy_matrix_cc_to_ubuf` with CRMODE_DEQSCALE_VDEQ16|
|same file|1057–1102|Vector path passes the UB parameter address shifted by5 and sets the address-level ReLU flag; it does not rewrite or decode every multiplier word in C++|

The source does **not** prove that the six discrepancies originate in `mad`, nor
that VDEQ16 performs a particular hidden truncation. In particular, the bit46
constant is not source evidence of MCB=1. Scalar DEQ and vector VDEQ register
setup are different branches; they must not be conflated.

## Raw INT32 feasibility and remaining gates

Planned bounded ABI: original A INT8 ND[16,2560], exact frozen Tile0 B INT8
ND[2560,256], output INT32 ND[16,256]. No Scale argument or re-quantization.
The first output path to check is Local NZ INT32, then a bit-preserving NZ→ND
writeback. The SDK offers this source path. The real Host query may still reject it.

Host configuration: GM ND INT8 A/B, no transpose, VECIN NZ DT_INT32 C,
INT32 disabled Bias, shape16/256/2560, split16/256/128, Scratch202240.
No SetDequantType call is made: the installed enum has SCALAR/TENSOR, not a
fabricated NONE value. Raw admission is independent of proving the FP16 codec's
hardware rounding rule.

|Per-core planned UB|Bytes|
|---|---:|
|Scratch|202240|
|Raw INT32 Local output|16384|
|Arena temporary|512|
|Three redzone pairs|1536|
|Untouched Reserve|8192|
|Total|228864|
|Unallocated remainder under262144|33280|

Scale storage is zero. Output indices are `(n//16)*256 + m*16 + n%16` in Local NZ
and `m*256+n` in GM ND, covering4096 INT32 values /16384 bytes. Actual depth,
step, L1/shared-pool equality, L0A/B/C, transLength and all capacity checks must
pass new GetTiling; L0B remains capped at65536. These are proposed budgets, not a
substitute for the query. Common Tiling fields are compared with the frozen FP16
run and differences reported, never silently normalized away.

The standalone manager's counts do not authorize a new Op's TilingContext.
Initial diagnostic launch would be one Block, within the already accepted policy.
A new formal Op Host must recheck actual Core policy, exact admitted Raw Tiling,
Workspace request and nonoverlapping Arena before any compute launch. Its identity
check is **NOT_RUN** now. Per the requested gate, no compute Kernel, OPP or device
Runner is generated in this delivery.

A subsequent independent Raw device test must keep original input SHA256,
Guard/Poison/Readonly/Repeat and compare **all4096** values with fresh CPU INT64
products, including the six original locations. A Raw match supports investigation
of the conversion path but is not direct observation of the original FP16 Kernel's
L0C. Output type, UB layout and possibly Tiling changed; those confounders must be
recorded. Raw mismatch also does not by itself prove a silicon MMAD fault—input,
layout, resource and repeat checks remain necessary.

If Raw is rejected, preserve the exact rejection. Do not automatically substitute
a Vector Cast+Mul as a VDEQ16 probe. A pure VDEQ test needs a proven legal way to
populate CO1/L0C with known INT32 values; GM→UB conversion is not the same path.
A future known-input probe should span both signs, odd/even neighbors, powers of2,
and half-midpoint neighborhoods under the original Scale words. It must have its
own evidence and cannot repair the Stage4V Golden or enable Benchmark.
