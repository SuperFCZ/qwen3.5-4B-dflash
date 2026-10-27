# Conditional K2560/N64 fusion plan (not implemented)

Proceed only after STAGE3B2_LOCAL_NZ_MATCH. Stage3B-1 already established the
ordered FP32 reduction on real NPU; it does not establish Local Matmul output.
Native Reshape_1 semantics remain a separate UNRESOLVED diagnostic.

## A: preserve accepted partial rounding

For g=0..19, independently reset the INT32 Matmul accumulator, compute K128,
select its own GM UINT64 Scale[64], obtain scaled FP16 Local NZ, convert that
partial to FP32 and add to a persistent FP32 sum in the same NZ layout. After
group 19, convert once to FP16 and use a bit-preserving NZ→ND write.

Target: `FP16(sum_in_order_g FP32(FP16(Z_g * effective_scale_g)))` with an
initial +0 accumulator and each addition rounded to FP32. Strict reference is
Stage3A/3B-1, not a newly generated or relaxed Golden. Full-X activation is
quantized once. Offline group packing may change storage order but not bytes,
values or scale. A single K2560 integer Matmul followed by one coefficient
cannot represent 20 different group coefficients.

## Resource envelope and unresolved minimum

| Storage | Proposed persistent/serial payload |
| --- | ---: |
| Local FP16 partial | 2 KiB |
| FP32 cast temporary | 4 KiB |
| FP32 sum | 4 KiB |
| Separate final FP16 output | 2 KiB |
| Explicit user UB | 12 KiB |
| Conservative headroom | 8 KiB |
| SDK budget at actual UB=256 KiB | at most 236 KiB (241664 B) |
| One A/B/L0C payload | 2 / 8 / 4 KiB |
| Scale vector per group | 512 B, within SDK scratch |

This is a **maximum permissible SDK allocation**, not a minimum-memory proof.
Do not copy the old 248 KiB Stage2B scratch. Query actual UB/L1/L0 resources,
regenerate MatmulApiTiling for Local NZ and the explicit scratch budget, then
inspect GetTransLength, SDK workspace partition offsets, alignment, temporary
conversion buffers, scale offset and all depth/db fields. Every partition must
fit and be disjoint from user buffers. The currently collected output files
alone cannot prove the complete input/LocalWorkspace partition minimum. If
tiling rejects the proposed budget, report that condition; do not silently
reuse a larger scratch or claim the 12 KiB fits.

User GM workspace for partial results can be removed; the platform-required
Matmul system workspace cannot be assumed zero. At fixed N64 the avoided
partial writes+reduction reads are 40+40 KiB. This is a traffic count only.

## Serial lifecycle first

Keep one Matmul object and all SDK scratch alive across groups only if its
documented lifecycle permits the selected Iterate/GetTensorC/End sequence.
Audit whether End resets scale state and whether SetQuantVector/CopyQuantTensor
refreshes the coefficient buffer on every group. Explicitly test two groups
with different scales before twenty; cached group-0 coefficients must fail.

Complete the prior group's L0C→UB transfer before Cast, complete Cast/Add before
reusing its FP16/FP32 temporary, and complete any SDK scratch/Scale DMA before
reset/reuse. Start with existing PIPE_ALL boundaries and synchronous output
contract. Do not assume Iterate alone completes Cube work, remove barriers,
invent event IDs, or overlap a new group with consumption of its predecessor.
Events and buffer lifetimes require actual selected implementation evidence
and device guards/repeat checks. There is no raw prefetch or double-bank change.

Final acceptance must also verify the first/last group, all scale updates,
ordered accumulation and strict final bits against Stage3B-1; no timed phase
is justified before that passes.

## B: independent future numeric scheme

Scaled FP32 partials are not supplied by the demonstrated VDEQ16→half path.
A possible separate scheme would obtain raw INT32 output, convert to FP32,
multiply by each effective coefficient using Vector arithmetic and add in
order, then round once. Its Local INT32 transfer, resource use and arithmetic
need separate validation. It must use Stage3A's effective-scale FP32 reference,
not scheme A's partial-rounding reference. Ideal original FP32 coefficients
are another independent change. No B code or automatic A→B fallback is added.

All fusion compilation, twenty-group device execution and performance remain
NOT_RUN. This plan makes no full-N19456 or Draft performance prediction.
