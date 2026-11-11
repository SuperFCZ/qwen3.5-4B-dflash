# Stage4W: real Draft Layer0 integration boundary

This experiment is based on1cb521f and keeps the production model, Target,
Verify, Draft OM, all historical Kernels/OPPs/Golden and reports read-only.
The accepted operator's exact spelling is `DFlashFullKBNz`. The installed API,
normal Runner and private OPP are discovered and hash-bound through the completed
fresh pairing report, including its Source/Build/Smoke/Benchmark predecessors.

## Actual Draft path

`models/dflash_v1/draft_quantization.py` pins the five-layer W8A16 checkpoint:
`naveenrajk/Qwen3.5-4B-DFlash-W8A16`, revision
`b4bf9f0f7f79bd8589d34b457e136f644c122180`. Config and545870208-byte safetensors
must match the production SHA256 lock. It is distinct from the six-layer FP16
Draft. Offset-binary compressed INT32 words decode four signed INT8 codes along K;
BF16 checkpoint scales are converted to FP16 by the production loader. The
original weights retain Group128 scales. Nothing in this stage replaces them.

`GroupQuantLinear.forward` -> `weight_quant_linear` consumes FP16 activations,
INT8 NK weights and GN FP16 scales through native WeightQuantBatchMatmulV2 with
`inner_precise=0`, K-group128. There is no CPU fallback. A same-source effective
FP16 matrix is `RN16(q_group[n,k] * FP16(checkpoint_scale[n,k//128]))`; it is NOT
the original unquantized training weight.

`models/dflash_v1/modeling_dflash.py` runs MLP after post-attention normalization.
`incremental.py::PackedDraftLayer` concatenates gate then up output channels,
and `DraftGraph.forward` executes post_norm -> gate/up -> FP16 SiLU -> FP16
multiply -> down -> residual add. This experiment ends at the down projection;
the residual, normalization, attention, Target/Verify and vocabulary head are not
replaced. The two FP16 nonlinear boundaries are retained, not folded into a
single FP32 expression.

Capture uses the existing `a2_capture.py`, with a real AscendCL frozen Draft
replay report (fake_acl must be false), real features/KV/anchor, and actual Target
embedding rows. It replays the native eager DraftGraph through layer4 down,
stopping before the vocabulary head. It does not run or modify Target. It is an
actual Draft forward, but is **not** an internal full-OM activation dump/parity
proof. Stage4W consumes layer0 gate/up and down; all A2 inventory hashes, repeat
identities, replay inputs, embeddings and production source identities are
checked. The captured layer0 q/scale arrays must match the real checkpoint
element-for-element, independently of the capture's claimed provenance.

Minimum acceptance dataset: two distinct real activations and replay identities
per context gear, C16 and C64. Random unit fixtures never satisfy this gate.
Existing A2 captures may be reused only if every source/input/checkpoint identity
still verifies; no path substitution or generated activation fallback exists.

## Scale ABI proof and conversion

The accepted B-NZ Kernel `candidate/op_kernel/d_flash_full_kb_nz.cpp` binds
`sg` to19456 UINT64 words and calls `SetQuantVector(sg[tile*256])` once per Tile.
There is no row offset or per-row post-scale. Its A is[16,2560] and B is
[76,80,16,16,32]. Consequently the supported combined coefficient is
`sA * sW[channel]`, **shared by all16 rows**. An arbitrary matrix
`sA[row]*sW[channel]` cannot be represented by this vector unless row scales
coincide (or degenerate all-zero channel). The current ABI does not support
rowwise activation scales. Per-row launches would require another M shape and
Host/resource/Kernel validation; a rowwise epilogue/Scale ABI would be a different
Kernel and rounding contract. Neither is silently introduced here.

The existing whole-X quantizer is reused: one binary64 maxabs over all16×2560
elements, divide by127, binary64 RNE, clip[-127,127], all-zero scale1. Full-K
weight requantization reuses `reference_v.quantize_weight` on the effective FP16
weight, one scale over full K per output channel. `reference_v.encode` is reused:
FP32 combined scale, original low-mantissa truncation and Sign bit46. B mapping:
`NZ[t,k//32,n//16,n%16,k%32] = q_fullk[t*256+n,k]`, with SDK B transpose=true.
Every element is inverse-checked;49,807,360 weight bytes, no padding.

Checkpoint→effective FP16→per-channel W8 is a **new quantization scheme** relative
to original Group128 W8A16. The report retains activation reconstruction error,
clipped/endpoint ratios, per-row range and hypothetical row scales, weight error,
all actual weight scales, combined/effective scales and UINT64 words. Zero,
nonfinite, underflow and overflow cases are checked; no new hardware rounding
model is fitted to outputs.

## Shape and resource scope

| Projection | M at C16/C64 | K | N | Current fixed B-NZ support |
|---|---:|---:|---:|---|
| Layer0 gate/up |16 /16|2560|19456|Yes, new real data needs fresh safety/numeric checks|
| Layer0 down |16 /16|9728|2560|No: separate Host Tiling/resource/Kernel admission required|
| Other layers' gate/up |16 /16|2560|19456|Shape compatible; new weights/activations unvalidated|
| FC |16 /64|12800|2560|No|
| KV fused |32 /80|2560|2048|No|
| Q |16 /16|2560|4096|No|
| O |16 /16|4096|2560|No|

C is the number of **context** rows. The gate/up/noise block remains M16, so C64
does not justify an M64 Gate Kernel or a larger UB budget. Current N256 Gate
Scratch202240B, L0B65536B, Reserve8192B and frozen Host/core/workspace formulas are
unchanged. Every new Gate process re-queries formal TilingContext and must match
the accepted fields and7-Block policy; Standalone Manager8 is irrelevant.

Down has76 K128 groups instead of20,10 N256 tiles instead of76. A is155648B,
B is24,903,680B, channel UINT64 Scale20480B, FP16 output81920B. Its K is32-aligned
and N256 divides N, but this geometric observation proves neither Full-K Tiling
nor L1 depth/UB/L0 validity. It must obtain its own formal resource proof. Do not
pad/split Down into current Gate launches and sum rounded FP16 results: that
changes the Full-K and per-channel numerical contract. No Down device code is
created in Stage4W.

## Numerical and performance acceptance

Each real sample gets an independent exact INT64 dot (range-checked INT32),
original encoded-scale FP32 multiply→RN16 candidate reference, and independent
FP64 effective-FP16 reference. Gate NPU errors are enumerated for **all311296**
elements: row/channel, bits, ULP and absolute error. The historical201 errors
remain historical; neither their count nor coordinates are imposed on new data.
Finite complete output, two poison patterns, readonly, GM/UB/Arena guards,
formal resources, synchronization and bitwise Repeat remain mandatory. A CPU
bitwise mismatch remains NUMERIC_MISMATCH_UNRESOLVED. Model-quality GO has no
new threshold and remains UNRESOLVED, including when candidate arithmetic matches.

The native eager MLP tool checks original W8A16 gate/up, nonlinear product and
down bitwise against their real capture. It separately checks effective-FP16
gate and down against independently computed FP64 dot using the fixed Stage4R
policy (0.002+0.002*abs(ref), relativeL2<=0.001); it does not apply this policy to
W8A8 or tune it to a result. All schemes' gate/up, SiLU, multiply and down outputs
and metrics are saved. Native internal UB guard coverage is not exposed by
PyTorch and is not claimed equivalent to the custom Kernel's full Guard protocol.

The third NPU MLP observation is explicitly **mixed**: actual new W8A8 Gate
output from the independent ACLNN process -> real NPU SiLU/multiply -> original
W8A16 Down. The complete W8A8 Gate/Down scheme exists as a CPU design reference
only. Its CPU SiLU formula is not presented as a bitwise NPU SiLU implementation.

Timing ledgers stay separate:
- CPU activation quant and combined Scale generation/encoding (5/30,3 rounds).
- Static checkpoint conversion and B packing+inverse-check costs (prepare).
- W8A8 Gate resident ACLNN Execute+Sync and GetWorkspaceSize,3×30 new samples.
- Native eager W8A16/FP16 complete MLP and gate/Swiglu/down component timings,
  and mixed suffix timings: Torch dispatcher/allocation+Execute+Sync included.
- Complete W8A8 and integrated mixed MLP latency: NOT_RUN. No addition of
  segmented medians, no fresh speedup using historical0.472195ms, no pure-device
  time without timestamps, no claim that CPU preprocessing is free or online NPU.

## Next26 projections

1. Complete the multi-sample Gate/Up and mixed MLP error ledger, retaining
   real C16/C64 outliers and Scale/endpoint distributions. Establish model-quality
   thresholds independently before any production acceptance.
2. Develop independent Down Full-K K9728/N2560 Host feasibility and then Kernel
   correctness. Keep the current Scale/rounding/Guard/synchronization contract;
   test all output channels and real nonlinear inputs before benchmarking.
3. Validate the remaining four layers' Gate/Up/Down on their actual upstream
   activations, including activations shifted by prior W8A8 projections. Replaying
   only original W8A16 activations cannot establish accumulated model error.
4. Admit Q/O/KV/FC independently by their shapes above. Packed inventory is
   `1 FC + 5*(KV,Q,O,GateUp,Down) =26`; source checkpoint has36 unfused linears.
   C64 changes FC/KV resources and online quantization scope, not Gate/Down M.
5. Only after those numerical/resource gates, design an explicit online NPU
   quant/Scale producer, orchestration/lifetime contract, and fresh complete MLP
   timing. Full Draft OM integration, Target/Verify, token agreement, acceptance
   length and end-to-end Decode are separate, currently NOT_RUN milestones.

No compute Kernel, private OPP, production routing or historical source is
modified by this stage. Source locks include the production entry points and
the reused numeric/layout routines; server reports bind actual historical
Kernel/API/Host/Runner binaries and CANN/Driver identities.
