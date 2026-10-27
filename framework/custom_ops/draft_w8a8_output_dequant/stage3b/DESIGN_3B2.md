# Stage 3B-2: design only; blocked on Stage 3B-1 NPU correctness

No fusion kernel is implemented or run here. Scope is M16/K2560/N64, with the
same full-X activation quantization and 20 independent K128 Group128 scales.
Neither a full N19456 implementation nor performance conclusions are enabled.

## A: preserve accepted FP16 partial semantics first

For group g, compute an independent INT32 integer accumulator Z_g, apply that
group's encoded coefficient through the accepted enhanced DataCopy VDEQ16
route, obtain FP16 partial P_g, convert P_g to FP32 and add in order to S.
Initialize S to +0 and finally convert S once to FP16. Stage 3A's partial
reference and Stage 3B-1's output remain the strict references.

The proposed dataflow is **not** a standalone Vector Cast+Mul dequantizer and
is **not** the 910B Fixpipe specialization. On 310P3, IsNeedUB selects the
MatmulQuantProcessor/DataCopy implementation. Any Vector Cast/Add in the
proposal accumulates already dequantized FP16 partials; any NZ-to-ND Vector
operations must be distinguished from application of Scale.

### Local output is a separate contract

Do not mechanically replace GM GetTensorC with LocalTensor GetTensorC. The
previously supplied CANN excerpts distinguish Local ND CopyOutNZ2ND calling
CopyCo12Co2WithoutQuant from Local NZ CopyOutNZ2NZ calling the quantized path.
Those overloads and their exact architecture/template conditions must be
revalidated against the installed, hash-bound files collected by Stage 3B.
Merely finding a VDEQ16 symbol does not establish the selected overload.

A promising **UNRESOLVED** route is FP16 Local NZ output with quantization,
pointwise FP32 accumulation in the same NZ layout, then one final bit-preserving
NZ-to-ND layout conversion. Before writing a 20-group kernel, a single K128
Local NZ probe must prove encoded per-channel Scale consumption, descriptor
and stride mapping, all 1024 FP16 bits and guard/readonly/repeat behavior.
If the actual overload cannot quantize Local output, stop this proposal rather
than treating an unscaled output as a valid partial.

### Memory, tiling and events

- A can be prepacked offline as INT8 `[20,16,128]`, B as INT8 `[20,128,64]`, and
  Scale as UINT64 `[20,64]`. These are permutations/slices of the accepted
  Stage 3A data; activation quantization is performed once over full X, never
  recalculated per group. The A K2560 stride must not accidentally be passed
  as a contiguous K128 tile.
- Reset the integer Matmul accumulator for each group. Do not use integer
  partial-sum accumulation across groups with different scales. Preserve S in
  a distinct FP32 UB allocation outside Matmul's internal workspace.
- Provisional user UB: FP16 partial 2 KiB, conversion 4 KiB, FP32 sum 4 KiB,
  final output 2 KiB, plus scale/layout scratch if required. The previous
  Stage 2B 253952-byte SDK UB budget cannot simply be reused with another
  12 KiB allocation. Query real platform capacities and subtract all user and
  SDK requirements before generating a new, independent fusion tiling.
- One logical M16/K128/N64 Cube block contains 2 KiB L0A, 8 KiB L0B and 4 KiB
  L0C data. These are payload estimates, not a claimed SDK allocation: buffer
  depths, L1 staging, alignment and any double buffering must be measured from
  the selected SDK tiling and implementation.
- Verify completion of GetTensorC/End and each scale/local workspace lifetime
  from the actual implementation before reuse. Iterate must not be assumed
  synchronous. Do not remove PIPE_ALL or substitute guessed event IDs. First
  use a serial group schedule; overlap is a later independent variable.

Avoiding 20 GM FP16 partial writes and the subsequent reduction's reads removes
40 KiB + 40 KiB of traffic per N64 tile. This is a byte-count estimate, **not**
a latency estimate or evidence of speedup.

## B: separate FP32 scaled-partial precision candidate

Do not assume VDEQ16 outputs scaled FP32. It produces FP16 in the validated
path. A distinct possible route is unscaled INT32 Local output, INT32→FP32,
FP32 channel multiplication and ordered FP32 accumulation. Its Local output
type/layout, transfer and resource contracts require their own verification.
This scheme explicitly uses Vector output scaling and must be named as such.

Start with the **effective encoded scale** used by Stage 3A, so the sole numeric
change is elimination of each group's FP16 rounding. Compare with Stage 3A's
effective-scale FP32 reference. Using the original ideal FP32 coefficient is
another independent variant. CPU Reference must preserve separate FP32 Mul
and Add rounding; do not silently replace them with a fused arithmetic rule.
K128 INT8 dot products fit within the exact integer range of FP32, but this
does not prove later products or accumulated outputs are bitwise equivalent
to scheme A. B has its own correctness target and cannot relax A's gate.

In general, sum_g(Z_g * Scale_g) cannot be replaced by one K2560 integer
Matmul and one Scale. Neither shared output shape nor equal group size makes
the 20 coefficients equal.

## Decision gates

1. Stage 3B-1 must return STAGE3B1_REDUCTION_MATCH on real 310P3.
2. Review the four fixed SDK files and record actual overloads, intrinsic,
   architecture guards, SHA256 and line citations. No source review is
   automatically granted by this collector.
3. Verify one-group Local output quantization and layout before a fused loop.
4. Only then implement 20-group scheme A in a new OPP, validate against saved
   Stage 3A/3B-1 results, and consider B separately.

Native Full-K semantics remain UNRESOLVED independently (Reshape_1 descriptor).
That optional cross-scheme diagnostic is not a blocker for these candidate
correctness gates. All fusion compilation, NPU and performance stages are
NOT_RUN.
