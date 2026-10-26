# Stage 3A — K2560/N64 correctness using twenty independent K128 calls

**Not a fused Kernel and not a performance experiment.** This stage reuses the accepted Stage2B private OPP and Runner. It leaves Stage 1/2A/2B/2C, A5.1, Native Golden, Source Lock and historical reports unchanged. Local CPU regressions do not establish Stage3A NPU PASS; device validation remains NOT_RUN until the user runs it on Ascend310P3/CANN9.

## Required candidate evidence

The default accepted Stage2C report is:

```
/home/w00949577/qwen35-dflash-run/w8a8-stage2c-20261026-104943/report.json
```

Stage2B defaults to `w8a8-stage2b-20261026-100533` with its build/basic/zero/nonpower reports. The reader verifies Stage2C's saved candidate outputs, Guard/Readonly/Poison/Repeat, source identities and bindings to the accepted Stage2B build. No earlier device suite is rerun. The original X/q/NZ/Scale files bound by Stage2C remain mandatory. Native OM/Golden files are not required to validate these candidate inputs; they are evaluated separately as optional diagnostics.

## Fixed arithmetic and CPU-only preparation

Only layer-0 gate_up, M16/K2560, N[0:64] is supported. One global activation scale is computed from the full original FP16 X[16,2560], using the established binary64 max_abs/127, all-zero scale=1, nearest-even rounding and clip[-127,127] contract. The complete quantized X is formed before any group is selected. Reference verification may recompute this same global formula; no group estimates its own scale.

For each g=0..19:

- select A[:,128g:128(g+1)] from that same quantized full X;
- select signed q_W[0:64,128g:128(g+1)] and verify its exact NZ inverse;
- use original FP16 `scale_W[g,0:64]`;
- combined coefficient = FP32(FP64(scale_A) × FP64(scale_W));
- use the unchanged Stage2A experimental uint64 codec; preserve both original and effective FP32 coefficients;
- compute exact INT64 dot Z_g; retain the INT32 oracle and two FP32 scaled partials;
- save candidate FP16 partial `RN16(FP32(Z_g)*effective_scale_g)` for strict device comparison.

Group0's input/reference byte hashes must also equal accepted Stage2C group0. Nonfinite references fail preparation; they are not repaired, saturated or replaced.

Four independent full-K CPU references are saved **both as FP32 accumulators and final FP16 results**:

1. **ideal**: add all twenty FP32(Z_g) × original combined Scale products in FP32; final FP16 once. No intermediate FP16 group rounding.
2. **effective**: same using candidate effective Scales; final FP16 once.
3. **partial**: first round each effective product to FP16, convert each partial to FP32, sum in g=0..19 order, final FP16 once.
4. **w8a16**: original FP16 X and RN16(q × original group Scale), one continuous sequential FP32 reduction over K0..2559, final FP16 once. This is a deterministic CPU mathematical reference, not a claim of identical Native reduction order.

All additions use explicitly ordered FP32 operations. `cpu_error_decomposition` compares effective vs ideal for coefficient encoding error; partial vs effective for early FP16 rounding error; and W8A8 variants vs full W8A16 CPU. It reports FP32 differences and final-FP16 bits/ULP/error separately. These relative error norms are **not additive independent costs**.

## Device validation and host aggregation

Each group starts a fresh accepted Stage2B Runner process, which performs two untimed, independently poisoned calls. Every group requires exact candidate FP16 bits, finite output, Guard, Readonly, Poison and Repeat. There are twenty Runner processes / forty checked K128 calls, no warmup or timing.

A numerical mismatch is retained as FAIL_NUMERICAL and the remaining independent groups still run. An execution, Guard or readonly failure stops further device calls; remaining groups are explicitly NOT_RUN_AFTER_EXECUTION_FAILURE. Native audit status never causes that stop.

When all twenty output pairs are available and finite, the tool reads the saved NPU FP16 files, converts them to FP32, adds them in g=0..19 order, and saves both FP32 and final FP16 totals for each repeat. Aggregation occurs on **CPU**, not in a new device Kernel. Both sums must exactly match the partial CPU reference; every individual group must also pass. Cancelling numerical errors in the final sum cannot hide a failed group.

Only these conditions produce `STAGE3A_20_GROUP_MATCH`. No claim is made about a fused implementation, full N19456 execution, Draft or performance.

## Optional Native Full-K semantics audit

Current source review confirms the intended exporter behavior: `a2_export.py` exports standalone `weight_quant_reference`; NativeLinear returns `weight_quant_linear`; the wrapper uses Group128/inner_precise=0 with no bias or activation. The Native OM Runner validates a single FP16 input and FP16 output. **Current source alone is not treated as proof of the frozen server artifact.**

The server audit additionally uses the existing `accepted()`, `validate_om_manifest()` and layout validators, then requires:

- the exact layer-0 gate_up M16/K2560/N19456 case, Group128, same X/q/NZ/Scale identities;
- matching AIR/deployment/OM identity and native layout audit, including GN [20,19456], NK logical weight and NZ storage;
- a SHA-bound `dynamo.pbtxt` in the frozen AIR manifest payload inventory (bounded parser, no TorchAir import or export);
- exactly one WeightQuantBatchMatmulV2 and one output; explicit group/precision/transpose attributes and no optional bias/output-quantization inputs;
- original FP16 X reaches the projection through only identity/reshape views; output reaches NetOutput through only value-preserving views. Bias/Add/Mul/activation is a contradiction; unsupported/ambiguous representation remains unresolved, never guessed;
- explicit FP16 ND [16,19456] output descriptors, frozen runtime IO validation, 622592-byte Golden, four identical repeat/tail/postcheck artifacts and corresponding W8A16/Native parity evidence.

Native numerical implementation/reduction order is still opaque. This audit establishes the same **full-K pure-linear W8A16 comparison scope** for the frozen input, not equality with W8A8 or an implementation-level FP32 instruction proof.

The report always separates:

- `native_semantics_status`: VERIFIED / UNRESOLVED / MISMATCH;
- `native_comparison_status`: COMPLETED / SKIPPED;
- named checks with PASS / FAIL / UNKNOWN, paths, SHA256 and reasons.

Missing or ambiguous evidence gives `NATIVE_COMPARISON_SKIPPED_SEMANTICS_UNRESOLVED`. Explicit contradictions give `NATIVE_SEMANTICS_MISMATCH`. Neither changes candidate acceptance or suppresses a candidate group. Unrecognized graph formats are not inferred from node names or shapes alone.

Only after VERIFIED and complete twenty-group aggregation may the final [16,64] total be compared against the first64 columns of frozen [16,19456] Native Full-K output. This is a cross-scheme diagnostic: relative L2, cosine, RMSE, max_abs, FP16 ULP/bits, never a Native bitwise gate. A full-K Golden column slice is never used as a K128 partial reference. No Native OM is executed or regenerated.

## Server commands

First prepare and verify CPU references and perform the read-only Native audit (no NPU call):

```bash
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
export S3A=/home/w00949577/qwen35-dflash-run/w8a8-stage3a-$(date +%Y%m%d-%H%M%S)
bash framework/custom_ops/draft_w8a8_output_dequant/stage3a/run.sh \
  --prepare-only --device 0 --output-dir "$S3A/cpu"
```

After `CPU_20_GROUP_REFERENCES_READY_NPU_NOT_RUN`, reuse those exact references for the NPU run:

```bash
bash framework/custom_ops/draft_w8a8_output_dequant/stage3a/run.sh \
  --cpu-report "$S3A/cpu/report.json" --device 0 --output-dir "$S3A/npu"
```

Default Stage2C and Stage2B paths are the accepted directories above. `--stage2c-report`, `--stage2b-root`, and individual Stage2B report flags can explicitly override them. Native-only evidence defaults reuse the prior A5.1 fallback and Gate Off/Raw accepted pair; override with `--a51-evidence` / `--prefetch-evidence` if needed. Missing Native-only evidence does not prevent candidate execution. Every output directory must be new; existing references/reports are never overwritten.

Return `report.json`, `analysis.md`, `native-semantics.json` from both stages. For failures preserve `execute/group-NN/runner.log`, `execution.json`, `actual-*.bin` and the corresponding CPU `data/group-NN/case.json`. The full manifest is `cpu/data/references.json`; final sums are `npu/npu-partials-host-sum-{0,1}.fp32.bin` and `.fp16.bin`.

Local tests use CPU/simulated-command fixtures, including rounding cancellation, wrong NZ group, missing/duplicate/reordered partials, Native pure-graph positive/negative checks and Native-independent candidate execution. They do not establish NPU PASS.
