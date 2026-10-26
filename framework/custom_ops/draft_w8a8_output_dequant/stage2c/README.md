# Stage 2C: real Qwen layer-0 Gate/Up, one K128 group / N64

This independent entry **reuses the accepted Stage2B Kernel, private OPP and ACLNN Runner without building or installing anything**. Stage 1, Stage 2A/2B, W8A16, checkpoints, Frozen Golden and Source Lock remain unchanged. Local work is CPU-only; **Stage2C NPU correctness is NOT_RUN until the server executes this entry**. No timing, profiling, K256/K2560 or full Gate/Draft execution exists here.

## Bound evidence

The default Stage2B directory is the supplied accepted run:

```
/home/w00949577/qwen35-dflash-run/w8a8-stage2b-20261026-100533/
```

The reader requires `build/report.json`, `basic/report.json`, `zero/report.json` and `nonpower/report.json`. It verifies identities, source/config/build/device bindings, installed kernel/header/API/Host libraries, and saved execution guards/readonly/poison/repeat. It recomputes the existing groups' CPU comparisons from their saved outputs; it never re-executes Stage 1 or Stage 2B. Every group must match the same accepted build. Individual report paths may be overridden explicitly with `--build-report`, `--basic-report`, `--zero-report`, `--nonpower-report` if the files have different locations.

Frozen Gate evidence defaults reuse the earlier accepted A5.1 + Off/Raw reports:

```
--a51-evidence /home/w00949577/qwen35-dflash-run/a51-fallback-20261012-063342/suite.json
--prefetch-evidence /home/w00949577/qwen35-dflash-run/gate-raw-prefetch-reverse-20261020-025853/suite.json
```

The existing `fp16_gate_evidence.accepted()` reader validates the C16 layer-0-gate_up Capture, original INT8/NZ/GN scales, Native OM/full-K Golden and accepted W8 artifacts. It performs read-only evidence validation, not another old-operator test. Missing or modified historical artifacts cause a failure, never fabricated replacement data. Both paths can be overridden explicitly; no random build directory is guessed.

## Exact preparation contract

- Read original FP16 X `[16,2560]`; derive **one** scale from all 40,960 elements:
  `scale_A = max(abs(X_full)) / 127` in binary64; all-zero X uses scale 1.
- Use the existing whole-X quantizer: binary64 division, round-to-nearest-even, clip `[-127,127]`. Quantize the whole tensor, then select `A = quantized_X[:,0:128]`. Never recompute scale from the selected K128.
- Select original signed INT8 `Q_W[0:64,0:128]`, independently verify against NZ `[80,1216,16,32]` inverse mapping, and transpose into contiguous ND B `[128,64]` for the unchanged Stage2B ABI.
- Select original FP16 group Scale `scale_W[0,0:64]` from GN `[20,19456]`.
- Form `combined = FP32(FP64(scale_A) * FP64(scale_W))`.
- Reuse the unchanged Stage2A codec candidate:
  `uint64(float32_bits(combined) & 0xFFFFE000) | (1ULL << 46)`.

The codec remains `EXPERIMENTAL_CODEC_CANDIDATE`; matching synthetic cases is not universal precision/exceptional-value validation. No input sanitization, replacement scale, retry codec or adaptive Golden is used.

Three separate references are prepared before the device call:

1. **Ideal W8A8**: exact CPU INT64 dot of quantized A and INT8 B; multiply in FP32 by the original combined FP32 Scale; final FP16 RNE.
2. **Candidate W8A8**: same exact integer dot; multiply in FP32 by the encoded candidate's effective FP32 Scale; final FP16 RNE. This is the strict device acceptance reference.
3. **Original W8A16 K128 CPU**: RN16(original INT8 weight × original FP16 group Scale), multiply original FP16 X, accumulate sequentially over K128 in FP32, final FP16 RNE. This is a deterministic partial-group mathematical reference, **not a claim that CPU accumulation is bitwise identical to a separately exported Native K128 OM**. The full K2560 Native Golden is kept as identity evidence and is never sliced into a partial-sum Golden.

Nonfinite inputs or generated references block this bounded test; the script does not silently introduce saturation or flush rules. Data includes the full-X copy and quantized full-X, selected weight/Scale, effective FP16 W8A16 weights, integer oracle and three references, all SHA-bound. Before comparison, it recomputes quantization and all references and checks the slices against the original frozen tensors.

## Device gate and reports

A fresh process runs only the existing Stage2B Runner on the new small input files. It retains two different output poisons and two untimed executions with Guard, complete input/Scale readonly and repeat checks. `STAGE2C_REAL_K128_MATCH` requires all 1024 outputs to match the candidate reference by FP16 bits for both repeats. The original Stage2B Host plan must also appear in the log.

`report.json` and `analysis.md` separate:

- NPU vs candidate: bits, ULP, max_abs, RMSE, relative L2, cosine, nonfinite counts and first difference;
- NPU vs ideal W8A8, and NPU vs W8A16 K128 CPU: the same diagnostic metrics, without cross-scheme acceptance tolerances;
- effective combined Scale vs original combined FP32 Scale: changed count, signed/absolute error, RMSE, p50/p95/p99, per-channel relative error and relative L2;
- combined FP32 vs the binary64 product: separate coefficient-storage rounding;
- 64 original weight Scales, combined/effective coefficients, encoded words; full-X clipping and endpoint ratios (not conflated), selected-group endpoint ratio, reconstruction errors.

Relative Scale errors exclude zero denominators and report their counts explicitly. Scale coefficients are compared in the same units; effective combined coefficients are not directly compared to uncombined weight Scale. Integer oracle values in Stage2C are CPU evidence, not a new NPU INT32 audit output.

## Server command: reuse, no build

```bash
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only &&
CANN_ROOT=/usr/local/Ascend/cann-9.0.0 \
bash framework/custom_ops/draft_w8a8_output_dequant/stage2c/run.sh \
  --stage2b-root /home/w00949577/qwen35-dflash-run/w8a8-stage2b-20261026-100533 \
  --device 0 \
  --output-dir "/home/w00949577/qwen35-dflash-run/w8a8-stage2c-$(date +%Y%m%d-%H%M%S)"
```

Optional `--prepare-only` produces the three CPU references and metrics without launching the Runner; use a separate new output directory. Every invocation refuses existing output directories and overlap with protected evidence. K range, N range, layer, shape and quantization scheme have no CLI overrides.

Return `report.json` and `analysis.md`. On failure also preserve `execute/runner.log`, `execute/execution.json`, `data/case.json` and `execute/actual-0.bin`, `actual-1.bin`. For Scale investigation, the small `data/scale_f32.bin`, `scale_effective_f32.bin` and `quant_u64_doc_candidate.bin` are sufficient alongside the report. No original artifact should be edited or deleted.

Local regression:

```bash
PYTHONPATH=framework/custom_ops/draft_quant/test python3 -B -m pytest -q \
  framework/custom_ops/draft_w8a8_output_dequant/stage2c/test
```

Passing this single-group test does not establish full K2560 numerical equivalence, 20-group accumulation semantics or performance. Those remain NOT_RUN and are not automatically started.
