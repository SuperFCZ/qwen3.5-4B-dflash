# W8A8 K128 experiments

**Active stage: [independent raw INT32 ND validation](RAW_INT32.md), selected with `--raw-only`.** This stage builds only the two-input accumulator and has no FP16/SetQuantVector/codec gate. Output conversion and performance remain deferred.

The rest of this document describes the older, separately gated output tensor-dequant candidate; it is not the active raw-only workflow.

Independent Ascend310P3 / dav_2002 / CANN 9.0.0 experiment. **CANN compilation, NPU correctness and performance: NOT_RUN.** CPU fixtures do not establish SDK compatibility. No production W8A16 operator, OPP, Source Lock or Golden is changed.

The question is whether this device supports **INT8 × INT8 → INT32 Cube → per-N-channel FP16 output conversion** through Matmul `SetQuantVector`. This is a different numerical scheme from W8A16. Output scaling must not be described as a bitwise replacement for RN16(weight × scale) before Matmul.

## Fixed scope and ABI

| Item | Contract |
| --- | --- |
| Shape | M16/K128/N64, one core, requested base M16/N64/K128 |
| A | signed INT8 ND `[16,128]` |
| B, first build | signed INT8 ND `[128,64]`, non-transposed |
| B, separate later build | signed INT8 NZ `[4,4,16,32]`, logical `[64,128]`, transpose=true |
| Scale | 64 contiguous GM uint64 parameters, zero offset; encoding must be source-reviewed |
| Raw audit op | `DFlashW8A8Accumulator(a,b,scale) -> INT32 ND [16,64]` |
| Scaled op | `DFlashW8A8Fixpipe(a,b,scale) -> FP16 ND [16,64]` |

Both operators have independent names and are packaged together in a new private OPP per build. ND and NZ use separate build directories and processes. The raw audit has the same integer inputs and fixed shape; its separate launch is not evidence about the scaled kernel's instruction timing. Actual INT32 L0C / integer Mmad and the 310P3 output path require SDK source evidence and successful device tests.

The kernel contains no weight Cast/Brcb/Mul or AntiQuant call. It uses GM INT8 A/B and SDK Matmul. The scaled variant uses `SetQuantVector(GlobalTensor<uint64_t>)`; the raw variant does not. No W8A16 fallback is available. Host records actual base/singleCore tiling, UB workspace, L1/L0 capacities and buffer depths in each `runner.log`; it rejects a different tiling or overflowing allocation.

## Source review before building

`preflight` reads a fixed, small list of version, Matmul and dav_m200 files. It does not walk the SDK tree. Missing files are reported; additional known files may be supplied with `--sdk-file SDK_RELATIVE_PATH` (repeatable). Source snapshots, paths, SHA256, tool identities and this experiment's source hashes are saved only to the new server output directory.

Preflight deliberately returns `UNRESOLVED`, even if it finds the symbols. It writes `review-template.json`. Complete a separate `review.json` using the **actual CANN 9.0.0 source**, with reviewer, findings and exact source line citations. Do not simply mark every field supported. An explicit unsupported required claim blocks the build with `BLOCKED_UNSUPPORTED`; unknown claims remain `UNRESOLVED`. Review ND independently; NZ may remain unresolved until its own review. Private SDK extracts and review artifacts must not be committed to Git.

Required claims cover the chip branch, INT8 Cube / INT32 L0C, Host output type, tensor quantization, parameter encoding, FP16 conversion, alignment/lifetime and memory requirements. In particular:

- `host_scaled_c_dtype` has **no default**. Establish whether this SDK's `MatmulApiTiling.SetCType` requires `DT_INT32` or `DT_FLOAT16` for the half-output quant path. Do not infer it from another SDK example. Kernel ABI remains half in either case. Raw Host/kernel C is INT32.
- The codec adapter requires a reviewed low32 IEEE FP32 scale field, explicit high32 fixed flags (including signed output / zero offset as applicable), stored and effective mantissa precision, multiplication rounding, overflow, subnormal and N64 vector layout. No codec is selected automatically. `constant_u64` must be an integer and cannot overlap low32. This adapter supports 10/23 mantissa bits with low-bit truncation, positive finite scales and signed accumulators; another actual encoding requires a separate adapter change, not a guessed setting.
- `product_rounding` distinguishes FP32 product then FP16 RNE from exact product then FP16 RNE. Overflow and subnormal policies are separate reviewed fields. These supported **reference recipes are not hardware support claims**. If the source does not resolve a field, report it as unknown and return the precise missing implementation. Do not tune the reference to observed outputs to obtain PASS.
- The NPU tests subsequently check the reviewed prediction. Unexpected output is a numerical failure with raw output files and bit patterns, never an automatic recipe adjustment.

Build revalidates citations, original SDK files, snapshots, version/tool/source hashes. It installs only under its new output's `opp/`, validates both nonempty objects/JSONs/ACLNN headers, links a fresh runner against that OPP, and records installed artifacts and libraries. It never installs into the toolkit or a production vendor directory.

## Independent numerical contract

Synthetic cases: scale 1, scale 0.5, channel-dependent powers, all-zero A, signed INT8 extremes, scale precision probe, positive overflow, and subnormal output. Negative values and mixed channel scales are retained. Every case runs raw INT32 twice before scaled FP16 twice, with distinct poisons, guards, full input/scale readonly checks, device-error logs and exact repeat comparison. INT32 mismatch stops before the quantized-output launch. Any failure stops the suite; no performance loop exists.

CPU oracle uses INT64 dot products, bounded here by 2,097,152, hence exactly representable in INT32 and FP32. Output records separately:

1. NPU raw INT32 vs exact INT64, mismatch count and integer error.
2. NPU FP16 vs source-reviewed **encoded/effective scale** reference: strict bits, ULP and absolute error. No tolerance.
3. NPU FP16 vs ideal FP32 combined-scale product followed by FP16 RNE: diagnostic comparison only. Nonfinite counts are explicit; nonfinite error metrics are null rather than misleading finite numbers. Reviewed expected infinity still requires the exact sign/bit pattern.
4. For frozen data, W8A8 ideal/NPU vs independent W8A16 **K128** CPU reference: max_abs, RMSE, cosine, relative L2 error and bit diagnostics. Cross-scheme differences do not alter either scheme's acceptance standard.

Frozen preparation reuses the existing A5.1/Off-Raw evidence validator, original X, Q, NZ, GN scale, Native OM / full-K Golden identities. NZ inverse for the selected N64/K128 is checked. It never uses full-K Golden columns as a partial-K reference.

Activation quantization is fixed: compute `scale_A=max(abs(X_full))/127` in binary64 over **all `[16,2560]`**, use scale 1 for all-zero X, binary64 division, nearest-even rounding, then clip `[-127,127]`. Quantize the whole X before taking its first K128. Report clipping and endpoint saturation ratios for full X and endpoint ratio for the selected group. The first group's combined output coefficients are `FP32(FP64(scale_A)*FP64(scale_W_half[n]))`. Only positive finite coefficients supported by the reviewed adapter are accepted. Zero/negative scales are rejected, never replaced.

W8A16 partial reference independently uses RN16(INT8 × FP16 group scale), original FP16 X, sequential FP32 K128 accumulation, final FP16 RNE. It is explicitly a CPU partial reference, not a newly invented Native Golden.

## Server stages

Run from the repository after `git pull --ff-only`, with NumPy available. Every command uses a new directory. These commands do not run Gate/Draft/Simulator or timing. `MODEL_PYTHON` may select the server interpreter. Keep the `RUN` variable in this shell for subsequent stages.

First, source evidence only:

```bash
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
export RUN=/home/w00949577/qwen35-dflash-run/w8a8-fixpipe-$(date +%Y%m%d-%H%M%S)
bash framework/custom_ops/draft_w8a8_fixpipe/run.sh preflight \
  --output-dir "$RUN/preflight"
```

Return `preflight/report.json`, `preflight/preflight.json`, and relevant SDK line ranges before completing the review. If a required file is missing, repeat preflight into a **new** directory with explicit `--sdk-file` paths; do not rescan the SDK. Keep the reviewed file beside, not inside, the source tree. For the following commands, `PF` and `REVIEW` refer to the completed, source-cited review and its original preflight:

```bash
export PF="$RUN/preflight/preflight.json"
export REVIEW="$RUN/review.json"
# After the ND source review is complete:
bash framework/custom_ops/draft_w8a8_fixpipe/run.sh build --layout nd \
  --preflight "$PF" --review "$REVIEW" --output-dir "$RUN/build-nd"
# Only a successful isolated build permits this smoke:
bash framework/custom_ops/draft_w8a8_fixpipe/run.sh smoke --layout nd --dataset synthetic \
  --preflight "$PF" --review "$REVIEW" --build-report "$RUN/build-nd/report.json" \
  --device 0 --output-dir "$RUN/synthetic-nd"
```

After `SYNTHETIC_ND_PASS`, frozen ND is optional and uses the **same reviewed build**:

```bash
bash framework/custom_ops/draft_w8a8_fixpipe/run.sh smoke --layout nd --dataset frozen \
  --preflight "$PF" --review "$REVIEW" --build-report "$RUN/build-nd/report.json" \
  --synthetic-proof "$RUN/synthetic-nd/report.json" --device 0 \
  --a51-evidence /home/w00949577/qwen35-dflash-run/a51-fallback-20261012-063342/suite.json \
  --prefetch-evidence /home/w00949577/qwen35-dflash-run/gate-raw-prefetch-reverse-20261020-025853/suite.json \
  --output-dir "$RUN/frozen-nd"
```

NZ requires its own source-reviewed layout claim and ND synthetic proof from the same preflight/device/settings. If the review changes, save a separate review file; do not overwrite the ND review used by existing reports. `NZ_REVIEW` is that separately completed review:

```bash
export NZ_REVIEW="$RUN/review-nz.json"
bash framework/custom_ops/draft_w8a8_fixpipe/run.sh build --layout nz \
  --preflight "$PF" --review "$NZ_REVIEW" --nd-proof "$RUN/synthetic-nd/report.json" \
  --device 0 --output-dir "$RUN/build-nz"
bash framework/custom_ops/draft_w8a8_fixpipe/run.sh smoke --layout nz --dataset synthetic \
  --preflight "$PF" --review "$NZ_REVIEW" --build-report "$RUN/build-nz/report.json" \
  --nd-proof "$RUN/synthetic-nd/report.json" --device 0 --output-dir "$RUN/synthetic-nz"
# Only SYNTHETIC_NZ_PASS permits frozen NZ:
bash framework/custom_ops/draft_w8a8_fixpipe/run.sh smoke --layout nz --dataset frozen \
  --preflight "$PF" --review "$NZ_REVIEW" --build-report "$RUN/build-nz/report.json" \
  --nd-proof "$RUN/synthetic-nd/report.json" --synthetic-proof "$RUN/synthetic-nz/report.json" \
  --device 0 --output-dir "$RUN/frozen-nz"
```

Return each `report.json`, build/install/cmake logs, and failing case's `case.json`, raw/quant `execution.json`, `runner.log` and output binaries. Passing a command does not transfer PASS to another layout, SDK, OPP or model. NPU commands verify the prior evidence instead of trusting a status string.

## Deferred decision

K256 / twenty groups, performance, full Gate, Draft and Simulator are **NOT_RUN and not implemented**. If K128 succeeds, evaluate each group's FP16 rounding before any FP32 reduction: it differs from keeping an unrounded scaled partial. Two separate K128 outputs alone cost 2 × 16 × 64 × 2 = 4096 B of GM writes; a separate accumulator adds GM reads and its own output traffic. At 20 groups, FP16 partial writes are 40,960 B per N64 tile before reads/reduction. These are traffic counts, not latency predictions. Do not assume scaled FP32 FixPipe output or preservation of the on-chip accumulator across scale changes.

Local regression (CPU / runner syntax stubs only):

```bash
PYTHONPATH=framework/custom_ops/draft_quant/test python3 -B -m pytest -q \
  framework/custom_ops/draft_w8a8_fixpipe/test/test_w8a8.py
```
