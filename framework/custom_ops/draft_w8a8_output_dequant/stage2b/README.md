# Stage 2B: DFlashW8A8OutputDequant

Independent Ascend310P3 / CANN 9.0.0 / `__NPU_ARCH__ == 2002` K128 experiment. **New operator CANN compilation, Host Tiling, installation and NPU correctness: NOT_RUN locally. Performance: NOT_RUN and no timing entry exists.** Stage 1 Raw, its OPP/Golden, and Stage 2A files/reports remain unchanged.

The branch already contains Stage 2A (`6640e6e`); this experiment imports its existing `dot`, `encode_doc_candidate` and `convert` functions. It does not reimplement or auto-correct the codec. The prior Stage 1 six-case PASS is read and verified offline (identities, saved outputs and CPU INT64 comparisons); its Runner/Kernel is never executed or rebuilt here. Stage 1 success is not evidence of Stage 2B FP16 output correctness.

## Candidate implementation

| Component | Fixed configuration |
| --- | --- |
| Op | `DFlashW8A8OutputDequant` only |
| A/B | signed INT8 ND `[16,128]`, `[128,64]`, non-transposed |
| Scale | UINT64 ND `[64]`, 512 bytes |
| O | FP16 ND `[16,64]`, 2048 bytes |
| Host | candidate `SetCType(GM,ND,DT_FLOAT16)`, `SetDequantType(TENSOR)` |
| Kernel | `Matmul<int8,int8,half>` types, INT32 bias type with bias disabled; GM `SetQuantVector` |
| Tiling | requested singleCore/base M16/N64/K128, blockDim=1, full K128 |
| Workspace | SDK-provided platform capacities; UB local workspace, strict L1/L0/depth and Scale offset bounds |

No explicit weight Cast/Brcb/Mul, no alternative Vector output kernel, no Host C=INT32 retry, and no Raw operator are generated. A failed compiler/tiler remains a failure with its original log. The output route is the SDK's 2002 UB/DataCopy tensor-dequant path identified in Stage 2A; it is not advertised as a separate hardware FixPipe bypass or as eliminating all Vector work.

The only codec attempted is labelled **EXPERIMENTAL_CODEC_CANDIDATE** throughout:

```
encoded[n] = uint64(float32_bits(scale[n]) & 0xFFFFE000) | (1ULL << 46)
```

It truncates the low 13 mantissa bits in the candidate model. Neither bit 46 semantics nor general hardware rounding/overflow behavior are assumed verified. Host DT_FLOAT16 is likewise a candidate until its actual Tiling/device call accepts it.

## Golden and test sequence

Stage 2A's existing five cases are checked for file size/SHA256 and independently recomputed with its unchanged CPU model. Stage 2B copies selected cases into a new directory. It records the original identities, INT64 accumulator range, 64 encoded words, and two independent FP16 goldens:

- ideal: FP32(exact INT64 dot) × ideal FP32 Scale, then FP16 RNE;
- candidate: FP32(exact INT64 dot) × candidate effective Scale, then FP16 RNE.

The candidate Golden is the strict bitwise gate. Ideal-reference bit/ULP/error is reported independently; a difference due to the explicitly chosen codec is not silently overwritten or treated as original W8A16 accuracy. NaN/Inf output fails these bounded finite fixtures. Signed zero is compared by bits; no zero normalization or ULP tolerance is used. An output mismatch reports both references and stops immediately. Raw integer accumulators are CPU oracles here; this three-input FP16 operator does not expose a new NPU INT32 audit output.

Three separate smoke groups:

1. **basic**: original scale=1, scale=0.5, per-channel powers; plus `per_channel_blocks` using identical A/B with block-dependent powers. Original powers depend on `n % 8`, the supplement on `n // 8`; together they help detect channel mapping aliases across N16 boundaries. Both goldens agree for these exact powers. The suite reports output-vs-unscaled diagnostics and strict per-channel-reference matching.
2. **zero**: original alternating zeros/ones plus an all-zero Scale supplement. No automatic transition from basic.
3. **nonpower**: original `mantissa_probe` (Scale=1.0009). Ideal/effective coefficients and reference differences remain visible.

Zero and nonpower each require a complete verified basic report from the exact same Stage2B build, device and upstream evidence. The basic proof's output comparisons are recomputed, not accepted from a PASS string. Each selected case executes twice with distinct output poisons (`0x7f`, `0xff`) whose writeback is checked before launch; full x/w/scale readonly, named Guards, repeated output equality and all 1024 FP16 elements are checked. No warmup, benchmark, Raw rerun, profiling or larger K is performed.

A successful group is `STAGE2B_BASIC_MATCH`, `STAGE2B_ZERO_MATCH`, or `STAGE2B_NONPOWER_MATCH`. These mean the candidate model matched those concrete cases on that run. The codec label stays experimental; a basic match is not universal HARDWARE_VERIFIED and does not assert exceptional-value behavior, full Gate accuracy or performance.

## Server commands

Defaults bind the actual reports supplied for this stage:

- Stage 1: `/home/w00949577/qwen35-dflash-run/w8a8-raw-reviewed-20261026-060329/smoke-20261026-064554/report.json`
- Stage 2A SDK index: `/home/w00949577/qwen35-dflash-run/w8a8-stage2a-20261026-083731/sdk-index.json`
- Stage 2A CPU manifest: `/home/w00949577/qwen35-dflash-run/w8a8-stage2a-20261026-083731/cpu-cases/manifest.json`

Use `--stage1`, `--stage2a-index`, `--stage2a-cases` to explicitly select relocated evidence. No path is guessed from random build names. The SDK index retains `UNRESOLVED_SOURCE_REVIEW_REQUIRED`; the user's bounded experimental authorization permits this new candidate test without rewriting that status or bypassing the old Stage 1 review gate.

Build and install only the new OPP under the new run directory, then compile its dedicated ACLNN runner:

```bash
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
export S2B=/home/w00949577/qwen35-dflash-run/w8a8-stage2b-$(date +%Y%m%d-%H%M%S)
bash framework/custom_ops/draft_w8a8_output_dequant/stage2b/run.sh build \
  --device 0 --output-dir "$S2B/build"
```

Only after `STAGE2B_BUILT_NPU_NOT_RUN`, run basic:

```bash
bash framework/custom_ops/draft_w8a8_output_dequant/stage2b/run.sh smoke \
  --build-report "$S2B/build/report.json" --group basic --device 0 \
  --output-dir "$S2B/basic"
```

Only after `STAGE2B_BASIC_MATCH`, explicitly run additional groups independently:

```bash
bash framework/custom_ops/draft_w8a8_output_dequant/stage2b/run.sh smoke \
  --build-report "$S2B/build/report.json" --basic-report "$S2B/basic/report.json" \
  --group zero --device 0 --output-dir "$S2B/zero"

bash framework/custom_ops/draft_w8a8_output_dequant/stage2b/run.sh smoke \
  --build-report "$S2B/build/report.json" --basic-report "$S2B/basic/report.json" \
  --group nonpower --device 0 --output-dir "$S2B/nonpower"
```

Every stage refuses existing output directories and overlap with protected evidence. Old libraries remain untouched. The new process uses only its private `ASCEND_CUSTOM_OPP_PATH`, bound Runner and installed kernel/header/library identities. Installed OPP path derivation uses the existing CANN9 generic filesystem checks, not the A-specific operator name. Build, install, Runner and NPU errors retain separate phases/returncodes/logs.

## Return only relevant failure evidence

- Always: failing stage's `report.json`.
- Generation/compiler failure: `build/msopgen.log`, `build/build.log`.
- Install/Runner failure: `install.log`, `cmake.log`, `runner-build.log`, `ldd.log` as applicable.
- Tiling/device/numeric failure: `execute/<case>/runner.log`, `execution.json`, and `data/<case>/case.json`. The report contains comparison bits/ULP/max_abs/first difference against both references; preserve `actual-0.bin` and `actual-1.bin` for follow-up inspection.

Do not delete prior evidence or regenerate goldens to get PASS. Host emits `STAGE2B_PLAN` only for an accepted plan and explicit diagnostics for the candidate C type, GetTiling, fixed base/singleCore or resource rejection. A mismatch stops the selected suite; later cases remain unexecuted.

Local verification (CPU and C++ declaration stubs, not CANN):

```bash
PYTHONPATH=framework/custom_ops/draft_quant/test python3 -B -m pytest -q \
  framework/custom_ops/draft_w8a8_output_dequant/tests \
  framework/custom_ops/draft_w8a8_output_dequant/stage2b/test
```
