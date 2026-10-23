# Stage 1: raw INT8 Cube / INT32 ND only

This is the active W8A8 experiment. **CANN compilation, NPU correctness and performance remain NOT_RUN locally.** It does not require output quantization or hardware FixPipe support. Stage 2 (UB Vector output conversion/fusion) and Stage 3 (performance comparison) are not implemented or automatically started.

`--raw-only` selects a separate two-input ABI:

```
DFlashW8A8Accumulator(A: INT8 ND[16,128], B: INT8 ND[128,64])
    -> Y: INT32 ND[16,64]
```

The new private OPP contains only this operator. There is no Scale tensor, uint64 parameter encoding, FP16 SetCType, SetDequantType or SetQuantVector in its hand-written Host/kernel path. A/B are non-transposed. Host C/Bias types are INT32, bias is disabled; requested singleCore/base dimensions are M16/N64/K128 with blockDim=1. The existing strict tiling and UB/L1/L0 capacity checks remain. Host logs its actual SDK tiling and buffer plan.

The previous experimental three-input raw/scaled pair remains available without `--raw-only`, but is not built or loaded by this route. These ABIs must never share a loaded OPP process. Build checks the generated raw source inventory, the single-op definition, installed object/JSON/header identities, and the linked libraries. The dedicated raw runner includes only the accumulator ACLNN header and uses its two-input GetWorkspaceSize call. Existing W8A16 source, OPP, Golden, and Source Lock are unchanged.

## Source-review gate

A fresh raw preflight collects a fixed small list of integer Matmul / datacopy files, version data and tool identities. No recursive SDK scanner is added. Missing paths stay explicit; only known additional files may be supplied via repeatable `--sdk-file`.

The raw review has exactly five claims:

- `chip`: actual CANN9 / 310P3 / dav_2002 branch;
- `integer_cube`: INT8×INT8 integer Mmad, INT32 L0C;
- `nd_io`: non-transposed GM ND inputs and unscaled INT32 ND GM output;
- `tiling`: matching Host types and legal M16/N64/K128 configuration;
- `buffers`: SDK UB allocation, capacities, alignment, lifetime and completion.

All claims still require real source paths, SHA256, line ranges and exact citations. They are initially UNRESOLVED; unsupported evidence blocks the experiment. The raw gate has **no** FP16/Scale/codec/FixPipe claim and does not call the codec validator. An old combined review cannot substitute for the raw review, nor can raw evidence authorize FP16/NZ. `msopgen --version` returning 2 is diagnostic only; it does not establish that generation fails. Generation is actually tested during build.

Source identity has changed with this implementation, so take a new preflight. Keep old reports read-only. Save source-cited review files outside the repository; do not commit private SDK source extracts.

## CPU and device checks

Six synthetic cases cover dense signed INT8, zero, K-boundary basis rows, maximum positive and negative dot products, and exact cancellation. CPU INT64 oracle is recomputed from the hashed input files; a saved oracle must agree. The absolute accumulation bound is 2,097,152, within INT32.

Each case executes twice without timing, with different output poisons. The runner checks x/w/output/workspace Guards, complete input readonly and exact repeat equality. Python requires all 1,024 INT32 outputs to equal the INT64 oracle, recording mismatch count, maximum integer error and first differing index/values. A failed case stops the suite. Success is only `RAW_INT32_ND_PASS`; it says nothing about FP16 output, NZ, full Gate, Draft or performance.

## Server commands

Run after `git pull --ff-only`. Every stage creates a new output directory; no production install is used.

```bash
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
export RAW_RUN=/home/w00949577/qwen35-dflash-run/w8a8-raw-$(date +%Y%m%d-%H%M%S)
bash framework/custom_ops/draft_w8a8_fixpipe/run.sh preflight --raw-only \
  --output-dir "$RAW_RUN/preflight"
```

Complete `$RAW_RUN/review-raw.json` from `preflight/review-template.json` **only after checking the actual integer-path source**. Keep its fixed settings and identity binding; do not copy support conclusions from a different SDK or claim CPU tests establish hardware compatibility. If evidence is missing, return the preflight and required source excerpts before build. The tool does not auto-fill SUPPORTED.

After the raw source review is complete, compile/install only the accumulator:

```bash
bash framework/custom_ops/draft_w8a8_fixpipe/run.sh build --raw-only \
  --preflight "$RAW_RUN/preflight/preflight.json" \
  --review "$RAW_RUN/review-raw.json" \
  --output-dir "$RAW_RUN/build"
```

After `BUILT_NOT_NPU_VERIFIED`, run the six-case integer smoke:

```bash
bash framework/custom_ops/draft_w8a8_fixpipe/run.sh smoke --raw-only --device 0 \
  --preflight "$RAW_RUN/preflight/preflight.json" \
  --review "$RAW_RUN/review-raw.json" \
  --build-report "$RAW_RUN/build/report.json" \
  --output-dir "$RAW_RUN/smoke"
```

Return preflight/build/smoke `report.json`, `preflight.json`, build/install/runner-build logs, and failing case's `runner.log`, `execution.json`, `case.json` and outputs. CLI rejects `--raw-only --layout nz` and `--raw-only --dataset frozen`. No output conversion or timing follows smoke automatically.

CPU tests (including C++ runner syntax with declarations, **not real CANN headers**):

```bash
PYTHONPATH=framework/custom_ops/draft_quant/test python3 -B -m pytest -q \
  framework/custom_ops/draft_w8a8_fixpipe/test/test_w8a8.py
```
