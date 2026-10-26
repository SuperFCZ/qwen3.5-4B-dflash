# Stage 2A: 310P3 W8A8 Matmul output dequantization

Scope: CANN 9.0.0, Ascend310P3, one M16/K128/N64 group. **This is a CPU/source-inspection gate only: NPU NOT_RUN.** The Stage 1 `draft_w8a8_fixpipe/raw/` operator and previous build/smoke artifacts are untouched.

```bash
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
export RUN=/home/w00949577/qwen35-dflash-run/w8a8-stage2a-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RUN"
python3 framework/custom_ops/draft_w8a8_output_dequant/stage2a.py sdk-index --cann-root "$CANN_ROOT" --out "$RUN/sdk-index.json"
python3 -m unittest discover -s framework/custom_ops/draft_w8a8_output_dequant/tests -v
python3 framework/custom_ops/draft_w8a8_output_dequant/stage2a.py cpu-cases --out "$RUN/cpu-cases"
```

The SDK index only lists known paths, symbol hit line numbers, and SHA256. It never exports proprietary source contents or declares support automatically. On the real device, inspect the exact CANN code at returned lines and record local source citations with separate review status for:
1. Selected dav_2002 / `IsNeedUB()`/DataCopy specialization for INT8 x INT8 -> half (not presumed FixPipe).
2. `SetDequantType(DequantType::TENSOR)`, `SetQuantVector(GlobalTensor<uint64_t>)`, exact C type used by Host tiling.
3. GM uint64 vector per-column mapping and coefficient encoding, including bit 46 and effective mantissa precision.
4. FP16 rounding/overflow/subnormal behavior.
5. Buffers/workspace/synchronization/lifetimes for M16 K128 N64.

Public CANN 9.0 examples describe a possible `(f32bits & 0xFFFFE000) | (1ULL << 46)` representation. The CPU fixture labels this strictly as `doc_candidate`, not a verified 310P3 encoding. It computes `Z=int64(A)@int64(B)` and FP16 of FP32(Z)*scale, with per-N-channel scales, five deterministic cases and hashes. Tests passing does **not** establish the installed SDK's output semantics.

Once all five evidence claims are confirmed, create an independent 3-input `DFlashW8A8OutputDequant` op, fresh OPP/runner/reports, and NPU smoke with positive/negative values, distinct per-channel and zero scales, input readonly, guards, poisoned outputs, repeats and numerical reference. Leave K2560, 20 group fusion and performance deferred until correctness passes. Never write SDK snippets or device logs to public GitHub.
