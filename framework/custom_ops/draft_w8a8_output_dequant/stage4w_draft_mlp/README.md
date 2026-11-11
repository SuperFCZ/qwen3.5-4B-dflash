# Stage4W real Draft Layer0 Gate/Up and MLP evidence

Independent tooling only; no historical Kernel, OPP, Golden, model or OM edits.
See [INTEGRATION_REVIEW.md](INTEGRATION_REVIEW.md) for numerical and resource
boundaries. **Gate/Up is supported by the frozen operator. Down is not.** Native
W8A16, same-source FP16 and an explicitly mixed W8A8-Gate/W8A16-Down path are
observed separately; complete W8A8 MLP device correctness/performance stays
`NOT_RUN_DOWN_KERNEL_UNSUPPORTED`. Quality remains UNRESOLVED.

The current accepted0.472195ms is a historical frozen-input measurement, not an
estimate for these new activations. Every sample receives a new CPU reference;
the old201 one-ULP differences are not used as an allowance for new results.

## Inputs and capture

Use a Python3.9+ server environment with NumPy, the working torch/torch_npu and
real native WeightQuant support already used by the Draft. Stage4W's CPU readers
do not require torch or safetensors. The on-device capture does. Missing actual
checkpoint, real replay reports or operators BLOCKS; there is no random/CPU
substitute. The checkpoint must be the production-pinned five-layer W8 model.

Set these to **existing actual paths**, not generated synthetic fixtures:

```bash
export AI_RUN_DIR=/home/w00949577/qwen35-dflash-run
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
export DRAFT_DIR=/actual/pinned/W8A16/checkpoint
export TARGET_DIR=/actual/original/Target/checkpoint
export C16_REPLAY_0=/actual/sample0/C16/private.json
export C16_REPLAY_1=/actual/sample1/C16/private.json
export C64_REPLAY_0=/actual/sample0/C64/private.json
export C64_REPLAY_1=/actual/sample1/C64/private.json
S=framework/custom_ops/draft_w8a8_output_dequant/stage4w_draft_mlp
R=$(mktemp -d "$AI_RUN_DIR/real-draft-mlp-XXXXXX")
PAIR=/home/w00949577/qwen35-dflash-run/w8a8-bnz-fp16-paired-20261110-120437/benchmark/report.json
```

The real replay reports are produced by the existing Draft replay/debug flow;
their `fake_acl=false`, snapshot files, contract, original OM identity and gear
must already be valid. The wrapper replays actual Draft activations. It does not
create features or fake a C64 input by padding a C16 sample. Supply the original
export's `--feature-layers` if it differs from the Draft's selected layers.

```bash
set -euo pipefail
python3 -B "$S/capture_real.py" --draft-dir "$DRAFT_DIR" --target-dir "$TARGET_DIR" --replay-report "$C16_REPLAY_0" --context-rows 16 --output-dir "$R/c16-0"
python3 -B "$S/capture_real.py" --draft-dir "$DRAFT_DIR" --target-dir "$TARGET_DIR" --replay-report "$C16_REPLAY_1" --context-rows 16 --output-dir "$R/c16-1"
python3 -B "$S/capture_real.py" --draft-dir "$DRAFT_DIR" --target-dir "$TARGET_DIR" --replay-report "$C64_REPLAY_0" --context-rows 64 --output-dir "$R/c64-0"
python3 -B "$S/capture_real.py" --draft-dir "$DRAFT_DIR" --target-dir "$TARGET_DIR" --replay-report "$C64_REPLAY_1" --context-rows 64 --output-dir "$R/c64-1"
python3 -B "$S/prepare_real.py" --draft-dir "$DRAFT_DIR" \
  --capture "$R/c16-0/capture/manifest.json" --capture "$R/c16-1/capture/manifest.json" \
  --capture "$R/c64-0/capture/manifest.json" --capture "$R/c64-1/capture/manifest.json" \
  --output-dir "$R/prepare"
```

Existing accepted A2 manifests may be used directly instead. Both gears require
at least two different X and replay hashes. Prepare rechecks actual checkpoint
codes/scales, every B-NZ element, full FP64 and INT64 references, whole-X shared
scale, rounding, finite values and bytes/SHA256. Gate/Down conversion is recorded
once and reused by samples. CPU full-W8 MLP is a design reference only.

## Independent normal OPP binding → Host → new real-data Smoke

Every command needs a **new** output directory. Failures return nonzero, preserve
raw logs/artifacts and must stop subsequent stages. Source deeply revalidates
the latest paired experiment and its original BNZ/FP16 evidence. Build performs
ldd/nm/private-symbol/runtime identity checks on the already accepted binaries;
it neither recompiles nor installs an OPP. Host calls only GetWorkspaceSize.
Actual7-Block TilingContext, SDK/Driver, Tiling and Workspace must remain exact.

```bash
COMMON=(--prepare-report "$R/prepare/prepare-report.json" --paired-report "$PAIR" --cann-root "$CANN_ROOT" --device 0)
python3 -B "$S/run_real_gate.py" source "${COMMON[@]}" --output-dir "$R/source"
python3 -B "$S/run_real_gate.py" build "${COMMON[@]}" --source-report "$R/source/report.json" --output-dir "$R/build"
python3 -B "$S/run_real_gate.py" host "${COMMON[@]}" --source-report "$R/source/report.json" --build-report "$R/build/report.json" --output-dir "$R/host"
python3 -B "$S/run_real_gate.py" smoke "${COMMON[@]}" --source-report "$R/source/report.json" --build-report "$R/build/report.json" --host-report "$R/host/report.json" --output-dir "$R/gate-smoke"
```

Smoke executes **full Gate/Up** for each real M16 input, twice with independent
Poison, unchanged normal Kernel guards, Readonly, Repeat, complete finite output,
sync and Workspace checks. All FP16 differences are retained. A stable finite
numeric mismatch is reported as unresolved, not transformed into a Golden PASS;
subsequent Gate timing is exploratory and cannot approve model quality. An
unexpected safety/Repeat/resource error blocks all timing.

## Real single-layer boundaries and native baseline

```bash
env -u ASCEND_CUSTOM_OPP_PATH python3 -B "$S/run_mlp_reference.py" smoke \
  --prepare-report "$R/prepare/prepare-report.json" --gate-smoke-report "$R/gate-smoke/report.json" \
  --output-dir "$R/mlp-smoke" --device 0
```

This separate Torch process checks the original W8A16 captured gate/up,
nonlinear product (Down input) and Down result exactly. It saves gate/up, SiLU,
multiply and Down for two repeats of original W8A16, effective FP16 and the
explicitly mixed bridge. The effective FP16 projections must pass independent
FP64 checks before timing. PyTorch's internal UB allocations are not exposed;
surrounding input/bridge-copy GM checks are not mislabeled as custom full Guard
coverage. Native complete MLP excludes residual addition and the rest of Draft.

## Separate timing ledgers, only after Smoke

```bash
python3 -B "$S/measure_preprocessing.py" --prepare-report "$R/prepare/prepare-report.json" --output-dir "$R/cpu-preprocessing"
python3 -B "$S/run_real_gate.py" benchmark "${COMMON[@]}" --source-report "$R/source/report.json" --build-report "$R/build/report.json" --host-report "$R/host/report.json" --smoke-report "$R/gate-smoke/report.json" --output-dir "$R/gate-benchmark"
env -u ASCEND_CUSTOM_OPP_PATH python3 -B "$S/run_mlp_reference.py" benchmark \
  --prepare-report "$R/prepare/prepare-report.json" --gate-smoke-report "$R/gate-smoke/report.json" \
  --smoke-report "$R/mlp-smoke/report.json" --output-dir "$R/mlp-benchmark" --device 0
python3 -B "$S/summarize_real.py" --prepare-report "$R/prepare/prepare-report.json" \
  --gate-smoke "$R/gate-smoke/report.json" --mlp-smoke "$R/mlp-smoke/report.json" \
  --gate-benchmark "$R/gate-benchmark/report.json" --mlp-benchmark "$R/mlp-benchmark/report.json" \
  --preprocessing "$R/cpu-preprocessing/report.json" --output-dir "$R/summary"
```

Gate timing reuses the accepted continuous-v1 Runner:2 poisoned prechecks,
5 warmups,30 Execute+Sync samples, tail and poisoned postcheck. Three rounds
per real input, reverse sample order in round1,90 fresh samples each. No transfers,
Host checks, extra logs, profiler or IO inside measured calls; internal Kernel
guards remain. The independent native timings include Torch dispatcher and
allocation cost, and must not be presented as the same ACLNN metric. All native
timed results are rechecked against Smoke and FP16 independently validated output.
No W8A8 complete MLP time is inferred by adding Gate and suffix medians.

## Exact success states (all new server execution currently NOT_RUN)

| Stage | Successful status |
|---|---|
| capture |REAL_DRAFT_CAPTURE_BOUND|
| prepare |REAL_MLP_CPU_PREPARED_NPU_NOT_RUN|
| source |REAL_MLP_SOURCE_BOUND|
| build |REAL_MLP_RUNNERS_REVERIFIED|
| formal Host |REAL_MLP_HOST_MATCH|
| new Gate Smoke |REAL_GATE_SAFETY_COMPLETE_QUALITY_UNRESOLVED|
| native/mixed MLP Smoke |REAL_MLP_BOUNDARIES_RECORDED_W8_DOWN_NOT_RUN|
| CPU preprocessing |CPU_ONLINE_PREPROCESSING_MEASURED|
| Gate benchmark |REAL_GATE_RESIDENT_MEASURED_QUALITY_UNRESOLVED|
| native/suffix benchmark |REAL_NATIVE_MLP_TIMED_W8_DOWN_NOT_RUN|
| assembled reports |REAL_MLP_REPORT_COMPILED_W8_DOWN_NOT_RUN|

On failure return the failing `report.json` (or `prepare-report.json`), relevant
`runner.log`/`native.log`/`capture.log`, `execution.json`, resource admission and
numeric JSON. Keep binary outputs and identities on the server. For a numeric
failure provide the full mismatch JSON plus the first failing boundary identity;
do not replace its reference. For provenance failures provide only the path/hash
failure and manifest metadata, not checkpoint/SDK source contents.

## Local validation

```bash
PYTHONPATH=framework/custom_ops/draft_quant/test python3 -B -m pytest -q -p no:cacheprovider "$S/test"
python3 -B -m pyflakes "$S"
```

Unit arrays are synthetic CPU fixtures only. They test all49,807,360 B packing
elements, Group128 effective weights, codec/RNE, integer dot, C16/C64 diversity,
checkpoint/identity rejection, CLI prerequisites and preserved safety gates.
The local Docker has no NPU or torch_npu. Its local CANN check compiles the
unchanged Runner translation unit against the actual private API; link/device
limitations are recorded in `local-validation.json`, not concealed with stubs
or ignored unresolved symbols. No new SDK investigation or compute Kernel is
required to supply the frozen ABI with new data.
