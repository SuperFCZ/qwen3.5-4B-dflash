# Stage4R: fresh native FP16 numerical admission and fair paired timing

No historical FP16 comparison, Native OM Golden or old PROF folder is read.
Source inputs are the intact frozen X/NK/NZ/GN chain admitted by current W8A8
proofs. Missing critical frozen data blocks with its actual path; missing old
FP16 Profile does not. This experiment does not modify Stage4B–4P or install
an OPP. Native aclnnMm links only selected CANN libraries; W8A8 runs separately.

## Independent numerical contract (fixed before measurement)

Effective weights are rebuilt OFFLINE as RN16(signed INT8 NK * original FP16
GN scale), with every NZ inverse mapping and NK↔KN FP16 bit checked. Input X
is the original FP16[16,2560], not the quantized W8A8 Activation. CPU reference
is a full FP64 dot with these actual FP16 operands, calculated in N256 chunks.
No Native output is used to construct this reference.

Predeclared engineering acceptance:

- all reference and output values finite, output shape[16,19456];
- every element abs_error <=0.002 +0.002*abs(FP64 reference);
- global relative L2 <=0.001 (zero reference requires zero error).

These limits are not a bitwise FP64→FP16 theorem, not a guarantee of Native
internal accumulation order, and not model-quality acceptance. They cannot be
changed by a CLI tolerance option or calibrated from timing. Failure BLOCKS
performance rather than loosening a Golden.

The new native Runner performs full-array CPU-reference checks INSIDE the
process before warmup and at tail/postcheck, alongside Guard/Readonly/Poison
readback/Repeat checks. Two untimed outputs must independently pass and match
one another. Only then does the driver copy that admitted output into a NEW
strict-repeat Golden. Timed runs check both the FP64 tolerance contract and
that frozen byte identity. A first unvalidated device result is never Golden.

The chosen W8A8 implementation retains its own strict Stage4B Golden. No
FP16↔W8A8 bitwise equality gate is applied. Cross-scheme errors are diagnostics.
FP16 uses cube_math_type0 and pure aclnnMm, without bias/activation/post-op.
Native internal tiling, scheduling and accumulation are not claimed identical
to custom W8A8, including its640-word group Guard overhead.

## Explicit candidate and scope

`--candidate N128` is the default known measured candidate. `--candidate N256`
selects the separately accepted Stage4P normal OPP; it does not assert N256 is
faster. Stage4Q timing is NOT a prerequisite. After Q is measured, the user
may explicitly choose the admitted better candidate for a NEW R run. Different
choices cannot be mixed in one report or changed between R phases.

Pairing after independent Smoke: FP16→W8, W8→FP16, FP16→W8, warmup5/reps30,
90 new samples per implementation. Execute+sync and prepare reported separately.
Inputs resident; uploads, validation, CPU conversion and file IO excluded.
Timed logging and workspace drift reject samples. Report includes both errors
versus FP64, raw samples, round ratios, workspaces, identity and payload bytes.
This is implementation latency at equal shape, not pure INT8/FP16 hardware
throughput. Online quantization/combined Scale/encoding, full Draft/Decode
remain NOT_RUN. No R profiler is run by these commands.

## Server phases

Use fresh directories, keep candidate identical in every phase. Defaults bind
current Stage4N/Stage4P correctness evidence, not old FP16 history.

```bash
set -euo pipefail
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
S4R=/home/w00949577/qwen35-dflash-run/fp16-fresh-paired-$(date +%Y%m%d-%H%M%S)
R=framework/custom_ops/draft_w8a8_output_dequant/stage4r/run.sh
CANDIDATE=N128
bash "$R" prepare --device 0 --candidate "$CANDIDATE" --output-dir "$S4R/prepare"
# Require STAGE4R_CPU_PREPARED
bash "$R" build --device 0 --candidate "$CANDIDATE" --prepare-report "$S4R/prepare/prepare-report.json" --output-dir "$S4R/build"
# Require STAGE4R_RUNNERS_BUILT
bash "$R" smoke --device 0 --candidate "$CANDIDATE" --prepare-report "$S4R/prepare/prepare-report.json" --build-report "$S4R/build/build-report.json" --output-dir "$S4R/smoke"
# Require STAGE4R_NUMERICS_ADMITTED
bash "$R" benchmark --device 0 --candidate "$CANDIDATE" --prepare-report "$S4R/prepare/prepare-report.json" --build-report "$S4R/build/build-report.json" --smoke-report "$S4R/smoke/smoke-report.json" --output-dir "$S4R/benchmark"
# Require STAGE4R_PAIRED_MEASURED
```

Any failure is BLOCKED/nonzero. Do not advance, rewrite old evidence, or reuse
a failed directory. Return phase reports, data/reference.json,
comparison.json and preprocessing-inventory.json. Preserve cmake.log,
compile.log, ldd.log, runner.log, execution.json and all raw outputs on failure.
An independent-correctness failure is distinct from compilation unavailability.

## Local limits

CPU doubles compile and exercise the actual native Runner, including numeric
failure before timing, invalid logging, postcheck and zero-reference behavior.
Effective-weight/layout and independent-dot tests do not use a device.
The local Docker CANN Toolkit lacks public aclnn_mm.h and libopapi.so in the
checked native install locations: real native CMake configuration is BLOCKED
locally (log retained). No SDK package is installed and no undefined symbols
are ignored. This does not block Q. Actual server native ABI/library binding,
NPU numerical acceptance and performance remain NOT_RUN.
