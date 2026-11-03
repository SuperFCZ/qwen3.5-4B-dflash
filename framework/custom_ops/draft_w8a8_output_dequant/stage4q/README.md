# Stage4Q: new paired N128 XOR / N256 XOR timing

No compute Kernel is modified, built or installed. Separate Benchmark
executables link the admitted Stage4N normal XOR OPP and Stage4P normal OPP;
fault artifacts are never timed. Stage4N Build/Smoke/Fault/Benchmark/Profile
and Stage4P Prepare/Build/Host/Bounded/Full/Fault are read-only revalidated.
The original Stage4B Golden and saved Activation must match; all76 N256 B/
Scale tiles are checked against their two N128 logical-channel halves.
Both retain640 uint32 group checks, original barriers/End/workspace, FP16
partials and ordered FP32 accumulation. Actual7 Blocks are required.

Protocol:2 poisoned strict prechecks, warmup5, continuous30 execute+stream
synchronize samples, checked timed tail and poisoned postcheck. Prepare is
separate. Readonly/Guard/Golden/Repeat/workspace/log failures reject the run.
Each variant runs in its own process/private-library environment. Three
orders: N128→N256, N256→N128, N128→N256. Each contributes90 NEW samples.
Historical4.299590ms is not used in the ratio.

Materiality: aggregate median N128/N256 >=1.03 and each round ratio >1;
otherwise NO_GO_NO_MATERIAL_IMPROVEMENT. A measured report can be valid while
its optimization decision is NO_GO. Do not respond by blindly increasing N.
Guard invocation counts also shrink with N256; this is not pure Cube
throughput attribution. PREPACKED_RESIDENT_W8A8_KERNEL_ONLY excludes online
activation quantization/scale encoding, upload and full Draft/Decode.

Profile is a separate phase and its latencies are not formal samples. It uses
Stage4L's row-provenance reviewer, preserving original CSV type/units,
Task/Stream/Core fields, duplicate flags and timelines. No mixed op_summary/
task_time medians or additive Pipe/critical-path interpretation.

## Server

All phase directories must be new. Stop on any failed state.

```bash
set -euo pipefail
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
S4Q=/home/w00949577/qwen35-dflash-run/w8a8-xor-width-pair-$(date +%Y%m%d-%H%M%S)
Q=framework/custom_ops/draft_w8a8_output_dequant/stage4q/run.sh
bash "$Q" build --device 0 --output-dir "$S4Q/build"
# Require STAGE4Q_RUNNERS_BUILT
bash "$Q" smoke --device 0 --runner-build-report "$S4Q/build/build-report.json" --output-dir "$S4Q/smoke"
# Require STAGE4Q_SMOKE_MATCH
bash "$Q" benchmark --device 0 --runner-build-report "$S4Q/build/build-report.json" --smoke-report "$S4Q/smoke/smoke-report.json" --output-dir "$S4Q/benchmark"
# Require STAGE4Q_PAIRED_MEASURED
bash "$Q" profile --device 0 --runner-build-report "$S4Q/build/build-report.json" --benchmark-report "$S4Q/benchmark/benchmark-report.json" --output-dir "$S4Q/profile"
# Require STAGE4Q_PROFILE_COMPLETED; individual counters may remain UNAVAILABLE
```

Optional explicit all-stage command: `bash framework/custom_ops/draft_w8a8_output_dequant/stage4q/all_stages.sh NEW_RUN_ROOT 0`.
It checks the exact status and complete phase after every step, refuses root
reuse, and performs no retry/repair. No automation is started by installation.

Return build/smoke/benchmark/profile reports, benchmark/comparison.json and
preprocessing-inventory.json. On failures retain variant CMake/build/ldd logs,
runner.log, execution.json, resource-admission.json and actual/tail/postcheck
outputs. Profile CSVs/logs/binaries stay private, never in Git.

Local validation: CPU ABI doubles exercise both real Runner sources, failed
Golden and timed logging. Docker also compiles/links both runners against
real CANN runtime headers/libraries with PRIVATE OP API STUBS ONLY because
server private OPPs are not present locally. This is
LOCAL_ONLY_NOT_SERVER_VERIFIED, not a real operator link or NPU test.
Server CANN Runner build, Smoke, Benchmark and Profile are NOT_RUN at delivery.
