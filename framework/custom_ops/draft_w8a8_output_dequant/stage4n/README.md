# Stage4N: N128 uint32 scalar XOR/OR group Guards

Independent candidate, not a production adoption. Reuses accepted Stage4J
Full OPP and Golden; creates a private XOR OPP, a separate fault-instrumented
OPP, and three independent Runners. No historical source/evidence is modified.
Stage4M Compare/GetCmpMask/bitmap directions remain stopped. No uint64 read,
Vector Guard, new UB allocation, reserve use, Scratch/Workspace change,
Workspace hoist, synchronization removal or numerical change.

## Exact allowed candidate change

At the original group boundary, replace only the five short-circuit
CheckGuard tests with one GroupDelta decision. Each of five GuardDelta calls
unconditionally reads64 uint32 prefix and64 suffix words through GetValue.
Each word contributes `(word ^ guardWord)` to a uint32 OR accumulator. The
five results are ORed; any nonzero result returns before the next group.
Equal corruptions cannot cancel. Both good and bad paths read all640 words.
Detection is at the same group boundary, not at the first corrupted word;
this deliberately differs from the original early-exit scan.

Only scalar locals/parameters are added. Tile-end five checks, final extra
p0/cast checks, Arena checks, initialization, per-group workspace/A/B/Scale,
Iterate/GetTensorC/End, barriers/events and FP16→ordered FP32 arithmetic are
unchanged. The candidate contains no fault injection or witness writes.
`guard_transform.py` materializes exact source differences into the new run
folder; generated project files, unchanged ABI and T1 Host contract are
compared again before private OPP admission. All three Kernel identities must
be distinct. Different ELF hashes alone do not establish instruction savings.

## Local SDK contract

Root: `/usr/local/Ascend/cann-9.0.0/aarch64-linux/asc` in qwen35-cann9-host.
`impl/basic_api/kernel_tensor_impl.h`669–719 marks GetValue as PIPE_S and
returns the typed UB value in the non-split path. Lines757–823 implement the
Scalar SetValue used ONLY by the diagnostic artifact. SHA256:
`1e5d6669d91e5687db76fcace1cb1b2a9af0dfde92c4b5ee65dd2b23b99b9e0a`.
`impl/basic_api/kernel_macros.h`150–173 defines MIX and split-core selection;
SHA256 `500ea245ecddc927f644ae8cb2a75abe914164b402d372c7293297135ef0d3f9`.
The new helper has a compile-time non-split MIX requirement. No unsupported
vector interface is used; XOR/OR are ordinary unsigned32 C++ operations.

Server admission checks these exact hashes, plus the already traced event
and block-sync wrapper identities. A mismatch blocks instead of assuming
local/server equality. SDK original source is not stored in the repository.
CANN operator compile and NPU validation remain NOT_RUN at development time.

## Independent device fault gate

The Fault OPP has the same resource sizes/ABI but explicit diagnostic code;
it is NEVER benchmarked and cannot substitute for the candidate.
All device fault runs use Tile0, core_limit1, two poisoned repeats:

- Mode0: at EACH of20 actual Matmul group boundaries, first require clean
  Guards, then sweep640 positions ×32 single-bit corruptions. Invoke the same
  GroupDelta helper, require the exact nonzero mask, restore each word, and
  finally run the normal candidate check. All409600 injections must complete
  before final Tile0 Golden can be produced. Unselected columns remain Poison.
- Modes1..20: inject0x80000001 at the final Sum suffix word in the selected
  group. A diagnostic abort branch writes a32B witness identifying magic,
  group, delta,640-word scope, buffer/side/word and completed group count,
  then returns. Every other output byte must remain that repeat's Poison.
  A missing/wrong-group witness fails; an early unrelated return cannot pass.

Witness transfer reuses the existing converted body and adds a diagnostic
S→MTE3 event. This extra synchronization and the witness writes exist ONLY
in the fault artifact, not the timing candidate. Intentional injected damage
is labeled FAULT_INSTRUMENTED_ABORT_WITNESS, not ordinary full-Golden success.
The exhaustive sweep can be slow. A timeout/device error blocks the gate;
no automatic timeout increase, skipped cases or reduced coverage is allowed.

These tests complement rather than replace the candidate's full311296-element
Golden Smoke, external GM Guards, Readonly, Poison, Repeat and actual resource
admission. Source/control-flow tests and CPU stubs do not prove device safety.

## Performance and stop rule

Only successful Smoke AND complete device fault evidence authorize Benchmark.
Full/Xor use auto Core scheduling with the accepted seven-Block policy,
continuous-v1, warmup5/repetitions30, three fresh orders Full→Xor,
Xor→Full, Full→Xor. Each version contributes90 fresh samples. Formal timing
excludes Profile and the fault OPP. Weak remains historical diagnostic context
only; Stage4L is not rerun or used as a denominator.

Predeclared materiality criterion: aggregate Full/Xor median ratio >=1.03,
and every round ratio >1. Otherwise comparison.json returns
NO_GO_NO_MATERIAL_IMPROVEMENT and this implementation should not be repeatedly
rewritten. This practical threshold is not statistical confidence or automatic
production approval. Pipe metrics and source call counts are not wall-time
attribution. All results are PREPACKED_RESIDENT_W8A8_KERNEL_ONLY.

## Server commands

Every output directory must be new. Inspect each successful phase before
advancing; a failure exits nonzero. Build creates only the two new private
OPPs and Runners, not Stage4J or Stage4L.

```bash
set -euo pipefail
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
S4N=/home/w00949577/qwen35-dflash-run/w8a8-n128-xor-guard-$(date +%Y%m%d-%H%M%S)
ENTRY=framework/custom_ops/draft_w8a8_output_dequant/stage4n/run.sh
bash "$ENTRY" build --device 0 --output-dir "$S4N/build"
```

After STAGE4N_RUNNERS_BUILT:

```bash
bash "$ENTRY" smoke --device 0 \
 --runner-build-report "$S4N/build/build-report.json" --output-dir "$S4N/smoke"
```

After STAGE4N_SMOKE_MATCH:

```bash
bash "$ENTRY" faults --device 0 \
 --runner-build-report "$S4N/build/build-report.json" \
 --smoke-report "$S4N/smoke/smoke-report.json" --output-dir "$S4N/faults"
```

Only after STAGE4N_DEVICE_FAULT_COVERAGE_MATCH:

```bash
bash "$ENTRY" benchmark --device 0 \
 --runner-build-report "$S4N/build/build-report.json" \
 --smoke-report "$S4N/smoke/smoke-report.json" \
 --fault-report "$S4N/faults/fault-report.json" --output-dir "$S4N/benchmark"
```

Optional separately profiled paired Runners, after accepted Benchmark:

```bash
bash "$ENTRY" profile --device 0 \
 --runner-build-report "$S4N/build/build-report.json" \
 --benchmark-report "$S4N/benchmark/benchmark-report.json" --output-dir "$S4N/profile"
```

Return build/smoke/fault/benchmark reports and comparison.json. Preserve
source-Xor/source-Fault audits, opp-Xor/opp-Fault build/install logs and failed
runner.log, execution.json, resource-admission.json and actual-*.bin. No SDK
source, binaries or server logs should be committed publicly.

CPU tests: `PYTHONPATH=framework/custom_ops/draft_quant/test python3 -B -m pytest -q framework/custom_ops/draft_w8a8_output_dequant/stage4n/test`.
They compile actual transformed C++ with cached-partial Matmul doubles; cover
all positions/bits/groups with a640-load assertion, multi-bit non-cancellation,
transient corruption, last group, abort timing, source constraints, Runner ABI
and strict absence of a performance gate without device fault proof.
