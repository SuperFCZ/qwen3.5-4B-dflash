# Stage4K: full-Guard N64 versus N128 T1

Scope: **PREPACKED_RESIDENT_W8A8_KERNEL_ONLY**. No new compute Kernel,
Host Tiling or OPP installation. N64 links the exact accepted Stage4B private
OPP; N128 links the exact accepted Stage4J private OPP. Separate executables,
private-library environments and processes prevent registration collisions.
No weak-Guard variant exists. CANN Runner compilation, NPU timing and Profile
remain NOT_RUN until performed on the server.

Admission revalidates Stage4B Full, Stage4C measured baseline and Stage4J
Host/Bounded/Full output/resource proofs, Stage4I T1 and current SDK hashes.
Every original Golden and input hash remains bound; the N128 B/Scale mapping
is rechecked against the two corresponding N64 tiles. Both variants must use
one original Stage4B Golden and saved full-X activation. Stage4J's original
correctness Runner and build identities are retained in evidence. No history
is rerun or rewritten. Historical latency values are never samples.

Both independent Benchmark Runners derive from Stage4C's accepted Runner:
N128 changes only ABI, packed shapes, operation/log names and fixed tile_id=-1;
mode=0/core_limit=0 in both. Formal policy is read from actual TilingContext
logs and must match the accepted seven-Block policy. No Core fallback.
Workspace admission, complete kernel Guards, Arena checks, Readonly, Poison,
Repeat and all311296 FP16 Golden elements remain mandatory.

Each timed process uploads once, runs two poisoned checked calls, warms up5,
then measures30 uninterrupted calls. Execute+stream synchronize and prepare
latencies are recorded separately. Last timed output and a fresh poisoned
postcheck are strictly compared. No timed callback performs readback, Poison,
file IO or Host guard scans. If the SDK/Host writes to runner.log during the
continuous interval the process fails; diagnostic output is not silenced.

Exact paired order: N64→N128, N128→N64, N64→N128. Each variant contributes90
new samples. comparison.json contains median/p95/min/max, raw samples,
per-round medians/ratios and the aggregate N64/N128 ratio. Each run records
actual Blocks/Workspace and the build/evidence chain binds Runner/Kernel/OPP,
inputs, Golden, device, CANN and driver information. Environment records must
match between build/smoke/benchmark/profile. Original SDK and server logs are
private outputs and must not be committed.

Profile runs only after accepted paired Benchmark. Each variant gets a
separate msprof PipeUtilization run with independent correctness checks. Raw
CSV/timelines and hashes are retained with actual counter names/units;
missing metrics are UNAVAILABLE. Profile timing is excluded from formal
samples. Pipe activity is not additive wall time or proven removable wait.
Full Guard code is preserved, but N128 halves the number of tile/group Guard
scans as well as Matmul invocations. The result is not pure Matmul attribution.

Online activation quantization, combined Scale generation/encoding, Draft
and Decode remain NOT_RUN. preprocessing-inventory.json identifies static
weight packing, input-dependent preprocessing, bytes and timed exclusions.

## Server stages

All directories must be new. Run each block only after the preceding stage
succeeds. Any failure exits nonzero; do not advance or modify old evidence.
Default evidence roots are the accepted Stage4B, Stage4C baseline, Stage4I.1
and Stage4J directories specified in this experiment.

```bash
set -euo pipefail
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
S4K=/home/w00949577/qwen35-dflash-run/w8a8-n64-n128-paired-$(date +%Y%m%d-%H%M%S)
ENTRY=framework/custom_ops/draft_w8a8_output_dequant/stage4k/run.sh
bash "$ENTRY" build --device 0 --output-dir "$S4K/build"
```

After STAGE4K_RUNNERS_BUILT (only Runner compilation):

```bash
bash "$ENTRY" smoke --device 0 \
  --runner-build-report "$S4K/build/build-report.json" --output-dir "$S4K/smoke"
```

After STAGE4K_SMOKE_MATCH:

```bash
bash "$ENTRY" benchmark --device 0 \
  --runner-build-report "$S4K/build/build-report.json" \
  --smoke-report "$S4K/smoke/smoke-report.json" --output-dir "$S4K/benchmark"
```

After STAGE4K_PAIRED_MEASURED:

```bash
bash "$ENTRY" profile --device 0 \
  --runner-build-report "$S4K/build/build-report.json" \
  --benchmark-report "$S4K/benchmark/benchmark-report.json" --output-dir "$S4K/profile"
```

Expected final collection state: STAGE4K_PROFILE_COMPLETED. Counter
availability is reported separately; completion does not invent missing
counters or prove a particular bottleneck.

Return build-report.json, smoke-report.json, benchmark-report.json,
benchmark/comparison.json, profile-report.json and preprocessing-inventory.json.
On failure retain the relevant variant CMake/build/ldd logs, and
round-*/N64-or-N128/case/{runner.log,execution.json,resource-admission.json,
actual-0.bin,actual-1.bin,benchmark-last.bin,postcheck.bin}. Partial timed samples
from failed checks cannot authorize Profile or a speedup claim.

## CPU regression

```bash
PYTHONPATH=framework/custom_ops/draft_quant/test python3 -B -m pytest -q \
  framework/custom_ops/draft_w8a8_output_dequant/stage4k/test
```

Tests compile both real Runner sources against CPU-only ACLNN ABI doubles,
exercise exact2+5+30+1 calls, forbid interval copies/allocations/memset, reject
bad Golden and unexpected logging, verify pairing/statistics, and isolate
profile operator rows. Fake timing is never treated as NPU performance.
