# Gate Raw Prefetch — Scheduler A / static OM / 8 blocks

本实验只改变 Gate Raw Prefetch。Off 和 Raw 都用 Scheduler A；只有实际 **ATC 8-block** 的静态 OM 才能计时。先执行下方“Server commands”中的 `gate8 source`，其余阶段逐条手动执行。C16/C64 各五份真实输入、各自 Native Golden；每档、每层分别报告正序与反序，不合并样本。两档都过门槛后才允许手动进入完整 Draft。旧 7-block 数据和历史完整 OM 延迟只作背景，不参与新倍率。

This is a **new experiment**, not a change to the production operator, not Scheduler A versus Scheduler B, and not an ACLNN timing run. It tests the five Gate/Up projections of the frozen A-only five-layer Draft on Ascend310P3 / CANN 9.0.0. Both Off and Raw use the same Scheduler A Host source and `LaunchBlockCount`, auto core limit 0. If an actual ATC callback does not launch eight blocks the graph is **not admitted**. No SetBlockDim override.

**Current server status:** all new stages NOT_RUN. Historical A/C16 and A/C64 full Draft ~31.2262 / 32.0923 ms and the provisional ~10.31 ms Gate task sums are context only. The historical ~1.086× 7-block standalone result is not a denominator or a qualification of these OMs.

## What is frozen

Source verifies OLD Scheduler Flow's first five status/source/hash/predecessor chains, Resume admit/full-smoke/benchmark (including the full Golden and actual timing records), the profiling admission and v3 per-gear projection bindings. It follows the accepted static graph/probe reports; it does not trust the current branch name or a repository default header as an installed binary identity. The source report contains the experiment checkout, source file manifest, host and device generated configuration identities, actual API/OPP/kernel and exact historical graph/case identities.

- Historical 52-probe suite: `6a9ec59e3f6e557ab4ee12aac390dc84dd7f3547f8bb26e733553577bb85301b`. Read the five accepted **C16 layer-0..4-gate_up** single-A AIRs from the suite's artifact references. Verify single-op scope, one A / zero custom Head and Head-Audit, zero native WeightQuant nodes, actual public X FP16 ND [16,2560] → Y FP16 ND [16,19456], deployment links, compiler/command/flags, immutable node and every AIR payload. Bind the one graph node to its **NZ and GN hash pair**, logical NK hash, actual shapes and Native capture. No suffix-to-layer mapping, random tensor, requantization, new Torch export or FP16 approximation. Missing/unprovable single graph → BLOCKED_UNPROVEN_GRAPH_IDENTITY; no ACLNN fallback.
- 5 independent graph bundles, **10 new OMs** after Off/Raw ATC. Reuse these *graphs only* across the two data gears: both have M16/K2560/N19456, and each corresponding pair of layers has identical INT8 NZ weights (49,807,360 B) and FP16 Group128 GN scale (778,240 B). Source requires independent C16/C64 X and original Native Golden per layer, and unique layers. Every C64 call stages **its own** X/Golden.
- Gate uses N64/K256, batched dequant, Broadcast, Loop DataCopy, pipeline mode prefetch, KV M80 A-UB policy unchanged, core_limit=0. The separate Gate switch selects serial vs raw lookahead. Gate raw header is the accepted Off header with precisely one macro changed. Header SHA Off `22d5541d1739f799588fbc076c866363d131d68a95e6e148a7cc0d1c02d692a1`, Raw `26017103ea631e701088f9d7e44e8645737228b4ba869b062c5726b47bb543e6`. Device source SHA `a4861bb5f7e38aceced9e4d89fc6c2519cb4a80e0914d7dd418c50389e8f1a4f`. Actual Kernel source is copied from the frozen generated project, including its original consumers' fences, two independent Group128 scales per K tile, ascending FP32 accumulation, one W16, final FP16 output. No additional synchronization change.
- Historical packaged Off ELF SHA `831a9ce01371ef0a70962e6576489f8d16ec07daa253b99f1b2e5e627f71bca3`; API SHA `b55f51b0eab58f780fbf287bc35162cd9ba718c3c3133dd694068574f81d5505`. A full OM SHA C16 `8e042a09444a9fba6b9a9aa3ddf11ce325698bd51861bc98f3ce7bdf38e20aaa`, C64 `3f2917b8bdaf83c06906a5359e1203362c1f65d3cf1d7f1a3e35a432152502c7`.

## Build and ATC evidence

Build creates two new private installation trees and two independent Gate OM runners. Both rebuild only the **DFlashGroupQuantLinear_ascend310p** device target in a fresh copy of the accepted Scheduler A project. The only build-system change to that frozen template restores its original op_kernel CMake target. The Off rebuild must reproduce the accepted device ELF byte-for-byte before use. Off preserves the entire accepted package identity at new paths. Raw must produce a different installed ELF **and different executable bytes**, not just a new Host DSO. It installs the newly built Host tiling/proto and device object/metadata into its own clone, and checks every packaged dynamic Gate implementation/header used by the ATC source route. SDK-generated dispatch config/keys must stay identical; other operators' device kernels, private API, headers, paths and aliases are frozen. No global installer. No Head device rebuild. The original full OM runner is reused for optional full stages.

There are **two different kernel identities**:

1. OPP target object + SDK registration JSON (matching sha256); retain generated source, compile recipes, verbose compiler/link commands, configured headers and the complete final package file inventory.
2. ATC-specialized code in each new OM. ATC runs from a private process, with a separate cache/output dir, using the same frozen graph/flags except model/output relocation. Keep stdout/stderr and original callback sidecars. ATC mode 1 reads the resulting OM graph; bind the exact named GroupQuant nodes to the immutable node audit and require tiled AIcoreEngine, `tvm_blockdim=8`, `_kernelname`, `_tbeKernelSize` and `_kernel_bin_id`. Only the complete ELF of the declared length, valid header/section extent, symbol and matching full SHA (last 64 hex of the node's ID) is accepted. A name alone, text fragment, guessed offset, private OPP path, or `GetCoreNumAic()==8` is insufficient. The earlier server observation (19,608-byte specialized ELF, SHA beginning d8957439) is an example from one GroupQuant node, **not** a universal Gate hash/offset. Source binds the actual five historical Gate specializations for both gears. The rebuilt Off Gate's executable sections must match the historical full A Gate for that layer; ELF-level metadata equality is reported separately. All new OMs still require new numerical acceptance. Raw's embedded Gate code must differ from Off. If ATC packing/attributes differ from the proven format, stop; do not substitute the OPP ELF digest.

Use actual post-SetBlockDim ATC records for both variants. Require 304 N tiles, 8 cyclic owners × 38, no gaps/duplicates, no cap, and unchanged PlatformInfo/capacities. The original parent tiling contains (M,K,N,tileK) and uint64 matmulUbBytes. A CPU-only SDK marker program in each Build verifies its real serializer: 224 B parent, 24 B prefix, 200 B nested TCube, 50 distinct marker fields. Read the new callback's parent bytes; compare the **entire nested Cube** against the accepted Off bytes. Do not compare entire Off/Raw parent hashes for equality. Require base (16,64,256), depth (1,1), steps (1,1,1,1), DB (2,2,1), transLength 34816; L1/L0A/L0B/L0C used=40960/16384/65536/4096 B. L0B is at capacity. Gate system Workspace must equal the actual anchor (2097152 B).

| Per-core UB bytes | Off | Raw |
|---|---:|---:|
| banks | 1 | 2 |
| INT8 / scale | 16384 / 256 | 32768 / 512 |
| single W16 | 32768 | 32768 |
| Broadcast buffer / reserved | 2304 / 8192 | 2304 / 8192 |
| user total / Matmul | 59904 / 202240 | 76544 / 185600 |

Other bytes/plan changes are rejected. The Gate operator Workspace, model work allocation, and model weight allocation are separately reported.

## Correctness, timings and decisions

Single Gate runner uses `aclmdlLoadFromFileWithMem → aclmdlExecuteAsync → aclrtSynchronizeStream` (device 0). One process, private OPP and model load per variant/layer/run. The existing full OM socket handshake checks all actual pre-maps against a freshly bound private manifest, pinned CANN/Driver/OS and the previously admitted 22 exact model-runtime DSOs; post-maps must also pass. No unknown OPP/DSO, alternate install, fake Execute, profiler or CPU fallback. After confirming shape/format of the frozen graph, check actual public dimensions/dtype/count, both device Poisons (read back before each correctness call), all 311296 finite FP16 values, whole output bitwise Native / Repeat, input payload including padding, GM guard prefixes/suffixes and model workspace/weight memory guards. Weights/scale *inside* an OM are bound by immutable constant hashes and compilation/OM identity. No claim to scan each internal constant's own GM guard. Historical Golden is never rewritten.

Both variants × all five layers must finish that gear's smoke before timing. Each benchmark process: two poisoned correctness calls; five warmup; thirty uninterrupted Execute+Sync samples; verify timed tail; one new poisoned postcheck. No upload, download, CPU scan, profiler, log write or disk I/O in timed samples. Model loading and setup excluded. Fail on timed log or memory drift. For each layer retain separately the Off→Raw and Raw→Off series (30 samples per variant **per order**, 600 per gear). No borrowing the C16 data case for C64; no mixing orders or exceptional retries.

Compare is predeclared per gear/order: all five layer medians strictly improve, **no layer P95 regression**, and sum(Off independent medians)/sum(Raw independent medians) ≥1.03. No threshold adjustments. Only if both gears qualify: `ELIGIBLE_FOR_FULL_DRAFT_EXPERIMENT`. If not: `NO_GO_FOR_FULL_DRAFT` (exit 3, no full stage). Neither isolated sum is full Draft latency; eligibility is not adoption.

## Server commands (manual, one stage at a time)

Use the same server interpreter/device/mount namespace as the accepted history. The code checks actual sources and previous reports rather than pretending the server SDK matches a version string. Keep sufficient disk for generated projects, ten single OMs, inputs/outputs/logs and later, optionally, four full AIR copies/OMs. Do not place the new run in the repo or in an old run. Do not stack private OPPs in the parent shell.

```bash
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only origin feature/ascend310p-custom-ops
PY=/home/w00949577/qwen35-dflash-run/venv-hf513-probe/bin/python
S=$PWD/framework/custom_ops/draft_gate_raw_atc8
R=$(mktemp -d /home/w00949577/qwen35-dflash-run/gate-raw-atc8-XXXXXX)
echo "R=$R"
gate8() { TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B "$S/run_gate8.py" "$1" --root "$R"; }
gate8 source
```

Default read-only history roots beneath `/home/w00949577/qwen35-dflash-run/`: `a-core-scheduler-flow-F6ia0q`, `a-scheduler-om-resume-0foBgF`, `a-only-draft-profile-OC8pCW`. CLI options `--flow-root`, `--resume-root`, `--profile-root` are explicit overrides, not permission to relax any SHA gate. Subsequent commands in the same shell (do **not** paste as a batch):

| Command | Required result | What happens |
|---|---|---|
| `gate8 source` | GATE8_SOURCE_PASS | Reverify historical/profiling and single graph sources. No device work. |
| `gate8 build` | GATE8_BUILD_PASS | New Off/Raw private packages, real Gate-only device compilation, Host/Runner/serializer. No NPU Execute. |
| `gate8 export` | GATE8_EXPORT_PASS | Compile 5 × 2 single Gate OMs, inspect actual ATC plans and embedded device code. No model Execute. |
| `gate8 c16-smoke` | GATE8_C16_SMOKE_PASS | 10 real untimed C16 cases, two executes each. |
| `gate8 c16-forward` | GATE8_C16_FORWARD_PASS | Five C16 pairs, Off then Raw for each layer. |
| `gate8 c16-reverse` | GATE8_C16_REVERSE_PASS | Five C16 pairs, Raw then Off. |
| `gate8 c64-smoke` | GATE8_C64_SMOKE_PASS | 10 real untimed cases with the C64 captures. |
| `gate8 c64-forward` | GATE8_C64_FORWARD_PASS | Five C64 pairs, Off then Raw. |
| `gate8 c64-reverse` | GATE8_C64_REVERSE_PASS | Five C64 pairs, Raw then Off. |
| `gate8 compare` | ELIGIBLE_FOR_FULL_DRAFT_EXPERIMENT **or** NO_GO_FOR_FULL_DRAFT | Offline only; inspect `$R/compare/comparison-c{16,64}.md` and the JSON with every sample. |

### Optional full Draft (separate manual gates)

Only after a two-gear GO, the following commands become admissible:

| Command | Required result | Additional contract |
|---|---|---|
| `gate8 full-export` | GATE8_FULL_EXPORT_PASS | New Off/Raw × C16/C64 from the identical historical A-only AIR. Same public ABI, 26 A / 0 C / 0 C-Audit and Head/weight payloads. Only the five hash-bound Gate nodes may differ in executable bytes. Other 21 actual ATC plans, nested/parent resources and embedded executable sections must match. If dormant branch/codegen changes non-Gate code too, BLOCKED_FULL_NON_GATE_CODE_DRIFT, no device run pending a separate review. |
| `gate8 full-smoke` | GATE8_FULL_SMOKE_PASS | 4 models, two independent Poison checks each. Existing guarded full Runner verifies all 15 INT64 token IDs and all ten complete logical KV buffers, including not-updated regions; full input readonly and guard checks. |
| `gate8 full-benchmark` | GATE8_FULL_BENCHMARK_PASS | Only after full Smoke: fresh three alternating rounds, 5/30, 90 per variant per gear. See `full-comparison.json`. Never substitute 31.2262/32.0923 ms as new measurements. |

## Reports and failure handoff

Every stage owns a fresh `$R/<stage>/report.json`. Schema: [report.schema.json](report.schema.json); code additionally validates predecessor order, hashes, typed fields, arrays and numerical contracts. Common fields: status/phase/error/integrity_error/device_execution/predecessors/sources, root/history_roots, atc/benchmark/profile/production. Failure stops immediately; nonzero exit. Attempted launch is not proof of completion. Do not “resume” by editing reports or reusing directories. New source hashes require a new experimental root.

For a failure return the terminal summary plus the phase's traceback and active_case; also the inner worker traceback/log if any. Build retains CMake/SDK command logs and the installed object/metadata. Export retains both ATC streams, original + shape-grouped callbacks, OM dump/compiled files and OPP/embedded ELF identities. Device attempts retain invocation, pre/post maps when produced, raw outputs, actual numerical difference and Guard log. Missing post-sync evidence is never reconstructed. Compare lists per-layer Median/P95/min/max, 8-block/resource proofs, identities, all sample series and each gear's decision.

## Local checks

```bash
python3 -B -m unittest discover -s framework/custom_ops/draft_gate_raw_atc8/test -v
# In the existing Linux ARM64 SDK container only, after sourcing set_env.sh:
python3 -B /src/framework/custom_ops/draft_gate_raw_atc8/local_check.py \
  --template /work/build/scheduler-flow-O4x84N/A/project --output-parent /work/build
```

[Local validation](LOCAL_VALIDATION.md) is compile/CPU evidence only. No new server Host callback, model launch, device correctness or timing has been obtained by Codex.
