# Stage 4G: read-only ELF and lifecycle investigation

Status: **STATIC_REVIEW_PARTIAL / MACHINE_EVIDENCE_PENDING**. No new Kernel,
Host Tiling, OPP, numerical Golden, benchmark or simulator is created. Historical
Stage4B–4F files are read-only. Workspace hoisting is not adopted.

The reported Stage4F medians show no benefit: A1−A0 = +0.430630 ms and
B1−B0 = +0.014425 ms. These user-reported medians alone do not establish
significance, a compiler explanation, or the critical-path cost of any API.
The reported equal ELF sizes and different complete hashes establish neither
identical instructions nor more executed instructions. The common generated
filename is not a content identity.

## Offline binary inspection

`inspect_binary.py` reads the exact four `variants.*.installed.kernels` from the
supplied Stage4F Build Report. It verifies the report's source audit, recorded
identities and original binary_comparison, without rebuilding anything. An
optional accepted Benchmark Report supplies the actual Host configuration and
paired statistics; its original validator is reused read-only.

The inspection distinguishes:

1. Whole ELF SHA256.
2. ELF machine/type and executable section addresses, sizes and byte hashes.
3. Allocated data, relocation, symbol and debug section hashes, independently.
4. Existing disassembler output and tool identity/version/capability failures.
5. Source-PC attribution and dynamic execution cost: remain UNRESOLVED until
   actual output is reviewed. Static instruction-like line counts are not loop
   execution counts, cycle costs or proof of optimization.

AIC/AIV identity comes only from explicit compiler metadata. UNKNOWN remains
UNRESOLVED. A coupled Cube+Vector workload is not evidence of a separate AIV
ELF. Multiple indistinguishable objects are not paired by list order. ELF64 LE
with ordinary section tables is supported; unsupported layouts block the
inspection instead of inventing section offsets or architecture.

Same executable bytes are insufficient to ignore relocation/data differences.
Different executable bytes can be addresses or scheduling/register-allocation
changes, not additional operations. Identical file sizes do not imply identical
instruction counts. No guessed load bias or subtraction of historical PC bases
is used. A zero disassembler return code alone does not prove device support.

Observed tool limitation: an `elf64-hiipu` listing may contain only addresses
followed by `<not available>`, even with return code zero and a registered hiipu
target. Those are undecoded-address placeholders, not instructions. They set
`unknown_diagnostic=true`, are counted separately as `unavailable_address_lines`,
and keep the analysis UNRESOLVED even in a mixed listing. The address increment
and `.text` byte difference must not be converted to an instruction count.
Existing reports are not rewritten; no repeat disassembly is needed merely to
correct the old diagnostic flag. First inspect already installed tool options
and alternatives, without downloading a new SDK or guessing a different ISA.

The tool checks four fixed CANN compiler locations and PATH for llvm-objdump /
ccec-objdump; it does not scan the SDK tree. Ambiguous tools require an explicit
`--objdump` path. Unsupported output is saved and remains UNRESOLVED. No online
SDK download or substituted toolchain is installed. Raw disassembly and logs go
only to the new **private server output directory**, never this repository.

## Current lifecycle dependency conclusions

SDK identities and reviewed line references are retained in the unchanged
Stage4F `sdk-contract.json`. Additional scheduler evidence from the user:

- `impl/adv_api/detail/matmul/scheduler/base/scheduler_norm.h:93–162`
  SHA256 `bb565c78b53644f63cdf89cef18f4d622f11f4fca755ce03b989347f5f5083cd`.
- `impl/adv_api/detail/matmul/stage/compute/mmad_compute.h:26–65`
  SHA256 `c5f5dd39c23f845d4a8e4088030c6e173bf0273370d81152819432a952c66c5b`.

These are identities/locations, not republished SDK text. Current Stage4B
operation sequence is in its own Kernel lines 83–99. The table distinguishes
required data dependencies from implementation-specific completion mechanisms.

| Operation | Producer → consumer / resource | Current protection and evidence | Change admission |
|---|---|---|---|
| SetTensorA/B | GM descriptors and input cache/Scheduler state → input loading | impl_base:505–542; input update resets input cache and Scheduler | Early rebinding while previous work runs: UNSUPPORTED; not pure pointer mutation |
| Iterate(false) | A/B L1 → L0A/B → INT32 L0C | scheduler_norm:93–162 dispatches input loading, L0 allocation, queues and CubeCompute; actual Mmad call is in MmadCompute | API overhead cannot be equated with Cube MAC time |
| SetQuantVector / coefficient copy | GM uint64[64] → SDK Scratch coefficient region → VDEQ16 output | quant_processor_datacopy:49–54,113–158; local coefficient location and MTE3_MTE2/MTE2_V handling | Updating the pointer is not prefetch. Earlier overwriting of the coefficient region: UNSUPPORTED without last-consumer proof |
| GetTensorC(Local NZ) | L0C → FP16 user Partial | scheduler_base:363–386 queues L0C, issues output copy, frees output Tensor; Stage4B retains PIPE_ALL | FreeTensor is not by itself proof that every external pipeline dependency completed |
| End | input/output queues, L0 cache, quant processor and conditional sync state | scheduler_base:149–167 | Keep unchanged. Omitted release/wait/reset behavior is not licensed by no-op quant Destroy |
| Cast | FP16 Partial → FP32 temporary | existing post-GetTensorC/End PIPE_ALL; then PIPE_V | RAW: output must be ready before Cast. WAR: next output must not overwrite Partial while Cast reads it |
| Add | prior FP32 Sum + current FP32 temporary → next Sum | PIPE_V before Add, PIPE_ALL after; fixed group order | Preserve RAW/WAW and FP16-partial boundary; no tree reduction/FMA substitution |
| V_S / guards | completed Vector writes → Scalar GetValue checks | explicit V_S Set/Wait before guards | Keep full protocol in correctness baseline. Diagnostic guard deletion is not proof that sync may be deleted |
| Next group | reuse Scratch, coefficient area, input queues and Partial/temporary | accepted sequence finishes current output/End/Cast/Add/checks before rebinding | This is the demonstrated safe point. An earlier legal overlap point is UNRESOLVED |

The internal `DOUBLE_BUFFER` class manages buffers within its selected SDK
schedule; its existence is not a contract for pipelining independent external
K128 calls with different Scale. Current SDK evidence contains no verified
cross-call snapshot/ownership protocol for that change.

Potential GM→L1 loading, L1→L0 loading, Cube and output conversion are distinct
activities. Overlap requires a proven disjoint buffer bank, producer completion,
consumer-release event and legal queue/state lifecycle. Pipe counters alone
neither prove these dependencies nor provide exposed waiting time.

## Resources and next-candidate decision

Per-core UB budget remains 239104 Scratch + 12288 user + 2560 guards + 8192
reserve = 262144 bytes. Scratch is an allocation cap, not measured live usage.
The reserved 8192 bytes are not free memory. An additional B K128×N64 tile plus
64 uint64 coefficients already needs 8704 bytes before alignment/guards/A.

Per-group payload lower bounds: INT8 A 2048 B, INT8 B 8192 B, INT32 L0C 4096 B,
FP16 Partial 2048 B. L1/L0 depths and db factors must come from the actual Host
plan; lower bounds cannot replace capacity/lifetime checks. Existing per-core
system Arena and address formulas remain unchanged.

Decision: **NO_NEW_KERNEL — UNSUPPORTED_FOR_CHANGE_PENDING_EVIDENCE** for
fine-grained barrier replacement, deleting/repositioning End, cross-call B/Scale
prefetch and new double buffers. This labels modification admission, not blanket
hardware incapability. No specific replacement event is guessed. Do not repeat
Workspace hoisting or infer that all remaining Scalar is lifecycle overhead.

After actual disassembly is returned, first establish whether Workspace code
changed, whether the changed bytes map to the moved statement, and whether it
lies in the repeated group body. Then prioritize a single evidence-backed
change; preserve Stage4B full guards and a separate original diagnostic baseline.
No new candidate is automatically selected by this analysis script.

## Server read-only command

```bash
set -euo pipefail
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only
S4F=/home/w00949577/qwen35-dflash-run/w8a8-workspace-tile-20261029-064832
S4G=/home/w00949577/qwen35-dflash-run/w8a8-lifecycle-offline-$(date +%Y%m%d-%H%M%S)
python3 -B framework/custom_ops/draft_w8a8_output_dequant/stage4g/inspect_binary.py \
  --build-report "$S4F/build/build-report.json" \
  --benchmark-report "$S4F/benchmark/benchmark-report.json" \
  --cann-root /usr/local/Ascend/cann-9.0.0 \
  --output-dir "$S4G"
printf 'Private output: %s\n' "$S4G"
```

If tool selection is ambiguous/missing, return `tool_candidates` and
`tool_selection`; choose an actual installed device-capable tool with
`--objdump /actual/path`, rerun into a new directory. Do not overwrite reports.
Return `binary-analysis.json` and relevant disassembly/tool-version files in
this private chat, not public GitHub. These files may include private paths,
source line mappings and compiled code. No historical artifact is edited.

Local tests: `PYTHONPATH=framework/custom_ops/draft_quant/test python3 -B -m
pytest -q framework/custom_ops/draft_w8a8_output_dequant/stage4g/test` (one
command). Synthetic ELF fixtures validate debug-vs-code and relocation
separation, unsupported disassembly, ambiguous matching, identity rejection and
fixed tool discovery. They do not establish CANN disassembler support or NPU
behavior. Actual machine-instruction analysis awaits the server results.
