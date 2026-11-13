# Down Scale byte-count fix (baseline35797f7)

The server evidence supplied for `w8a8-down-fullk-41phZ3` confirms:

- Host report: BLOCKED, phase=host, integrity_error=null, NPU=NOT_RUN.
- Candidate normal Runner, `--host-only`, returned1 for run-0.
- Retained Scale identity:20480B, SHA256
  `444d2f92d8836ca074fdf18253e2c38f98368bbcc3d80b47a2823cf9299a8fd1`.
- Runner log: `FAIL: input file has wrong size: .../quant_u64_doc_candidate.bin`.
  Supplied log identity is129B,
  `dd4a931a10054957aecb7bf3ce618ad04023102a52a89fe4a42e10f47e1cbce7`.

These are user-returned report/log excerpts, not a claim of direct local access
to the server. Source order confirms that the bad ReadBytes call precedes
aclInit, device allocation and GetWorkspaceSize. This failure does not indicate
a rejection of Down Tiling or Workspace, and it does not prove Formal Host PASS.

## Minimal correction

Both normal and fault Runner Scale ReadBytes and Tensor::Make payload sizes are
20480 =2560×sizeof(uint64_t). No file is padded, replaced or regenerated to match
the erroneous length. The strict file-size and CopyIn checks are unchanged.

The transform previously omitted Gate Scale155648→Down Scale20480. Down's A
also becomes155648B, from Gate A40960B. The regex substitution is **one pass over
the original Gate source**:40960→155648 and155648→20480 occur simultaneously.
Replacement output must never be processed again. All generated Candidate/Fault
sources are still checked byte-for-byte against that transform. Only the four
Runner size literals change; Kernel, Host, Tiling, Guard and Workspace code do not.

| Payload | Contract | Bytes |
|---|---|---:|
| A |INT8[16,9728]|155648|
| B |INT8 NZ carrier[10,304,16,16,32]|24903680|
| Scale |UINT64[2560]|20480|
| Initial inputs |A+B+Scale, excludes guards/Workspace/output|25079808|
| Output / full expected file |FP16[16,2560]|81920|

Workspace remains separately queried and admitted:
`arena=align512(system)`, `stride=arena+1024`,
`total=system+blocks*stride`. It is not derived from input payload bytes.
Per-core arenas, framework system space and allocation limits are unchanged.
New regression checks multiple system sizes and Auto/2-Core plans, including
rejection when a payload size is incorrectly substituted for Workspace total.

CPU regressions check both variants' read lengths, logical/storage shapes,
dtype widths, allocation payloads and reported initial input sum. A compiled
CPU harness uses the actual shared ReadBytes and actual Runner Tensor::Make with
explicit CPU ACL mocks: a2560-word file succeeds and allocates20480B plus the
existing guards;20479B,20481B and155648B files fail. These fixtures are not NPU
allocation/execution evidence. The old wrong read/allocate constants and a
cascaded transformation of A are rejected by mutation fixtures.

## Fresh server rerun

The whole Stage4X source tree is hash-locked. Recreate Source, Prepare, Hostplan,
Build and Host under the new commit. Do not patch old report hashes or replace
the Runner inside the old Build. The Stage4W real captures, checkpoint and MLP
evidence remain read-only and reusable through the existing verification chain.
New Prepare uses unchanged numerical rules and writes its own files.

```bash
set -eo pipefail
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
source "$CANN_ROOT/set_env.sh"
set -u
export DEVICE_ID=0
S=framework/custom_ops/draft_w8a8_output_dequant/stage4x_down
W=/home/w00949577/qwen35-dflash-run/real-draft-mlp-mountfix-pqbjQq
X=$(mktemp -d /home/w00949577/qwen35-dflash-run/w8a8-down-scale-bytes-XXXXXX)

bash "$S/server_stage.sh" source "$X" "$W"
bash "$S/server_stage.sh" prepare "$X" "$W"
bash "$S/server_stage.sh" hostplan "$X" "$W"
bash "$S/server_stage.sh" build "$X" "$W"
bash "$S/server_stage.sh" host "$X" "$W"
```

Stop on failure. New Host success must be `STAGE4X_FORMAL_HOST_ADMITTED`, with
both Candidate/Fault Host-only rows accepted and NPU still NOT_RUN. Only then
continue the separate tile0, smoke, fault, mlp and benchmark commands in README.
All current numerical/strict-Benchmark gates remain intact.

If blocked, return the new host/report.json, failing run's runner.log and
execution.json, and resource-admission.json if it was produced. Keep old
`w8a8-down-fullk-41phZ3` unchanged. This bugfix supplies no new server Host/NPU
result and does not upgrade the old BLOCKED report.
