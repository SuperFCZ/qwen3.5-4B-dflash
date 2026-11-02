# Stage4O: N256/BaseN256 Host-only feasibility

Stage4N XOR/OR is retained as the current best admitted full-group-Guard
candidate, with GO_FURTHER_REVIEW rather than production adoption. This
experiment reads and verifies its Build, Smoke, Fault, Benchmark and Profile
identities, actual outputs, source hashes and installed OPPs. It writes a new
`stage4n-freeze.json`; it does not edit history or rerun a historical operator.

No compute Kernel, OpDef, private OPP or Benchmark Runner exists here. NPU
Kernel execution, Benchmark, Profile and N256 Kernel compilation are NOT_RUN.
Only a standalone Host C++ executable is compiled and queried on the server.

## One fixed candidate; no fallback

- Shape/OrgShape/FixSplit: M16/N256/K128, including BaseN256.
- A/B: GM INT8 ND, B non-transposed.
- C: VECIN NZ FP16; BiasType INT32, Bias disabled; DequantType TENSOR.
- Requested Scratch202240B, user UB49152B, Guards2560B, reserve8192B.
- Total262144B. No reserve borrowing, reduced Guards or new sync policy.

Actual setter results, GetTiling return code, serialized TCubeTiling and all
available fields are recorded. Only GetTiling=0 and exact requested shapes
can pass. If the SDK refuses BaseN256 or returns BaseN128, stop. No T2,
N128-control retry or different dtype/output/dequant configuration is tried.

Resource admission uses actual PlatformAscendC capacities and returned depth,
step and db values. Rounded INT8 K32 and M/N16 storage contributes to L1 and
L0 bounds. N256 B is32768B per group; dbL0B=2 would require65536B (calculation,
not a predicted SDK return). Scale requires2048B beyond returned transLength;
two B-conversion tiles plus an entire FP16 C staging tile are conservatively
bounded inside Scratch. Sharing fields are retained. The SHA-bound dav_2002
GetShareUbSize consumer returns zero even if the Host raw field is nonzero;
coefficient storage remains a view inside Scratch, not a separate allocation.

These are Host acceptance/capacity bounds, not proof of new Kernel lifetimes
or NPU safety. Genuine resource results remain NOT_RUN until the server query.

## Core Policy distinction

The probe uses the approved no-argument PlatformAscendCManager::GetInstance
on the selected real device; no custom SoC or simulated capacities are passed.
The manager's aic/aiv/arch must match Stage4N's formal TilingContext snapshot,
and its all count must contain the accepted scheduling limit without exceeding
aic. A known manager8/formal7 difference is recorded, never converted into
permission to launch8 Blocks. Workspace system/arena/stride/total is calculated
for the accepted formal count and checked against the existing512MiB cap.

`REFERENCE_BOUNDS_MATCH` is not a new N256 TilingContext callback test. There
is no N256 Op Host yet. The future independent Kernel must requery and check
formal TilingContext/Core/Workspace exactly before device execution.

## Frozen layout

A remains saved INT8[20,16,128], with full-X scale unchanged. N256 logical
B[76,20,128,256] and Scale[76,20,256] are checked group-by-group against original
frozen NK/NZ/GN and the two adjacent accepted N128 packings. No new weight or
activation quantization is selected. Whole logical packing digests are saved;
large new packed files and a new Golden are not written.

All76×20=1520 group mappings are checked. Local NZ is[16,16,16], with
`(n//16)*256+row*16+n%16`; full GM index remains `row*19456+tile*256+n`.
Gate ends at tile37; Up starts at tile38 (9728/256=38); last tile75 ends at19456.
All311296 output addresses have one owner in the CPU index model. This is not
new N256 numerical acceptance.

## Independent FP16 readiness review

`fp16-review.json` examines the historical
`gate-same-weight-fp16-w8-20261021-014911/comparison.json` when available.
It compares frozen source identities, reconstructs all effective FP16 weight
bits against existing NK/KN files without writing weights, checks runtime
FP16 inputs, actual Ascend310P3 records, continuous-v1 5/30 sessions, all six
historical rotations/180 samples, and repeat/tail/postcheck output identities.
Missing or mismatched evidence remains UNRESOLVED and does not block the
independent N256 Host resource experiment.

Source review establishes pure native aclnnMm with FP16 inputs/output,
cube_math_type0 and offline W_eff=RN16(q*FP16_GN_scale), not an FP16 checkpoint.
Native internal accumulation/reduction order is not established by the
aclnnMm call alone. It does not prove the old Native OM graph's semantics, or that the current
server/driver is unchanged from that historical run. The reported0.782460ms
is historical and cannot be used as a fresh paired denominator.

Future fair comparison must use separate processes and newly admitted Runners:

- Native FP16: original FP16 X, verified effective FP16 ND_KN weights, its own
  accepted numerical reference. Guard/readonly/repeat and independent CPU
  numeric diagnostics; no W8A8 Golden substitution.
- N128 XOR: exact accepted Stage4N OPP/inputs and strict original W8A8 Golden.
- Two strict poisoned prechecks, warmup5, continuous30, timed-tail check and
  poisoned postcheck; at least three alternating fresh paired rounds.
- Strengthen the old FP16 Runner: its Golden comparison currently occurs in
  Python after process exit. New Runner must check Golden before warmup and
  at tail/postcheck, verify Poison readback and reject timed logging.
- Record actual device/driver/CANN/Runner/weights/OPP identities. Native SDK
  scheduling is recorded, not assumed to equal custom7 Blocks.
- Compare shape/workload latency and cross-scheme numerical error separately.
  Native FP16 has no asserted equivalent640-word group Guard scheme. Measured
  latency includes each implementation's actual internal checks; it is not
  pure dtype, dequantization or bandwidth attribution.

Cost inventory for that future measurement: static weight preparation and
upload excluded; FP16 weight payload99614720B versus W8 codes49807360B;
FP16 X81920B versus saved INT8 A40960B; combined UINT64 Scale3112960B is a
precomputed W8A8 input. Dynamic activation quantization, combined Scale
calculation/encoding, data transfer, full Draft/Decode remain separately
accounted and NOT_RUN. Neither scheme's prepacked latency is end-to-end.

## Server query

```bash
set -euo pipefail
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
S4O=/home/w00949577/qwen35-dflash-run/w8a8-n256-host-$(date +%Y%m%d-%H%M%S)
bash framework/custom_ops/draft_w8a8_output_dequant/stage4o/run.sh \
 --device 0 --cann-root "$CANN_ROOT" --output-dir "$S4O"
```

Outputs: tiling-report.json, resource-report.json, layout-report.json,
sdk-binding.json, stage4n-freeze.json, fp16-review.json and comparison.md.
Compiler, linker and actual dependency identities retain Stage4I's strict
checks (including CMake3.16 Makefiles dependencies, CANN_ROOT and SHA256).
The existing reviewed contract ledger is read from Stage4I, not overwritten.

Only actual Host success plus all layout/resource/reference checks yields
STAGE4O_N256_HOST_TILING_FEASIBLE. Otherwise STAGE4O_N256_BLOCKED with the
precise return code/blockers. Even success does not authorize automatic Kernel
implementation or performance tests. Return these reports and, on failure,
configure.log, compile.log or host-query.log. No SDK text is uploaded to Git.

CPU tests: `PYTHONPATH=framework/custom_ops/draft_quant/test python3 -B -m pytest -q framework/custom_ops/draft_w8a8_output_dequant/stage4o/test`.
Synthetic Host stubs use test capacities only; they are never exported as real
platform evidence. The actual server probe never supplies such capacities.

Local validation:22 CPU tests passed on macOS and Docker ARM64. The real local
SDK compiled/linked the Host C++ executable after loading CANN set_env.sh;
the executable was NOT run. An earlier bare-environment attempt failed to
resolve HAL dependencies and its log is retained outside Git. No dependency
was added or linker error ignored. Local Host libraries are not asserted
byte-identical to the server; only the server query can admit N256.
