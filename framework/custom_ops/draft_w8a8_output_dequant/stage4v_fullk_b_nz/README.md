# Full-K B-only NZ input-layout A/B

Baseline8252c1c, frozen Arena experiment
`/home/w00949577/qwen35-dflash-run/w8a8-fullk-arena-ab-20261110-054934`.
The new baseline is its **candidate normal** `DFlashFullKGuardAggregate`, not the
older strict/Full-K normal Runner or fault OPP. All its Source/Build/Host/Tile0/
Full/Fault/Benchmark evidence and actual artifacts are reverified read-only.
Historical1.573055ms/P951.57905ms is context, never a new paired denominator.

## Representation choice — necessary transpose metadata, unchanged mathematics

A remains GM ND INT8. Original logical B is `[76,K2560,N256]`.
The candidate stores **Q=B^T**, in SDK NZ layout, and consistently sets:

* Kernel B type `MatmulType<GM,NZ,int8_t,true>`;
* Kernel `SetTensorB(...,true)`;
* Host `SetBType(GM,NZ,DT_INT8,true)`.

Thus the logical operation is still `A @ Q^T = A @ B`. This transpose flag
interprets pretransposed storage; it is not an additional mathematical change.
A, Scale, output channel order, Arena XOR/OR, Guard boundaries, End, application
barriers/events, M16/K2560/N19456 and76 Tiles remain unchanged.

This choice matters: simply changing **nontransposed** INT8 B from ND to NZ still
selects `CopyNZ2NZWithTransData` in this SDK. Its interface compiles, but it retains
UB staging/Vector transpose and does not prove conversion elimination. We do
not call that variant unsupported, nor silently use it as the performance test.
The selected **transposed-storage NZ** branch directly invokes GM→L1 CopyNZ2NZ.
Exact source/queue/layout proof is in SDK_LAYOUT_REVIEW.md and sdk-review.json.

## Physical layout and byte accounting

For original `B[t,k,n]`:

```
NZ[t, k//32, n//16, n%16, k%32] = B[t,k,n]
physical shape = [76,80,16,16,32]
flat byte offset = ((((t*80+k//32)*16+n//16)*16+n%16)*32+k%32)
```

The custom Op ABI carries those bytes in a **contiguous ND rank5 container**,
explicitly validated by Host/Runner. Matmul itself sees NZ. This avoids relying
on an ACL implicit format conversion; we do not label the container FRACTAL_NZ
while guessing its logical/storage metadata. New B descriptor dimensions are
`[76,80,16,16,32]`; all three Matmul transpose declarations above agree.

Logical elements49,807,360; physical INT8 bytes49,807,360; padding0. Old and new
files retained together occupy99,614,720 payload bytes. Each variant uploads
49,807,360 B bytes once per Runner process. A40960, Scale155648, output622592
bytes remain unchanged. No activation quantization or Scale generation occurs.

Prepare reads the actual original B, packs each Tile, unpacks it and compares
**every byte**, hashes logical/physical Tiles and complete files, and rechecks
the original identities. Signed values, all76 Tile boundaries and Gate/Up
boundary at Tile38 are preserved. No old frozen weight/NZ file from a different
quantization scheme is substituted.

## Resource and execution gates

`hostplan` is an independently compiled no-argument PlatformAscendCManager query
on the selected real device. It requests exactly the new B format/transpose and
records actual GetTiling, setters, fields, capacities and resource checks. It is
**not Formal TilingContext** and its8-Core Manager result is not launch policy.
Any SDK rejection stops the candidate; no alternate width/type/config fallback.

The later new normal/fault private Host callbacks must match that actual new T1
contract and the frozen formal8/8/7/selected7 policy. New Tiling fields are not
filled from the old ND plan. Actual returned depths, steps, L1 pool, L0 double
buffers, transLength/Scale range, UB and Workspace are checked. Workspace may
legitimately differ between variants, but must match each admitted plan exactly
and never drift between that variant's runs.

Scratch202240, explicit user allocation212480, guards1536 and Reserve8192 are
kept. Total accounted UB220672. The old two-B-tile conservative Scratch envelope
is retained, not newly allocated or claimed as an active NZ consumer. B direct
copy adds no UB temporary. L1 basic blocks stay A2048/B32768 bytes for the fixed
Base16/256/128. Actual depth×block bytes must equal reported shareL1Size and fit.
L0B live bound32768×dbL0B must fit65536; no additional L0B or overlap is introduced.
The SDK's L0 pools and application lifecycle remain unchanged; all actual fields
are re-admitted before launch.

Tile0 then Full Smoke require two repetitions, all Guard/Readonly/Poison/sync
checks, exact frozen baseline FP16 bits and unchanged six/201 CPU differences.
Numeric status remains **NUMERIC_MISMATCH_UNRESOLVED**. Any new difference blocks
Benchmark. Since B input resources/queues change, the separate diagnostic OPP
also repeats the existing UB/Arena sweep and abort protocol before timing.
Fault artifacts cannot enter Benchmark. No Golden or tolerance changes.

## Server commands

```bash
set -eo pipefail
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
source "$CANN_ROOT/set_env.sh"
set -u
D=framework/custom_ops/draft_w8a8_output_dequant/stage4v_fullk_b_nz
OUT=/home/w00949577/qwen35-dflash-run/w8a8-fullk-bnz-$(date +%Y%m%d-%H%M%S)
COMMON=(--cann-root "$CANN_ROOT" --device 0 \
 --arena-root /home/w00949577/qwen35-dflash-run/w8a8-fullk-arena-ab-20261110-054934)
python3 -B "$D/run_b_nz.py" source "${COMMON[@]}" --output-dir "$OUT/source"
GATES=(--source-report "$OUT/source/report.json")
python3 -B "$D/run_b_nz.py" prepare "${COMMON[@]}" "${GATES[@]}" --output-dir "$OUT/prepare"
GATES+=(--prepare-report "$OUT/prepare/report.json")
python3 -B "$D/run_b_nz.py" hostplan "${COMMON[@]}" "${GATES[@]}" --output-dir "$OUT/hostplan"
GATES+=(--hostplan-report "$OUT/hostplan/report.json")
python3 -B "$D/run_b_nz.py" build "${COMMON[@]}" "${GATES[@]}" --output-dir "$OUT/build"
GATES+=(--build-report "$OUT/build/report.json")
python3 -B "$D/run_b_nz.py" host "${COMMON[@]}" "${GATES[@]}" --output-dir "$OUT/host"
GATES+=(--host-report "$OUT/host/report.json")
python3 -B "$D/run_b_nz.py" tile0 "${COMMON[@]}" "${GATES[@]}" --output-dir "$OUT/tile0"
GATES+=(--tile-report "$OUT/tile0/report.json")
python3 -B "$D/run_b_nz.py" full "${COMMON[@]}" "${GATES[@]}" --output-dir "$OUT/full"
GATES+=(--full-report "$OUT/full/report.json")
python3 -B "$D/run_b_nz.py" fault "${COMMON[@]}" "${GATES[@]}" --output-dir "$OUT/fault"
```

Only after successful Full and Fault review:

```bash
python3 -B "$D/run_b_nz.py" benchmark "${COMMON[@]}" "${GATES[@]}" \
 --fault-report "$OUT/fault/report.json" --output-dir "$OUT/benchmark"
```

Exact successful statuses:
`B_NZ_SOURCE_REVIEWED`, `B_NZ_PREPACK_VERIFIED`,
`B_NZ_STANDALONE_RESOURCE_FEASIBLE`, `B_NZ_BUILT_NPU_NOT_RUN`,
`B_NZ_FORMAL_HOST_MATCH`, `B_NZ_TILE0_MATCH`, `B_NZ_FULL_MATCH`,
`B_NZ_FAULT_COVERAGE_MATCH`, `B_NZ_PAIRED_MEASURED`.

Each stage creates a new directory, checks predecessors and retains failure
logs/outputs. On failure return report.json and the relevant query/compile/
runner.log, execution.json and resource-admission.json. Retain packing data,
all original output/witness/numeric files and the CANN build tree. No silent
fallback or overwriting an old experiment.

## Performance and amortization

Baseline/Candidate use separate processes/private normal OPPs. Three rounds:
A→B, B→A, A→B. Each has two Poison prechecks,5 warmups,30 measured Execute+Sync
calls, Timed Tail and fresh Poison Postcheck. All180 samples are new. Prepare
latency is separate; no Host IO/checks/logging or profiler inside measured calls.
Internal Guard work stays included. Median/P95/min/max, per-round ratios, actual
per-variant Workspace and input bytes are saved. >=1.03 and all rounds>1 is only
GO_FURTHER_REVIEW, not production approval.

Prepack records CPU permutation, reads, writes, inverse verification and full
prepare wall time separately. The complete wall time includes bookkeeping/hash
cost; file flush is not fsync durability timing. Neither cost is called free.
If measured per-call saving is positive, report:

```
ceil(prepack_cost_ms / (fresh_baseline_ms - fresh_candidate_ms))
```

Both permutation-only and complete-prepare versions are shown. If no positive
saving, no finite amortization is reported. This model excludes upload, online
Activation/Scale work and full Draft scheduling; it is not an end-to-end claim.
B weights may be reused across calls, but input-resident Kernel timing alone
cannot establish application-level payoff.

## Local checks and evidence limits

```bash
PYTHONPATH=framework/custom_ops/draft_quant/test python3 -B -m pytest -q -p no:cacheprovider "$D/test"
# Existing Docker, fresh directory: compilation only, no installation/execution
bash "$D/compile_check/check.sh" /work/build/NEW_BNZ_INTERFACE
bash "$D/local_check.sh" /work/build/NEW_BNZ_PACKAGES
```

Local tests exercise full-size lossless packing, all1520 SDK K-block source
windows, actual candidate source against a bounded CPU memory/Matmul model,
Gate/Up boundaries, Guard fault checks and CLI gates. CPU events/dequantization
are not hardware emulation. Real CANN builds use a clearly labeled synthetic
T1 only for compilation. No local NPU, real Host query, installation or timing.

An initial local build found a generated filename mismatch (`full_kb_nz`, not
`full_k_b_nz`) and duplicate Op registration; it was corrected in a fresh build.
This was our naming error, not an SDK rejection. An earlier compile probe also
had an AuxGetC0Size namespace typo, corrected without changing the configuration.
Failed local logs remain separate; no rejection was hidden or bypassed.

If actual Host/GetTiling or Smoke rejects the selected direct NZ-transposed
route, stop and report exact evidence. The next independent candidate is bounded
Matmul lifecycle/Scalar synchronization investigation, not a simultaneous change
in this A/B. End, barriers, events, Guard, A and Scale stay fixed here.
