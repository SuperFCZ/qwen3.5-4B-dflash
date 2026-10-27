# Stage 3B-2 single-group Local Output Probe

Independent `DFlashW8A8LocalDequant`, M16/K128/N64 only. No earlier Kernel,
Host, OPP, Golden, report or Source Lock is changed. CANN compilation, Local
output Tiling and NPU correctness are **NOT_RUN** until server execution.
The accepted Stage 3B-1 result is an evidence prerequisite, not proof that this
new Local NZ Matmul works. There is no performance stage or 20-group kernel.

## Source decision

The supplied CANN9/dav_m200 excerpts support a **Local NZ compile probe**:

| Source under `aarch64-linux/asc/impl/` | Observed branch | Remaining condition |
| --- | --- | --- |
| `adv_api/detail/matmul/stage/copy_cube_out/copy_cube_out_datacopy.h:107-136` | Local NZ calls CopyCo12Co2WithQuant | Current CType/template and Host Tiling must compile and run |
| Same file, `:163-200` | Local ND calls CopyCo12Co2WithoutQuant | Do not substitute Local ND and assume Scale is consumed |
| Same file, `:245-294` | INT8 source updates quant parameters; VDEQF16 has its own output-copy branch | N-block coefficient offsets/layout need numerical witnesses |
| `.../copy_cube_out/quant/quant_processor_datacopy.h:49-53,114-124` | INT8/half SetQuantVector selects VDEQF16; UpdateDataCopyParamForQuant sets VDEQ16 and local coefficient address | CopyQuantTensor allocation/copy/event lifetime remains subject to actual SDK body and runtime |
| `basic_api/dav_m200/kernel_operator_data_copy_impl.h:542-572` | INT32→half VDEQ16 branch calls copy_matrix_cc_to_ubuf with CRMODE_DEQSCALE_VDEQ16; vector-only target is excluded | Compile the AI Core branch; do not assert support for vector-only code |
| `.../copy_cube_out/copy_cube_out_datacopy_wrapper.h:290-360` | Separate NZ/ND layout handling contains Vector operations | Layout instructions are not application of the dequant coefficient |

The last low-level file's supplied SHA256 is
`8a890aa80c39380519bcba461bfd19a67060e650253622d8b49b4b2c01ea3c8f`.
Other hashes are **not guessed**: `source-review/local-review.json` binds each
original source and snapshot identity from the already accepted Stage 3B-1
build and contains numbered excerpts. All source/snapshot pairs must still
match. No SDK files are rescanned or copied into this repository.

`local-excerpts.txt` includes the fixed quant-processor range through line 260
and basic DataCopy range 1050-1105, to expose the coefficient buffer and
set_deqscale implementation that the previous grep output omitted. Return
these excerpts for review. The collector checks expected route symbols to
detect changed/incomplete evidence, but **does not use those symbols to grant
SOURCE_REVIEWED or hardware SUPPORTED**. Its status is
`STATIC_BRANCH_EVIDENCE_LOCAL_COMBINATION_UNRESOLVED`.

This is enhanced output DataCopy dequantization, not a standalone Vector
Cast/Mul coefficient calculation and not the 910B Fixpipe specialization.
The probe implements neither of those alternatives.

## Candidate configuration and resources

- ABI: A INT8 ND `[16,128]`, B INT8 ND `[128,64]`, Scale UINT64 `[64]`,
  output FP16 ND `[16,64]`; B is not transposed.
- Matmul C: `VECIN / NZ / half`. Host SetCType uses the same position/format,
  DT_FLOAT16, and SetDequantType(TENSOR). L0C remains the integer accumulation
  selected by INT8 Matmul. Host enforces single/base 16/64/128 and blockDim 1.
- SetQuantVector receives the unchanged 64 candidate uint64 words. Offset,
  scalar substitute and explicit weight/output Cast/Mul are not introduced.
- A single Iterate/GetTensorC fills Local NZ; PIPE_ALL, End, PIPE_ALL precede
  output DMA. This deliberately avoids speculative overlap. SDK coefficient
  scratch and local output remain allocated for the entire call.
- FP16 NZ `[N/16,M,16]` is the **layout under test**. Sixty-four 32-byte
  DataCopy blocks map `NZ[nb,m,0:16]` to `ND[m,nb*16:nb*16+16]`. This only
  rearranges bits. It does not use an SDK Local ND output overload, Vector
  scaling, scalar element loads, or change the rounding boundary.
- Explicit UB: 2 KiB Local NZ output. Reserved headroom: 8 KiB. SDK scratch:
  `floor32(actual_UB - 2048 - 8192)` = 251904 B on the expected 256 KiB UB.
  **Scratch is an allocation cap, not a proven SDK minimum or used-byte count.**
  Host records all actual capacities and returned tiling. It checks aligned
  transLength + 512-byte Scale storage, conservative layout space, L1 depth
  and L0 double-buffer capacities. It never passes the previous 248 KiB
  scratch allocation together with additional user buffers.
- Payload lower bounds per Cube tile: A 2048 B, B 8192 B, L0C 4096 B. Actual
  L1/L0 allocations depend on returned depth/db values. System Matmul workspace
  comes from the real platform API and ACLNN's size is recorded by the runner.

## Numerical and evidence gates

The existing offline Stage2B evidence validator checks all accepted basic,
zero and nonpower proofs, private OPP, SDK/tool/source identities. It never
executes those operators. This probe executes only:

1. scale_one;
2. scale_half;
3. per_channel_powers (within each N16 block);
4. per_channel_blocks (across N16 blocks);
5. mantissa_probe;
6. real Gate/Up group 0, from accepted Stage2C.

Each uses the original inputs, candidate reference and two saved GM outputs.
The real case retains Stage2C's full-X quantization and frozen slice validator;
it is not rebuilt from a different activation scale. New output must match
both the candidate Golden and accepted GM bytes, with two identical repeats,
Guard, Readonly and Poison readback. Nonuniform witnesses must distinguish
both an ignored coefficient and broadcast of coefficient zero to every N.
The nonpower ideal-scale comparison stays diagnostic; no Golden is adjusted.
Any numerical or execution error stops remaining cases without a fallback.

## Server execution

Run in the already validated CANN9/310P3 environment. All output directories
must be new; prior reports, SDK and OPP are read-only inputs. Build installs
only this new operator in its own output directory.

```bash
set -euo pipefail
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
S3B2=/home/w00949577/qwen35-dflash-run/w8a8-local-nz-$(date +%Y%m%d-%H%M%S)
ENTRY=framework/custom_ops/draft_w8a8_output_dequant/stage3b2/run.sh
bash "$ENTRY" prepare --device 0 --output-dir "$S3B2/prepare"
bash "$ENTRY" build --device 0 --output-dir "$S3B2/build"
bash "$ENTRY" smoke --device 0 --local-build-report "$S3B2/build/report.json" \
  --output-dir "$S3B2/smoke"
printf 'Evidence: %s\n' "$S3B2"
```

Defaults bind Stage3B-1 `w8a8-stage3b-20261027-024844`, Stage2B
`w8a8-stage2b-20261026-100533`, and Stage2C
`w8a8-stage2c-20261026-104943/report.json`. Override `--stage3b1-root`,
`--stage2b-root`, `--stage2c-report` only to select other accepted evidence.
`--build-report` refers to the historical **Stage2B** build; the new build uses
the deliberately distinct `--local-build-report` option.

Expected stages: `STAGE3B2_PREPARED_NPU_NOT_RUN`,
`STAGE3B2_BUILT_NPU_NOT_RUN`, then only on actual strict device success
`STAGE3B2_LOCAL_NZ_MATCH`. A build/tiling failure is recorded as FAILED with
logs; only analysis of its actual unsupported condition may classify it as
UNSUPPORTED. A generic compiler error is not automatically a hardware verdict.

Return `prepare/source-review/{local-review.json,local-excerpts.txt}`,
`build/report.json`, `smoke/report.json` and the per-case
`smoke/execute/*/{runner.log,execution.json}`. On compile failure include
`build/msopgen.log` and `build/build.log`; on numerical failure preserve both
actual output files and source case identities. No full Gate/Draft/Simulator
or timing is launched. The conditional fusion resource plan is in
[FUSION_PLAN.md](FUSION_PLAN.md).

CPU checks: `python3 -B -m pytest -q
framework/custom_ops/draft_w8a8_output_dequant/stage3b2/test` (one command).
Runner syntax stubs and CPU layout/arithmetic fixtures do not establish CANN
overloads, asynchronous event correctness, Local output support or NPU PASS.
