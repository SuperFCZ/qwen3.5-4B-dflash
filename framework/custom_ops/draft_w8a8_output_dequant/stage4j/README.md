# Stage4J: independent N128 T1 correctness

`DFlashW8A8N128Gate` retains the Stage4B **full Guard** algorithm. This is a
new private OPP, Host, Kernel and ACLNN Runner. Historical stages, OPPs,
Source Locks and Golden files are read-only. CANN operator build and NPU
correctness are **NOT_RUN** at development delivery. Benchmark, Profile,
online activation quantization and Draft integration have no entry point.

## Fixed data and arithmetic

| Input/output | Layout | Bytes |
|---|---|---:|
| A INT8 | `[20,16,128]`, group,row,k | 40960 |
| B INT8 | `[152,20,128,128]`, tile,group,k,n | 49807360 |
| Scale UINT64 | `[152,20,128]`, tile,group,n | 3112960 |
| O FP16 | `[16,19456]`, row,global channel | 622592 |

Prepare binds the accepted Stage4I.1 four reports, its serialized T1 and SDK
identities, and Stage4B's accepted NPU outputs/resource records. It reads saved
full-X INT8 A, never calculates a new activation scale. Every N128 tile/group
is checked against frozen NK and NZ weights, original GN scales and both
corresponding N64 inputs. Packing concatenates the **logical N axes**, not
arbitrary byte streams. B/Scale whole-file SHA256 must equal the Stage4I
layout report. All 3040 groups are checked.

An independent exact INT64 dot and the unchanged effective-scale codec
recompute every FP16 partial. These must match the saved Stage4B CPU partials
in both N64 halves. Ordered FP32 accumulation must match the original full
Stage4B Golden. The final Golden identity remains the original file; Prepare
does not write a new final Golden. Native OM comparison is not enabled.

The device path is unchanged apart from N128 sizes and the explicit bounded
selection: INT8 Cube, per-group `SetQuantVector`, VDEQ16 to FP16 Local NZ,
Cast FP32, ordered g=0..19 Add, then final FP16 Cast. Each group retains
`SetLocalWorkspace`, A/B/Scale updates, `Iterate(false)`, `GetTensorC`, `End`,
every barrier/event and all five Guard scans. No intermediate GM partial,
FP16 accumulation, FMA, async overlap, double buffering or T2 fallback.

Local NZ index is `(n//16)*256 + row*16 + n%16` for eight N16 blocks.
The GM output index is `row*19456 + tile*128 + n`; Gate/Up split is Tile76.

## Resource and source admission

| Allocation per Core | Bytes |
|---|---:|
| SDK Scratch | 226816 |
| Two FP16 local buffers | 8192 |
| FP32 conversion temporary | 8192 |
| FP32 sequential accumulator | 8192 |
| Five prefix/suffix Guard pairs | 2560 |
| Unallocated reserve | 8192 |
| Total | 262144 |

The accepted dav_2002 consumption contract records raw `shareUbSize=1024`,
effective SDK shared UB reservation zero and coefficient scratch view
`[16384,17408)`. Its source SHA256 checks run before each new phase; the
server Host archive must match the accepted identity. No SDK text is copied
into this directory.

Build generates a small `t1_contract.h` from the **verified real report**.
The new Host compares all reported T1 fields (including base/singleCore,
depth/db, steps, sharing and transLength), UB/L1/L0 capacities, Scratch and
system Workspace size against those recorded values. Any difference rejects
Tiling; it does not choose a different configuration. The actual Host values
are printed as `STAGE4J_T1 ...` and bound in each execution report.

Core counts are deliberately not copied from the standalone GetInstance
probe. Formal `TilingContext → PlatformAscendC` supplies aic/aiv/all/arch.
The unchanged policy requires dav_2002, nonzero counts, aic==aiv, all<=aic;
selected=GetCoreNum(). Auto=min(152,selected), plus a separate 2-Core run.
The new Host-only proof must also match the accepted Stage4B **formal** Core
tuple. A changed tuple blocks NPU admission for review, rather than taking
the standalone probe's eight-Core result as a new scheduling allowance.
The established system-plus-private-Arena size and pointer formulas are
unchanged. Capacity equality can block a differing TilingContext; such a
failure requires review, never a fallback or skipped check.

`source_audit.py` derives the exact allowed Kernel/Host/Runner changes from
Stage4B and rejects any extra change. Each phase writes `source-audit.json`
and `source.diff`. Build records generated Host contract identity, installed
Kernel ELF/SHA256, metadata, API, Host library and complete private OPP files.
Compiler macro evidence retains its existing limitations; absent pragma
expansion is not treated as proof of a bad Workspace pointer.

## Separate gates

1. **prepare**: CPU layout/numerics and historical identities.
2. **build**: independent operator/Runner build and installation under the
   new output directory only. No Kernel launch.
3. **host-only**: one mode21/core_limit1 workspace query. The Runner uses ACL
   input buffers as before, but does not submit the operator. This proves the
   new Op Host contract, not numerical correctness.
4. **bounded**: auto Core schedule; Tiles0,75,76,151 individually, each with
   mode1 (Group0), mode20 (Group19), then mode0 (final). Every selected tile
   runs all20 groups in every mode. Other tiles do no Matmul work and must
   remain Poison. Each mode runs two different poisoned repeats: 12 modes,
   24 executions, with only2048 active elements per run. This cannot claim a
   complete Gate PASS.
5. **full**: requires and revalidates all bounded output proofs from this
   exact Build/Host/input/Golden. Auto then2-Core, each mode21 scheduling
   witness followed by mode0 numerical result, each twice. Both complete
   numerical results must match all311296 Golden elements and one another.

Modes use an independent `tile_id` attribute: -1=full, 0..151=one tile.
Mode0 returns final output; modes1..20 retain the selected group's partial
until all20 groups complete; mode21 is a scheduling witness only. The
provided bounded command fixes only the approved four tiles and Groups0/19.

Readonly, output Poison, GM Guard, per-group and per-tile UB Guards, Arena
Guards, workspace admission and Repeat remain gates. In bounded runs, only
active columns are compared between repeats, since unused columns carry
different Poison patterns; all unused bytes must retain the correct pattern.
No phase proceeds after a failure. No CPU fallback or tolerance relaxation.

## Server commands

Use the accepted server environment. Each output subdirectory must be new;
never reuse a failed directory. The defaults bind the accepted Stage4B
`w8a8-full-gate-20261027-093617/smoke/report.json` and Stage4I.1
`w8a8-n128-host-shareub-20261030-083337` under the existing run root.

```bash
set -euo pipefail
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
S4J=/home/w00949577/qwen35-dflash-run/w8a8-n128-t1-$(date +%Y%m%d-%H%M%S)
ENTRY=framework/custom_ops/draft_w8a8_output_dequant/stage4j/run.sh

bash "$ENTRY" prepare --device 0 --output-dir "$S4J/prepare"
```

Only after `STAGE4J_PREPARED_NPU_NOT_RUN`:

```bash
bash "$ENTRY" build --device 0 \
  --prepare-report "$S4J/prepare/report.json" --output-dir "$S4J/build"
```

Only after `STAGE4J_BUILT_NPU_NOT_RUN`:

```bash
bash "$ENTRY" host-only --device 0 \
  --build-report "$S4J/build/report.json" --output-dir "$S4J/host-only"
```

Only after `STAGE4J_HOST_TILING_ONLY`; inspect actual T1/Core/Workspace logs:

```bash
bash "$ENTRY" bounded --device 0 \
  --build-report "$S4J/build/report.json" \
  --host-report "$S4J/host-only/report.json" --output-dir "$S4J/bounded"
```

Only after `STAGE4J_BOUNDED_MATCH`:

```bash
bash "$ENTRY" full --device 0 \
  --build-report "$S4J/build/report.json" \
  --host-report "$S4J/host-only/report.json" \
  --bounded-report "$S4J/bounded/report.json" --output-dir "$S4J/full"
```

Only real complete device acceptance may emit `STAGE4J_FULL_GATE_MATCH`.
Stage4K performance work is not started by any command.

## Reports and failures

Return each phase's `report.json`. Prepare also writes `data/case.json` and
`data/progress.json`; Build retains `msopgen.log`, `build.log`, `install.log`,
Runner CMake/build/ldd logs and the generated `op_host/t1_contract.h`.
Do not publish SDK or server logs to GitHub.

On Host/NPU failure retain the failing folder under
`execute/limit-*/tile-*/mode-*` including `runner.log`, `execution.json`,
`resource-admission.json`, and any `actual-*.bin`. Numerical reports include
first row/global-channel/Tile, expected/actual bits, Scale words and bound
CPU partial bits. A final-sum mismatch cannot identify which device group
failed by itself: it records `UNRESOLVED_FINAL_SUM` with all20 candidate
partial/Scale entries instead of inventing an actual intermediate result.
Group0/19 actual outputs from bounded diagnostics remain independent evidence.
Failed outputs and all historical records remain unmodified.

## Local regression

```bash
PYTHONPATH=framework/custom_ops/draft_quant/test python3 -B -m pytest -q \
  framework/custom_ops/draft_w8a8_output_dequant/stage4j/test
python3 -B framework/custom_ops/draft_w8a8_output_dequant/stage4j/source_audit.py
bash -n framework/custom_ops/draft_w8a8_output_dequant/stage4j/run.sh
```

CPU tests exercise full synthetic frozen packing, strict historical binding,
N128/NZ/ND index coverage, codec/rounding, exact T1 admission, phase gates,
all Guard words and fault injection. A C++ test double runs the actual new
Kernel control flow with independently cached partials; it validates buffer
bounds, A/B/Scale offsets, End/reset, output coverage and retention. It does
not emulate Cube, VDEQ16, physical concurrency, DMA completion or NPU safety.
