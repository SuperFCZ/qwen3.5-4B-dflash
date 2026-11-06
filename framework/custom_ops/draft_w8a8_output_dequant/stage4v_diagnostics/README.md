# Stage4V independent numerical diagnosis

**Cause UNRESOLVED. No replacement Golden. No Benchmark admission.**
Frozen source run:
`/home/w00949577/qwen35-dflash-run/w8a8-full-k-l1fix-20261106-073850`.
The six1-ULP discrepancies, negative odd dots and dot+1 matches are user-reported
observations until the all-element analyzer reads the bound server files.

## Read-only all4096 analysis

`analyze.py` verifies the failed Bounded report and linked Prepare/Host/Build/source
and installed-file identities. It finds actual-0 through the bound failure report,
not a whole-tree search. It reads full ND[16,19456] actual/expected files and selects
columns0:256 for **every row**, never treating the first8192 bytes as Tile0. Inactive
columns must retain Poison. It checks expected-0 against the original frozen Golden
and independently reconstructs original model A. It recomputes INT64 A@B for all4096
positions and verifies dot_i32; effective Scale must match actual word bits.
Nothing in the frozen directory is rewritten. First-precheck failure may mean no
second device output exists; no Repeat or complete-safety PASS is fabricated.

The fixed14-model suite is in `stage4v_rounding_models.py`:

- A: the original FP32-input / FP32-product / RN16 contract.
- B: all signed integers truncated toward zero to an even integer, followed by A.
- C: exact integer product with one RN16; FP32 input with exact product and RN16;
  and a documented Cast-mode category (odd rounding), without claiming VDEQ uses it.
- D: explicitly hypothetical conversion/product rounding,22/23-significand-bit
  product truncation, two's-complement LSB clearing and the user-supplied
  negative-odd+1 hypothesis. No parameters are optimized against device values.

Rounding is implemented with exact rational arithmetic and explicit IEEE-style
encodings; it does not depend on a potentially double-rounded NumPy float64→half
conversion for its alternative predictions. Tests exhaust all finite half encodings,
midpoints, subnormals, overflow, signs and selected FP32 integer boundaries.
The original A implementation is independently cross-checked with NumPy.

For each model the report counts explained original mismatches, unexplained ones,
new errors among originally matching elements, total bit mismatches, ULP histogram,
sign/parity distribution and midpoint-distance distributions. It reports full-tile
prediction-equivalence classes. Midpoints are the nearest magnitude-space half
rounding boundaries, including asymmetric spacing at powers of2; signed distance
means toward/away from greater magnitude, not the sign of the product.

The test suite contains **synthetic mathematical counterexamples, not user data**:
unchanged negative odd INT32 values, one fewer product-rounding step,1-ULP FP16
movement toward zero, and the same result obtained by dot+1. Thus the stated
pattern cannot logically prove an accumulator error. Real-data model fits remain
observational even if they cover all4096 elements.

Outputs in a new directory:

- `analysis-report.json`: evidence identities, Scale fields and all summary results.
- `model-summary.json`: compact statistics and indistinguishable models.
- `original-mismatches.json`: every original mismatch with dot, Scale, predictions
  and exact midpoint information.
- `elements.json`: all4096 records, including the originally correct elements.

No predicted `.bin` or file called a new Golden is generated. The script never
calls Stage4V Build/Smoke/Benchmark and cannot set any normal-stage PASS status.

## Source / Raw INT32 review

See [SDK_REVIEW.md](SDK_REVIEW.md) and `sdk-review.json` for exact CANN9 file paths,
line ranges, SHA256 and official version900 documentation links. MCB is bit36,
not bit46. Software dispatch and documented parameter fields are distinguished
from undisclosed hardware rounding.

`raw_host.py` builds and queries only a standalone Host executable, using actual
selected-device PlatformAscendCManager resources, exact SDK identities and CMake
compiled-header dependency binding. It binds original A and the Tile0 B byte slice;
no weights or Activation are re-quantized. No Raw OPP, device Runner or compute
Kernel exists here. The original normal OPP remains untouched.

Possible Host outcome:
`RAW_HOST_CANDIDATE_FEASIBLE_CONTEXT_PENDING` means standalone Tiling/resources
passed; it does **not** mean the future Raw Op's TilingContext, Build or NPU passed.
`SOURCE_REVIEWED`, `host_compile=COMPLETED`, standalone feasibility,
`formal_tiling_context=NOT_RUN`, `build_pass=NOT_RUN`, `npu=NOT_RUN` remain separate.
A rejected query stays BLOCKED, without an automatic fallback probe.

## Server commands

No prior Stage4V operation is rerun. Run these read-only-analysis / Host-only steps
in fresh sibling directories after pulling the diagnostic commit:

```bash
set -euo pipefail
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
FROZEN=/home/w00949577/qwen35-dflash-run/w8a8-full-k-l1fix-20261106-073850
OUT=/home/w00949577/qwen35-dflash-run/w8a8-full-k-numeric-diag-$(date +%Y%m%d-%H%M%S)
D=framework/custom_ops/draft_w8a8_output_dequant/stage4v_diagnostics
python3 -B "$D/analyze.py" --frozen-root "$FROZEN" --output-dir "$OUT/analysis"
bash "$D/run_raw_host.sh" --frozen-root "$FROZEN" \
  --device 0 --output-dir "$OUT/raw-host"
```

Return `analysis/model-summary.json`, `analysis/original-mismatches.json` and
`raw-host/raw-host-report.json`. If blocked, include the reported failing log
(configure/compile/ldd/query) and keep the raw artifacts. Do not paste SDK originals
or large raw weight files into GitHub. The cause remains unresolved until these
actual results and independent device accumulator/conversion evidence are reviewed.

Local checks:

```bash
PYTHONPATH=framework/custom_ops/draft_quant/test python3 -B -m pytest -q -p no:cacheprovider \
 framework/custom_ops/draft_w8a8_output_dequant/stage4v_diagnostics/test
bash framework/custom_ops/draft_w8a8_output_dequant/stage4v_diagnostics/local_check/check.sh NEW_DIR
```

Local Host compilation and SDK interface syntax checks do not query a device or
establish NPU safety. Raw budgets and fixtures are not hard-coded platform results.
All actual Tile0 analysis, Raw Host query, formal TilingContext, OPP Build, NPU,
fault/repeat validation and performance remain NOT_RUN locally.

Local validation on2026-11-06: macOS12/12 and Docker ARM64 12/12 tests passed.
The standalone Host executable compiled/linked, the Raw Matmul interface syntax
check returned0, and20 fixed SDK identities/ranges matched locally. No Host query,
compute OPP or NPU execution occurred. See `local-status.json`.

## Import-identity bugfix

The original `dependency_identity.py` collided with Stage4I during the historical
`evidence` import chain. Normal startup could reuse the cached Stage4I validator
and demand `stage4i_host.dir`; importing the diagnostic file first could instead
create a circular import. The diagnostic validator is now uniquely named
`raw_host_dependency_identity.py`, with byte-identical parsing code. No old-name
shim, sys.modules deletion, historical sys.path edit or weakened dependency gate
is used. Stage4I is untouched.

Raw Host now checks the validator module file, import-spec origin and both
function code origins against its own Diagnostics directory at startup. It binds
the file SHA256 to the source inventory, rechecks before Host query and in final
integrity validation, and writes `dependency_validator` to `raw-host-report.json`.
The expected module is `raw_host_dependency_identity`, with source SHA256
`dfb876acd73a8ac46b957c5425ea06b4ddc0550c3c418d30d7aad98952e7e8ea`.

The sibling-module audit also found `models.py` conflicting with the repository's
`models` package. It is now `stage4v_rounding_models.py`; its body is byte-identical.
`evidence`, `sdk_review`, `raw_resources` and `raw_host` have no duplicate module
file names in this repository/dependency chain, and their resolved paths are
checked by the import-order tests. No broader import refactor is included.

`import-rename-proof.json` records the old/new names and unchanged hashes. Existing
4096-element analysis results remain valid and need not be regenerated. Historical
source paths belong to commit21e6369; the reports are not rewritten to impersonate
the new source inventory. The rounding cause remains UNRESOLVED.

Fresh-process regression covers Raw Host first, Stage4I first, validator first,
application `models` first, both CLI help entries, correct/wrong targets, compiler
depfiles, Make dependency pairs, missing/disagreeing records, escaped/modified SDK
headers, post-check tampering and substituted validator functions/modules.
macOS40/40 and Docker ARM64 40/40 tests pass. Read-only validation of an existing
local CMake3.16 Raw Host build also accepted the actual197280-byte object and132
SDK headers. This is not a new Host resource or NPU PASS.

Rerun **Raw Host only**, in a new directory (do not rerun or overwrite the original
analysis or BLOCKED run):

```bash
set -euo pipefail
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
FROZEN=/home/w00949577/qwen35-dflash-run/w8a8-full-k-l1fix-20261106-073850
OUT=/home/w00949577/qwen35-dflash-run/w8a8-raw-host-importfix-$(date +%Y%m%d-%H%M%S)
D=framework/custom_ops/draft_w8a8_output_dequant/stage4v_diagnostics
bash "$D/run_raw_host.sh" --frozen-root "$FROZEN" --device 0 --output-dir "$OUT"
```

Inspect `dependency_validator`, `dependency_binding.object`, `dependency_binding.route`,
then the real query return code/resources in `raw-host-report.json`. Return that
report and the failing configure/compile/ldd/query log if blocked.
