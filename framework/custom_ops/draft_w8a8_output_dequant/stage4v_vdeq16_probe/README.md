# Stage4V VDEQ16 root-cause review

**Current decision: BLOCKED_L0C_INJECTION_CONTRACT_UNRESOLVED.**
The reviewed CANN9.0.0 mixed INT32→half VDEQ16 path consumes L0C, not UB. A
same-type UB→L0C wrapper exists and compiles, but its transfer extent and public
supported-path contract have not been established. The SDK overflow checker
explicitly bypasses this path. No execution Kernel/OPP/device Runner is created
on the strength of that compilation. See `SDK_REVIEW.md` and `sdk-review.json`.

This directory provides everything that can be admitted offline at this boundary:
fixed-file source hashing, a real SDK serializer self-check, all-field saved
Tiling comparison, strict binding of the completed Raw NPU experiment, full4096
frozen numerical analysis and closed execution gates. It neither modifies nor
reruns the original FP16/Raw operator. There is no performance mode, relaxed
Golden, generated device Golden, CPU fallback or hidden NPU launch.

The historical commits are distinct: Stage4V L1fix3138313; independent Raw279c306;
import fixbe92dee; separate Driver identity fix5001c1a. Evidence admission uses
actual report/file hashes, not a claim based on commit labels.

## Source and numerical findings

* GetTensorC → Scheduler GetResult → CubeOutBuffer's live L0C → Local NZ
  CopyCubeOut → QuantProcessor →16 N16 DataCopy operations →
  `copy_matrix_cc_to_ubuf(..., CRMODE_DEQSCALE_VDEQ16)` is the reviewed path.
* Scale words are carried unchanged. Address shifting in `set_deqscale` setup
  does not mean an INT32 arithmetic shift. Bit36 is MCB; bit46 is Sign, not MCB.
* SDK wrappers do not specify hidden multiplier precision or the FP16 tie rule.
  No fitted model is promoted to hardware fact.
* A hypothetical injection probe would need an exact unquantized L0C round-trip
  echo and full byte/range proof before its FP16 comparisons are interpretable.
* The original half Matmul GetTensorC is DstT-typed; recomputing it does not inject
  the independently measured Raw INT32 values. Changing CType cannot establish
  equivalence of the original live L0C. An external-CO1 Iterate overload does
  exist, but needs a new CO1 Host/resource and completion contract; see the
  explicitly bounded alternative design in SDK_REVIEW.md.

The existing saved standalone TCubeTiling binaries contain50 fields. The formal
Op reports bind27 fields; the remaining23 formal fields were not saved. Reports
explicitly preserve this limitation instead of presenting standalone values as
actual formal Op serialization. transLength/shareUbSize changes are identified
as output-workspace related, with unexplained exact deltas left UNRESOLVED.
Launch Block count and Workspace/Core policy are compared separately from
usedCoreNum and the single-core Matmul fields.

## Stages and outputs

| Command | Successful offline status | Scope |
|---|---|---|
| review | SOURCE_REVIEWED | Current fixed SDK files/versions checked; real SDK serializer linked and run on CPU; device admission remains BLOCKED |
| tiling | SAVED_TILING_COMPARED_FORMAL_SCOPE_PARTIAL | All50 saved serialized fields,27 formal-field bindings,23 formal omissions and launch-context differences |
| analyze | FROZEN_RAW_AND_FP16_ANALYZED_CAUSE_UNRESOLVED | Same comparisons plus4096 frozen numeric rows and six-coordinate detail |
| host/build/bounded | BLOCKED, nonzero exit | Explicit refusal; no new device APIs, OPP or Kernel are invoked |

`review` compiles a small ordinary Host executable that exercises TCubeTiling
setters/SaveToBuffer. It does not initialize ACL, obtain PlatformInfo, run
GetTiling, install an OPP or execute a Kernel. Offsets are derived with unique
markers and checked for default/padding bytes rather than guessed as an ABI.
Libraries are resolved within selected CANN_ROOT and identities recorded.

`tiling` and `analyze` require the successful review report, unchanged sources and
SDK, the original frozen failed FP16 experiment, and the actual Raw
`RAW_INT32_NPU_MATCH` report. They revalidate original frozen A/B/Scale/Oracle,
Raw Formal Host/Build, normal OPP/Runner and dependency identities, both Raw
outputs and all4096 exact integer values, safety flags, actual Workspace/Core
admission, and the six reported INT32 coordinates. They do not rerun either
operator. Missing or changed evidence blocks the new offline phase.

The analyzer keeps original UINT64 words and the original FP16 actual/CPU
reference. `elements.json` records raw INT32, Scale Word, actual/CPU/model bits,
ULP/absolute error and exact-rational midpoint distance for each element.
`numeric-summary.json` includes model definitions, original6-error explanations,
new errors among previously matching elements, sign/parity and midpoint
statistics, and indistinguishable models. `six-positions.json` isolates the six
requested coordinates. These are **original FP16 observations**, not new VDEQ16
device outputs. No new Scale words are generated or admitted.

## Server commands

Use the repository root and a new experiment directory. These commands do only
source/serialization/evidence analysis, not NPU execution. A changed/missing
report or SDK identity stops the process; historical evidence is never edited.

```bash
set -euo pipefail
git pull --ff-only
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
D=framework/custom_ops/draft_w8a8_output_dequant/stage4v_vdeq16_probe
OUT=/home/w00949577/qwen35-dflash-run/w8a8-vdeq16-review-$(date +%Y%m%d-%H%M%S)
FROZEN=/home/w00949577/qwen35-dflash-run/w8a8-full-k-l1fix-20261106-073850
RAW=/home/w00949577/qwen35-dflash-run/w8a8-independent-raw-int32-driverfix-20261109-022500

bash "$D/run.sh" review --output-dir "$OUT/review"
bash "$D/run.sh" tiling --review-report "$OUT/review/report.json" \
  --frozen-root "$FROZEN" --raw-root "$RAW" --output-dir "$OUT/tiling"
bash "$D/run.sh" analyze --review-report "$OUT/review/report.json" \
  --frozen-root "$FROZEN" --raw-root "$RAW" --output-dir "$OUT/analysis"
```

Return `review/report.json`, `tiling/tiling-comparison.json`,
`analysis/numeric-summary.json` and `analysis/six-positions.json`.
For a failure, return that phase's `report.json` and the named serializer
`compile.log`, `ldd.log` or `run.log` if applicable. Keep all original binary files
on the server. Do not upload SDK headers, generated SDK excerpts or private logs
to the public repository.

No executable `Host → Build → Bounded` recipe is supplied because the API/resource
admission is not satisfied. The CLI rejects those phases explicitly even if the
source review has completed. No `API_HOST_FEASIBLE`, `BUILT_NPU_NOT_RUN` or
`VDEQ16_NPU_MATCH` can be emitted by the current implementation.

## Local checks

```bash
D=framework/custom_ops/draft_w8a8_output_dequant/stage4v_vdeq16_probe
PYTHONPATH=framework/custom_ops/draft_quant/test python3 -B -m pytest -q -p no:cacheprovider "$D/test"
# In existing Linux ARM64 CANN container; new output directory only:
bash "$D/local_check/check.sh" /work/build/NEW_VDEQ_REVIEW
```

The interface compile check has no global Kernel entry or raw intrinsic call.
CPU tests cover field36/46 separation, positive/negative odd/even integers,
binary16 midpoint ties and Scale magnitudes, complete4096 analysis, invalid
frozen field combinations, complete serialization/coverage limits, SDK tampering
and closed device gates. Synthetic numeric observations never establish hardware
precision or a new Golden. Actual server frozen-file analysis and all new device
stages remain NOT_RUN locally. See `local-validation.json` for completed local
checks; the source/serialization evidence does not resolve hardware rounding.
