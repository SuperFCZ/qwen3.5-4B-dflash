# Proposed ABI and acceptance — no execution implementation

`abi-design.json` is a design artifact, **not** an msopgen operator description.
Proposed unique name: `DFlashExternalCo1Vdeq16Phase2Probe`; distinct ACLNN symbols,
private OPP root and separate processes from Phase1. Nothing is registered here.

| Tensor | Shape / dtype / format | Bytes | Layout / safety |
|---|---|---:|---|
| A |[16,2560] INT8 ND|40960|exact original frozen bytes, readonly |
| B |[2560,256] INT8 ND|655360|exact Tile0 bytes, no transpose/re-quantization |
| Scale |[256] UINT64 ND|2048|exact frozen words, readonly, per-N16 slices |
| Raw Echo |[16,256] INT32 ND|16384|compact stride256; strict historical Raw bitwise |
| FP16 |[16,256] FP16 ND|8192|compact stride256; all elements saved and compared |
| Witness |[32] UINT64 ND|256|proposed protocolV2; not Phase1's old offsets |

Input payload698368B; output payload24832B, plus separate GM guards and actual
Workspace. This is not a latency/inference-cost claim. Host must verify all
shapes, dtypes, strides, source identities, actual device, coupled Core policy,
BlockDim1, full50 serialized TCube fields and the new ABI's Workspace. Historical
4195328B is a reference, never an automatic accepted new request. Reuse of prior
Tiling numbers cannot replace new Formal Host admission.

WitnessV2 keeps8 header words, then10 event IDs and10 completion/visit counts,
then4 phase flags. Fields include failure code, actual L0C address/extent,
explicit user-buffer end, Guard completion and **no-L0C-canary** flag. The two
new events are MTE3_S and S_V. Validate actual IDs and ownership; invalid metadata,
old protocol version or missing completion blocks. Metadata must move intact
from A to B before FP16 overwrite and be published only after all final checks.

## Gates

1. **Source Review**: immutable Echo/Raw/CO1/Frozen/offline-analysis evidence,
   current SDK hashes, exact API/geometry/alias contracts and independent design
   review. Current event failure/ownership blocker must be closed.
2. **CPU Regression**: all-field/schema/ABI and source audits, frozen identities,
   Scale byte slicing, N16 physical bounds, NZ/ND bijections, Guard faults,
   event/lifetime rejection cases and report protocol. CPU results are not NPU.
3. **Private OPP Build**: future unique real Kernel/Host/API/Runner identities,
   SDK/compiler/runtime bindings. No fake intrinsic, simulated VDEQ or fallback.
4. **Formal Host**: actual registered TilingContext;50 fields/capacities/Workspace,
   parent248B/cube200B/offset48 verified through actual SDK serializer, not a
   copied standalone struct. No Kernel execution at this stage.
5. **Device Observation**: two independent poisoned runs; save all outputs before
   rejecting a check when safe readback is possible. Kernel synchronization error,
   Raw mismatch, any Guard/readonly/witness failure or missing output blocks.
6. **Numerical Analysis**: only admitted complete observations may be interpreted.
   A relationship classification is not a replacement Golden or Benchmark gate.

The Runner must require real device execution/no CPU fallback, stream completion,
all4096 Raw values matching both historical Raw and independent INT64 Oracle,
Raw Repeat equality, complete4096 FP16 outputs, FP16 bitwise Repeat equality,
all input readonly, GM/UB/Workspace guards, exact ranges and valid WitnessV2.
FP16 poison patterns0x7e01 and0x7fff are distinct NaNs; the frozen case has finite
references, so remaining NaNs/infinities or poison indicate an invalid/unexpected
observation and cannot be ignored. INT32 poisons0x7f7f7f7f/0x80808080 are outside
the safe dot range. Use raw-byte comparisons, not tolerance expansion. Maintain
L0C_RANGE_ONLY_NO_CANARIES; Echo equality does not create missing Canary coverage.

A partial Raw output written before later failure is preserved but not promoted
to overall PASS. If Raw does not match, do not explain FP16 differences. If Raw
matches and all safety/Repeat checks pass, report **DIAGNOSTIC_OBSERVATION_VALID**
separately from the relationships MATCHES_FROZEN_ACTUAL / MATCHES_CPU_GOLDEN /
MATCHES_BOTH / MATCHES_NEITHER. There is no unconditional VDEQ16_NPU_MATCH rule
that can quietly choose the more favorable comparator. No new mathematical
Golden is created from a first device output.

## CPU plan

Already implemented pure contract tests cover:

* exact2048-byte Scale slices and no mutation; identity tampering rejection;
* every source/destination/Scale N16 range, alignment, endpoint and unique element;
* complete FP16 NZ→ND coverage, including Gate layout stride distinction;
* all512 UB redzone words with every single bit (16384 mathematical fault cases);
* migration before Raw DMA completion, VDEQ before S_V, End before consumers,
  output publication before completion and early event release all rejected;
* no stale completion token may authorize a later producer; IDs must be0..7;
* resource sum239104, no alias in Guard/Reserve, no extra L0C;
* ABI sizes, separate numeric references and wrong-status/failed-safety rejection.

Later implementation tests must additionally compile the actual device sources
against the admitted SDK, test actual generated ABI/report producers across
languages, preserve all prior tests and audit source changes. Mocked events are
only an abstract ordering checker. They never prove `wait_flag` completed on NPU.

## Numerical report design

For each4096-element pair, save Raw INT32, original UINT64 word, FP16 actual bits,
original Stage4V actual bits, original CPU bits, declared model predictions,
absolute error, ordered-half ULP distance, direction, and exact-rational
midpoint distance. Reference extraction is `historical_full.reshape(16,19456)
[:,0:256]`, not the first4096 words. Store all raw files and SHA256 identities.

Report all six coordinates separately: (0,84),(1,53),(1,201),(5,138),(7,235),
(15,210), plus new mismatches among the original4090 matches. Stratify by sign,
odd/even integer, Scale magnitude and midpoint conditions. Reuse predeclared
mathematical models; do not select/edit a model or Scale to manufacture PASS.

| Evidence category | Allowed conclusion |
|---|---|
| Direct observation |Specific raw FP16/INT32 bytes, Repeat, Guards and identities |
| Source/documentation |Selected operation, address-space/type/parameter contracts |
| Mathematical derivation |N16 coverage, exact midpoint distances, declared model results |
| Hypothesis |Which hidden rounding/truncation mechanism could explain differences |

Reproducing the original six differences on an Echo-verified CO1 value would
support a dequant/output-path explanation. It does not prove the original FP16
Matmul's live L0C was identical or identify every internal rounding stage.
Matches with CPU instead would require further path/parameter/producer analysis;
matches with neither are preserved, not fitted away. Hidden VDEQ precision and
tie-breaking remain UNKNOWN unless independently evidenced.

## Outstanding issues / risks

* **Implementation blocker**: event ownership and resource-exhaustion cleanup,
  including app S_V versus SDK Fetch(S_V), need a complete reviewed reservation
  schedule. Nominal arrows alone do not prove every error path safe.
* **Review gate**: this design has not passed an independent review. No Kernel
  development is authorized by the current BLOCKED report.
* **Evidence gate**: local access cannot read server experiment files. Run the
  read-only audit to verify actual SHA256s; supplied summaries alone are insufficient.
* **Future execution gates**: new ABI Formal Host/Workspace, compiler behavior,
  alias migration and all numerical/safety observations remain NOT_RUN.
* **Coverage limitation**: no L0C Canary protection; cannot claim unknown OOB
  absence solely from full Echo equality. Range/alignment proof is required.
* **Arithmetic unknown**: SDK does not expose full internal rounding precision.
  This is an investigation target, not permission to invent a hardware model.

No additional Buffer, relaxed tolerance, old report rewrite, profiling run or
full-SDK scan is proposed to work around any of these limitations.
