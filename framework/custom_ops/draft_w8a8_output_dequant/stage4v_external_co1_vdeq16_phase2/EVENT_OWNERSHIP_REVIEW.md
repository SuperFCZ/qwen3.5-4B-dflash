# Phase2.1 Event ownership review

Scope: coupled dav_2002, one TPipe, one application instruction stream, no pool
reset or concurrent allocator in the interval being reasoned about. Source is the
installed CANN9.0.0, not another architecture/version. No Kernel is implemented.

## Exact source binding

Root `/usr/local/Ascend/cann-9.0.0/aarch64-linux/asc/`:

| ID | Path | SHA256 | Lines |
|---|---|---|---|
| TP |impl/basic_api/kernel_tpipe_impl.h|7a91fe65d16c3775fa57d15ebb63a37b275c0c9ee1c269bca4d76355e2873c8d|443–484|
| EV |impl/basic_api/kernel_event.h|d974ecee61e39402dad6fa368e036b538bc1f69b05989b6b3a9fc9c946bd609c|125–132,227–237,291–327,890–915,922–951,1030–1048,1093–1119|
| DC |impl/basic_api/dav_m200/kernel_operator_data_copy_impl.h|8a890aa80c39380519bcba461bfd19a67060e650253622d8b49b4b2c01ea3c8f|542–572,1057–1107|
| QP |impl/adv_api/detail/matmul/stage/copy_cube_out/quant/quant_processor_datacopy.h|10438170dde8cb131bf3ba9443955d101d5e7936aa27dc38127677cb1b898c7c|113–124,143–158|

Hashes equal the existing `sdk-review.json`. The directly relevant enum entries
EV31–71 were additionally read in the same hash-bound file: the target HardEvent
names have distinct numeric values. No SDK tree scan or raw source publication
was performed. The user reports the server audit at `w8a8-co1-phase2-design-
20261109-114559/design-review-report.json` completed with all12 hashes matched.
That historical report remains unchanged; the new audit rechecks current files
and evidence rather than rewriting its old source inventory.

## Answers from implementation, not names

1. **Alloc reserves**: TP447–453 selects `eventPool_ + EventToIndex(evt)`, finds a
   zero bit with sff0, and sets that bit via sbitset1 before returning.
2. **Fetch does not reserve**: TP467–484 selects the same pool and returns its
   first zero bit; it performs no occupancy write.
3. **Fetch cannot return a still-reserved valid ID**, under this single-owner
   stable-bitmap contract. It can return an ID after its bit was cleared, even
   if the previous hardware operation using that ID has not completed.
4. **Release is bookkeeping only**: TP456–464 clears the occupancy bit. There is
   no WaitFlag, cancellation, hardware-completion check or memory drain there.
5. **Pools are separate by numeric HardEvent on2002**: EV125–132 and227–237 use
   the full HardEvent count and direct enum index. M_V and V_M, for example, are
   different pools; the same numeric ID in two different pools is not this
   same-pool collision. This is not generalized to split-core architectures.
6. **Same-type collisions are possible**: Fetch leaves its choice software-free.
   Two Fetch calls can return the same ID; an application Alloc during an SDK
   in-flight use can select it. Early Release similarly exposes an active ID.
   The bitmap neither records pending Set/Wait nor enforces hardware lifetime.
7. **Direct VDEQ16 fetches S_V**: DC1075–1107 programs the coefficient address,
   then Fetch(S_V), SetFlag(S_V), WaitFlag(S_V), then dispatches the matrix copy.
   It does not Alloc or Release that ID. DC559–562 selects the VDEQ16 opcode.
8. **Sixteen calls are sequential at the C++ issue level**, with a complete S_V
   Set/Wait pair emitted for each call. The focused source alone does not prove
   that each V copy physically finishes before the next C++ call changes the
   dequant register or before a free ID is reused. Do not call this sixteen
   hardware-completed operations merely because the loop returned.
9. **Internal pairing exists**, but a matching emitted pair is not a Scalar
   completion witness. EV291–327 shows the set_flag/wait_flag calls and actual
   source/destination pipelines, not a C++ completion return value.
10. **Manual S_V can finish and release before SDK Fetch conditionally**: emit its
    S_V pair, then a separately reserved V_S pair behind the relevant V wait/work.
    Once the Scalar V_S wait has actually completed, the prior V work and S_V wait
    precede that observation. Only then Release the manual S_V ID. Releasing it
    earlier is not justified. Availability and the failure drain for that V_S
    ticket must be secured before starting the work.

The SDK Matmul QuantProcessor's coefficient copy additionally Fetches
MTE3_MTE2 and MTE2_V (QP151–157). That is a separate routine from the direct
mixed-type DataCopy. Phase2 plans its own unchanged GM→UB Scale copy and direct
VDEQ16; it must not assume QuantProcessor secretly supplies those fences.
All ranges/debug assertions are admission requirements, not production error
handling: Alloc's range assertion is debug-only, and its code then mutates the
bitmap. Check Fetch availability before Alloc, with no intervening pool mutation,
and verify the returned ID. When all usable bits are occupied, no valid ID is
proven; do not assume a clean -1 failure return or call an out-of-range Alloc.

## Time-indexed S_V ledger

Let I={0..7}, R0 be existing reservations, a the valid app-selected ID, and
s=min(I minus the current reservations). Values here are variables, not hard-coded
runtime IDs. `in use` and `wait completed` are distinct from software occupancy.

| Time | Free / allocated bitmap | SDK Fetch | Hardware obligation | Release eligibility |
|---|---|---|---|---|
| A0 preflight |I−R0 / R0|not called|none owned by app yet|idle reservations only|
| A1 app Alloc(S_V) |I−(R0∪{a}) / R0∪{a}|would exclude a|app has not signaled|a may release while unused|
| A2 migration + app S_V Set/Wait issued |same|must not enter SDK yet|a wait/work may still be pending|a NOT releasable|
| A3 matched V_S completion observed on S |same|not yet called|app S_V wait completed|a releasable|
| A4 app Release |I−R0 / R0|may now choose a safely|no old app obligation|no app-owned S_V|
| B0 conversion window |I−R0 / R0|s, require valid|no app S_V exists|SDK Fetch is not owned/released by app|
| j.0 before DataCopy, j=0..15 |same frozen bitmap|s each time if bitmap unchanged|previous j−1 completed by below join|none for fetched s|
| j.1 internal Set/Wait issued |same|still returns s even though in use|s wait and VDEQ16 may be pending|NO app allocation/release on S_V|
| j.2 appended V_S Set/Wait completes on S |same|s still software-free|prior S_V wait and V copy completed|next call may reuse s|
| after j15.2 |same|not called again|no conversion use outstanding|return to reviewed allocation policy|

The conservative proposed schedule adds a V_S Scalar completion join after **each
N16 VDEQ16 call**. This is correctness instrumentation, not optimization. It
avoids claiming hardware serialization from SDK C++ ordering alone. Do not
change SDK Fetch/Set/Wait or add a hard-coded internal ID. During the conversion
window, forbid application S_V Alloc/Release/Fetch-use and any other same-pool
user; the SDK must have at least one valid free slot. SchemeB eliminates the
manual S_V transfer entirely. SchemeA requires the A1–A4 handoff above.

`phase21_contract.sv_trace()` (empty prior reservations in its example) generates all sixteen modeled intervals, recording
free, allocated, Fetch candidate, in-use, issued/completed-wait and releasable
sets. These tables are **abstract proofs of conditions**, never device telemetry.

## Every manual HardEvent pool (B)

For each X below, use the same ledger transitions: preflight idle reservation
`a_X`; signal issued→in use; wait issued→still in use; Scalar-observed completion
→releasable; release only after last use. During a pending use, bitmap=allocated
and Fetch(X) must exclude `a_X`. An unrelated pool remains unchanged.

| X | Producer / consumer | Completion needed before releasing/reusing a_X |
|---|---|---|
| M_V |MMAD M→V|followed by V_S observed on S; V wait emission alone is insufficient|
| V_M |last V reader→future M writer|proposed V_M→M_V→V_S round trip, reusing previously completed reserved tickets; S observes the final wait before release|
| V_S |V work/Guards→S|actual matched S wait completion|
| MTE2_V |Scale load→V|V-side wait plus V_S Scalar observation before release if reused|
| V_MTE3 |UB producer→GM DMA|MTE3 chain completion, observed by MTE3_S|
| MTE3_MTE2 |Arena initialization write→read|MTE2 chain completion, observed by MTE2_S|
| MTE2_S |Arena/Scale/input DMA→S|actual matched S wait completion|
| S_MTE3 |Witness stores→GM publication|MTE3_S observation after publication; no premature return|
| MTE3_S |GM write→S|actual matched S wait completion|

For terminal V_M ownership, the proposed extra M_V→V_S round trip is issued
behind the M-side V_M wait. It closes that wait on Scalar without inventing an
M-side return guarantee. Reuse of the earlier M_V ticket requires its previous
chain already completed; this does not prove unrelated SDK MTE1/queue cleanup.

Reserve required control/drain IDs before user asynchronous producers; no late
Alloc for ordinary cleanup. The simple proposal retains them until terminal
completion. But successful app reservation proves only app capacity, **not**
that Matmul's own Init/Iterate/End can operate with the remaining slots. These
four files do not give the selected Matmul path's peak reservation demand for
each shared pool. Phase1's dynamic reservations and real PASS do not prove this
new pre-reservation schedule. That separate SDK-demand bound remains BLOCKED.

## Conflict verdict

S_V conflict is **not inevitable**. Under stable single-TPipe ownership, no app
S_V activity during the SDK window, one valid free SDK slot, and per-call Scalar
completion joins, the app/SDK conflict is excluded. Early release, concurrent
same-pool allocation, exhaustion or beginning SDK conversion before A's handoff
is unsafe. The minimum A fix is completion-before-release and a frozen SDK-use
window; B removes the application S_V need. This closes the narrowly defined
S_V collision question conditionally, not the entire Kernel cleanup/resource
contract or implementation admission.
