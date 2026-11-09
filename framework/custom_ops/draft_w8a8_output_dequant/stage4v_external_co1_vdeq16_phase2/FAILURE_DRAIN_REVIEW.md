# Phase2.1 failure drain review

This is a review of obligations, not a cleanup Kernel. Source IDs TP/EV/DC/QP and
full hashes are defined in EVENT_OWNERSHIP_REVIEW.md and sdk-review.json. No
End/return/PIPE_ALL shortcut is credited as universal completion. Existing
barriers are retained, but require separately identified completion edges.

## Failure classes

* **Preflight resource failure**: before user DMA, vector initialization or MMAD,
  acquire/validate all required app normal/drain IDs. Unused reservations can be
  released with no invented hardware wait. SDK registration/initialization may
  have its own event obligations; do not claim it was pure Scalar code unless
  that is separately proved. Failure there is not an automatic safe TPipe reset.
* **Ordinary data validation error** (Guard, bounds, metadata, numeric evidence):
  only branch at explicit checkpoints, never interrupt the middle of a SDK call.
  Mark failure, preserve all live tensors, drain the operations actually issued,
  then perform only proved SDK cleanup. Do not issue a wait for a producer that
  never ran; do not allocate a missing cleanup ID after work has started.
* **Hardware fault/timeout/no progress**: no finite in-Kernel completion guarantee
  is available from these headers. A stalled wait may never reach cleanup. Do
  not synthesize a success Witness, convert timeout into a successful return,
  or free/reuse device storage while a DMA may remain active. The future Host
  must record failure and require an independently verified runtime quiescence/
  abort/reset contract before reclaiming resources. No such Runner is added here.

## Reserved completion joins (healthy hardware, fully issued dependency graph)

* M completion: append reserved M_V, then reserved V_S; observe S completion.
* V completion: reserved V_S; observe S completion. Retained terminal V_M
  additionally needs the proposed M_V→V_S round trip before its ticket release.
* MTE2 completion: reserved MTE2_S; observe S completion.
* MTE3 completion: reserved MTE3_S; observe S completion.

Mappings are explicit in EV890–915,940–951,1030–1048,1102–1119 and ultimately
EV291–327. These joins establish only their named pipes' obligations. MTE3_S also
closes the preceding V_MTE3/S_MTE3 chain when they and the DMA are issued in the
specified order. MTE2_S closes a preceding MTE3_MTE2 chain for Arena reads.
Per-call V_S after VDEQ16 closes its preceding internal S_V wait and conversion.
This is conditional ordering reasoning, not an assertion of hardware progress.

The provided four files do **not** prove all SDK-internal MTE1/L1/L0A/L0B
ownership after every possible ordinary-error exit from the selected Matmul
Init/Iterate/End path. No broad search is performed to paper over that gap.
Consequently `all known app pipes joined` is not labeled `all SDK resources
released`. End may be called only once its resource/lifetime prerequisites have
been established; calling End is not itself that proof. Current overall status
remains BLOCKED on this boundary and the peak event-demand bound.

## A–H stage matrix

A state is determined by actual issued/completed markers, not just a function
name or program counter. `reserved` below means the preflight app ticket set;
SDK reservations and transient Fetch uses must be accounted separately.

| Stage | Issued / completed evidence | IDs still owned or transient | Live buffers / forbidden reuse | Ordinary-error drain and scope |
|---|---|---|---|---|
| A: MMAD not started |A0 before user async: no user work; A1 may already have Guard V, Arena MTE3/MTE2, Scale MTE2 outstanding |unused reserved set; any initialization tickets with issued work |all allocated UB; A/B diagnostic storage; no alias reuse during initialization |A0 release unused app tickets only; A1 V_S/MTE2_S/MTE3_S as needed. SDK Init cleanup remains separate unresolved proof|
| B: MMAD issued, M_V not complete |SDK issue returned, Cube and inputs may be in flight; no result reader admitted |M_V/V_S reserved; SDK input-pool obligations |CO1, Scratch, L1/L0A/B, A/B GM, all UB; no End/reset/free |M_V→V_S join, MTE2_S for outstanding GM→UB work; then prove SDK-owned input/MTE1 cleanup. Not unconditional safe release|
| C: Echo generated, Raw GM incomplete |M_V and Echo V_S complete only if observed; Raw MTE3 may read Echo UB |V_MTE3/MTE3_S active/reserved; other tickets held |Echo UB and Raw GM immutable until completion; CO1 and Scale live |MTE3_S; complete any V work via V_S. A cannot migrate yet. B diagnostics remain separate. SDK cleanup prerequisites still apply|
| D: Raw GM done, Witness not moved |Raw MTE3_S observed |A manual S_V not yet used; B has no migration |A Witness/Arena still in FP16; do not start VDEQ16. B diagnostics fixed |No new async migration required to abort A. Finish prior joins, retain live metadata, use failure-only publication after safe cleanup|
| E: Witness migration |A Scalar reads/writes may be incomplete; no VDEQ16 admitted until S_V/V_S handoff completes |A S_V/V_S tickets; B stage absent |A FP16 source and Echo destination both live; no overwriting either |complete Scalar sequence and S_V→V_S handoff before release; ordinary branches only after the copy, not midway. Hardware interruption follows fault policy. B eliminates this state|
| F: VDEQ16 in flight |0..16 calls issued; a call return does not prove V completion |transient SDK Fetch(S_V), reserved V_S/V_M |CO1, all Scale, current FP16 destination, diagnostics, Scratch; no End/free/rebind |reserved V_S after the last issued complete SDK call; confirms prior S_V wait and V data transfer before next call or cleanup. V_M/reverse edge retained. Unissued calls are not waited on|
| G: FP16 GM incomplete |all V consumers and their joins completed; FP16 MTE3 may still read UB |V_MTE3/MTE3_S and remaining terminal tickets |FP16 UB and GM output stay valid; diagnostics live |MTE3_S; no FP16 reuse/free before observation. No success publication just because transfer was issued|
| H: Witness not published / in flight |all prior output and safety checks must already have completed |S_MTE3/MTE3_S plus terminal reservations |Witness UB and GM, all guarded allocations until final checks |if no DMA yet, publish failure only for an error; if in flight, MTE3_S before release. An already written success-looking buffer is not Host PASS if execute/sync/guards fail|

Both consumer branches must finish before CO1 is released. SchemeB never aliases
numeric buffers, making the live-set table stable. SchemeA additionally needs
correct active metadata handles at D/E and after migration; a partially copied
Witness cannot be trusted as a valid final record.

## Cleanup ordering and admission

Proposed order: secure resources → initialize → issue operations → observe
required completion joins → record validation error if any → finish all already
issued work → prove SDK cleanup prerequisites → End → final Guard/Arena checks
with their own completion edges → publish **failure** or success as appropriate
→ complete publication → release event reservations and buffers. There is no
new numerical work or retry in an error tail.

Do not allocate cleanup buffers/IDs after discovering an error. Do not destroy
the TPipe while its buffers/events can still be consumed. If a Guard indicates
the diagnostic buffer itself may be corrupted, the Host must not trust its
Witness; retain logs/available raw outputs and fail closed. Failure after early
Raw publication preserves that data but cannot grant interpretation of FP16.

Required missing evidence is finite and specific:

1. For this fixed SDK/CO1/full-K path, a per-HardEvent peak reservation bound for
   SDK Init/Iterate/End with the proposed app ticket set (and a valid free S_V
   slot during the direct conversion window).
2. For the proposed checkpoints, an input/MTE1/queue lifetime proof sufficient
   to call End and destroy resources after ordinary error without crediting
   PIPE_ALL or return as universal drain.
3. Independent review and new Formal Host for B; future runtime fault/timeout
   quiescence policy cannot be fabricated by this source-only review.

A permanent hardware failure cannot be made drainable by adding an event wait.
Its safe outcome is BLOCKED/no success evidence, not an unproved cleanup PASS.
`phase21_contract.Drain` tests these requirements using external completion/proof
tokens; it deliberately cannot invent them from a barrier call or a timeout.
