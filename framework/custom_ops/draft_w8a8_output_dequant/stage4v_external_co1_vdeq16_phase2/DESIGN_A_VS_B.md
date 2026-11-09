# Phase2.1 design comparison and selection

**Select B for independent review, not for execution or implementation admission.**
It removes migration and application S_V from the normal conversion path. Space
arithmetic and simpler ownership are reasons to prefer it, not proof of actual
TPipe allocation, event capacity or device safety. Overall status remains BLOCKED.

| Item | A: alias migration | B: dedicated diagnostic allocation |
|---|---|---|
| Explicit UB |230912B|232192B|
| Reserve, untouched |8192B|8192B|
| Accounted UB |239104B|240384B|
| Remaining margin below fixed top reserve |23040B|21760B|
| Added payload / guards |none|Arena512 + Witness256 + prefix/suffix512 =1280B|
| UB TBuf objects |4|5; plus separate external CO1 TBuf|
| L0C active |16384B|16384B, unchanged|
| Metadata location |FP16 then Echo|fixed independent buffer throughout|
| Raw output → metadata dependency |MTE3_S before reuse|no numerical-buffer alias; MTE3_S still needed for completion/error drain|
| App S_V |migration handoff and completion-before-release needed|none; internal SDK S_V only|
| Error state E |two live metadata copies / migration in progress|eliminated|
| Event resource problem |all normal/drain tickets plus app/SDK S_V handoff|manual ticket demand remains; SDK still requires a free S_V ID|
| Formal Host |new ABI/lifetimes require re-admission|new ABI **and increased allocation** require re-admission|
| Hardware timeout |no invented successful drain|same limitation|

## Exact B allocation

Keep all four original raw allocations and addresses unchanged:
Scratch `[0,202752)`, Echo `[202752,219648)`, Scale `[219648,222208)`,
FP16 `[222208,230912)`, including their existing prefix/suffix Guards. Append
one diagnostic TBuf of1280B, before Matmul registration:

| Region | Absolute half-open UB range | Bytes |
|---|---|---:|
| diagnostic prefix Guard |[230912,231168)|256|
| Arena temporary |[231168,231680)|512|
| Witness |[231680,231936)|256|
| diagnostic suffix Guard |[231936,232192)|256|

All boundaries are32B aligned. The two payload views are disjoint and never
alias Echo, Scale or FP16. Guards now total2560B =640 uint32 words. Total payload
229632 + Guards2560 =232192; adding Reserve8192 gives240384. Reserve remains
`[253952,262144)`; the21760B gap is unallocated, not borrowed Reserve.

L0C stays first/only caller allocation `[0,16384)`,1024B aligned, with no Canary
claim. Additional UB does not imply additional L1/L0A/L0B/L0C. Nonetheless, actual
TPipe addresses/high-water behavior and Matmul registration must be verified;
source accounting uses the already-bound TPipe/shared-resource contracts. Effective
shared UB0 and external CO1 shared-L0C reservation0 must still match the actual
selected path; no extra SDK consumer is silently ignored. Scratch remains202240.

New Formal Host must report actual PlatformInfo/Core policy,50 TCube fields,
transLength, L1 depths/steps, L0A/B double buffers, full allocation ledger and
actual ACLNN Workspace for the three-output ABI. Workspace4195328 is only a
reference. B cannot reuse an old239104B Host PASS as proof of240384B admission.
Failure rejects; no buffer shrink, Guard removal, type/Base change or Reserve use.

## Simpler B normal path (specification only)

Preflight app resources before user asynchronous work; initialize all five Guard
pairs and the dedicated diagnostics. Keep original producer, Scale bytes, M_V
and Raw Echo semantics. Echo and FP16 are now live independently; no scalar
migration or manual S_V is needed. Before the sixteen direct VDEQ16 calls,
check a valid free S_V ID and freeze application same-pool activity. Keep each
SDK pair unchanged and append a reserved V_S Scalar completion join per call.
End follows both completed consumers and retained reverse/cleanup dependencies.
Arena uses its permanent diagnostic view; final Witness uses its own permanent
view, finalized and published only after completed output/safety checks.

S_MTE3 still protects Scalar Witness publication. MTE3_S still protects output
completion; dropping aliases does not authorize dropping DMA completion checks.
The SDK internal S_V candidate can be recorded as a prechecked/source-inferred
value while its bitmap is frozen, but must not be described as directly hooked
hardware telemetry. No SDK instrumentation is implemented.

## Why not approve implementation yet?

B removes one identifiable collision/migration mechanism. It does not establish
SDK event demand while app drain tickets are held, all SDK-internal ordinary-error
cleanup, or new actual resource admission. These remain exact blockers in
FAILURE_DRAIN_REVIEW.md. Prefer B rather than forcing A to keep the old total,
but wait for independent review. No device implementation, OPP or Runner exists.
