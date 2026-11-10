# Bounded normal-path review, performance priority

This revision does not require a proof of recovery from every hardware fault.
It requires safe normal execution and a terminal path for ordinary safety checks.
The old Phase2.1 BLOCKED report is not modified. New observations narrow its gap;
they do not automatically re-admit Scheme B.

Root `/usr/local/Ascend/cann-9.0.0`. Hashes were measured in local Docker.
Existing event/DataCopy/shared-resource identities remain pinned by Phase2's
12-file manifest. Additional identities below are LOCAL_ONLY until a server
read-only audit matches them. No SDK contents are committed.

| Relative path | SHA256 | Reviewed lines |
|---|---|---|
|aarch64-linux/asc/impl/basic_api/kernel_tquebind_impl.h|36097569b96d84e44a2688b04711bf326c9f61d84d9f524d188ee3775afd34e9|244–248,316–329,409–452,543–558|
|aarch64-linux/asc/impl/adv_api/detail/matmul/resource/cube_in_buffer/cube_in_buffer_normal.h|e98fc3cbc75b916088a956922a6d49137477a1f211ba45572a8095874f954bd7|40–85,137–153|
|aarch64-linux/asc/impl/adv_api/detail/matmul/resource/cube_in_buffer/cube_in_buffer_utils.h|f790d435011a69734d73259c3936ee52cc1ccb08899c7551076f2129df459350|42–63,95–145|
|aarch64-linux/asc/impl/adv_api/detail/matmul/stage/copy_cube_in/base/copy_cube_in_using_ub.h|4f57a5faf40eca08996956691ffe91be90f90ab4500707bb28ce43b85dce1b8e|33–74,76–112|
|aarch64-linux/asc/impl/adv_api/detail/matmul/stage/copy_cube_in/copy_tile_to_cube/data_copy_wrapper_using_ub_nd.h|6d5315d222c1f4357f45adbc00ad76380ec0d7d6d3bcffd570ff6cd9867c18d6|42–134|
|aarch64-linux/asc/impl/adv_api/detail/matmul/resource/l0_buffer/tbuf_pool_l0_v200.h|dc547d6d1635d4456519ddca46b1b8f2b1607fe2cbec8b82b5aff11b76983f45|29–59|
|aarch64-linux/asc/impl/adv_api/detail/matmul/resource/l0_buffer/tbuf_pool_l0_base.h|3531b6fbc126040850669e092c7429422cd0559eee6e593c43b717f19066f0ce|35–51,58–63,103–125|
|aarch64-linux/asc/impl/basic_api/kernel_event.h|d974ecee61e39402dad6fa368e036b538bc1f69b05989b6b3a9fc9c946bd609c|503–576|
|aarch64-linux/asc/impl/adv_api/detail/matmul/scheduler/base/scheduler_norm.h|bb565c78b53644f63cdf89cef18f4d622f11f4fca755ce03b989347f5f5083cd|93–122,164–207|
|aarch64-linux/asc/impl/adv_api/detail/matmul/scheduler/base/scheduler_base.h|e37c5029455b8ffac29e2a51652592d02830504e77ce68b6d3fe7d4902d9a2a4|61–118,149–167|

## New source findings

1. `scheduler_norm.h:164–207` executes the multi-K loop; input LoadData precedes
   L0 EnQue/DeQue and CubeCompute. `copy_cube_in_using_ub.h:76–112` calls the
   matching input-buffer enqueue then dequeue. These are SDK operations, not
   application-owned tickets that may be reclaimed arbitrarily.
2. `scheduler_base.h:149–159` End destroys CopyCubeInA/B and resets L0 cache.
   The NORMAL input-buffer specialization `cube_in_buffer_normal.h:74–85`
   frees a cached tensor when present and invokes both queue FreeAllEvent calls.
   `kernel_tquebind_impl.h:543–558` requires enqueue IDs already invalid and
   waits/releases every remaining free-buffer event. Thus normal input cleanup
   has actual waits. This is conditional on correct selected specialization and
   completed enqueue/dequeue protocol, not a universal exception-drain API.
3. `cube_in_buffer_utils.h:42–63` constructs GM→L1 queues with an ND→NZ mask.
   `kernel_event.h:503–527` maps those queues on2002 to MTE3_MTE1/MTE1_MTE3,
   not the GM→UB MTE2_V/V_MTE2 pair. The new syntax probe static-asserts this.
   Normal queue counts depend on cacheNum/reduceAxisCnt/major-cache selection
   (`cube_in_buffer_normal.h:40–71`); they cannot be inferred from buffer bytes
   alone or treated as nine application reservations.
4. `tbuf_pool_l0_v200.h:47–53` Allocates two M_MTE1 IDs and initially signals
   them; its destructor40–44 waits/releases them. But base Allocate58–63 and
   Free115–125 use ping/pong0/1. Base EnQue/DeQue105–113 also use MTE1_M0/1.
   Do not pre-reserve or borrow these SDK-private pools/IDs. Merely counting
   total available IDs without this placement constraint is insufficient.
5. `data_copy_wrapper_using_ub_nd.h:42–134` uses transient Fetch for V_MTE2,
   V_MTE3 and MTE3_V in the plain input path, and calls additional block-copy,
   padding and NDTrans2NZ helpers. Some pools overlap proposed app fences. This
   review does not yet bound all helper event demand and consumption lifetimes.
   `Fetch` does not reserve; preflight free-count checks must not be promoted to
   proof that every later SDK transient use has completed.
6. The existing hash-bound direct VDEQ16 wrapper uses S_V Fetch/Set/Wait before
   conversion (dav_m200 DataCopy1057–1107). B eliminates application S_V; per-call
   V_S Scalar observation remains the conservative completion condition. This
   resolves that *local* handoff under stable pool ownership, not the preceding
   input-copy helper graph automatically.

## Exact remaining basic-safety gate

The unresolved item is the concrete chosen Normal/IsNeedUB/ND call graph's
per-HardEvent ledger at three boundaries: before Init/Iterate, after Iterate
before app M_V/V_S consumption, and after both consumers before End/destruction.
It must cover the actual CubeInBuffer specialization and helpers called by
CopyND2NZWithVecOp, not all SDK versions. The review has not supplied this ledger.
Normal app reservations must leave enough SDK IDs AND preserve fixed L0 IDs.
A compiler accepting Iterate/DataCopy does not establish either proposition.

The smallest next proof is a fixed-specialization compile assertion plus a
bounded inventory of its direct helper allocations/Fetch pairs and their
completion edges. Then retain pre-owned app tickets through observed terminal
completion, never release a potentially active ID. A source guard/error flag
must converge on the same completed terminal path; no additional MMAD or silent
retry. Do not add a late resource allocation that can fail after work starts.
New Formal Host must admit the actual B buffer/Workspace layout before any launch.

This gate is intentionally narrower than complete production failure recovery.
FP16 mismatch/root-cause/quality review is deferred independently, and cannot
be used either to block a safe exploration unnecessarily or to excuse unsafe
resource reuse. There is no evidence that the API is unsupported; the correct
claim is that this candidate's normal execution resource proof is unfinished.

## Normal resource ledger (proposed, not measured anew)

Scratch202240 + Echo16384 + Scale2048 + FP168192 + Diagnostics768 =229632 payload.
Five prefix/suffix Guard pairs2560 produce232192 explicit UB. Reserve8192 gives
240384 total. External CO116384 is separate L0C; no canary claim. Existing numerical
payload starts remain256/203008/219904/222464. Diagnostics raw230912, Arena231168,
Witness231680, suffix231936, end232192. No alias migration or Reserve use.
SDK effective shared UB0 and external shared L0C0 must still be admitted by the
new real TilingContext; historical Workspace4195328 is reference only. One Tile,
one Block, original M16/K2560/N256 and Base16/256/128; no full-layer extrapolation.

## Decision

`BLOCKED_NORMAL_PATH_EVENT_BUDGET`. No device Kernel or timing Runner is created.
This is a permitted safety stop, not a performance result or an assertion that
VDEQ16 cannot be implemented. Source/CPU/interface checks are independent evidence
levels. Device timing remains NOT_RUN; no old Smoke samples become latency data.
