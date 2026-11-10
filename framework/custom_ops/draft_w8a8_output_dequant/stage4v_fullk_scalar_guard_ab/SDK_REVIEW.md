# SDK/API decision and exact detection proof

Actual local root `/usr/local/Ascend/cann-9.0.0`; full hashes and bounded ranges
are in sdk-review.json and are rechecked on the server Source gate.

| API/source | Relevant evidence | Decision |
|---|---|---|
| kernel_operator_vec_reduce_intf_impl.h |181–220 BlockReduceSum/Max;625–645,688–708 Reduce contracts require half/float on2002|No direct uint32 compare/reduce candidate|
| kernel_operator_vec_cmpsel_intf_impl.h |429–454:2002 CompareScalar input half/float, outputuint8, count byte multiple256|No uint32 overload assumption; no unresolved bitmap footprint bypass|
| kernel_tensor_impl.h |669–719,748–825 existing scalar GetValue/SetValue implementation|Reuse original uint32 accesses, no new normal API|
| kernel_event.h |890–951,1030–1048,1093–1119 exact producer/consumer pipe mappings|All original normal events preserved; extra diagnostic-only completion fences|
| kernel_tpipe_impl.h |443–484 Alloc reserves, Fetch does not reserve, Release does not wait|Fault transient pairs serial, with completion before reuse; no new persistent reservation|

The normal optimization uses only uint32 C++ XOR/OR plus the existing GetValue.
No Vector interface is enabled merely because a generic template compiles.
Numeric uint32→float conversion would lose low bits; reinterpreted-float compare
still needs fully proven bitmap stores and exceptional-value behavior. Previous
CompareScalar/GetCmpMask review blockers are not relaxed or re-labeled supported.
This is a bounded route decision, not proof that no other vector construction
could ever work.

For a side containing128 words x_i and sentinel g:

    D = OR_i (x_i XOR g)
    D == 0  iff  every x_i == g

Each XOR is an exact32-bit difference; OR's set bits cannot cancel. Any single
bit at any position makes D nonzero, as does any nonempty multiword corruption.
By contrast XOR across words would miss two identical bit flips. No tolerance,
float conversion, saturated arithmetic or hash collision is introduced.

The original CheckArena loads512B from the selected GM side, keeps PIPE_ALL and
MTE2_S, then scans. Candidate keeps the identical load and waits, same indices,
same original sentinel, same side/call return boundary. No numerical or Guard
writer is introduced during the scan. It reads further only on failure, within
that valid512B payload. UB Delta/AllDelta and Boundary's V_S are byte-identical
after necessary operator-name normalization.

Normal memory budget is unchanged: Scratch202240 + partial8192 + temporary512 +
redzones1536 + untouched Reserve8192 =220672B. No new LocalTensor/TBuf, no hidden
Reserve borrowing, and no changes to L1/L0/Workspace/Tiling. Additional scalar
variables are compiler-managed state, not a claimed device instruction saving.
Actual Formal Host still checks the frozen capacity/T1 contract.

Fault-only ArenaStore initializes the existing temporary with Duplicate, closes
V→S before SetValue, closes S→MTE3 before GM writes, preserves the MTE3→MTE2 edge,
and observes MTE3→S completion before reusing the source buffer. CheckArena's
MTE2→S remains unchanged. Restored GM is rechecked in sweeps. Abort modes publish
their failure witness then observe GM write completion. Those additions exist
only in the separate fault OPP, which is never eligible for latency measurement.
