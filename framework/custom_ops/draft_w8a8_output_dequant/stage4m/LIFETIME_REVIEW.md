# Bounded local SDK follow-up: candidate remains blocked

Decision: **BLOCKED_BITMAP_STORE_CONTRACT**. No candidate Kernel, CANN compile,
NPU launch or performance run is authorized by this review. No change to
Reserve, barriers, End, Scale, Scratch, historical source or Golden.

The four comparison/reduction file hashes in README were subsequently
confirmed identical by the server owner. Newly inspected files below remain
LOCAL_ONLY_NOT_SERVER_VERIFIED. This review uses local CANN9.0.0 in
qwen35-cann9-host, `/usr/local/Ascend/cann-9.0.0/aarch64-linux/asc`.

## Findings

| Question | What is established | What is not established |
|---|---|---|
| Logical compare output | 64 FP32 comparisons correspond to64 EQ bits; count64 consumes256B and uses one repeat. | Logical8B result is NOT a physical write upper bound; neither8B nor32B store footprint has been established. |
| Existing temporary storage | ReinterpretCast preserves buffer address/handle and byte length. A320B view fits in the8192B converted body, outside its two Guards. No allocation/Reserve change. | A320B view is safe only if each compare write obeys the unproved slot bound; capacity alone is insufficient. |
| Old data lifetime | Stage4J lines94–95 perform Cast, PIPE_V, Add, PIPE_ALL. The converted value has no further arithmetic consumer in that group. Sum and selected partial remain live and must not be borrowed. | Complete new alias lifetime through mask writes and scalar consumption cannot be certified without the output write contract and compiler validation. |
| V→S completion | Compare interface is annotated PIPE_V; V_S maps to set_flag/wait_flag PIPE_V→PIPE_S; GetValue is PIPE_S. In non-split MIX compilation the public wrappers do not skip this event. | This is source dispatch proof, not a hardware test. Future compilation must reject SPLIT_CORE_CUBE/VEC and bind these files; new compiled behavior is NOT_RUN. |
| S reads→next overwrite | Proposed scalar all-bits decision must precede advance to the next group; next Cast overwrites converted only after the original Matmul lifecycle. | Do not infer that V_S itself protects future S→V reuse. The new alias/control dependency must also survive compiler validation. No new S_V or moved barrier is silently added. |

Proposed placement remains: existing Add + ALL; ten comparisons; existing
FetchEventID/SetFlag/WaitFlag V_S; complete bitmap check/failure return; next
group. Comparisons must not be inserted *after* the existing V_S wait. Every
original synchronization call and the full Tile/Arena checks remain.

The `ASCEND_IS_AIC` early returns found in GetValue and the public event
wrappers are **split-core compilation branches**, not a claim that an
AI_CORE task on310P3 cannot read UB. `kernel_macros.h` selects MIX when neither
SPLIT_CORE_CUBE nor SPLIT_CORE_VEC is defined. Only under that explicit
condition does the traced path reach the UB load and V→S event calls.

## Direct evidence ledger (metadata only)

| Relative path | SHA256 | Lines / relevant meaning |
|---|---|---|
| impl/basic_api/kernel_event.h | d974ecee61e39402dad6fa368e036b538bc1f69b05989b6b3a9fc9c946bd609c | 291–327 underlying set/wait calls;914–916 and1117–1119 V_S selects PIPE_V→PIPE_S;287 ALL/PIPE_V barrier dispatch for this architecture |
| impl/basic_api/kernel_operator_block_sync_intf_impl.h | 7f0fa56ad683b86f2ded10b984645e698d05a04e541c04d25cedb69c4be70c6f | 47–100 public wrappers, split-core early returns and dispatch |
| impl/basic_api/kernel_tensor_impl.h | 1e5d6669d91e5687db76fcace1cb1b2a9af0dfde92c4b5ee65dd2b23b99b9e0a | 645–668 physical address;669–719 S-pipe GetValue;727–749 reinterpreted address/length/handle |
| impl/basic_api/kernel_macros.h | 500ea245ecddc927f644ae8cb2a75abe914164b402d372c7293297135ef0d3f9 | 150–173 MIX/AIC/AIV constants and split macro branches |
| include/basic_api/kernel_struct_unary.h | 3e6aaa0894bbd707c236bfa84a4ffefb1c942c0c601995678fd11b5472509ca7 | 27–54 repeat-parameter fields/defaults; not an intrinsic store-size specification |
| impl/basic_api/utils/kernel_check_vec_binary_scalar.h | 7e1e17ef1779d565e562b1c88379d0a6672d2cd9c58b2c19179c3ca916f3b603 | 18 CPU-debug guard;47–61 forwards pointers, sizes, positions and count to checker |
| impl/basic_api/utils/kernel_check_vec_binary_util.h | b56d0b1fe6fb851ae1154a6948d997f798d64ec4aeb465a7b7723b1185f375fe | 219–221 checker declarations only, not its allocation/alignment validation body |

Already server-matched cmpsel interface:2002 dispatch at25–26, PIPE_V pragma
before namespace, and count overload429–454. Already matched dav_m200
implementation:197–243 EQ intrinsic selection,265–293 repeat arithmetic.
No SDK source excerpts or implementation files are copied into Git.

## Exact stop condition

The direct visible chain terminates at `vcmpvs_eq`; the CPU-debug validation
chain terminates at a declaration of `CheckFuncVecBinaryScalarCmpImpl`.
The bounded search of the direct basic_api checker/include dependencies did
not expose its body. This is not a claim that no specification exists anywhere,
and a CPU checker would not by itself prove the hardware store transaction.

The missing evidence is a target-specific contract for the one-repeat
float→uint8 compare result: destination alignment, maximum modified byte
range, valid result-bit placement and padding behavior. Without it, reserving
32B per logical8B result is still an assumption. Increasing slot size or using
Reserve does not cure the missing contract and is not allowed here.

Therefore stop before implementing an integrated Vector Guard. Do not ask
for more Profile rows: they cannot close an API memory contract. No new SDK
full-tree scanner, disassembler or unsupported integer reduction is introduced.
If the exact contract becomes available, bind it and the event/alias sources,
then separately admit a minimal compile/write-footprint/fault-injection probe.
Such a probe, rather than the existing NumPy model, must establish device
mask writes, special-value comparisons and all group-boundary detection.
