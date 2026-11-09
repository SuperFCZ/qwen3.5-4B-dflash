# CANN9.0.0 VDEQ16 bounded review

Decision: **BLOCKED_L0C_INJECTION_CONTRACT_UNRESOLVED**. No compute Kernel, OPP,
ACLNN Runner, installation, device query or NPU execution is admitted here.
A standalone UB INT32 -> FP16 VDEQ16 call is not supported by the reviewed
mixed-type overload. An UB -> L0C wrapper exists and type-checks; this is **not**
a claim that the wrapper is unavailable. Its public supported-path/transfer-extent
contract has not been closed. This distinction is intentional.

All paths below are relative to `/usr/local/Ascend/cann-9.0.0`. Complete SHA256,
line ranges and previously recorded server identities are in `sdk-review.json`.
Runtime review hashes actual files and rejects mismatches. Local files alone do
not establish server identity; the server runs the same fixed-file checks. Files
with no previously recorded server SHA are marked accordingly, not retroactively
claimed to have been server-reviewed. The already accepted Raw Host report binds
many of the same SDK files but does not prove the new injection route safe.

## Actual path and boundaries

| Evidence ID | Function/lines | Established contract |
|---|---|---|
| base | SetQuantVector334–346; GetTensorC603–614 | Stores the vector via QuantProcessor; typed Local DstT output invokes Scheduler GetResult |
| scheduler | ScheduleOnce93–122; ComputeMultiIter164–206 | Allocates CubeOutBuffer; multi-K path calls CubeCompute using that tensor |
| type_map | GetMmDstType120–122 | INT8 input maps the accumulator type to INT32, independent of output DstT |
| compute / mad_device | Compute26–69 / MmadCal368–391 | Selected integer MMAD combination includes INT8×INT8→INT32; C operand is the L0C address space |
| scheduler_base | GetResultImpl363–385 | Gets the CubeOutBuffer L0C tensor, queue synchronization, CopyCubeOut, then releases it |
| quant | SetQuantVector49–63; Update113–124 | INT8→half chooses VDEQF16 and sets VDEQ16 with a local quant-vector address |
| quant | CopyQuantTensorImpl143–158 | Scale words copied unchanged from GM into scratch at transLength, MTE3_MTE2 and MTE2_V synchronization |
| copyout | CopyOutImpl75–88; CopyOutNZ2NZ107–136 | Original CType is Local NZ, so this is the NZ→NZ branch, not a Vector NZ→ND conversion |
| copyout | CopyCo12Co2WithQuant245–294 | VDEQF16 emits one matrix block per N16, offset256 elements for M16, quant-vector address advanced128B |
| datacopy_interface | CheckTensorL0C2UB1099–1105; mixed-type1111–1121 | Source must be L0C and destination UB; passes INT32/half physical pointers to implementation |
| datacopy_device | DataCopyL0C2UBImpl1057–1107 | VDEQ16 configures the dequant-vector address and S_V event, then matrix/vector dispatch |
| datacopy_device | DataCopyMatrixL0C2UBImpl542–572 | Calls copy_matrix_cc_to_ubuf with CRMODE_DEQSCALE_VDEQ16 for INT32→half |
| params | DataCopyEnhancedParams113–137 | Default no ReLU, storeMode0, no pad; deqTensorAddr is an address, not decoded coefficients |

There is no C++ INT32 arithmetic shift or multiplier truncation in this selected
handoff. `deqTensorAddr >> 5` is address encoding, **not** a right shift of the
integer dot. The instruction implements the dequantization arithmetic; its
intermediate precision, possible truncation and FP16 tie rule are not exposed by
these headers. No generic Cast rule is substituted for VDEQ16 behavior.

The original Stage4V Kernel calls GetTensorC into half Local NZ and later performs
ordinary32B copies to GM ND. It does not execute a separate Vector Cast/Mul to
produce the FP16 partial. CopyCubeOut's VDEQF16 loop uses16 calls for N256, each
with one N16 coefficient vector. A future direct-L0C probe must match those
parameters rather than assume one aggregated transfer is equivalent.

## UINT64 fields and precision limits

CANN900 [Enhanced Data Transfer, Tables5/6](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0104.html)
identifies M in bits0–31, N in bits32–35, MCB in bit36 and Sign in bit46. Sign is
specified for the8-bit dequantization modes. MCB0 selects integer-to-float
conversion; bit46 is not MCB. This public9.0 documentation is supplemental field
semantics, not a description of hidden arithmetic precision.

The frozen codec's raw words remain unchanged. The analyzer reports every field,
checks the original frozen configuration, and keeps MCB1/other configurations out
of device admission. All six reported integers fit exactly in binary32. This
rules out an ordinary IEEE binary32 input-conversion loss at those coordinates;
it does not establish the instruction's actual internal representation. Fitting
“negative odd plus1” or another mathematical model remains observational.

## Frozen INT32 injection alternatives

| Candidate | Result | Exact missing/established evidence |
|---|---|---|
| UB INT32 directly to half using VDEQ16 | NOT_SUPPORTED_BY_REVIEWED_OVERLOAD | Interface1099–1121 checks source L0C, not UB |
| GM→UB→L0C INT32, then VDEQ16 | BLOCKED_CONTRACT_UNRESOLVED | Same-type interface941–995 routes UB→L0C; dav_m2001290–1297→1117–1130→939–951 reaches copy_matrix_ubuf_to_cc with CRMODE_NONE. Thus an implementation exists. However transfer extents/units/limits for this injection path have no completed public contract in the reviewed material. |
| Rely on SDK capacity checker for injection | NOT_A_PROOF | kernel_check_data_copy_util.h314–325 explicitly bypasses bounds calculation for UB→L0C and L1→L0C. Zero reported offsets must not be interpreted as zero memory use. |
| Infer injection support from reverse transfer | NOT_A_PROOF | Public CANN900 Tables7–9 enumerate L0C→UB and other paths but do not establish the required reverse integer injection contract. Reverse-path parameter semantics cannot be silently assumed. |
| Call the low-level intrinsic manually | REJECTED | Bypasses the missing public contract; no invented intrinsic or address-space cast is allowed |
| Recompute original A/B via FP16 CType Matmul | NOT_INDEPENDENT_INT32_INJECTION | Repeats the suspect path; cannot guarantee its internal integer tensor equals the frozen Raw result |
| Capture int32 with the existing half GetTensorC | UNRESOLVED_PUBLIC_API | GetTensorC is DstT-typed and frees its L0C result after CopyOut. The reviewed API does not expose the same live L0C as an independently typed INT32 tensor. Switching CType creates a different path again. |

`local_check/interfaces.cpp` instantiates only public wrappers for compiler type
checking. It has no `__global__` entry, no allocations, no CPU model pretending to
be VDEQ16 and no executable Kernel. Successful syntax checking cannot close the
transfer-extent or hardware-rounding gaps. No further SDK tree scan is prescribed.
The exact remaining evidence is the dav_2002 UB→L0C integer injection contract:
supported use, matrix layout, block/stride units, complete bounds and completion
ordering. Without it this experiment remains BLOCKED.

## Conditional probe design, not implemented/admitted

If that contract is established later, retain original4096 INT32 and256 UINT64
words byte for byte. A candidate would pack ND→NZ without arithmetic, inject
INT32 into L0C, copy it back without quantization into a distinct INT32 echo,
then apply16 N16 VDEQ16 transfers with original words and save FP16 ND. Echo must
match all4096 input integers before the FP16 observation can inform root cause.
Each execution must check readonly inputs, both Poison patterns, GM/UB guards,
all output coverage and Repeat; two independent executions are mandatory.

Provisional buffer accounting only: UB input16384 + raw echo16384 + Scale2048 +
FP16 output8192 =43008B payload; four256B prefix/suffix pairs add2048B; untouched
8192B reserve gives53248B. Active L0C data would require16384B. These sizes do not
prove unknown instruction extents/alignment or account for an as-yet-unapproved
L0C guard scheme. No original workspace/UB/L1/L0 budget is changed and no memory
is allocated by this delivery.

Compare observations separately with frozen original FP16, original CPU model,
and the predeclared rational models. A result reproducing original FP16 would
support a dequant/output-path explanation **conditional on the echo and path
contracts**; it would not identify hidden arithmetic stages. An echo failure
invalidates that inference. Original actual output is a comparator, not a newly
admitted mathematical Golden. No result grants Stage4V benchmark admission.

## Tiling evidence limitation

Installed TCubeTiling declares50 scalar int32 fields. The offline Host serializer
self-check derives each byte offset through actual SDK setters and SaveToBuffer,
checks zero defaults/padding and confirms a200B encoding locally. The comparator
reads both preserved standalone serialized Tiling files and compares all50
fields, including batch/cache/layout fields omitted from the old27-field list.

Formal Op reports enforce/print only27 TCube fields. Neither old experiment saved
a complete formal serialized TCube blob for the other23. The new tool explicitly
reports those23 as UNRESOLVED for **formal** equality; it never fills them from
standalone defaults. Changed transLength/shareUbSize are labeled CType/workspace
related with exact allocation delta still unresolved, not automatically certified
as harmless. Other differences remain unexplained. No equality claim about the
complete two execution paths is produced, even if all scheduling fields match.

## Additional CO1 Matmul route: real interface, not yet admitted

A further direct-include check found a genuine alternative; it must not be
reported as a nonexistent API. `MatmulImplBase::Iterate(bool, LocalTensor<T>)`
(base648–658) binds the supplied tensor **only when CType position is L0C**.
`cube_out_external`29–65 implements this specialization: SetTensor retains the
caller tensor; allocation returns that tensor and queue/free methods are no-ops.
With the original VECIN CType that binding branch is not taken. This changes
CType/ownership and bypasses the original queue completion behavior, so it cannot
be silently substituted into the existing Kernel.

A possible later experiment is: original A/B -> CO1-owned full-K Matmul -> strict
INT32 echo -> manual public VDEQ16 with the frozen words -> FP16. Echo would have
to match the frozen Raw result at every element, not just six locations. It
would test known-value dequantization conditionally without directly uploading
Raw INT32 into L0C. However, it requires its own real CO1 Host Tiling/resource
admission and a proven Cube-to-Vector completion/ownership contract, because the
normal queue operations are absent. Neither exists in the accepted VECIN Host
reports. No such Kernel or new arithmetic is implemented here. Its status is
`BLOCKED_CO1_HOST_AND_COMPLETION_CONTRACT_PENDING`, not UNSUPPORTED.
