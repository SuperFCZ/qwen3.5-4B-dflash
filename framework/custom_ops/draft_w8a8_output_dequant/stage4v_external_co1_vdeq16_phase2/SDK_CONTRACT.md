# Selected CANN9.0.0 interface contract

Evidence root: `/usr/local/Ascend/cann-9.0.0`. `sdk-review.json` records every
selected file's canonical-relative path, full SHA256, byte count, exact line
ranges and interface. `audit_phase2.py` rehashes the current installation, rejects
changed identities and binds the completed Echo reports separately. No SDK source
snapshot or device code is included. Until that command runs on the server,
local matches are `LOCAL_ONLY_NOT_SERVER_VERIFIED` for this new audit.

## Interfaces, meaning and remaining scope

| Conclusion | SDK ID / lines | Contract / status |
|---|---|---|
| Mixed INT32→half public overload | interface1099–1121 | DataCopy(LocalTensor<T> dst, LocalTensor<U> src, DataCopyParams, DataCopyEnhancedParams), with PrimT<T>=half and PrimT<U>=int32_t; source L0C, destination UB. SOURCE_SUPPORTED |
| Fixed parameters | params44–68,115–137 | Four uint16 fields; matrix mode, VDEQ16, UB coefficient address, isRelu=false, sidStoreMode0, no padding. SOURCE_SUPPORTED |
| Actual instruction path | device542–572 | Coupled dav_m200 selects DataCopyMatrixL0C2UBImpl and CRMODE_DEQSCALE_VDEQ16. Vector-only architecture branch rejects. SOURCE_SUPPORTED, device observation NOT_RUN |
| Scale address | quant113–124,143–158; device1075–1093 | Original SDK carries UINT64 words into UB, obtains the physical address, programs dequant state and S_V. Address `>>5` is address-unit encoding, not shifting integer data. SOURCE_SUPPORTED |
| N16 mapping | copyout107–136,276–287 | M16 gives baseBlockHeight1. VDEQF16 sets count1/len1/dstStride0; source and destination typed offsets advance256 elements, Scale address advances128B. SOURCE_SUPPORTED |
| Source vs destination byte extent | extent286–300,327–353; device542–572 | Conversion halves the UB burst byte unit; it does not halve the integer parameter1. Range equations below are derived for this exact shape/type. CPU_RANGE_PROOF + SDK_SOURCE_SUPPORT |
| NZ→GM ND | Phase1 frozen source plus design geometry | NZ indices are identical for Echo and FP16, but element sizes differ. Future GM row stride is256. SOURCE/MATH_SUPPORTED; new writeback NOT_RUN |
| Scalar witness migration | tensor646–668,669–719,748–825 | UB GetValue/SetValue operate through S in the required MIX target. Do not use split AIC-only assumptions. Proposed copy concerns metadata only, never Scale or numeric data. NEW_USE_NOT_DEVICE_VERIFIED |
| Allocation/ownership | tpipe298–347,920–935; external29–65; scheduler91–118 | Caller owns external CO1, start0,16384B; effective shared-L0C reservation0 for CO1. No rebind/reset until both consumers finish. SOURCE_SUPPORTED |
| End | scheduler149–167 | Cleanup is not completion proof. End follows completed Echo and VDEQ16; it cannot replace M_V, V_S or reverse-dependency edges. SOURCE_SUPPORTED |
| Event mappings | event291–327,890–951,1030–1119 | Concrete set_flag/wait_flag pipeline mappings exist, including new MTE3_S and S_V alias edges. SOURCE_SUPPORTED |
| Event ownership | event125–132,227–237; tpipe443–484 | On2002 distinct HardEvent values index distinct software pools. Alloc reserves, Fetch only looks, Release clears reservation. Complete outstanding waits before release/reuse. SDK internal VDEQ16 Fetch(S_V) also needs free space. PARTIAL: full SDK/app reservation/failure schedule remains BLOCKED |

The signature is a public DataCopy overload; direct calls to
`copy_matrix_cc_to_ubuf`, UB→L0C injection, or ordinary Cast/Mul are not candidates.
The existing Matmul producer and its coupled/CFG_NORM contract stay fixed.

## Parameter derivation (new for VDEQ16, not inferred from DEQ_NONE)

For each j=0..15, propose **one** transfer:

* source CO1 element offset `j*256`, corresponding bytes `[j*1024,(j+1)*1024)`;
* destination FP16 NZ element offset `j*256`, relative bytes `[j*512,(j+1)*512)`;
* raw coefficient channels `[16*j,16*(j+1))`, UB bytes `[Q+128*j,Q+128*(j+1))`;
* `blockCount=1, blockLen=1, srcStride=0, dstStride=0`;
* `BlockMode::BLOCK_MODE_MATRIX`, `DeqScale::VDEQ16`, `deqTensorAddr=Q+128*j`,
  `isRelu=false`, `sidStoreMode=0`, no pad; scalar deqValue is unused.

CANN900 [Enhanced Data Transfer, Tables4–6](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0104.html)
provides supplementary units: matrix INT32 source blockLen/srcStride units are
1024B; UB dstStride units are32B. INT32→FP16 conversion halves the UB length unit
to512B. A coefficient vector is128B and its address requires32B alignment. Hence
`blockLen=1` remains one matrix block, not0,512 or1024. Zero strides avoid gaps.
This agrees with the actual9.0.0 CopyCubeOut VDEQF16 loop rather than relying on
another release's examples.

Across16 transfers: exactly16384B source is read,8192B FP16 is written and2048B
Scale is addressed, with no gaps or overlap. Matrix geometry is16×16 per N16.
NZ index `(n//16)*256 + m*16 + n%16` maps to ND `m*256+n`. For GM FP16 writeback,
row m/block j reads NZ `(j*256+m*16)` and writes compact ND `(m*256+j*16)`,16
half elements/32B per transfer. Echo uses the same element indices but64B per
16-element write. Physical source0 plus1024*j and destination222464 plus512*j
are appropriately aligned. This arithmetic proves intended coverage; it does not
prove a future device implementation performed it.

## Scale words and arithmetic limits

The256 words are copied byte-for-byte from frozen evidence: no re-encoding,
rounding, fitting, swapping or float preprocessing. M is bits0–31, N bits32–35,
MCB bit36 and Sign bit46. Sign is documented for8-bit modes; it is not MCB.
Frozen words retain their original bit46. Decode fields only for analysis,
never to replace the UINT64 input. Do not introduce new MCB/Sign configurations
in the initial experiment.

The SDK wrapper does not expose VDEQ16 internal multiplication precision or
FP16 tie-breaking. Both remain **UNKNOWN**. RNE/RTZ/round-to-odd and integer
parity models are explicitly hypotheses, not an instruction specification.

## Finite scope and open issue

Nominal DataCopy parameters, Scale mapping, geometry and alias synchronization
edges have source support. The complete manual event ownership plan still needs
independent review against SDK event demand: a persistent app S_V reservation
must not starve/conflict with the sixteen internal Fetch(S_V) operations. Releasing
an ID after merely issuing a wait is not credited as completed consumption in
this design. Late allocation failure while MTE3/V work is outstanding requires
an already-proven drain path, not just End or an assumed C++ return barrier.

This is a specific implementation-admission blocker, not a claim that the API
is unsupported. No additional SDK tree search is requested. Review is bounded
to these event-pool methods, the selected Matmul input/output lifetimes and the
DataCopy S_V implementation. The model tests deliberately reject premature
release and missing completion; they cannot supply hardware timing evidence.
