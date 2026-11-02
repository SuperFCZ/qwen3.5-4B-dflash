# Stage4M design review — no device experiment admitted yet

Status: **BLOCKED_PENDING_COMPARE_CONTRACT_AND_SERVER_IDENTITIES**.
No Kernel, OPP, CANN build, NPU smoke, benchmark or profile is created/run by
this directory. Stage4B–4L remain unchanged. Weak Guard remains diagnostic.

## What the supplied Stage4L measurements establish

The user reports Full/Weak N64 medians12.852710/4.477270 ms and N128
7.031495/2.860490 ms. These paired observations support net effects of the
compiled variants, including an interaction of4.204435 ms. They do not
separate instruction cost, hidden wait, DMA or critical-path contribution.
The FP16 and W8A16 numbers are historical, unpaired context only.

The actual Stage4L CSVs are on the server, not present on this development
machine. Profile-supported conclusions therefore remain UNRESOLVED until
rows are returned. `profile_review.py` rereads hash-bound raw rows using the
existing Stage4L reviewer, keeping each source file, row type, column schema,
unit, Task/Core identifiers and duplicate flags. It computes no pooled median
or Pipe sum. Its grouped examples are evidence for subsequent interpretation,
not an assertion that all rows within a group may safely be averaged.

## Installed local SDK evidence (not SDK source excerpts)

Local container qwen35-cann9-host, root
`/usr/local/Ascend/cann-9.0.0/aarch64-linux/asc`.
All files below are **LOCAL_ONLY_NOT_SERVER_VERIFIED** for this new review.
The historical local Toolkit has the matched internal build identifier, but
that is not proof that these newly inspected files match the server.

| Relative file | Lines | Finding | SHA256 |
|---|---|---|---|
| impl/basic_api/kernel_operator_vec_cmpsel_intf_impl.h | 429–454 | On2002, count-form CompareScalar accepts half/float input and uint8 output; count*sizeof(T) must be a multiple of256. Calls VcmpvsImpl. | 4f852c20951d3b2cf33e812c4b320f2b7188a653bb3f409b71d2d3a737c998c5 |
| impl/basic_api/dav_m200/kernel_operator_vec_cmpsel_impl.h | 197–243;265–293 | EQ routes to vcmpvs_eq; repeat count derives from source bytes/256, result bit packing is implied by offsets. This body alone does not prove physical destination write granularity. | bc1978e6c8e124b5f5811a7e8dcb59e78ad36b3b220011e0133451c8bcb78578 |
| impl/basic_api/kernel_operator_vec_reduce_intf_impl.h | 181–220;625–705 | Reviewed BlockReduce and ReduceMax/Min/Sum interfaces restrict primitive inputs to half/float. Direct uint32 reduction is not admitted. | 87f073691c5296f19c6b2c23473635c886fc25409fbe26f6c6b43f603b1a503a |
| impl/basic_api/dav_m200/kernel_operator_vec_reduce_impl.h | 129–174 | WholeReduceMax/Min dispatch to vcmax/vcmin; generic template syntax alone is not integer support evidence. | f645d34d3a30e41bcac4cdd18001446b19ffae2eeb5fbdeed20123f540d30917 |

Direct uint32 CompareScalar/Reduce: **UNSUPPORTED by the reviewed public
interface contract**. Do not evade checks through a raw intrinsic or assume
another architecture's int32 overload applies.

## Candidate: exact sentinel equality bitmap, no vector reduction

The sentinel0x5a39c671 is a finite nonzero FP32 normal bit pattern. Under exact
IEEE float equality it has a unique encoding. Reinterpreting its storage as
FP32 and comparing EQ therefore detects any different uint32 pattern,
including NaN/Inf/zero corruption. This is **not** a numeric uint32→float Cast:
that conversion loses low bits and has a CPU counterexample in the tests.
It is not a universally valid trick for arbitrary Guard patterns (notably
signed zero or NaN sentinels). The device comparison/NaN behavior still needs
confirmation; CPU IEEE behavior is not a device support verdict.

Proposed per-group algorithm, not implemented as a Kernel:

1. Retain the exact current Add and PIPE_ALL completion boundary.
2. Compare all ten256B prefix/suffix regions,64 float-reinterpreted words each,
   against the exact bit-reinterpreted sentinel; full256B repeats avoid tails.
3. Write each64-bit EQ bitmap into a separate proposed32B-aligned slot of the
   existing Cast temporary body: ten slots,320B total. Do not touch its Guards,
   FP32 Sum, selected FP16 partial or SDK Scratch body.
4. Use the existing group V_S SetFlag/WaitFlag **after** these comparisons;
   retain all original barriers, events, End and A/B/Scale binding order.
5. Read/check all bitmap bits (conceptually20 uint32 scalar reads rather than
   640 data-word reads), fail before the next group on any zero bit.
6. Leave Tile-end/Arena/final extra checks unchanged. The next Cast overwrites
   the temporary body normally, after the preceding scalar decision completes.

20 versus640 is a source-level load-count hypothesis, not a compiled
instruction count or predicted speedup. No performance claim is made.

| Producer → consumer | Storage | Proposed lifetime / remaining proof |
|---|---|---|
| Matmul/GetTensorC/End → Guard comparison | five raw buffer redzones | Original ALL completion retained; input compare type/address contract needs binding |
| Cast → Add | FP32 temporary → Sum | Original PIPE_V then ALL retained; Add must finish before temporary body is borrowed |
| Compare → scalar bitmap check | temporary body[0:320) | Existing later V_S is the proposed completion fence; verify compare PIPE_V result completion and GetValue visibility |
| scalar decision → next Cast/group | temporary/matmul resources | No advance on mismatch; no End or Scale lifetime change; verify alias/write footprint |

## UB budget and blockers

Scratch226816 + two FP16 buffers8192 + FP32 temporary8192 + Sum8192 +
Guards2560 + Reserve8192 =262144B. No new allocation is proposed. The320B
bitmap is a **temporal view**, not a reduction of Guard or reserve space.
This accounting is only a design: it does not establish legal device aliasing.

Before a device experiment can be created, close:

- Server hashes for the direct API/implementation files above.
- Actual destination footprint,32B alignment and comparison result packing;
  the direct next check is CheckFuncVecBinaryScalarCmp for the count overload
  referenced at cmpsel_intf_impl.h:450, plus the corresponding vcmpvs contract.
- Guaranteed completion/visibility of the vector result under existing V_S,
  and consumer completion before next temporary overwrite. No new barrier or
  reordered original event is silently authorized.
- Exact float comparison of all corrupted bit patterns, including signaling
  NaNs, without changing numerical error state. This remains a compiler/NPU
  probe obligation even after source-contract review.

If a solution requires separate UB or new synchronization not proven here,
remain blocked and report it. Do not borrow the8192B reserve or change Scratch.
No broad SDK scan or alternative Tile width is needed.

## CPU coverage and later admission

Tests cover every one of640 words ×32 bit positions ×20 group boundaries,
random multi-bit values and special floats, incorrect numeric casts, transient
damage repaired before Tile end, and resource/index bounds. The model does
not emulate vector mask stores, real DMA/events, compiler or NPU FP semantics.

Only after remaining contracts close: isolated compile probe, then every
prefix/suffix word at every group with single/multiple-bit and transient
fault injection on NPU, then full Golden/Guard/Readonly/Poison/Repeat. A later
three-way N128 Full/Vector/Weak fresh paired Benchmark must preserve full
coverage in the Vector candidate. Historical7.031495/2.860490 ms are not new
sample denominators. No such experiment is generated in this review.

```bash
PYTHONPATH=framework/custom_ops/draft_quant/test python3 -B -m pytest -q \
 framework/custom_ops/draft_w8a8_output_dequant/stage4m/test
```
