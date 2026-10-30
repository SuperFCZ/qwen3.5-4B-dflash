# Stage4I.1: Host shareUbSize versus dav_2002 UB consumption

Decision: the reviewed **device-side consumption contract** supports admission
of nonnegative int32 Host shareUbSize on the fixed arch2002 path, without adding
that raw field to the UB allocation total. This is not a generic reserved-field
exception. Host Query/resource feasibility must still pass again on the real
server. No N128 Kernel, benchmark or profile is created or run here.

## Identity and evidence boundary

The local container is `qwen35-cann9-host`; CANN root is
`/usr/local/Ascend/cann-9.0.0`. `shared-ub-contract.json` records relative paths,
local SHA256, server-provided SHA256, function line ranges and brief conclusions.
All admission headers were checked against the server's identical hashes. The
new binding verifies their actual contents on every run and fails closed on
changed/missing/escaped files. Only metadata is committed, never SDK source.

The Host `libtiling_api.a` differs locally:
local `a2e5bb0bc70ffef0abeb33d864213ad069d65ad6156bf8a02daf37ed29992079`,
server `90f7a3ba4fd55e4c3c99a5b4ad62d5e2e16a434eb62eb880232276b0b84a2ebf`.
It remains LOCAL_ONLY_NOT_SERVER_VERIFIED in the local review. Actual target
library identity is recorded separately by bind_shared_ub. No local Host query
or disassembly was used to infer the server's field-generation algorithm.

## Call/resource chain (paths relative to aarch64-linux/asc)

| Link | Source location | Proven consequence |
|---|---|---|
| REGIST_MATMUL_OBJ to local object initialization | include/adv_api/matmul/matmul_intf.h:18–31; impl/adv_api/detail/kfc/kfc_register_obj.h:125–138,369–377 | Initializes Matmul using the supplied TPipe/Tiling on the coupled path |
| Matmul Init to Scheduler | impl/adv_api/detail/matmul/matmul_impl_base.h:388–404 | Initializes the shape accessor and buffers |
| Shape accessor binding | impl/adv_api/detail/matmul/param/matmul_shape_tiling.h:18–37 | Scheduler reads MatmulTiling, not a raw Host field directly |
| **Architecture-specific accessor** | impl/adv_api/tiling/matmul/matmul_constant_tiling_struct.h:206–217 | **GetShareUbSize returns0 on arch2002**; the raw Host field is not the effective shared UB length |
| Scheduler passes shared lengths | impl/adv_api/detail/matmul/scheduler/base/scheduler_base.h:91–118,137–139 | Sends effective0 for UB to InitShareBufStart, then InitShareBufEnd |
| Shared-pool cursor | impl/basic_api/kernel_tpipe_base.h:220–247; impl/basic_api/kernel_tpipe_impl.h:552–594 | First shared region ends at start+supplied length; mode0 does NOT itself skip accounting. Effective length0 means no UB cursor advance from this reservation |
| User TBuf allocations | impl/basic_api/kernel_tpipe_impl.h:298–346,920–934 | Allocations advance the UB cursor by aligned lengths; initial shared-pool starts are unset |
| Scratch binding and coefficient view | impl/adv_api/detail/matmul/resource/local_workspace/matmul_local_workspace.h:175–183,227–232,285–317 | SetLocalWorkspace keeps a view of caller-owned Scratch; offset views do not allocate a second UB region |
| Coefficient copy | impl/adv_api/detail/matmul/stage/copy_cube_out/quant/quant_processor_datacopy.h:42,109–111,113–124,143–158 | Init/Destroy allocate no coefficient queue; CopyQuantTensorImpl takes baseUseN uint64 entries at transLength inside Scratch |

The raw values512/1024 are not themselves the proof. Nor is a public comment
calling the field reserved. The decisive evidence is the hashed arch2002
accessor and its actual consumer chain. Other architectures or share modes
are not admitted by this policy.

## Host generation: what remains unknown

The supplied server Host query establishes GetTiling=0 and the raw values for
N64/T1/T2. The installed MatmulApiTiling header declares SetBufferSpace;
its resource algorithm is in the compiled Host library. The installed local
headers do not expose the complete same-server generation implementation.
Thus the exact formula assigning shareUbSize and its internal relation to the
Host algorithm's ubSize constraint remain UNRESOLVED. Equality to baseN*8 in
these three observations is not used as proof of that formula.

The device allocation conclusion does not require inventing the Host formula:
whatever valid nonnegative int32 raw value was returned, the reviewed arch2002
accessor supplies0 to shared-pool allocation. Real coefficient storage remains
inside the explicitly allocated Scratch and must satisfy its range checks.
SetBufferSpace is not treated as allocating or aliasing memory by itself.

## Accounting and correction of the intermediate hypothesis

An intermediate reading of AuxShareBufStart alone suggested charging the raw
Host value on top of Scratch. That was incomplete: it missed the upstream
arch2002 accessor. **The earlier hypothetical512/1024 reserve shortfall is
withdrawn. No budget or Kernel change was made based on that hypothesis.**

| Item | N64 | T1 | T2 |
|---|---:|---:|---:|
| Raw Host shareUbSize (user's real query) | 512 | 1024 | 512 |
| Effective SDK shared UB reservation (source-proven arch2002) | 0 | 0 | 0 |
| Scratch allocation | 239104 | 226816 | 226816 |
| User buffers | 12288 | 24576 | 24576 |
| Guards | 2560 | 2560 | 2560 |
| Reserve retained | 8192 | 8192 | 8192 |
| Total budget | 262144 | 262144 | 262144 |
| transLength (user's query) | 8192 | 16384 | 16384 |
| Per-base coefficient view bytes | 512 | 1024 | 512 |
| Coefficient view interval in Scratch | [8192,8704) | [16384,17408) | [16384,16896) |

Coefficient views can reuse input-conversion scratch ranges over time; they
are not claimed to be permanently disjoint inside Scratch. The existing serial
Matmul lifecycle, SDK MTE3_MTE2/MTE2_V sequence and application barriers remain
unchanged. This review adds no asynchronous reuse permission.

T2 retains the existing conservative **logical-N** capacity check through
17408, even though each baseN64 copy uses512 bytes. That capacity bound does
not introduce another allocation. Adding either the raw shareUbSize or the
coefficient view again to the full Scratch allocation would double-charge or
invent storage. The total is allocated Scratch + user buffers + guards +
reserve + effective SDK reservation (0).

This explains the accepted N64 device evidence without erasing its512 raw
field. The historical numerical PASS alone would not prove ownership; the
source chain supplies that proof. Runtime UB high-water measurement is still
NOT_RUN, and Stage4J numerical/event validation remains separate.

## Minimal code change and test scope

- Original N64/T1/T2 shapes, Scratch caps, Probe and CMake are unchanged.
- Raw Host shareUbSize is preserved in the report as reported_shareUbSize.
- Effective SDK reservation, coefficient interval and accounting policy are
  recorded separately. Stage4I's existing transLength, layout, L1/L0, reserve,
  platform and strict field checks remain.
- Changed consumption headers block before Host query; local-only source
  hashes are not silently approved as server evidence.
- CPU regression covers the three real field tuples, full reserve, no extra
  coefficient charge, negative/overflow Host fields, wrong architecture/mode,
  coefficient overflow, alignment, capacity and identity/alias failures.

Run the unchanged Stage4I Host-only entry in a **new** output directory after
pulling this commit. Success may establish STAGE4I_N128_HOST_TILING_FEASIBLE;
it does not authorize creating Stage4J automatically. Old blocked reports and
Stage4B–4H evidence remain read-only. NPU/Benchmark/Profile stay NOT_RUN.
