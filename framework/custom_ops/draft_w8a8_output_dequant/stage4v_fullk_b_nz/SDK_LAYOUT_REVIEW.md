# B-only NZ source, layout and lifecycle review

Selected route: A GM ND INT8; B GM NZ INT8 **transposed storage**; C VECIN NZ FP16;
CFG_NORM, dav_2002 MIX. The logical B remains original KN, while stored Q=BT is
interpreted with all three B transpose settings true. This is the required paired
representation metadata, not a change to A@B or a second arithmetic optimization.

All paths below are relative to `/usr/local/Ascend/cann-9.0.0`. Hashes were read
from the installed local SDK; Source on the server must match them. New headers
are LOCAL_ONLY_NOT_SERVER_VERIFIED until that gate. No SDK source is published.

| File | SHA256 | Reviewed lines |
|---|---|---|
|aarch64-linux/asc/impl/adv_api/detail/matmul/utils/matmul_type_def.h|bbcd98829a26cb3d97281f995a5c8c9d61fa9811356ba1f0058c169aeaa23081|30-52|
|aarch64-linux/asc/impl/adv_api/detail/matmul/stage/copy_cube_in/copy_tile_to_cube/copy_tile_to_cube_using_ub.h|8fed124b939a1a2477622e1414362a95222c37dbfb24ad895494cacb58b33726|39-111|
|aarch64-linux/asc/impl/adv_api/detail/matmul/stage/copy_cube_in/copy_tile_to_cube/data_copy_wrapper_nz.h|92c549855058706b47e50f282006ba176b5d474b6e62420827b9f7f319b4eac9|26-45,101-133|
|aarch64-linux/asc/impl/adv_api/detail/matmul/stage/copy_cube_in/copy_tile_to_cube/data_copy_wrapper_utils.h|52ec7a9a05788443aaa54917b1543594e2440e510e88f0cc56d1dfe5816447c6|209-273,302-370|
|aarch64-linux/asc/impl/adv_api/detail/matmul/stage/copy_cube_in/base/copy_cube_in_params_base.h|83df6f7e85f295d082a911e4d5e5e220c9e0f31f786a93e8f6922cbba1f7c287|410-484,513-558|
|aarch64-linux/asc/impl/adv_api/detail/matmul/stage/copy_cube_in/base/copy_cube_in_using_ub.h|4f57a5faf40eca08996956691ffe91be90f90ab4500707bb28ce43b85dce1b8e|33-124|
|aarch64-linux/asc/impl/adv_api/detail/matmul/resource/cube_in_buffer/cube_in_buffer_utils.h|f790d435011a69734d73259c3936ee52cc1ccb08899c7551076f2129df459350|42-63,109-145|
|aarch64-linux/asc/impl/adv_api/detail/matmul/resource/local_workspace/matmul_local_workspace.h|dbffdce3b74c3e18414c1c5e89e4a4260ec64654eeccc58f172ed89f7ae6c08b|155-183,248-321|
|aarch64-linux/asc/impl/adv_api/detail/matmul/scheduler/iterator/k_loop/k_loop_norm.h|c774fa805fcd2bc458e7333a50c66fc0d819b4c5d0508de4542d09dfbe4dbec5|39-84,156-198|
|aarch64-linux/asc/impl/basic_api/kernel_event.h|d974ecee61e39402dad6fa368e036b538bc1f69b05989b6b3a9fc9c946bd609c|503-576|
|aarch64-linux/asc/impl/basic_api/kernel_tquebind_impl.h|36097569b96d84e44a2688b04711bf326c9f61d84d9f524d188ee3775afd34e9|206-248,292-342,543-558|
|aarch64-linux/asc/impl/adv_api/detail/matmul/stage/copy_cube_out/quant/quant_processor_datacopy.h|10438170dde8cb131bf3ba9443955d101d5e7936aa27dc38127677cb1b898c7c|49-63,113-158|
|aarch64-linux/asc/impl/adv_api/detail/matmul/stage/copy_cube_out/copy_cube_out_datacopy.h|e572b74a29a2d09cc66f79fba676400e4900ef5ceadddf3f6abb99746ab14aad|107-136,245-294|
|aarch64-linux/asc/impl/adv_api/detail/matmul/policy/matmul_private_modules.h|e1934286246684af49b155a77ea9952ce94fb63c4439f57ff568ee24312c2fbd|45-71|
|aarch64-linux/asc/impl/adv_api/detail/matmul/policy/matmul_policy.h|4f418fa553c628a83e2056fe7d2ef5b01097d2743588e729f78c20fe7f56b133|28-55|
|aarch64-linux/asc/impl/adv_api/detail/matmul/utils/mx_matmul_utils.h|7b8de5eb9ad12e10016da2af6ffe661bff130afbf7725d1b70f42913daea4625|85-115|

## Selection, rather than an inference from the name NZ

`copy_cube_in_using_ub.h:33–38,77–112` selects the IsNeedUB Normal input wrapper
and preserves allocation/EnQue/DeQue ownership. `copy_tile_to_cube_using_ub.h:
43–54` dispatches the runtime transpose state. With B type/SetTensorB/SetBType all
true, it selects `CopyTileToCubeFromGM<true>`.

For NZ, lines92–111 contain a non-transposed INT8 special branch at98–102.
It is **not selected** here; the else branch calls `CopyNZ2NZ`. That wrapper
(`data_copy_wrapper_nz.h:39–45`) forwards to the GlobalTensor overload of
`CopyNZ2NZImpl` (`data_copy_wrapper_utils.h:302–345`). Its dav_2002 implementation
uses GM→L1 DataCopy, no B UB staging or TransData. This eliminates the baseline
nontransposed INT8 B branch `CopyND2NZWithTransData`; it does not eliminate A,
output VDEQ16, Guard initialization or every Vector operation in the Kernel.

Merely setting B.format=NZ while keeping B transpose false would instead select
`CopyNZ2NZWithTransData` (NZ wrapper101–133). That body still loads UB and calls
`TransDataNZBMatrix`. Its syntax probe passed, but it is not our selected route
and is not advertised as zero-conversion. No unsafe SDK modification is made to
force that branch to act like a direct copy.

## Exact physical-address proof

For B transpose=true, CopyCubeInParams B accessors report:

* OrgHeight=N256, OrgWidth=K2560 (410–442);
* BaseHeight=BaseN256, BaseWidth=BaseK128 (457–484).

The outer Normal K loop's tile height is BaseK128 (k_loop_norm156–198), with20
iterations for fullK2560. The B transpose dispatch swaps the copy's row/column
and height/width arguments. In our one-N-block Tile, the direct NZ call therefore
has row0, col=128*g, height256, width128, gRow256.

The SDK's direct NZ formula is `row*C0 + col*alignedGRow`, with row aligned to16,
C0=32 bytes/elements for INT8. Thus source offset is32768*g. It emits nburst4,
blockLen256 units32B, source/destination stride0:32768B per K block. Each block
contains4 K32 groups,256 N rows,32 consecutive K bytes. This yields the physical
layout `[K/32,N/16,16,32]` of Q[N,K], exactly
`NZ[k//32,n//16,n%16,k%32]=B[k,n]`.

All20 windows cover one655360B Tile without gaps/overlap; all76 Tiles cover
49,807,360B. N256 is16-aligned and K2560 is32-aligned, so no Padding is needed.
Tile starts and all K-window starts are512B-aligned relative to the GuardedBuffer
payload. Gate/Up boundary9728 is exactly Tile38; final Tile75 ends at channel19455.
The CPU test checks all original bytes and all1520 SDK-derived K-window inverses,
not just sample coordinates or a guessed generic Fractal formula.

## Queues and lifetime

CubeInQueType computes nd2nz from the actual input format. NZ→NZ gives0.
`kernel_event.h:503–527` therefore maps the GM→L1 queue to MTE2_MTE1 (ready) and
MTE1_MTE2 (reuse). Those match the selected direct GM DMA producer and L1→L0
consumer. The baseline ND/nontransposed B conversion used MTE3→L1 and the ND
queue mask's MTE3_MTE1/MTE1_MTE3 pair. This SDK-internal change is necessary for
the changed transfer path, not deletion of an application barrier.

Queue EnQue/DeQue and End's FreeAllEvent remain the standard selected SDK
implementation; no queue substitution or external Event ID is introduced.
Manual V_S/MTE2_S, PIPE_ALL, End and all Guard/Arena boundaries are byte-identical
outside the reviewed B type/SetTensorB statements. No prefetch or asynchronous
cross-Tile rebind is added. Local-interface assertions verify the Normal input
buffer selection, C0=32 and the direct queue event mapping.

## Numerical and resource contract

Input dtypes remain INT8/INT8; Normal scheduler still completes fullK before the
single GetTensorC. C type/position/format, UINT64 SetQuantVector and the selected
QuantProcessor/CopyCubeOut VDEQ16 branch are unchanged. B format/transpose do not
substitute FP32 multiplication or change output Scale words. Integer accumulation
cannot overflow for the frozen bounds; still, runtime bitwise equality with the
frozen baseline is mandatory and no hardware correctness is inferred from CPU.

The basic L1/L0 payload remains A2048B, B32768B, accumulator16384B. Tiling depth/
step/cache choices may change, so actual GetTiling is required. L0B double-buffer
payload bound remains65536B; no extra L0B is available. SDK full L0 pool reservation
is unchanged; tile payload bounds are not described as smaller physical pools.

Selected B does not need ND/transpose UB scratch. Existing202240B Scratch and
all user buffers/Guards/Reserve are nevertheless retained. The old conservative
extra-conversion envelope is a bound, not an assertion of active B consumers.
Actual transLength plus256 UINT64 coefficients must fit; shareL1Size must equal
the accounted Normal allocation and fit actual capacity. Raw shareUbSize is kept
while the identity-bound dav_2002 effective reservation remains0. New Formal
TilingContext must match the new Host plan, not the old ND plan or Manager's
core count. Any rejection/resource discrepancy is BLOCKED, with no fallback.

## Admission and interpretation

Exact interface/initialization and real CANN package compilation are local
checks only. GetTiling on a real310P3, formal Host, Smoke, fault coverage and
paired latency remain server gates. The first prototype's namespace typo and
msopgen filename mismatch were application errors, corrected without changing
the selected NZ configuration; original diagnostics are retained separately.

No claim is made that a format name alone removes work, that packing is free,
that all Scalar/Vector time is removable, or that the new layout is faster.
The only performance answer comes from new180-sample paired results and measured
packing cost. NUMERIC_MISMATCH_UNRESOLVED and non-production status remain.
