// UNVERIFIED CANN candidate: receiver source review gates compilation/execution.
// K128 only. This is NOT evidence for refreshing scale during partial sums.
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;
using namespace matmul;
using AType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using BType = MatmulType<TPosition::GM, CubeFormat::NZ, int8_t, true>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

extern "C" __global__ __aicore__ void d_flash_anti_quant_k128(
    GM_ADDR x, GM_ADDR w_nz, GM_ADDR scale, GM_ADDR offset,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(td, tiling);
    TPipe pipe;
    Matmul<AType, BType, CType, BiasType> mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &td.cubeTilingData);
    TBuf<TPosition::VECIN> scaleBuf, offsetBuf;
    TBuf<> sdkUb;
    pipe.InitBuffer(scaleBuf, 64 * sizeof(half));
    pipe.InitBuffer(offsetBuf, 64 * sizeof(half));
    pipe.InitBuffer(sdkUb, td.localMemSize);
    mm.SetLocalWorkspace(sdkUb.Get<uint8_t>(td.localMemSize));
    GlobalTensor<half> a, s, o, c;
    GlobalTensor<int8_t> b;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x), 16 * 128);
    b.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(w_nz), 64 * 128);
    s.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(scale), 64);
    o.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(offset), 64);
    c.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(y), 16 * 64);
    auto scaleLocal = scaleBuf.Get<half>(64);
    auto offsetLocal = offsetBuf.Get<half>(64);
    DataCopy(scaleLocal, s, 64);
    DataCopy(offsetLocal, o, 64);
    PipeBarrier<PIPE_ALL>();
    // Vectors remain alive and unchanged until Matmul has fully completed.
    mm.SetTensorA(a);
    mm.SetTensorB(b, true);
    mm.SetAntiQuantVector(offsetLocal, scaleLocal);
    mm.IterateAll(c);
    PipeBarrier<PIPE_ALL>();
    mm.End();
}
