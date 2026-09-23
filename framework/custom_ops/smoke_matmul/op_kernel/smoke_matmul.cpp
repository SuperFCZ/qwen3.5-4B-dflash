#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

using AType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
using BType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, true>;
using CType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
using BiasType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

extern "C" __global__ __aicore__ void smoke_matmul(
    GM_ADDR x, GM_ADDR w, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    AscendC::TPipe pipe;
    matmul::Matmul<AType, BType, CType, BiasType> mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tilingData.cubeTilingData);

    // Ascend 310P's ND output path uses temporary UB space for format conversion.
    AscendC::TBuf<> tempUb;
    pipe.InitBuffer(tempUb, tilingData.localMemSize);
    mm.SetLocalWorkspace(tempUb.Get<uint8_t>(tilingData.localMemSize));

    AscendC::GlobalTensor<half> xGm, wGm, yGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x), 16 * 256);
    wGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(w), 64 * 256);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(y), 16 * 64);
    mm.SetTensorA(xGm);
    mm.SetTensorB(wGm, true);
    mm.IterateAll(yGm);
    mm.End();
}
