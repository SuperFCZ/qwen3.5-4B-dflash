#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

namespace {
constexpr uint32_t kM = 16;
constexpr uint32_t kK = 256;
constexpr uint32_t kN = 64;
constexpr uint32_t kGroup = 128;
constexpr uint32_t kNzK0 = 32;
constexpr uint32_t kNzN0 = 16;
}  // namespace

using AType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
using BType = MatmulType<AscendC::TPosition::VECOUT, CubeFormat::ND, half, true>;
using CType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
using BiasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

extern "C" __global__ __aicore__ void d_flash_group_quant_linear(
    GM_ADDR x, GM_ADDR w_nz, GM_ADDR s, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    AscendC::TPipe pipe;
    Matmul<AType, BType, CType, BiasType> mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tilingData.cubeTilingData);

    AscendC::TBuf<AscendC::TPosition::VECIN> codesBuf, scaleBuf;
    AscendC::TBuf<AscendC::TPosition::VECOUT> weightBuf;
    AscendC::TBuf<> matmulUb;
    pipe.InitBuffer(codesBuf, kN * kK * sizeof(int8_t));
    pipe.InitBuffer(scaleBuf, (kK / kGroup) * kN * sizeof(half));
    pipe.InitBuffer(weightBuf, kN * kK * sizeof(half));
    pipe.InitBuffer(matmulUb, tilingData.matmulUbBytes);
    mm.SetLocalWorkspace(matmulUb.Get<uint8_t>(tilingData.matmulUbBytes));

    AscendC::GlobalTensor<half> xGm, sGm, yGm;
    AscendC::GlobalTensor<int8_t> qGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x), kM * kK);
    qGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(w_nz), kN * kK);
    sGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(s), (kK / kGroup) * kN);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(y), kM * kN);
    auto qLocal = codesBuf.Get<int8_t>(kN * kK);
    auto sLocal = scaleBuf.Get<half>((kK / kGroup) * kN);
    auto wLocal = weightBuf.Get<half>(kN * kK);

    AscendC::DataCopy(qLocal, qGm, kN * kK);
    AscendC::DataCopy(sLocal, sGm, (kK / kGroup) * kN);
    // Tiny correctness path: serialize DMA, scalar scale reads and vector work.
    // There is one owner and no reuse/overlap of these buffers before mm.End().
    AscendC::PipeBarrier<PIPE_ALL>();
    for (uint32_t group = 0; group < kK / kGroup; ++group) {
        for (uint32_t n = 0; n < kN; ++n) {
            const half scale = sLocal.GetValue(group * kN + n);
            for (uint32_t part = 0; part < kGroup / kNzK0; ++part) {
                const uint32_t k1 = group * (kGroup / kNzK0) + part;
                const uint32_t src = ((k1 * (kN / kNzN0) + n / kNzN0) * kNzN0 +
                                       n % kNzN0) * kNzK0;
                const uint32_t dst = n * kK + k1 * kNzK0;
                // Every signed INT8 value, including -128, is exact in FP16.
                AscendC::Cast(wLocal[dst], qLocal[src], AscendC::RoundMode::CAST_NONE, kNzK0);
                AscendC::PipeBarrier<PIPE_V>();
                // FP16 multiplication is the explicit dequantization boundary.
                AscendC::Muls(wLocal[dst], wLocal[dst], scale, kNzK0);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    // One Cube reduction over the entire K, shared by all 16 activation rows.
    // No per-group FP16 partial output, scalar GEMM, or dense W in GM.
    // Matching native WeightQuant's internal rounding still requires NPU tests.
    mm.SetTensorA(xGm);
    mm.SetTensorB(wLocal, true);
    mm.IterateAll(yGm);
    mm.End();
}
