#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "down_spec.h"
#include "panel_vector.h"

using namespace AscendC;
using namespace matmul;
using namespace down_panel;
using gate_panel::CastCodes;
using gate_panel::Pair;

namespace {
using A = MatmulType<TPosition::A1, CubeFormat::NZ, half>;
using B = MatmulType<TPosition::B1, CubeFormat::NZ, half, false>;
using C = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using Bias = MatmulType<TPosition::GM, CubeFormat::ND, float>;

__aicore__ inline void CacheA(LocalTensor<half> dst, GlobalTensor<half> src)
{
    // ND [M,K] -> A1 [K/16,M,16]. Scatter one full contiguous GM row at a time.
    const DataCopyParams scatter{K / 16, 1, 0, M - 1};
    for (uint32_t row = 0; row < M; ++row) DataCopy(dst[row * 16], src[row * K], scatter);
}
} // namespace

extern "C" __global__ __aicore__ void d_flash_down_panel(
    GM_ADDR x, GM_ADDR w_tiles, GM_ADDR s, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(td, tiling);
    const uint32_t block = GetBlockIdx(), blocks = GetBlockNum();
    if (!blocks || block >= Tiles || td.globalM != M || td.globalK != K ||
        td.globalN != N || td.tileK != PanelK) return;

    TPipe pipe;
    Matmul<A, B, C, Bias> mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &td.cubeTilingData);
    TBuf<TPosition::A1> abuf;
    TBuf<TPosition::B1> bbuf;
    TBuf<TPosition::VECIN> qbuf, sbuf;
    TBuf<TPosition::VECOUT> wbuf;
    TBuf<> scratch;
    pipe.InitBuffer(abuf, M * K * 2);
    pipe.InitBuffer(bbuf, Columns * PanelK * 2);
    pipe.InitBuffer(qbuf, CodeBytes);
    pipe.InitBuffer(sbuf, ScaleBytes);
    pipe.InitBuffer(wbuf, WeightBytes);
    pipe.InitBuffer(scratch, td.matmulUbBytes);
    mm.SetLocalWorkspace(scratch.Get<uint8_t>(td.matmulUbBytes));
    auto a = abuf.Get<half>(M * K);
    auto b = bbuf.Get<half>(Columns * PanelK);
    auto raw = qbuf.Get<int8_t>(Elements);
    auto scales = sbuf.Get<half>(Groups * Columns);
    auto weights = wbuf.Get<half>(W16Banks * Elements);
    GlobalTensor<int8_t> qgm;
    GlobalTensor<half> xgm, sgm, ygm;
    xgm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x), M * K);
    qgm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(w_tiles), N * K);
    sgm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(s), (K / Group) * N);
    ygm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y), M * N);
    CacheA(a, xgm);

    for (uint32_t nt = block; nt < Tiles; nt += blocks) {
        for (uint32_t kp = 0; kp < Panels; ++kp) {
            // No SDK consumer is invoked between these immediate event pairs.
            const auto inputReady = pipe.FetchEventID(HardEvent::MTE2_V);
            const auto rawConsumed = pipe.FetchEventID(HardEvent::V_MTE2);
            const auto weightReady = pipe.FetchEventID(HardEvent::V_MTE3);
            const auto weightStored = pipe.FetchEventID(HardEvent::MTE3_V);
            if (static_cast<uint32_t>(inputReady) >= QUE_MAX_EVENT ||
                static_cast<uint32_t>(rawConsumed) >= QUE_MAX_EVENT ||
                static_cast<uint32_t>(weightReady) >= QUE_MAX_EVENT ||
                static_cast<uint32_t>(weightStored) >= QUE_MAX_EVENT) {
                PipeBarrier<PIPE_ALL>();
                mm.End();
                return;
            }
            const uint32_t panel = nt * Panels + kp;
            const uint32_t start = panel * Columns * PanelK;
            DataCopy(scales, sgm[panel * Groups * Columns], Groups * Columns);
            DataCopy(raw, qgm[start], Elements);
            for (uint32_t piece = 0; piece < Pieces; ++piece) {
                Pair<HardEvent::MTE2_V>(inputReady);
                if constexpr (W16Banks == 1) {
                    if (piece != 0) Pair<HardEvent::MTE3_V>(weightStored);
                }
                auto w = weights[(piece % W16Banks) * Elements];
                CastCodes<Elements>(w, raw);
                if (piece + 1 < Pieces) {
                    Pair<HardEvent::V_MTE2>(rawConsumed);
                    DataCopy(raw, qgm[start + (piece + 1) * Elements], Elements);
                }
                PipeBarrier<PIPE_V>();
                const uint32_t channel = (piece / ChunksPerStripe) * ChunkN;
                const uint32_t firstGroup = (piece % ChunksPerStripe) * (ChunkK / Group);
                for (uint32_t g = 0; g < ChunkK / Group; ++g) {
                    auto values = w[g * Group * ChunkN];
                    Mul(values, values, scales[(firstGroup + g) * Columns + channel],
                        uint64_t(128), uint8_t(16), BinaryRepeatParams{1,1,0,8,8,0});
                }
                if constexpr (W16Banks == 2) {
                    // Finish store(piece-1) before the next piece reuses that bank.
                    if (piece > 0 && piece + 1 < Pieces) Pair<HardEvent::MTE3_V>(weightStored);
                }
                Pair<HardEvent::V_MTE3>(weightReady);
                DataCopy(b[piece * Elements], w, Elements);
            }
            PipeBarrier<PIPE_ALL>(); // A1 and all eight B1 pieces are ready
            mm.SetTensorA(a[kp * PanelK * M]);
            mm.SetTensorB(b, false);
            if (!mm.Iterate(kp != 0)) {
                PipeBarrier<PIPE_ALL>();
                mm.End();
                return;
            }
            PipeBarrier<PIPE_ALL>(); // retire B1 reads before its next K4864 panel; preserve FP32 L0C
        }
        mm.GetTensorC(ygm[nt * Columns]);
        PipeBarrier<PIPE_ALL>();
    }
    mm.End();
}
