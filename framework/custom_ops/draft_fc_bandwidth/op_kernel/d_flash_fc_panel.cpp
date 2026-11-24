#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "fc_spec.h"
#include "panel_vector.h"

using namespace AscendC;
using namespace matmul;
using namespace fc_panel;
using gate_panel::CastCodes;
using gate_panel::Pair;

namespace {
using LocalA = MatmulType<TPosition::A1, CubeFormat::NZ, half>;
using GlobalA = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using B = MatmulType<TPosition::B1, CubeFormat::NZ, half, false>;
using C = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using Bias = MatmulType<TPosition::GM, CubeFormat::ND, float>;

template<uint32_t M, uint32_t LocalK>
__aicore__ inline void LoadActivation(LocalTensor<half> a, GlobalTensor<half> x, uint32_t k0)
{
    // One contiguous GM segment per row -> local A1 [LocalK/16, M, 16].
    const DataCopyParams scatter{LocalK / 16, 1, 0, M - 1};
    for (uint32_t row = 0; row < M; ++row) {
        DataCopy(a[row * 16], x[row * K + k0], scatter);
    }
}

__aicore__ inline void ScalePiece(LocalTensor<half> w, LocalTensor<half> scales, uint32_t piece)
{
    PipeBarrier<PIPE_V>();
    const uint32_t channel = (piece / Parts) * 16;
    const uint32_t firstGroup = (piece % Parts) * GroupsPerPart;
    for (uint32_t g = 0; g < GroupsPerPart; ++g) {
        auto values = w[g * Group * 16];
        Mul(values, values, scales[(firstGroup + g) * Columns + channel], uint64_t(128), uint8_t(16),
            BinaryRepeatParams{1, 1, 0, 8, 8, 0});
    }
}

template<class Event>
__aicore__ inline void ConvertPiece(LocalTensor<half> w, LocalTensor<int8_t> raw,
    LocalTensor<half> scales, GlobalTensor<int8_t> codes, uint32_t start, uint32_t piece,
    Event ready, Event consumed)
{
    Pair<HardEvent::MTE2_V>(ready);
    CastCodes<Elements>(w, raw);
    if (piece + 1 < Pieces) {
        Pair<HardEvent::V_MTE2>(consumed);
        DataCopy(raw, codes[start + (piece + 1) * Elements], Elements);
    }
    ScalePiece(w, scales, piece);
}

template<uint32_t M, class Tiling>
__aicore__ inline void Project(GM_ADDR x, GM_ADDR w_tiles, GM_ADDR s, GM_ADDR y, Tiling& td)
{
    const uint32_t block = GetBlockIdx(), blocks = GetBlockNum();
    if (!blocks || block >= Tiles || td.globalM != M || td.globalK != K || td.globalN != N ||
        td.tileK != PanelK || td.matmulUbBytes != 262144 - UserBytes) return;
    using A = typename AscendC::Conditional<StreamA, GlobalA, LocalA>::type;
    TPipe pipe;
    Matmul<A, B, C, Bias> mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &td.cubeTilingData);
    TBuf<TPosition::A1> abuf;
    TBuf<TPosition::B1> bbuf;
    TBuf<TPosition::VECIN> qbuf, sbuf;
    TBuf<TPosition::VECOUT> wbuf;
    TBuf<> scratch;
    if constexpr (!StreamA) pipe.InitBuffer(abuf, M * Activation<M>::LocalK * 2);
    pipe.InitBuffer(bbuf, Columns * PanelK * 2);
    pipe.InitBuffer(qbuf, CodeBytes);
    pipe.InitBuffer(sbuf, ScaleBytes);
    pipe.InitBuffer(wbuf, WeightBytes);
    pipe.InitBuffer(scratch, td.matmulUbBytes);
    mm.SetLocalWorkspace(scratch.Get<uint8_t>(td.matmulUbBytes));
    auto b = bbuf.Get<half>(Columns * PanelK);
    auto raw = qbuf.Get<int8_t>(Elements);
    auto scales = sbuf.Get<half>(Groups * Columns);
    auto weights = wbuf.Get<half>(2 * Elements);
    GlobalTensor<int8_t> codes;
    GlobalTensor<half> xgm, sgm, ygm;
    xgm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x), M * K);
    codes.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(w_tiles), N * K);
    sgm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(s), (K / Group) * N);
    ygm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y), M * N);
    if constexpr (Activation<M>::Whole) {
        LoadActivation<M, K>(abuf.Get<half>(M * K), xgm, 0);
    }
    for (uint32_t nt = block; nt < Tiles; nt += blocks) {
        for (uint32_t kp = 0; kp < Panels; ++kp) {
            const auto ready = pipe.FetchEventID(HardEvent::MTE2_V);
            const auto consumed = pipe.FetchEventID(HardEvent::V_MTE2);
            const auto handoff = pipe.FetchEventID(HardEvent::V_MTE3);
            const auto stored = pipe.FetchEventID(HardEvent::MTE3_V);
            if (static_cast<uint32_t>(ready) >= QUE_MAX_EVENT || static_cast<uint32_t>(consumed) >= QUE_MAX_EVENT ||
                static_cast<uint32_t>(handoff) >= QUE_MAX_EVENT || static_cast<uint32_t>(stored) >= QUE_MAX_EVENT) {
                PipeBarrier<PIPE_ALL>(); mm.End(); return;
            }
            const uint32_t panel = nt * Panels + kp, start = panel * Columns * PanelK, k0 = kp * PanelK;
            if constexpr (!StreamA && !Activation<M>::Whole) {
                LoadActivation<M, PanelK>(abuf.Get<half>(M * PanelK), xgm, k0);
            }
            if (nt == block && kp == 0) {
                DataCopy(scales, sgm[panel * Groups * Columns], Groups * Columns);
                DataCopy(raw, codes[start], Elements);
                ConvertPiece(weights, raw, scales, codes, start, 0, ready, consumed);
            }
            Pair<HardEvent::V_MTE3>(handoff);
            DataCopy(b, weights, Elements);
            for (uint32_t piece = 1; piece < Pieces; ++piece) {
                auto w = weights[(piece % 2) * Elements];
                ConvertPiece(w, raw, scales, codes, start, piece, ready, consumed);
                if (piece + 1 < Pieces) Pair<HardEvent::MTE3_V>(stored);
                Pair<HardEvent::V_MTE3>(handoff);
                DataCopy(b[piece * Elements], w, Elements);
            }
            PipeBarrier<PIPE_ALL>(); // Current A/B ready, application UB is available for the next panel.
            const bool lastK = kp + 1 == Panels;
            const uint32_t next = lastK ? (nt + blocks) * Panels : panel + 1;
            const bool hasNext = next < Tiles * Panels;
            if (hasNext) {
                DataCopy(scales, sgm[next * Groups * Columns], Groups * Columns);
                DataCopy(raw, codes[next * Columns * PanelK], Elements);
                Pair<HardEvent::MTE2_V>(ready);
            }
            if constexpr (StreamA) mm.SetTensorA(xgm[k0]);
            else if constexpr (Activation<M>::Whole) mm.SetTensorA(abuf.Get<half>(M * K)[k0 * M]);
            else mm.SetTensorA(abuf.Get<half>(M * PanelK));
            mm.SetTensorB(b, false);
            if (!mm.Iterate(kp != 0)) {
                PipeBarrier<PIPE_ALL>(); mm.End(); return;
            }
            if (hasNext) {
                // No application event stays armed across the SDK call. W0/raw/scales are disjoint from B1 and SDK scratch.
                CastCodes<Elements>(weights, raw);
                ScalePiece(weights, scales, 0);
            }
            PipeBarrier<PIPE_ALL>(); // Cube has consumed A/B and the next raw bank can be overwritten.
            if (lastK) mm.GetTensorC(ygm[nt * Columns]);
            if (hasNext) DataCopy(raw, codes[next * Columns * PanelK + Elements], Elements);
            PipeBarrier<PIPE_ALL>();
        }
    }
    mm.End();
}
} // namespace

extern "C" __global__ __aicore__ void d_flash_fc_panel(GM_ADDR x, GM_ADDR w_tiles,
    GM_ADDR s, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(td, tiling);
    if (td.globalM == 16) Project<16>(x, w_tiles, s, y, td);
    else if (td.globalM == 64) Project<64>(x, w_tiles, s, y, td);
}
