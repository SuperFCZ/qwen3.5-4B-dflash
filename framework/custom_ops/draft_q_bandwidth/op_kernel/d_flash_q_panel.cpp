#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "q_spec.h"
#include "panel_vector.h"

using namespace AscendC;
using namespace matmul;
using namespace q_panel;
using gate_panel::CastCodes;
using gate_panel::Pair;

namespace {
using A = MatmulType<TPosition::A1, CubeFormat::NZ, half>;
using B = MatmulType<TPosition::B1, CubeFormat::NZ, half, false>;
using C = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using Bias = MatmulType<TPosition::GM, CubeFormat::ND, float>;

__aicore__ inline void CacheActivation(LocalTensor<half> dst, GlobalTensor<half> src)
{
    // A1[K/16,M,16]. A contiguous GM row supplies all 160 K16 blocks in one scatter.
    const DataCopyParams scatter{K / 16, 1, 0, M - 1};
    for (uint32_t row = 0; row < M; ++row) DataCopy(dst[row * 16], src[row * K], scatter);
}
} // namespace

extern "C" __global__ __aicore__ void d_flash_q_panel(GM_ADDR x, GM_ADDR w_tiles,
    GM_ADDR s, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(td, tiling);
    const uint32_t block = GetBlockIdx(), blocks = GetBlockNum();
    if (!blocks || block >= Tiles || td.globalM != M || td.globalK != K ||
        td.globalN != N || td.tileK != K) return;
    TPipe pipe;
    Matmul<A,B,C,Bias> mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &td.cubeTilingData);
    TBuf<TPosition::A1> abuf;
    TBuf<TPosition::B1> bbuf;
    TBuf<TPosition::VECIN> codes, sbuf;
    TBuf<TPosition::VECOUT> wbuf;
    TBuf<> scratch;
    pipe.InitBuffer(abuf, M*K*2);
    pipe.InitBuffer(bbuf, Columns*K*2);
    pipe.InitBuffer(codes, CodeBytes);
    pipe.InitBuffer(sbuf, ScaleBytes);
    pipe.InitBuffer(wbuf, WeightBytes);
    pipe.InitBuffer(scratch, td.matmulUbBytes);
    mm.SetLocalWorkspace(scratch.Get<uint8_t>(td.matmulUbBytes));
    auto a=abuf.Get<half>(M*K);
    auto b=bbuf.Get<half>(Columns*K);
    auto raw=codes.Get<int8_t>(RawBanks*Stripe);
    auto scales=sbuf.Get<half>(Groups*Columns);
    auto weights=wbuf.Get<half>(W16Banks*Stripe);
    GlobalTensor<int8_t> qgm;
    GlobalTensor<half> xgm, sgm, ygm;
    xgm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x), M*K);
    qgm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(w_tiles), N*K);
    sgm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(s), N*K/Group);
    ygm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y), M*N);
    CacheActivation(a,xgm);
    for (uint32_t nt=block; nt<Tiles; nt+=blocks) {
        const auto inputReady=pipe.FetchEventID(HardEvent::MTE2_V);
        const auto rawConsumed=pipe.FetchEventID(HardEvent::V_MTE2);
        const auto weightReady=pipe.FetchEventID(HardEvent::V_MTE3);
        const auto weightStored=pipe.FetchEventID(HardEvent::MTE3_V);
        if (static_cast<uint32_t>(inputReady)>=QUE_MAX_EVENT || static_cast<uint32_t>(rawConsumed)>=QUE_MAX_EVENT ||
            static_cast<uint32_t>(weightReady)>=QUE_MAX_EVENT || static_cast<uint32_t>(weightStored)>=QUE_MAX_EVENT) {
            PipeBarrier<PIPE_ALL>(); mm.End(); return;
        }
        DataCopy(scales,sgm[nt*Groups*Columns],Groups*Columns);
        DataCopy(raw,qgm[nt*Columns*K],Stripe);
        for (uint32_t stripe=0; stripe<4; ++stripe) {
            Pair<HardEvent::MTE2_V>(inputReady);
            auto current=raw[(stripe%RawBanks)*Stripe];
            if constexpr (Lookahead) {
                if (stripe+1<4) {
                    // Bank(stripe+1) was last read by Cast(stripe-1), except at the prologue.
                    if (stripe>0) Pair<HardEvent::V_MTE2>(rawConsumed);
                    DataCopy(raw[((stripe+1)%RawBanks)*Stripe],qgm[nt*Columns*K+(stripe+1)*Stripe],Stripe);
                }
            }
            auto w=weights[(stripe%W16Banks)*Stripe];
            CastCodes<Stripe>(w,current);
            if constexpr (!Lookahead) {
                if (stripe+1<4) {
                    Pair<HardEvent::V_MTE2>(rawConsumed);
                    DataCopy(raw,qgm[nt*Columns*K+(stripe+1)*Stripe],Stripe);
                }
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t g=0; g<Groups; ++g) {
                auto values=w[g*Group*16];
                Mul(values,values,scales[g*Columns+stripe*16],
                    uint64_t(128),uint8_t(16),BinaryRepeatParams{1,1,0,8,8,0});
            }
            // Retire store(stripe-1) before stripe+1 reuses its W16 bank.
            if (stripe==1 || stripe==2) Pair<HardEvent::MTE3_V>(weightStored);
            Pair<HardEvent::V_MTE3>(weightReady);
            DataCopy(b[stripe*Stripe],w,Stripe);
        }
        PipeBarrier<PIPE_ALL>();
        mm.SetTensorA(a);
        mm.SetTensorB(b,false);
        if (!mm.Iterate(false)) { PipeBarrier<PIPE_ALL>(); mm.End(); return; }
        PipeBarrier<PIPE_ALL>();
        mm.GetTensorC(ygm[nt*Columns]);
        PipeBarrier<PIPE_ALL>();
    }
    mm.End();
}
