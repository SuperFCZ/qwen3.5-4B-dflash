#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "panel_spec.h"
#include "panel_vector.h"

using namespace AscendC;
using namespace matmul;
using namespace gate_panel;
static_assert(FullL1, "full-K L1 kernel requires K2560");

namespace {
using A = MatmulType<TPosition::A1, CubeFormat::NZ, half>;
using B = MatmulType<TPosition::B1, CubeFormat::NZ, half, false>;
using C = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using Bias = MatmulType<TPosition::GM, CubeFormat::ND, float>;
constexpr uint32_t Stripe = K * 16;

__aicore__ inline void CacheActivation(LocalTensor<half> a, GlobalTensor<half> x)
{
    // [M,K] -> L1 [K/16,M,16]. One copy instruction gathers all 16 rows of a K16.
    const DataCopyParams gather{M, 1, K / 16 - 1, 0};
    for (uint32_t kg=0; kg<K/16; ++kg) DataCopy(a[kg*M*16], x[kg*16], gather);
}
} // namespace

extern "C" __global__ __aicore__ void d_flash_gate_panel(GM_ADDR x, GM_ADDR w_tiles,
    GM_ADDR s, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(td, tiling);
    const uint32_t block=GetBlockIdx(), blocks=GetBlockNum();
    if (!blocks || block>=Tiles || td.globalM!=M || td.globalK!=K || td.globalN!=N || td.tileK!=K) return;

    TPipe pipe;
    Matmul<A,B,C,Bias> mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &td.cubeTilingData);
    TBuf<TPosition::A1> abuf;
    TBuf<TPosition::B1> bbuf;
    TBuf<TPosition::VECIN> qbuf, sbuf;
    TBuf<TPosition::VECOUT> wbuf;
    TBuf<> scratch;
    pipe.InitBuffer(abuf, M*K*2);
    pipe.InitBuffer(bbuf, Columns*K*2);
    pipe.InitBuffer(qbuf, CodeBytes);
    pipe.InitBuffer(sbuf, ScaleBytes);
    pipe.InitBuffer(wbuf, WeightBytes);
    pipe.InitBuffer(scratch, td.matmulUbBytes);
    mm.SetLocalWorkspace(scratch.Get<uint8_t>(td.matmulUbBytes));

    const auto a=abuf.Get<half>(M*K);
    const auto b=bbuf.Get<half>(Columns*K);
    const auto q=qbuf.Get<int8_t>(Stripe);
    const auto scales=sbuf.Get<half>(Groups*Columns);
    const auto weights=wbuf.Get<half>(W16Banks * Stripe);
    GlobalTensor<int8_t> qgm;
    GlobalTensor<half> sgm, xgm, ygm;
    qgm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(w_tiles),N*K);
    sgm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(s),N*K/Group);
    xgm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x),M*K);
    ygm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y),M*N);
    CacheActivation(a,xgm);

    for(uint32_t nt=block;nt<Tiles;nt+=blocks) {
        // SDK may own event slots. These four directions are used only by immediate
        // set/wait pairs in the preparation region, which contains no Matmul call.
        const auto inputReady=pipe.FetchEventID(HardEvent::MTE2_V);
        const auto rawConsumed=pipe.FetchEventID(HardEvent::V_MTE2);
        const auto weightReady=pipe.FetchEventID(HardEvent::V_MTE3);
        const auto weightStored=pipe.FetchEventID(HardEvent::MTE3_V);
        if (static_cast<uint32_t>(inputReady)>=QUE_MAX_EVENT || static_cast<uint32_t>(rawConsumed)>=QUE_MAX_EVENT ||
            static_cast<uint32_t>(weightReady)>=QUE_MAX_EVENT || static_cast<uint32_t>(weightStored)>=QUE_MAX_EVENT) {
            PipeBarrier<PIPE_ALL>();
            mm.End();
            return;
        }
        DataCopy(scales,sgm[nt*Groups*Columns],Groups*Columns);
        DataCopy(q,qgm[nt*Columns*K],Stripe);
        for(uint32_t slab=0;slab<4;++slab) {
            Pair<HardEvent::MTE2_V>(inputReady);
            // The last reader of this UB bank is our copy to B1. Matmul only sees B1.
            if constexpr (W16Banks == 1) {
                if(slab!=0) Pair<HardEvent::MTE3_V>(weightStored);
            }
            auto w = weights[(slab % W16Banks) * Stripe];
            CastCodes<Stripe>(w,q);
            if(slab+1<4) {
                Pair<HardEvent::V_MTE2>(rawConsumed);
                DataCopy(q,qgm[nt*Columns*K+(slab+1)*Stripe],Stripe);
            }
            PipeBarrier<PIPE_V>();
            for(uint32_t g=0;g<Groups;++g) {
                auto part=w[g*Group*16];
                Mul(part,part,scales[g*Columns+slab*16],uint64_t(128),uint8_t(16),
                    BinaryRepeatParams{1,1,0,8,8,0});
            }
            if constexpr (W16Banks == 2) {
                // Slab 1 can compute in bank 1 while store 0 reads bank 0. Retire store
                // 0 here before slab 2 overwrites bank 0; likewise store 1 at slab 2.
                // The final all-pipe fence retires stores 2 and 3 before Cube starts.
                if (slab == 1 || slab == 2) Pair<HardEvent::MTE3_V>(weightStored);
            }
            Pair<HardEvent::V_MTE3>(weightReady);
            DataCopy(b[slab*Stripe],w,Stripe);
        }
        // All four disjoint B stripes and the block's A cache are now ready for MTE1.
        PipeBarrier<PIPE_ALL>();
        mm.SetTensorA(a);
        mm.SetTensorB(b,false);
        if(!mm.Iterate(false)) {
            PipeBarrier<PIPE_ALL>();
            mm.End();
            return;
        }
        PipeBarrier<PIPE_ALL>();
        mm.GetTensorC(ygm[nt*Columns]);
        // Finish L1 consumers and output copy before the next N64 reuses B1/L0C.
        PipeBarrier<PIPE_ALL>();
    }
    mm.End();
}
