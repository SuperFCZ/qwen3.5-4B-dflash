#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "packed_spec.h"
#include "panel_vector.h"

using namespace AscendC;
using namespace matmul;
using namespace packed_fc;
using gate_panel::CastCodes;
using gate_panel::Pair;

namespace {
using A = MatmulType<TPosition::A1, CubeFormat::NZ, half>;
using B = MatmulType<TPosition::B1, CubeFormat::NZ, half, false>;
using C = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using Bias = MatmulType<TPosition::GM, CubeFormat::ND, float>;
static_assert(Banked && Pieces * Elements == CodeBytes, "raw holds a complete W8 panel");

struct Events { TEventID input, produce, consume; };
__aicore__ inline bool Valid(TEventID id) { return static_cast<uint32_t>(id) < QUE_MAX_EVENT; }
__aicore__ inline void Release(TPipe& pipe, const Events& ids)
{
    if (Valid(ids.input)) pipe.ReleaseEventID<HardEvent::MTE2_V>(ids.input);
    if (Valid(ids.produce)) pipe.ReleaseEventID<HardEvent::V_MTE3>(ids.produce);
    if (Valid(ids.consume)) pipe.ReleaseEventID<HardEvent::MTE3_V>(ids.consume);
}

__aicore__ inline void LoadPanel(uint32_t panel, LocalTensor<half> a, LocalTensor<int8_t> raw, LocalTensor<half> scale,
    GlobalTensor<half> x, GlobalTensor<int8_t> codes, GlobalTensor<half> gn, const Events& ids)
{
    DataCopy(a, x[(panel % Panels) * PanelK * MRows], MRows * PanelK);
    DataCopy(scale, gn[panel * Groups * Columns], Groups * Columns);
    DataCopy(raw, codes[panel * Columns * PanelK], CodeBytes);
    Pair<HardEvent::MTE2_V>(ids.input);
}

__aicore__ inline void FillB(LocalTensor<half> b, LocalTensor<half> weights,
    LocalTensor<int8_t> raw, LocalTensor<half> scale, const Events& ids)
{
    for (uint32_t p=0; p<Pieces; ++p) {
        auto w=weights[(p % 2) * Elements];
        CastCodes<Elements>(w, raw[p * Elements]);
        PipeBarrier<PIPE_V>();
        for (uint32_t g=0; g<Groups; ++g) {
            auto v=w[g * Group * 16];
            Mul(v, v, scale[g * Columns + p * 16], uint64_t(128), uint8_t(16), BinaryRepeatParams{1,1,0,8,8,0});
        }
        // Retire store(p-1) before p+1 overwrites that W16 bank.
        if (p>0 && p+1<Pieces) Pair<HardEvent::MTE3_V>(ids.consume);
        Pair<HardEvent::V_MTE3>(ids.produce);
        DataCopy(b[p * Elements], w, Elements);
    }
}
} // namespace

extern "C" __global__ __aicore__ void d_flash_fc_packed_panel(GM_ADDR x, GM_ADDR w_tiles, GM_ADDR s,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(td, tiling);
    const uint32_t block=GetBlockIdx(), blocks=GetBlockNum();
    if (!blocks || block>=Tiles || td.globalM!=MRows || td.globalK!=K || td.globalN!=N ||
        td.tileK!=PanelK || td.matmulUbBytes!=262144-UserBytes) return;
    TPipe pipe;
    Matmul<A,B,C,Bias> mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &td.cubeTilingData);
    TBuf<TPosition::A1> ab0, ab1;
    TBuf<TPosition::B1> bb0, bb1;
    TBuf<TPosition::VECIN> qr, gs;
    TBuf<TPosition::VECOUT> wb;
    TBuf<> scratch;
    pipe.InitBuffer(ab0,MRows*PanelK*2);pipe.InitBuffer(ab1,MRows*PanelK*2);
    pipe.InitBuffer(bb0,Columns*PanelK*2);pipe.InitBuffer(bb1,Columns*PanelK*2);
    pipe.InitBuffer(qr,CodeBytes);pipe.InitBuffer(gs,ScaleBytes);pipe.InitBuffer(wb,WeightBytes);
    pipe.InitBuffer(scratch,td.matmulUbBytes);mm.SetLocalWorkspace(scratch.Get<uint8_t>(td.matmulUbBytes));
    const Events ids{pipe.AllocEventID<HardEvent::MTE2_V>(),pipe.AllocEventID<HardEvent::V_MTE3>(),pipe.AllocEventID<HardEvent::MTE3_V>()};
    if (!Valid(ids.input) || !Valid(ids.produce) || !Valid(ids.consume)) {
        PipeBarrier<PIPE_ALL>(); mm.End(); Release(pipe,ids); return;
    }
    auto a0=ab0.Get<half>(MRows*PanelK), a1=ab1.Get<half>(MRows*PanelK);
    auto b0=bb0.Get<half>(Columns*PanelK), b1=bb1.Get<half>(Columns*PanelK);
    auto raw=qr.Get<int8_t>(CodeBytes);auto scale=gs.Get<half>(Groups*Columns);
    auto weights=wb.Get<half>(2*Elements);
    GlobalTensor<half> xgm,gn,ygm;GlobalTensor<int8_t> codes;
    xgm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x),MRows*K);
    codes.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(w_tiles),N*K);
    gn.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(s),N*K/Group);
    ygm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y),MRows*N);
    LoadPanel(block*Panels,a0,raw,scale,xgm,codes,gn,ids);FillB(b0,weights,raw,scale,ids);
    bool second=false;
    for (uint32_t nt=block;nt<Tiles;nt+=blocks) {
        for (uint32_t kp=0;kp<Panels;++kp) {
            PipeBarrier<PIPE_ALL>(); // Current complete A1/B1 are visible before SDK consumption.
            const bool lastK=kp+1==Panels;
            const uint32_t next=lastK ? (nt+blocks)*Panels : nt*Panels+kp+1;
            const bool ahead=next<Tiles*Panels;
            if (ahead) LoadPanel(next,second?a0:a1,raw,scale,xgm,codes,gn,ids);
            mm.SetTensorA(second?a1:a0);mm.SetTensorB(second?b1:b0,false);
            if (!mm.Iterate(kp!=0)) { PipeBarrier<PIPE_ALL>(); mm.End(); Release(pipe,ids); return; }
            // Application events are reserved for this kernel. The SDK's current A1/B1 and
            // FP32 accumulator stay intact while the Vector/MTE3 prepare the OTHER complete B1.
            if (ahead) FillB(second?b0:b1,weights,raw,scale,ids);
            PipeBarrier<PIPE_ALL>();
            if (lastK) mm.GetTensorC(ygm[nt*Columns]);
            PipeBarrier<PIPE_ALL>();
            second=!second;
        }
    }
    mm.End();Release(pipe,ids);
}
