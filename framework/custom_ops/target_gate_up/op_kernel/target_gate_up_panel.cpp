#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace {
constexpr uint32_t K = 2560;
constexpr uint32_t N = 9216;
constexpr uint32_t C0 = 16;

template<bool CubeDequant> struct Result { using Type = int32_t; };
template<> struct Result<true> { using Type = half; };

template<bool Prefetch, bool CubeDequant, class Tiling>
__aicore__ inline void Project(GM_ADDR x, GM_ADDR weights, GM_ADDR rowScale,
                               GM_ADDR gateScale, GM_ADDR upScale, GM_ADDR gate,
                               GM_ADDR up, Tiling& t) {
    using A = MatmulType<TPosition::A1, CubeFormat::NZ, int8_t>;
    using B = MatmulType<TPosition::B1, CubeFormat::NZ, int8_t>;
    using Acc = typename Result<CubeDequant>::Type;
    using C = MatmulType<TPosition::VECIN, CubeFormat::NZ, Acc>;
    using Bias = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    Matmul<A,B,C,Bias> mm;
    TPipe pipe;
    const uint32_t m=t.rows, width=t.tileN, elems=m*width, panelBytes=K*width;
    const uint32_t panels=2*N/width, id=GetBlockIdx();
    if (id>=t.cores) return;
    const uint32_t begin=panels*id/t.cores, end=panels*(id+1)/t.cores;
    GlobalTensor<int8_t> activations, panelsGm;
    GlobalTensor<float> rowsGm, gateGm, upGm;
    GlobalTensor<half> gateOut, upOut;
    activations.SetGlobalBuffer((__gm__ int8_t*)x,m*K);
    panelsGm.SetGlobalBuffer((__gm__ int8_t*)weights,2*N*K);
    rowsGm.SetGlobalBuffer((__gm__ float*)rowScale,m);
    gateGm.SetGlobalBuffer((__gm__ float*)gateScale,1);
    upGm.SetGlobalBuffer((__gm__ float*)upScale,1);
    gateOut.SetGlobalBuffer((__gm__ half*)gate,m*N);
    upOut.SetGlobalBuffer((__gm__ half*)up,m*N);

    TQue<TPosition::A1,1> aQueue;
    TQue<TPosition::B1,Prefetch ? 2 : 1> bQueue;
    TQue<TPosition::VECIN,1> accQueue;
    TQue<TPosition::VECOUT,1> outputQueue;
    TBuf<TPosition::VECCALC> floatBuf, halfBuf, scaleBuf, halfScaleBuf, localWorkspace;
    pipe.InitBuffer(aQueue,1,m*K);
    pipe.InitBuffer(bQueue,Prefetch ? 2 : 1,panelBytes);
    pipe.InitBuffer(accQueue,1,elems*sizeof(Acc));
    if constexpr (!CubeDequant) pipe.InitBuffer(floatBuf,elems*sizeof(float));
    pipe.InitBuffer(halfBuf,elems*sizeof(half));
    pipe.InitBuffer(outputQueue,1,elems*sizeof(half));
    pipe.InitBuffer(scaleBuf,m*(1+C0)*sizeof(float));
    pipe.InitBuffer(halfScaleBuf,m*C0*sizeof(half));
    if (t.localBytes) pipe.InitBuffer(localWorkspace,t.localBytes);
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &t.cube);
    if (t.localBytes) mm.SetLocalWorkspace(localWorkspace.Get<uint8_t>());

    // Each core reuses A in L1 across its entire N slab. A float16 input was
    // quantized upstream by the receiver. Only its signed INT8 codes arrive here.
    auto a=aQueue.AllocTensor<int8_t>();
    if (t.packedA) {
        DataCopy(a,activations,m*K);
    } else {
        DataCopyParams copy{static_cast<uint16_t>(m),1,static_cast<uint16_t>(K/32-1),0};
        for (uint32_t k=0; k<K/32; ++k) DataCopy(a[k*m*32],activations[k*32],copy);
    }
    aQueue.EnQue(a);
    a=aQueue.DeQue<int8_t>();
    auto scales=scaleBuf.Get<float>();
    DataCopy(scales,rowsGm,m);
    const auto scalarReady=static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(scalarReady);
    WaitFlag<HardEvent::MTE2_S>(scalarReady);
    auto row=scales[m];
    const float gateWeight=gateGm.GetValue(0), upWeight=upGm.GetValue(0);
    for (uint32_t r=0; r<m; ++r) Duplicate(row[r*C0],scales.GetValue(r),C0);
    PipeBarrier<PIPE_V>();
    auto rowHalf=halfScaleBuf.Get<half>();
    Cast(rowHalf,row,RoundMode::CAST_NONE,m*C0);
    PipeBarrier<PIPE_V>();
    if constexpr (Prefetch) {
        auto first=bQueue.template AllocTensor<int8_t>();
        DataCopy(first,panelsGm[begin*panelBytes],panelBytes);
        bQueue.EnQue(first);
    }
    for (uint32_t tile=begin; tile<end; ++tile) {
        const bool isGate=tile<N/width;
        const float weightScale=isGate ? gateWeight : upWeight;
        if constexpr (CubeDequant) {
            union { float value; uint32_t bits; } scale;
            scale.value=weightScale;
            mm.SetQuantScalar(static_cast<uint64_t>(scale.bits & 0xffffe000u) | (uint64_t(1)<<46));
        }
        mm.SetTensorA(a);
        if constexpr (!Prefetch) {
            auto b=bQueue.template AllocTensor<int8_t>();
            DataCopy(b,panelsGm[tile*panelBytes],panelBytes);
            bQueue.EnQue(b);
        }
        auto currentB=bQueue.template DeQue<int8_t>();
        if constexpr (Prefetch) {
            if (tile+1<end) {
                auto next=bQueue.template AllocTensor<int8_t>();
                DataCopy(next,panelsGm[(tile+1)*panelBytes],panelBytes);
                bQueue.EnQue(next);
            }
        }
        mm.SetTensorB(currentB);
        auto acc=accQueue.AllocTensor<Acc>();
        while (mm.template Iterate<true>()) mm.template GetTensorC<true>(acc);
        mm.End();
        PipeBarrier<PIPE_ALL>();
        bQueue.FreeTensor(currentB);

        auto halves=halfBuf.Get<half>();
        // Preserve the independently observed half rounding before the per-row scale.
        if constexpr (!CubeDequant) {
            auto f=floatBuf.Get<float>();
            Cast(f,acc,RoundMode::CAST_NONE,elems);
            PipeBarrier<PIPE_V>();
            Muls(f,f,weightScale,elems);
            PipeBarrier<PIPE_V>();
            Cast(halves,f,RoundMode::CAST_NONE,elems);
            PipeBarrier<PIPE_V>();
        }
        // Cube output is NZ [N/16,M,16]. Reuse the row scales across N.
        for (uint32_t offset=0; offset<elems; offset+=m*C0) {
            if constexpr (CubeDequant) Mul(halves[offset],acc[offset],rowHalf,m*C0);
            else Mul(halves[offset],halves[offset],rowHalf,m*C0);
        }
        PipeBarrier<PIPE_V>();
        accQueue.FreeTensor(acc);
        auto dense=outputQueue.AllocTensor<half>();
        DataCopyParams reorder{static_cast<uint16_t>(m),1,0,static_cast<uint16_t>(width/C0-1)};
        for (uint32_t n=0; n<width; n+=C0) DataCopy(dense[n],halves[n*m],reorder);
        outputQueue.EnQue(dense);
        dense=outputQueue.DeQue<half>();
        const uint32_t col=(tile%(N/width))*width;
        DataCopyParams store{static_cast<uint16_t>(m),static_cast<uint16_t>(width/C0),
                             0,static_cast<uint16_t>((N-width)/C0)};
        if (isGate) DataCopy(gateOut[col],dense,store);
        else DataCopy(upOut[col],dense,store);
        outputQueue.FreeTensor(dense);
    }
    aQueue.FreeTensor(a);
}
}

extern "C" __global__ __aicore__ void target_gate_up_panel(GM_ADDR x, GM_ADDR panels,
    GM_ADDR rowScale, GM_ADDR gateScale, GM_ADDR upScale, GM_ADDR gate, GM_ADDR up,
    GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(t,tiling);
    if (t.prefetchB) Project<true,true>(x,panels,rowScale,gateScale,upScale,gate,up,t);
    else if (t.cubeDequant) Project<false,true>(x,panels,rowScale,gateScale,upScale,gate,up,t);
    else Project<false,false>(x,panels,rowScale,gateScale,upScale,gate,up,t);
}
