// Serial two-group experiment. Diagnostic selection happens AFTER both groups.
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;
using namespace matmul;
#if defined(__NPU_ARCH__)
static_assert(__NPU_ARCH__ == 2002, "Stage3B3 requires dav_2002");
#endif
#if defined(__DAV_M200_VEC__)
#error "L0C output requires the AI Core target"
#endif
using AType=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using BType=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using CType=MatmulType<TPosition::VECIN,CubeFormat::NZ,half>;
using BiasType=MatmulType<TPosition::GM,CubeFormat::ND,int32_t>;
constexpr uint32_t guardBytes=256,guardWord=0x5a39c671;
__aicore__ inline void InitGuard(LocalTensor<uint32_t> raw,uint32_t bytes) {
    Duplicate(raw,guardWord,64);Duplicate(raw[(guardBytes+bytes)/4],guardWord,64);
}
__aicore__ inline bool CheckGuard(LocalTensor<uint32_t> raw,uint32_t bytes) {
    for(uint32_t i=0;i<64;++i)
        if(raw.GetValue(i)!=guardWord||raw.GetValue((guardBytes+bytes)/4+i)!=guardWord)return false;
    return true;
}
extern "C" __global__ __aicore__ void d_flash_w8_a8_two_group(
    GM_ADDR a,GM_ADDR b,GM_ADDR scale,GM_ADDR o,GM_ADDR workspace,GM_ADDR tiling) {
    GET_TILING_DATA(td,tiling);
    if(td.outputMode>2||td.localMemSize!=239104)return;
    TPipe pipe;Matmul<AType,BType,CType,BiasType> mm;
    REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&td.cubeTilingData);
    TBuf<> scratchRaw;
    TBuf<TPosition::VECIN> p1Raw,p0FinalRaw;
    TBuf<TPosition::VECCALC> castRaw,sumRaw;
    pipe.InitBuffer(scratchRaw,td.localMemSize+2*guardBytes);
    pipe.InitBuffer(p1Raw,2048+2*guardBytes);pipe.InitBuffer(p0FinalRaw,2048+2*guardBytes);
    pipe.InitBuffer(castRaw,4096+2*guardBytes);pipe.InitBuffer(sumRaw,4096+2*guardBytes);
    InitGuard(scratchRaw.Get<uint32_t>(),td.localMemSize);
    InitGuard(p1Raw.Get<uint32_t>(),2048);InitGuard(p0FinalRaw.Get<uint32_t>(),2048);
    InitGuard(castRaw.Get<uint32_t>(),4096);InitGuard(sumRaw.Get<uint32_t>(),4096);
    auto scratch=scratchRaw.Get<uint8_t>()[guardBytes];scratch.SetSize(td.localMemSize);
    auto p0=p0FinalRaw.Get<half>()[guardBytes/2];p0.SetSize(1024);
    auto p1=p1Raw.Get<half>()[guardBytes/2];p1.SetSize(1024);
    auto converted=castRaw.Get<float>()[guardBytes/4];converted.SetSize(1024);
    auto sum=sumRaw.Get<float>()[guardBytes/4];sum.SetSize(1024);
    Duplicate(sum,0.0f,1024);PipeBarrier<PIPE_ALL>();
    GlobalTensor<int8_t> aGm,bGm;GlobalTensor<uint64_t> scales;GlobalTensor<half> out;
    aGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(a),2*2048);
    bGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(b),2*8192);
    scales.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(scale),2*64);
    out.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(o),1024);
    for(uint32_t g=0;g<2;++g) {
        // Rebind all inputs/Scale/workspace after End; false starts a fresh INT32 C.
        mm.SetLocalWorkspace(scratch);
        mm.SetTensorA(aGm[g*2048]);mm.SetTensorB(bGm[g*8192],false);
        mm.SetQuantVector(scales[g*64]);
        if(!mm.Iterate(false)) { mm.End();PipeBarrier<PIPE_ALL>();return; }
        auto partial=(g==0)?p0:p1;
        mm.GetTensorC(partial,0,true);
        PipeBarrier<PIPE_ALL>();mm.End();PipeBarrier<PIPE_ALL>();
        Cast(converted,partial,RoundMode::CAST_NONE,1024);PipeBarrier<PIPE_V>();
        Add(sum,sum,converted,1024);PipeBarrier<PIPE_ALL>();
        const auto groupReady=pipe.FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(groupReady);WaitFlag<HardEvent::V_S>(groupReady);
        if(!CheckGuard(scratchRaw.Get<uint32_t>(),td.localMemSize)||
           !CheckGuard(p1Raw.Get<uint32_t>(),2048)||!CheckGuard(p0FinalRaw.Get<uint32_t>(),2048)||
           !CheckGuard(castRaw.Get<uint32_t>(),4096)||!CheckGuard(sumRaw.Get<uint32_t>(),4096))return;
    }
    // Same two Matmuls, pointers, barriers and additions in all three modes.
    if(td.outputMode==0) { Cast(p0,sum,RoundMode::CAST_NONE,1024);PipeBarrier<PIPE_ALL>(); }
    const auto ready=pipe.FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(ready);WaitFlag<HardEvent::V_S>(ready);
    if(!CheckGuard(scratchRaw.Get<uint32_t>(),td.localMemSize)||
       !CheckGuard(p1Raw.Get<uint32_t>(),2048)||!CheckGuard(p0FinalRaw.Get<uint32_t>(),2048)||
       !CheckGuard(castRaw.Get<uint32_t>(),4096)||!CheckGuard(sumRaw.Get<uint32_t>(),4096))return;
    auto result=(td.outputMode==2)?p1:p0;
    // No GM partial write between groups. Only one selected ND output at the end.
    for(uint32_t nb=0;nb<4;++nb)
        for(uint32_t m=0;m<16;++m)DataCopy(out[m*64+nb*16],result[nb*256+m*16],16);
    PipeBarrier<PIPE_ALL>();
}
