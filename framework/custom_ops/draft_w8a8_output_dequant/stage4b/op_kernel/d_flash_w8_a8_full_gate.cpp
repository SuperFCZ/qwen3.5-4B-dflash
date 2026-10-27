// Full 304 N64 tiles, externally scheduled over real coupled AICores. No overlap.
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#define S4B_TEXT_IMPL(...) #__VA_ARGS__
#define S4B_TEXT(...) S4B_TEXT_IMPL(__VA_ARGS__)
// Capture the installed compiler's macro branch in build.log; not a hardware support verdict.
#pragma message("STAGE4B_REGIST_EXPANSION=" S4B_TEXT(REGIST_MATMUL_OBJ(S4B_PIPE,S4B_CORE_ARENA,S4B_MM,S4B_TILING)))
using namespace AscendC;
using namespace matmul;
#if defined(__NPU_ARCH__)
static_assert(__NPU_ARCH__ == 2002, "Stage4B requires dav_2002");
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
__aicore__ inline bool CheckArena(TPipe& pipe, LocalTensor<uint32_t> temp, GlobalTensor<uint32_t> arena, uint64_t bytes) {
    for(uint32_t side=0;side<2;++side) {
        DataCopy(temp,arena[side?(512+bytes)/4:0],128);PipeBarrier<PIPE_ALL>();
        const auto e=pipe.FetchEventID(HardEvent::MTE2_S);
        SetFlag<HardEvent::MTE2_S>(e);WaitFlag<HardEvent::MTE2_S>(e);
        for(uint32_t i=0;i<128;++i)if(temp.GetValue(i)!=guardWord)return false;
    }
    return true;
}
extern "C" __global__ __aicore__ void d_flash_w8_a8_full_gate(
    GM_ADDR a,GM_ADDR b,GM_ADDR scale,GM_ADDR o,GM_ADDR workspace,GM_ADDR tiling) {
    GET_TILING_DATA(td,tiling);
    const uint32_t core=GetBlockIdx();
    if(td.outputMode>21||td.localMemSize!=239104||td.launchBlocks<1||td.launchBlocks>304||
       GetBlockNum()!=td.launchBlocks||core>=td.launchBlocks||!workspace||!td.systemBytes||
       td.systemBytes>512ULL*1024*1024||td.arenaBytes!=(td.systemBytes+511)/512*512||
       td.arenaStride!=td.arenaBytes+1024||td.systemBytes+td.launchBlocks*td.arenaStride>512ULL*1024*1024)return;
    TPipe pipe;
    // Coupled dav_2002 only: initialize this core's GM guards before Matmul registration.
    // REGIST-before-InitBuffer is a separated-architecture constraint, not used here.
    TBuf<TPosition::VECIN> p0FinalRaw;pipe.InitBuffer(p0FinalRaw,2048+2*guardBytes);
    InitGuard(p0FinalRaw.Get<uint32_t>(),2048);
    auto p0=p0FinalRaw.Get<half>()[guardBytes/2];p0.SetSize(1024);
    GlobalTensor<uint32_t> coreArena;
    coreArena.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(workspace+core*td.arenaStride),td.arenaStride/4);
    auto initial=p0.ReinterpretCast<uint32_t>();Duplicate(initial,guardWord,128);PipeBarrier<PIPE_ALL>();
    DataCopy(coreArena,initial,128);DataCopy(coreArena[(512+td.arenaBytes)/4],initial,128);PipeBarrier<PIPE_ALL>();
    Matmul<AType,BType,CType,BiasType> mm;
    // Reserve a FULL platform system arena for each core; SDK-internal core offsets fit
    // within that complete arena. Never divide the opaque platform size by blockDim.
    REGIST_MATMUL_OBJ(&pipe,workspace+core*td.arenaStride+512,mm,&td.cubeTilingData);
    TBuf<> scratchRaw;TBuf<TPosition::VECIN> p1Raw;
    TBuf<TPosition::VECCALC> castRaw,sumRaw;
    pipe.InitBuffer(scratchRaw,td.localMemSize+2*guardBytes);
    pipe.InitBuffer(p1Raw,2048+2*guardBytes);
    pipe.InitBuffer(castRaw,4096+2*guardBytes);pipe.InitBuffer(sumRaw,4096+2*guardBytes);
    InitGuard(scratchRaw.Get<uint32_t>(),td.localMemSize);
    InitGuard(p1Raw.Get<uint32_t>(),2048);
    InitGuard(castRaw.Get<uint32_t>(),4096);InitGuard(sumRaw.Get<uint32_t>(),4096);
    auto scratch=scratchRaw.Get<uint8_t>()[guardBytes];scratch.SetSize(td.localMemSize);

    auto p1=p1Raw.Get<half>()[guardBytes/2];p1.SetSize(1024);
    auto converted=castRaw.Get<float>()[guardBytes/4];converted.SetSize(1024);
    auto sum=sumRaw.Get<float>()[guardBytes/4];sum.SetSize(1024);

    GlobalTensor<int8_t> aGm,bGm;GlobalTensor<uint64_t> scales;GlobalTensor<half> out;
    aGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(a),20*2048);
    bGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(b),304*20*8192);
    scales.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(scale),304*20*64);
    out.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(o),16*19456);
    if(!CheckArena(pipe,converted.ReinterpretCast<uint32_t>(),coreArena,td.arenaBytes))return;
    for(uint32_t slot=core;slot<304;slot+=td.launchBlocks) {
    // Persistent buffers are per core, but sum is per TILE, never carried across slots.
    Duplicate(sum,0.0f,1024);PipeBarrier<PIPE_ALL>();
    for(uint32_t g=0;g<20;++g) {
        // Rebind all inputs/Scale/workspace after End; false starts a fresh INT32 C.
        mm.SetLocalWorkspace(scratch);
        mm.SetTensorA(aGm[g*2048]);mm.SetTensorB(bGm[(slot*20+g)*8192],false);
        mm.SetQuantVector(scales[(slot*20+g)*64]);
        if(!mm.Iterate(false)) { mm.End();PipeBarrier<PIPE_ALL>();return; }
        // Retain exactly the selected diagnostic group; all others reuse p1.
        auto partial=(td.outputMode==g+1)?p0:p1;
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
    // All modes completed twenty Matmuls and ordered FP32 additions.
    if(td.outputMode==0) { Cast(p0,sum,RoundMode::CAST_NONE,1024);PipeBarrier<PIPE_ALL>(); }
    const auto ready=pipe.FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(ready);WaitFlag<HardEvent::V_S>(ready);
    if(!CheckGuard(scratchRaw.Get<uint32_t>(),td.localMemSize)||
       !CheckGuard(p1Raw.Get<uint32_t>(),2048)||!CheckGuard(p0FinalRaw.Get<uint32_t>(),2048)||
       !CheckGuard(castRaw.Get<uint32_t>(),4096)||!CheckGuard(sumRaw.Get<uint32_t>(),4096))return;
    if(!CheckArena(pipe,converted.ReinterpretCast<uint32_t>(),coreArena,td.arenaBytes))return;
    if(td.outputMode==21) { // Scheduler witness, after all 20 Matmuls; never numerical PASS by itself.
        Duplicate(p0,static_cast<half>(core+1+slot/16.0f),1024);PipeBarrier<PIPE_ALL>();
    }
    const auto checked=pipe.FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(checked);WaitFlag<HardEvent::V_S>(checked);
    if(!CheckGuard(p0FinalRaw.Get<uint32_t>(),2048)||!CheckGuard(castRaw.Get<uint32_t>(),4096))return;
    auto result=p0;
    // No GM partial write between groups. Only the final/selected ND output at the end.
    for(uint32_t nb=0;nb<4;++nb)
        for(uint32_t m=0;m<16;++m)DataCopy(out[m*19456+slot*64+nb*16],result[nb*256+m*16],16);
    PipeBarrier<PIPE_ALL>();
    } // next slot owned by this core
}
