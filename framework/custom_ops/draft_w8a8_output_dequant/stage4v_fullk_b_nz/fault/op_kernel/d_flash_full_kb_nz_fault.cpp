#define STAGE4V_FAULT 1
// New arithmetic contract: one Full-K INT32 accumulation and VDEQ16 per N256 Tile.
// No Group128 partials, no ordered FP32 reduction. Independent CPU Golden required.
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#ifndef STAGE4V_FAULT
#define STAGE4V_FAULT 0
#endif
#if STAGE4V_FAULT
#define S4V_COUNTER (&sweepCount)
#else
#define S4V_COUNTER nullptr
#endif
using namespace AscendC;
using namespace matmul;
#if defined(__NPU_ARCH__)
static_assert(__NPU_ARCH__==2002 && g_coreType==AscendC::MIX,"coupled dav_2002 only");
#endif
#if defined(__DAV_M200_VEC__)
#error "Full-K L0C VDEQ16 requires the coupled AI Core target"
#endif
using AType=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using BType=MatmulType<TPosition::GM,CubeFormat::NZ,int8_t,true>;
using CType=MatmulType<TPosition::VECIN,CubeFormat::NZ,half>;
using BiasType=MatmulType<TPosition::GM,CubeFormat::ND,int32_t>;
#if defined(__NPU_ARCH__)
static_assert(!IsBasic(CFG_NORM),"Full-K requires Normal Scheduler multi-K path");
#endif
constexpr uint32_t guardBytes=256,guardWord=0x5a39c671;
__aicore__ inline void InitGuard(LocalTensor<uint32_t> raw,uint32_t bytes) {
 Duplicate(raw,guardWord,64);Duplicate(raw[(guardBytes+bytes)/4],guardWord,64);
}
__aicore__ inline uint32_t Delta(LocalTensor<uint32_t> raw,uint32_t bytes) {
 uint32_t d=0;for(uint32_t i=0;i<64;++i){d|=raw.GetValue(i)^guardWord;d|=raw.GetValue((256+bytes)/4+i)^guardWord;}return d;
}
__aicore__ inline uint32_t AllDelta(LocalTensor<uint32_t>s,LocalTensor<uint32_t>p,LocalTensor<uint32_t>t) {
 return Delta(s,202240)|Delta(p,8192)|Delta(t,512);
}
__aicore__ inline bool CheckArena(TPipe&pipe,LocalTensor<uint32_t>t,GlobalTensor<uint32_t>a,uint64_t bytes) {
 for(uint32_t side=0;side<2;++side){
  DataCopy(t,a[side?(512+bytes)/4:0],128);PipeBarrier<PIPE_ALL>();
  const auto e=pipe.FetchEventID(HardEvent::MTE2_S);SetFlag<HardEvent::MTE2_S>(e);WaitFlag<HardEvent::MTE2_S>(e);
  uint32_t delta=0;for(uint32_t i=0;i<128;++i)delta|=t.GetValue(i)^guardWord;
  if(delta!=0)return false;
 }return true;
}
// Same three384-word boundaries in normal and fault builds. Fault build is NEVER timed.
__aicore__ inline bool Boundary(TPipe&pipe,uint32_t checkpoint,uint32_t mode,GM_ADDR o,
 LocalTensor<uint32_t>s,LocalTensor<uint32_t>p,LocalTensor<uint32_t>t,LocalTensor<uint32_t>meta,uint32_t* counter) {
 const auto e=pipe.FetchEventID(HardEvent::V_S);SetFlag<HardEvent::V_S>(e);WaitFlag<HardEvent::V_S>(e);
#if STAGE4V_FAULT
 if(AllDelta(s,p,t))return false;
 if(mode==2){
  for(uint32_t buffer=0;buffer<3;++buffer){auto raw=buffer==0?s:buffer==1?p:t;uint32_t bytes=buffer==0?202240:buffer==1?8192:512;
   for(uint32_t side=0;side<2;++side)for(uint32_t word=0;word<64;++word)for(uint32_t bit=0;bit<32;++bit){
    uint32_t i=side?(256+bytes)/4+word:word;raw.SetValue(i,guardWord^(uint32_t(1)<<bit));
    uint32_t d=AllDelta(s,p,t);raw.SetValue(i,guardWord);if(d!=(uint32_t(1)<<bit))return false; ++(*counter);
   }
  }
 }else if(mode==3+checkpoint){t.SetValue((256+512)/4+63,guardWord^0x80000001U);}
#endif
 uint32_t d=AllDelta(s,p,t);
#if STAGE4V_FAULT
 if(d && mode==3+checkpoint && d==0x80000001U){
  meta.SetValue(0,0x53564641U);meta.SetValue(1,checkpoint);meta.SetValue(2,d);meta.SetValue(3,384);
  meta.SetValue(4,2);meta.SetValue(5,1);meta.SetValue(6,63);meta.SetValue(7,checkpoint+1);
  const auto ready=pipe.FetchEventID(HardEvent::S_MTE3);SetFlag<HardEvent::S_MTE3>(ready);WaitFlag<HardEvent::S_MTE3>(ready);
  GlobalTensor<uint32_t>w;w.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(o),8);DataCopy(w,meta,8);PipeBarrier<PIPE_ALL>();
 }
#endif
 return d==0;
}

// Diagnostic OPP only. Modify actual GM Arena redzones, exercise the normal
// CheckArena path, restore after each sweep. No new UB; existing512B temp only.
__aicore__ inline void ArenaStore(TPipe&pipe,LocalTensor<uint32_t>t,GlobalTensor<uint32_t>a,
 uint64_t off,uint32_t word,uint32_t bits){
 Duplicate(t,guardWord,128);PipeBarrier<PIPE_ALL>();
 auto vs=pipe.FetchEventID(HardEvent::V_S);SetFlag<HardEvent::V_S>(vs);WaitFlag<HardEvent::V_S>(vs);
 t.SetValue(word,guardWord^bits);
 auto sm=pipe.FetchEventID(HardEvent::S_MTE3);SetFlag<HardEvent::S_MTE3>(sm);WaitFlag<HardEvent::S_MTE3>(sm);
 DataCopy(a[off],t,128);PipeBarrier<PIPE_ALL>();
 auto mm=pipe.FetchEventID(HardEvent::MTE3_MTE2);SetFlag<HardEvent::MTE3_MTE2>(mm);WaitFlag<HardEvent::MTE3_MTE2>(mm);
 auto ms=pipe.FetchEventID(HardEvent::MTE3_S);SetFlag<HardEvent::MTE3_S>(ms);WaitFlag<HardEvent::MTE3_S>(ms);
}
__aicore__ inline bool ArenaFault(TPipe&pipe,LocalTensor<uint32_t>t,GlobalTensor<uint32_t>a,
 uint64_t bytes,uint32_t checkpoint,uint32_t mode,GM_ADDR out,uint32_t& count){
 if(!CheckArena(pipe,t,a,bytes))return false;
 if(mode==2){
  for(uint32_t side=0;side<2;++side)for(uint32_t word=0;word<128;++word)for(uint32_t bit=0;bit<32;++bit){
   uint64_t off=side?(512+bytes)/4:0;
   ArenaStore(pipe,t,a,off,word,uint32_t(1)<<bit);
   bool detected=!CheckArena(pipe,t,a,bytes);
   ArenaStore(pipe,t,a,off,word,0);
   if(!detected||!CheckArena(pipe,t,a,bytes))return false;
   ++count;
  }
 }else if(mode==6+checkpoint){
  ArenaStore(pipe,t,a,(512+bytes)/4,127,0x80000001U);
  bool detected=!CheckArena(pipe,t,a,bytes);
  if(!detected)return false;
  // Restore redzone after detection; the wrapper still returns false so no
  // later numerical output can hide the abort. Witness is not a normal output.
  ArenaStore(pipe,t,a,(512+bytes)/4,127,0);
  t.SetValue(0,0x53474146U);t.SetValue(1,checkpoint);t.SetValue(2,0x80000001U);t.SetValue(3,256);
  t.SetValue(4,1);t.SetValue(5,127);t.SetValue(6,checkpoint);t.SetValue(7,1);
  auto sm=pipe.FetchEventID(HardEvent::S_MTE3);SetFlag<HardEvent::S_MTE3>(sm);WaitFlag<HardEvent::S_MTE3>(sm);
  GlobalTensor<uint32_t>w;w.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(out),8);DataCopy(w,t,8);PipeBarrier<PIPE_ALL>();
  auto ms=pipe.FetchEventID(HardEvent::MTE3_S);SetFlag<HardEvent::MTE3_S>(ms);WaitFlag<HardEvent::MTE3_S>(ms);
  return false;
 }
 return CheckArena(pipe,t,a,bytes);
}

extern "C" __global__ __aicore__ void d_flash_full_kb_nz_fault(GM_ADDR a,GM_ADDR b,GM_ADDR scale,GM_ADDR o,GM_ADDR workspace,GM_ADDR tiling) {
 GET_TILING_DATA(td,tiling);const uint32_t core=GetBlockIdx();
 if(td.tileId < -1||td.tileId>=76||td.localMemSize!=202240||td.launchBlocks<1||td.launchBlocks>76||GetBlockNum()!=td.launchBlocks||core>=td.launchBlocks||!workspace||!td.systemBytes||td.systemBytes>512ULL*1024*1024||td.arenaBytes!=(td.systemBytes+511)/512*512||td.arenaStride!=td.arenaBytes+1024||td.systemBytes+td.launchBlocks*td.arenaStride>512ULL*1024*1024)return;
#if STAGE4V_FAULT
 if(td.tileId!=0||td.launchBlocks!=1||td.outputMode<2||td.outputMode>7)return;
#else
 if(td.outputMode>1)return;
#endif
 TPipe pipe;TBuf<TPosition::VECIN> partialRaw;pipe.InitBuffer(partialRaw,8192+512);
 auto partial=partialRaw.Get<half>()[128];partial.SetSize(4096);InitGuard(partialRaw.Get<uint32_t>(),8192);
 GlobalTensor<uint32_t>arena;arena.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(workspace+core*td.arenaStride),td.arenaStride/4);
 auto init=partial.ReinterpretCast<uint32_t>();Duplicate(init,guardWord,128);PipeBarrier<PIPE_ALL>();
 DataCopy(arena,init,128);DataCopy(arena[(512+td.arenaBytes)/4],init,128);PipeBarrier<PIPE_ALL>();
 Matmul<AType,BType,CType,BiasType>mm;
 REGIST_MATMUL_OBJ(&pipe,workspace+core*td.arenaStride+512,mm,&td.cubeTilingData);
 TBuf<>scratchRaw;TBuf<TPosition::VECCALC>tempRaw;
 pipe.InitBuffer(scratchRaw,202240+512);pipe.InitBuffer(tempRaw,512+512);
 InitGuard(scratchRaw.Get<uint32_t>(),202240);InitGuard(tempRaw.Get<uint32_t>(),512);
 auto scratch=scratchRaw.Get<uint8_t>()[256];scratch.SetSize(202240);
 auto temp=tempRaw.Get<uint32_t>()[64];temp.SetSize(128);
 GlobalTensor<int8_t>ag,bg;GlobalTensor<uint64_t>sg;GlobalTensor<half>out;
 ag.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(a),40960);bg.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(b),49807360);
 sg.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(scale),19456);out.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(o),311296);
 uint32_t arenaSweepCount=0;
 PipeBarrier<PIPE_ALL>();if(!ArenaFault(pipe,temp,arena,td.arenaBytes,0,td.outputMode,o,arenaSweepCount))return;
 for(uint32_t tile=core;tile<76;tile+=td.launchBlocks){
  if(td.tileId>=0&&tile!=static_cast<uint32_t>(td.tileId))continue;
#if STAGE4V_FAULT
  uint32_t sweepCount=0;
#endif
  if(!Boundary(pipe,0,td.outputMode,o,scratchRaw.Get<uint32_t>(),partialRaw.Get<uint32_t>(),tempRaw.Get<uint32_t>(),temp,S4V_COUNTER))return;
  mm.SetLocalWorkspace(scratch);mm.SetTensorA(ag);mm.SetTensorB(bg[tile*2560*256],true);mm.SetQuantVector(sg[tile*256]);
  if(!mm.Iterate(false)){mm.End();PipeBarrier<PIPE_ALL>();return;}
  mm.GetTensorC(partial,0,true);PipeBarrier<PIPE_ALL>();mm.End();PipeBarrier<PIPE_ALL>();
  if(!Boundary(pipe,1,td.outputMode,o,scratchRaw.Get<uint32_t>(),partialRaw.Get<uint32_t>(),tempRaw.Get<uint32_t>(),temp,S4V_COUNTER))return;
  if(!ArenaFault(pipe,temp,arena,td.arenaBytes,1,td.outputMode,o,arenaSweepCount))return;
  if(td.outputMode==1){Duplicate(partial,static_cast<half>(core+1+tile/16.0f),4096);PipeBarrier<PIPE_ALL>();}
  if(!Boundary(pipe,2,td.outputMode,o,scratchRaw.Get<uint32_t>(),partialRaw.Get<uint32_t>(),tempRaw.Get<uint32_t>(),temp,S4V_COUNTER))return;
#if STAGE4V_FAULT
  if(td.outputMode==2){
   if(sweepCount!=36864||arenaSweepCount!=16384)return;
   temp.SetValue(0,0x53474753U);temp.SetValue(1,3);temp.SetValue(2,384);temp.SetValue(3,32);
   temp.SetValue(4,sweepCount);temp.SetValue(5,arenaSweepCount);temp.SetValue(6,2);temp.SetValue(7,1);
   const auto e=pipe.FetchEventID(HardEvent::S_MTE3);SetFlag<HardEvent::S_MTE3>(e);WaitFlag<HardEvent::S_MTE3>(e);
   GlobalTensor<uint32_t>w;w.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(o),8);DataCopy(w,temp,8);PipeBarrier<PIPE_ALL>();return;
  }
#endif
  for(uint32_t nb=0;nb<16;++nb)for(uint32_t m=0;m<16;++m)DataCopy(out[m*19456+tile*256+nb*16],partial[nb*256+m*16],16);
  PipeBarrier<PIPE_ALL>();
 }
}
