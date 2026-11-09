#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;using namespace matmul;
#if defined(__NPU_ARCH__)
static_assert(__NPU_ARCH__==2002 && g_coreType==AscendC::MIX,"coupled2002 required");
static_assert(!IsBasic(CFG_NORM),"Normal multi-K scheduler required");
#endif
#if defined(__DAV_M200_VEC__)
#error "Raw L0C requires AI Core"
#endif
using A=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using B=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using C=MatmulType<TPosition::VECIN,CubeFormat::NZ,int32_t>;
using Bias=MatmulType<TPosition::GM,CubeFormat::ND,int32_t>;
constexpr uint32_t word=0x5a39c671;
__aicore__ inline void InitGuard(LocalTensor<uint32_t>x,uint32_t bytes){Duplicate(x,word,64);Duplicate(x[(256+bytes)/4],word,64);}
__aicore__ inline uint32_t Delta(LocalTensor<uint32_t>x,uint32_t bytes){uint32_t d=0;for(uint32_t i=0;i<64;++i){d|=x.GetValue(i)^word;d|=x.GetValue((256+bytes)/4+i)^word;}return d;}
__aicore__ inline bool Guards(TPipe&p,LocalTensor<uint32_t>s,LocalTensor<uint32_t>o,LocalTensor<uint32_t>t){
 auto e=p.FetchEventID(HardEvent::V_S);SetFlag<HardEvent::V_S>(e);WaitFlag<HardEvent::V_S>(e);
 return (Delta(s,202240)|Delta(o,16384)|Delta(t,512))==0;
}
__aicore__ inline bool Arena(TPipe&p,LocalTensor<uint32_t>t,GlobalTensor<uint32_t>g,uint64_t bytes){
 for(uint32_t side=0;side<2;++side){DataCopy(t,g[side?(512+bytes)/4:0],128);PipeBarrier<PIPE_ALL>();
 auto e=p.FetchEventID(HardEvent::MTE2_S);SetFlag<HardEvent::MTE2_S>(e);WaitFlag<HardEvent::MTE2_S>(e);
 for(uint32_t i=0;i<128;++i)if(t.GetValue(i)!=word)return false;}return true;
}
extern "C" __global__ __aicore__ void d_flash_full_k_int32_probe(GM_ADDR a,GM_ADDR b,GM_ADDR o,GM_ADDR workspace,GM_ADDR tiling){
 GET_TILING_DATA(td,tiling);
 if(GetBlockIdx()!=0||GetBlockNum()!=1||td.launchBlocks!=1||td.tileId!=0||td.outputMode!=0||td.localMemSize!=202240||!workspace||!td.systemBytes||td.systemBytes>512ULL*1024*1024||td.arenaBytes!=(td.systemBytes+511)/512*512||td.arenaStride!=td.arenaBytes+1024||td.systemBytes+td.arenaStride>512ULL*1024*1024)return;
 TPipe pipe;TBuf<TPosition::VECIN>outRaw;pipe.InitBuffer(outRaw,16384+512);InitGuard(outRaw.Get<uint32_t>(),16384);
 auto partial=outRaw.Get<int32_t>()[64];partial.SetSize(4096);
 auto seed=partial.ReinterpretCast<uint32_t>();Duplicate(seed,word,128);PipeBarrier<PIPE_ALL>();
 GlobalTensor<uint32_t>arena;arena.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(workspace),td.arenaStride/4);
 DataCopy(arena,seed,128);DataCopy(arena[(512+td.arenaBytes)/4],seed,128);PipeBarrier<PIPE_ALL>();
 Matmul<A,B,C,Bias>mm;REGIST_MATMUL_OBJ(&pipe,workspace+512,mm,&td.cubeTilingData);
 TBuf<>scratchRaw;TBuf<TPosition::VECCALC>tempRaw;
 pipe.InitBuffer(scratchRaw,202240+512);pipe.InitBuffer(tempRaw,512+512);
 InitGuard(scratchRaw.Get<uint32_t>(),202240);InitGuard(tempRaw.Get<uint32_t>(),512);
 auto scratch=scratchRaw.Get<uint8_t>()[256];scratch.SetSize(202240);
 auto temp=tempRaw.Get<uint32_t>()[64];temp.SetSize(128);
 PipeBarrier<PIPE_ALL>();if(!Arena(pipe,temp,arena,td.arenaBytes)||!Guards(pipe,scratchRaw.Get<uint32_t>(),outRaw.Get<uint32_t>(),tempRaw.Get<uint32_t>()))return;
 GlobalTensor<int8_t>ag,bg;GlobalTensor<int32_t>dst;
 ag.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(a),40960);bg.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(b),655360);dst.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(o),4096);
 mm.SetLocalWorkspace(scratch);mm.SetTensorA(ag);mm.SetTensorB(bg,false);
 if(!mm.Iterate(false)){mm.End();PipeBarrier<PIPE_ALL>();return;}
 mm.GetTensorC(partial,0,true);PipeBarrier<PIPE_ALL>();mm.End();PipeBarrier<PIPE_ALL>();
 if(!Guards(pipe,scratchRaw.Get<uint32_t>(),outRaw.Get<uint32_t>(),tempRaw.Get<uint32_t>()))return;
 if(!Arena(pipe,temp,arena,td.arenaBytes)||!Guards(pipe,scratchRaw.Get<uint32_t>(),outRaw.Get<uint32_t>(),tempRaw.Get<uint32_t>()))return;
 for(uint32_t nb=0;nb<16;++nb)for(uint32_t m=0;m<16;++m)DataCopy(dst[m*256+nb*16],partial[nb*256+m*16],16);
 PipeBarrier<PIPE_ALL>();
}
