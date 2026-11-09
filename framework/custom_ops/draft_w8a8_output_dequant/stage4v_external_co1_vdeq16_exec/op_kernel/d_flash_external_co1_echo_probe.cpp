// ECHO_ONLY. VDEQ16 implementation is intentionally absent until real Echo PASS.
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;using namespace matmul;
#if defined(__NPU_ARCH__)
static_assert(__NPU_ARCH__==2002 && g_coreType==AscendC::MIX,"coupled2002 only");
static_assert(QUE_MAX_EVENT==8,"reviewed event ID range changed");
static_assert(!IsBasic(CFG_NORM),"Normal Full-K required");
static_assert(GetPhyType(TPosition::CO1)==Hardware::L0C,"CO1 mapping");
#endif
using A=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using C=MatmulType<TPosition::CO1,CubeFormat::NZ,int32_t>;
using Bias=MatmulType<TPosition::GM,CubeFormat::ND,int32_t>;
constexpr uint32_t guard=0x5a39c671;
__aicore__ inline void Init(LocalTensor<uint32_t>x,uint32_t bytes){Duplicate(x,guard,64);Duplicate(x[(256+bytes)/4],guard,64);}
__aicore__ inline uint32_t Delta(LocalTensor<uint32_t>x,uint32_t bytes){uint32_t d=0;for(uint32_t i=0;i<64;++i){d|=x.GetValue(i)^guard;d|=x.GetValue((256+bytes)/4+i)^guard;}return d;}
template<HardEvent E>__aicore__ inline bool Fence(TPipe&p,LocalTensor<uint64_t>meta,uint32_t slot){
 auto available=p.FetchEventID(E);if(available<0||available>=8)return false;
 auto id=p.AllocEventID<E>();if(id<0||id>=8)return false;
 meta.SetValue(8+slot,static_cast<uint64_t>(id));meta.SetValue(16+slot,meta.GetValue(16+slot)+1);
 SetFlag<E>(id);WaitFlag<E>(id);p.ReleaseEventID<E>(id);return true;
}
__aicore__ inline bool Guards(TPipe&p,LocalTensor<uint64_t>m,LocalTensor<uint32_t>s,LocalTensor<uint32_t>e,LocalTensor<uint32_t>q,LocalTensor<uint32_t>f){
 if(!Fence<HardEvent::V_S>(p,m,2))return false;
 return (Delta(s,202240)|Delta(e,16384)|Delta(q,2048)|Delta(f,8192))==0;
}
__aicore__ inline bool Arena(TPipe&p,LocalTensor<uint64_t>m,LocalTensor<uint32_t>tmp,GlobalTensor<uint32_t>g,uint64_t bytes){
 for(uint32_t side=0;side<2;++side){DataCopy(tmp,g[side?(512+bytes)/4:0],128);PipeBarrier<PIPE_ALL>();
 if(!Fence<HardEvent::MTE2_S>(p,m,6))return false;
 for(uint32_t i=0;i<128;++i)if(tmp.GetValue(i)!=guard)return false;}return true;
}
extern "C" __global__ __aicore__ void d_flash_external_co1_echo_probe(GM_ADDR a,GM_ADDR b,GM_ADDR scale,GM_ADDR o,GM_ADDR witness,GM_ADDR workspace,GM_ADDR tiling){
 GET_TILING_DATA(td,tiling);
 if(GetBlockIdx()!=0||GetBlockNum()!=1||td.launchBlocks!=1||td.outputMode!=0||td.tileId!=0||td.localMemSize!=202240||!workspace||!td.systemBytes||td.systemBytes>512ULL*1024*1024||td.arenaBytes!=(td.systemBytes+511)/512*512||td.arenaStride!=td.arenaBytes+1024||td.systemBytes+td.arenaStride>512ULL*1024*1024)return;
 TPipe pipe;TBuf<TPosition::CO1>co1Buffer;pipe.InitBuffer(co1Buffer,16384);
 auto co1=co1Buffer.Get<int32_t>();co1.SetSize(4096);
 if(co1.GetPhyAddr()!=0||co1.GetPhyAddr()%1024||co1.GetSize()!=4096||static_cast<TPosition>(co1.GetPosition())!=TPosition::CO1)return;
 TBuf<TPosition::VECIN>scratchRaw,echoRaw,scaleRaw;TBuf<TPosition::VECCALC>fp16Reserved;
 pipe.InitBuffer(scratchRaw,202240+512);pipe.InitBuffer(echoRaw,16384+512);pipe.InitBuffer(scaleRaw,2048+512);pipe.InitBuffer(fp16Reserved,8192+512);
 if(scratchRaw.Get<uint8_t>().GetPhyAddr()!=0||echoRaw.Get<uint8_t>().GetPhyAddr()!=202752||scaleRaw.Get<uint8_t>().GetPhyAddr()!=219648||fp16Reserved.Get<uint8_t>().GetPhyAddr()!=222208)return;
 Init(scratchRaw.Get<uint32_t>(),202240);Init(echoRaw.Get<uint32_t>(),16384);Init(scaleRaw.Get<uint32_t>(),2048);Init(fp16Reserved.Get<uint32_t>(),8192);PipeBarrier<PIPE_ALL>();
 auto scratch=scratchRaw.Get<uint8_t>()[256];scratch.SetSize(202240);
 auto echo=echoRaw.Get<int32_t>()[64];echo.SetSize(4096);
 auto quant=scaleRaw.Get<uint64_t>()[32];quant.SetSize(256);
 // Phase1-only aliases inside the unused FP16 payload: arena[0,512), witness[512,768).
 auto tmp=fp16Reserved.Get<uint32_t>()[64];tmp.SetSize(128);
 auto meta=fp16Reserved.Get<uint64_t>()[96];meta.SetSize(32);
 for(uint32_t i=0;i<32;++i)meta.SetValue(i,0);
 meta.SetValue(0,0x434f314543484f31ULL);meta.SetValue(1,1);meta.SetValue(2,1);meta.SetValue(3,co1.GetPhyAddr());meta.SetValue(4,16384);
 GlobalTensor<uint32_t>arena;arena.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(workspace),td.arenaStride/4);
 Duplicate(tmp,guard,128);if(!Fence<HardEvent::V_MTE3>(pipe,meta,4))return;
 DataCopy(arena,tmp,128);DataCopy(arena[(512+td.arenaBytes)/4],tmp,128);
 if(!Fence<HardEvent::MTE3_MTE2>(pipe,meta,5))return;
 GlobalTensor<uint64_t>failure;failure.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(witness),32);
 Matmul<A,A,C,Bias>mm;bool active=false;REGIST_MATMUL_OBJ(&pipe,workspace+512,mm,&td.cubeTilingData);
 #define ECHO_ABORT(code) {PipeBarrier<PIPE_ALL>();if(active){mm.End();active=false;}PipeBarrier<PIPE_ALL>();meta.SetValue(2,code);if(Fence<HardEvent::S_MTE3>(pipe,meta,7)){DataCopy(failure,meta,32);PipeBarrier<PIPE_ALL>();}return;}
 uint64_t ubEnd=(fp16Reserved.Get<uint8_t>().GetPhyAddr()+fp16Reserved.Get<uint8_t>().GetSize());if(ubEnd!=230912)ECHO_ABORT(10)meta.SetValue(5,ubEnd);
 if(!Arena(pipe,meta,tmp,arena,td.arenaBytes)||!Guards(pipe,meta,scratchRaw.Get<uint32_t>(),echoRaw.Get<uint32_t>(),scaleRaw.Get<uint32_t>(),fp16Reserved.Get<uint32_t>()))ECHO_ABORT(11)
 GlobalTensor<int8_t>ag,bg;GlobalTensor<uint64_t>sg,wg;GlobalTensor<int32_t>dst;
 ag.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(a),40960);bg.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(b),655360);
 sg.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(scale),256);wg.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(witness),32);dst.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(o),4096);
 DataCopy(quant,sg,256);if(!Fence<HardEvent::MTE2_V>(pipe,meta,3))ECHO_ABORT(12)
 Duplicate(echo.ReinterpretCast<uint32_t>(),static_cast<uint32_t>(0x7f7f7f7f),4096);PipeBarrier<PIPE_ALL>();
 mm.SetLocalWorkspace(scratch);mm.SetTensorA(ag);mm.SetTensorB(bg,false);
 active=true;if(!mm.Iterate(false,co1))ECHO_ABORT(13)
 if(!Fence<HardEvent::M_V>(pipe,meta,0))ECHO_ABORT(14)
 DataCopyParams cp{1,1,0,0};DataCopyEnhancedParams plain;plain.blockMode=BlockMode::BLOCK_MODE_MATRIX;
 plain.deqScale=DeqScale::DEQ_NONE;plain.isRelu=false;
 for(uint32_t nb=0;nb<16;++nb)DataCopy(echo[nb*256],co1[nb*256],cp,plain);
 PipeBarrier<PIPE_V>();
 if(!Guards(pipe,meta,scratchRaw.Get<uint32_t>(),echoRaw.Get<uint32_t>(),scaleRaw.Get<uint32_t>(),fp16Reserved.Get<uint32_t>()))ECHO_ABORT(11)
 if(!Fence<HardEvent::V_M>(pipe,meta,1))ECHO_ABORT(15) PipeBarrier<PIPE_ALL>();mm.End();active=false;PipeBarrier<PIPE_ALL>();
 if(!Arena(pipe,meta,tmp,arena,td.arenaBytes)||!Guards(pipe,meta,scratchRaw.Get<uint32_t>(),echoRaw.Get<uint32_t>(),scaleRaw.Get<uint32_t>(),fp16Reserved.Get<uint32_t>()))ECHO_ABORT(11)
 if(co1.GetPhyAddr()!=0||(fp16Reserved.Get<uint8_t>().GetPhyAddr()+fp16Reserved.Get<uint8_t>().GetSize())!=230912)ECHO_ABORT(16)
 if(!Fence<HardEvent::V_MTE3>(pipe,meta,4))ECHO_ABORT(17)
 for(uint32_t nb=0;nb<16;++nb)for(uint32_t m=0;m<16;++m)DataCopy(dst[m*256+nb*16],echo[nb*256+m*16],16);
 PipeBarrier<PIPE_ALL>();meta.SetValue(2,0);meta.SetValue(6,1);meta.SetValue(7,0); // no L0C canary coverage
 if(!Fence<HardEvent::S_MTE3>(pipe,meta,7))ECHO_ABORT(18) DataCopy(wg,meta,32);PipeBarrier<PIPE_ALL>();
#undef ECHO_ABORT
}
