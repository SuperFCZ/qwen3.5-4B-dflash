// SDK signature/constant checks only. No __global__, main, OPP or executable.
#include "kernel_operator.h"
#include "include/adv_api/matmul/matmul_intf.h"
using namespace AscendC;
static_assert(__NPU_ARCH__==2002,"dav_2002 only");
static_assert(QUE_MAX_EVENT==8,"reviewed event bound");
static_assert(GetPhyType(TPosition::CO1)==Hardware::L0C,"caller CO1");
static_assert(GetQueEvt(Hardware::GM,Hardware::L1,true,true)==HardEvent::MTE3_MTE1,"ND input forward");
static_assert(GetQueEvt(Hardware::GM,Hardware::L1,false,true)==HardEvent::MTE1_MTE3,"ND input reverse");
static_assert(Impl::Detail::MatmulFeatureTrait<CFG_NORM>::IsNeedUB(),"selected normal input path");
using A=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using C=MatmulType<TPosition::CO1,CubeFormat::NZ,int32_t>;
using Bias=MatmulType<TPosition::GM,CubeFormat::ND,int32_t>;
using MM=Matmul<A,A,C,Bias>;
__aicore__ bool signature_external(MM& mm,const LocalTensor<int32_t>& c){return mm.Iterate(false,c);}
__aicore__ void signature_vdeq(const LocalTensor<half>& dst,const LocalTensor<int32_t>& src,const LocalTensor<uint64_t>& scale){
 DataCopyParams cp{1,1,0,0};DataCopyEnhancedParams p;
 p.blockMode=BlockMode::BLOCK_MODE_MATRIX;p.deqScale=DeqScale::VDEQ16;p.isRelu=false;
 p.deqTensorAddr=reinterpret_cast<uint64_t>(scale.GetPhyAddr());DataCopy(dst,src,cp,p);
}
// Not a proposed completed schedule: just instantiate supported public signatures.
__aicore__ void signature_alloc(TPipe& pipe,TBuf<TPosition::CO1>& c,TBuf<TPosition::VECCALC>& d){
 pipe.InitBuffer(c,16384);pipe.InitBuffer(d,1280);
}
