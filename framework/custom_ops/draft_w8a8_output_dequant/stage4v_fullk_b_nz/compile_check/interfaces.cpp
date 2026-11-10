// Exact target signatures only; no Kernel entry, OPP or device launch.
#include "kernel_operator.h"
#include "include/adv_api/matmul/matmul_intf.h"
using namespace AscendC;
static_assert(__NPU_ARCH__==2002,"dav_2002 only");
static_assert(Impl::Detail::MatmulFeatureTrait<CFG_NORM>::IsNeedUB(),"selected UB route");
static_assert(!IsBasic(CFG_NORM),"Full-K Normal scheduler");
using A=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using B=MatmulType<TPosition::GM,CubeFormat::NZ,int8_t,true>;
using C=MatmulType<TPosition::VECIN,CubeFormat::NZ,half>;
using Bias=MatmulType<TPosition::GM,CubeFormat::ND,int32_t>;
using MM=Matmul<A,B,C,Bias>;
__aicore__ void exact_calls(MM& mm,const GlobalTensor<int8_t>& a,const GlobalTensor<int8_t>& b,const GlobalTensor<uint64_t>& s,const LocalTensor<uint8_t>& workspace,const LocalTensor<half>& partial){
 mm.SetLocalWorkspace(workspace);mm.SetTensorA(a);mm.SetTensorB(b,true);mm.SetQuantVector(s);
 if(mm.Iterate(false)){mm.GetTensorC(partial,0,true);mm.End();}
}
__aicore__ void exact_init(MM& mm,const TCubeTiling& tiling,TPipe& pipe){mm.Init(&tiling,&pipe);}

using TaggedB=MatmulInputBType<B>;
static_assert(Impl::Detail::GetCubeInBufferType<TaggedB,CFG_NORM>()==Impl::Detail::CubeInBufferType::NORMAL,"normal B input buffer");
static_assert(GetNdNzMask(CubeFormat::NZ,B::format)==0,"NZ does not request ND conversion queue mask");
static_assert(GetQueEvt(Hardware::GM,Hardware::L1,true,false)==HardEvent::MTE2_MTE1,"direct GM to L1 dependency");
static_assert(GetQueEvt(Hardware::GM,Hardware::L1,false,false)==HardEvent::MTE1_MTE2,"input reuse dependency");
static_assert(AscendC::AuxGetC0Size<int8_t>()==32,"int8 C0 layout");
