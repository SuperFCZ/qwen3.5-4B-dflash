// Syntax/type review only. No Kernel entry, OPP, launch or platform simulation.
#include "kernel_tpipe.h"
#include "include/adv_api/matmul/matmul_intf.h"
using namespace AscendC;
using A=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using B=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using C=MatmulType<TPosition::VECIN,CubeFormat::NZ,int32_t>;
using Bias=MatmulType<TPosition::GM,CubeFormat::ND,int32_t>;
using MM=Matmul<A,B,C,Bias>;
static_assert(__NPU_ARCH__==2002,"target architecture");
static_assert(Std::is_same<GetMmDstType<int8_t>::Type,int32_t>::value,"integer L0C");
static_assert(!Impl::Detail::IsQuantSenario<int32_t,int8_t>(),"NoQuant raw path");
static_assert(Impl::Detail::IsQuantSenario<half,int8_t>(),"distinct VDEQ16 path");
__aicore__ void raw_interfaces(MM& mm,const LocalTensor<int32_t>& output) {
 if(mm.Iterate(false))mm.GetTensorC(output,0,true);
 mm.End();
}
