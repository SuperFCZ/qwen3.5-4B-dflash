// Parse/instantiate SDK interfaces only. No __global__ entry, launch, main or OPP.
#include "kernel_tpipe.h"
#include "include/adv_api/matmul/matmul_intf.h"
using namespace AscendC;
static_assert(__NPU_ARCH__ == 2002, "dav_2002 only");
using A=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using B=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using C=MatmulType<TPosition::VECIN,CubeFormat::NZ,half>;
using Bias=MatmulType<TPosition::GM,CubeFormat::ND,int32_t>;
using MM=Matmul<A,B,C,Bias>;
static_assert(!ToMatmulConfig(CFG_NORM).enVecND2NZ, "reviewed workspace specialization");
static_assert(Impl::Detail::MatmulFeatureTrait<CFG_NORM>::IsNeedUB(), "reviewed UB output path");
// These are independent signature checks, deliberately not an executable group pipeline.
__aicore__ bool check_iterate_true(MM& mm) { return mm.Iterate<true>(false); }
__aicore__ bool check_iterate_false(MM& mm) { return mm.Iterate<false>(false); }
__aicore__ void check_local_true(MM& mm,const LocalTensor<half>& x) { mm.GetTensorC<true>(x,0,true); }
__aicore__ void check_local_false(MM& mm,const LocalTensor<half>& x) { mm.GetTensorC<false>(x,0,true); }
__aicore__ void check_quant(MM& mm,const GlobalTensor<uint64_t>& q) { mm.SetQuantVector(q); }
__aicore__ void check_end(MM& mm) { mm.End(); }
