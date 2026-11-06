// Compile-only resource selection proof. No Kernel entry or platform simulation.
#include "kernel_tpipe.h"
#include "include/adv_api/matmul/matmul_intf.h"
using namespace AscendC;
namespace D=AscendC::Impl::Detail;
using A=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using B=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using IA=MatmulInputAType<A,int8_t>;
using IB=MatmulInputBType<B,int8_t>;
static_assert(__NPU_ARCH__==2002,"reviewed architecture");
static_assert(D::GetCubeInBufferType<IA,CFG_NORM>()==D::CubeInBufferType::NORMAL,"A allocation contract");
static_assert(D::GetCubeInBufferType<IB,CFG_NORM>()==D::CubeInBufferType::NORMAL,"B allocation contract");
static_assert(CFG_NORM.doNorm && CFG_NORM.enableInit && !CFG_NORM.doIBShareNorm && !CFG_NORM.doMultiDataLoad,"Normal initialization");
static_assert(!HasScalePosition<A>::value && !HasScalePosition<B>::value,"no MX L1 scale tensors");
static_assert(!CFG_NORM.enableL1CacheUB && !CFG_NORM.enableL1BankConflictOptimise && !CFG_NORM.enVecND2NZ,"no extra L1 cache/padding route");
static_assert(D::MatmulFeatureTrait<CFG_NORM>::IsNeedUB(),"quant coefficient uses UB workspace");
static_assert(Std::is_same<typename decltype(GetTransBDataType<A,B,CFG_NORM>())::T,int8_t>::value,"INT8 B size");
