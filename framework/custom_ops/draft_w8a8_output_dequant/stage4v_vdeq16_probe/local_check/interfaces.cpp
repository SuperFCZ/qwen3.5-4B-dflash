// Compile-only interface instantiation. NO Kernel entry, launch, allocation or admission.
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;
static_assert(__NPU_ARCH__==2002 && g_coreType==AscendC::MIX,"dav_2002 coupled AI Core only");
__aicore__ inline void VdeqInterfaceSyntaxOnly(LocalTensor<int32_t> l0c,
    LocalTensor<int32_t> ubInt,LocalTensor<half> ubHalf,LocalTensor<uint64_t> words) {
    DataCopyParams p{1,1,0,0};
    DataCopyEnhancedParams plain;plain.blockMode=BlockMode::BLOCK_MODE_MATRIX;
    // Callable wrapper exists; its capacity/supported-path contract is NOT admitted.
    DataCopy(l0c,ubInt,p,plain);
    PipeBarrier<PIPE_ALL>();
    DataCopyEnhancedParams quant;quant.blockMode=BlockMode::BLOCK_MODE_MATRIX;
    quant.deqScale=DeqScale::VDEQ16;
    quant.deqTensorAddr=reinterpret_cast<uint64_t>(words.GetPhyAddr());
    DataCopy(ubHalf,l0c,p,quant);
    PipeBarrier<PIPE_ALL>();
}

using Ia=matmul::MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using Ic=matmul::MatmulType<TPosition::CO1,CubeFormat::NZ,int32_t>;
using Ibias=matmul::MatmulType<TPosition::GM,CubeFormat::ND,int32_t>;
__aicore__ inline void ExternalCo1SyntaxOnly(matmul::Matmul<Ia,Ia,Ic,Ibias>& mm,
    LocalTensor<int32_t> ownedCo1) { mm.Iterate(false,ownedCo1); }
