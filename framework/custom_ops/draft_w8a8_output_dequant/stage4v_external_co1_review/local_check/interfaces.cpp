// Syntax/type inspection only: no global Kernel, no launch, no runtime allocation.
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;
static_assert(__NPU_ARCH__==2002 && g_coreType==AscendC::MIX,"coupled dav_2002 only");
static_assert(GetPhyType(TPosition::CO1)==Hardware::L0C,"CO1 must map to physical L0C");
using A=matmul::MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using C=matmul::MatmulType<TPosition::CO1,CubeFormat::NZ,int32_t>;
using Bias=matmul::MatmulType<TPosition::GM,CubeFormat::ND,int32_t>;
// Independent API instantiations, NOT an admitted execution sequence.
__aicore__ inline void CheckExternalSignature(matmul::Matmul<A,A,C,Bias>& mm,LocalTensor<int32_t> x){mm.Iterate(false,x);}
__aicore__ inline void CheckBufferSignature(TPipe& p,TBuf<TPosition::CO1>& b){p.InitBuffer(b,16384);}
__aicore__ inline void CheckEventSignature(TPipe& p){
 auto m=p.AllocEventID<HardEvent::M_V>();SetFlag<HardEvent::M_V>(m);WaitFlag<HardEvent::M_V>(m);p.ReleaseEventID<HardEvent::M_V>(m);
 auto v=p.AllocEventID<HardEvent::V_M>();SetFlag<HardEvent::V_M>(v);WaitFlag<HardEvent::V_M>(v);p.ReleaseEventID<HardEvent::V_M>(v);
}
