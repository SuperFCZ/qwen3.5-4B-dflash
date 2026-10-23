// Independent integer-only ND ABI. No quantization parameters or conversion path.
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;
using namespace matmul;
using AType=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using BType=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using CType=MatmulType<TPosition::GM,CubeFormat::ND,int32_t>;
using BiasType=MatmulType<TPosition::GM,CubeFormat::ND,int32_t>;
extern "C" __global__ __aicore__ void d_flash_w8_a8_accumulator(
    GM_ADDR a,GM_ADDR b,GM_ADDR y,GM_ADDR workspace,GM_ADDR tiling) {
    GET_TILING_DATA(td,tiling);
    TPipe pipe;
    Matmul<AType,BType,CType,BiasType> mm;
    REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&td.cubeTilingData);
    TBuf<> scratch;pipe.InitBuffer(scratch,td.localMemSize);
    mm.SetLocalWorkspace(scratch.Get<uint8_t>(td.localMemSize));
    GlobalTensor<int8_t> aGm,bGm;
    GlobalTensor<int32_t> yGm;
    aGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(a),16*128);
    bGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(b),128*64);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(y),16*64);
    mm.SetTensorA(aGm);mm.SetTensorB(bGm,false);
    mm.IterateAll(yGm);
    PipeBarrier<PIPE_ALL>();mm.End();
}
