// Stage 2B experimental SDK output dequant. No explicit weight conversion.
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;
using namespace matmul;
#if defined(__NPU_ARCH__)
static_assert(__NPU_ARCH__ == 2002, "Stage 2B requires dav_2002");
#endif
using AType=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using BType=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using CType=MatmulType<TPosition::GM,CubeFormat::ND,half>;
using BiasType=MatmulType<TPosition::GM,CubeFormat::ND,int32_t>;
extern "C" __global__ __aicore__ void d_flash_w8_a8_output_dequant(
    GM_ADDR a,GM_ADDR b,GM_ADDR scale,GM_ADDR o,GM_ADDR workspace,GM_ADDR tiling) {
    GET_TILING_DATA(td,tiling);
    TPipe pipe;
    Matmul<AType,BType,CType,BiasType> mm;
    REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&td.cubeTilingData);
    TBuf<> scratch;pipe.InitBuffer(scratch,td.localMemSize);
    mm.SetLocalWorkspace(scratch.Get<uint8_t>(td.localMemSize));
    GlobalTensor<int8_t> aGm,bGm;
    GlobalTensor<half> oGm;
    GlobalTensor<uint64_t> scales;
    aGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(a),16*128);
    bGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(b),128*64);
    oGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(o),16*64);
    scales.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(scale),64);
    mm.SetTensorA(aGm);mm.SetTensorB(bGm,false);
    mm.SetQuantVector(scales);
    mm.IterateAll(oGm);
    PipeBarrier<PIPE_ALL>();mm.End();
}
