// Single K128 Local NZ output probe. No fusion, manual dequant or fallback.
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;
using namespace matmul;
#if defined(__NPU_ARCH__)
static_assert(__NPU_ARCH__ == 2002, "Stage3B2 requires dav_2002");
#endif
#if defined(__DAV_M200_VEC__)
#error "Local L0C DataCopy probe requires the AI Core path, not a vector-only target"
#endif
using AType=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using BType=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using CType=MatmulType<TPosition::VECIN,CubeFormat::NZ,half>;
using BiasType=MatmulType<TPosition::GM,CubeFormat::ND,int32_t>;
extern "C" __global__ __aicore__ void d_flash_w8_a8_local_dequant(
    GM_ADDR a,GM_ADDR b,GM_ADDR scale,GM_ADDR o,GM_ADDR workspace,GM_ADDR tiling) {
    GET_TILING_DATA(td,tiling);
    TPipe pipe;
    Matmul<AType,BType,CType,BiasType> mm;
    REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&td.cubeTilingData);
    TBuf<> scratch;TBuf<TPosition::VECIN> result;
    pipe.InitBuffer(scratch,td.localMemSize);pipe.InitBuffer(result,2048);
    mm.SetLocalWorkspace(scratch.Get<uint8_t>(td.localMemSize));
    GlobalTensor<int8_t> aGm,bGm;GlobalTensor<uint64_t> scales;GlobalTensor<half> out;
    aGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(a),2048);
    bGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(b),8192);
    scales.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(scale),64);
    out.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(o),1024);
    mm.SetTensorA(aGm);mm.SetTensorB(bGm,false);mm.SetQuantVector(scales);
    if(!mm.Iterate(false)) { mm.End();return; } // leave GM poison on failed iteration
    auto nz=result.Get<half>(1024);
    mm.GetTensorC(nz,0,true);
    PipeBarrier<PIPE_ALL>();mm.End();PipeBarrier<PIPE_ALL>();
    // FP16 NZ [N/16,M,16] for aligned M16/N64. Each copy is exactly 32B.
    // Only rearranges bits; never scales or rounds. The NPU probe verifies this layout.
    for(uint32_t nb=0;nb<4;++nb)
        for(uint32_t m=0;m<16;++m)
            DataCopy(out[m*64+nb*16],nz[nb*256+m*16],16);
    PipeBarrier<PIPE_ALL>();
}
