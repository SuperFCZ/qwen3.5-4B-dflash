#pragma once
#include "experiment_config.h" // generated only from source-reviewed settings
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;
using namespace matmul;
#if W8A8_RAW_OUTPUT
using OutputType=int32_t;
#else
using OutputType=half;
#endif
using AType=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
#if W8A8_B_NZ
using BType=MatmulType<TPosition::GM,CubeFormat::NZ,int8_t,true>;
#else
using BType=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
#endif
using CType=MatmulType<TPosition::GM,CubeFormat::ND,OutputType>;
using BiasType=MatmulType<TPosition::GM,CubeFormat::ND,int32_t>;
template<class Tiling>
__aicore__ inline void W8Process(GM_ADDR a,GM_ADDR b,GM_ADDR scale,GM_ADDR y,Tiling &td) {
    TPipe pipe;
    Matmul<AType,BType,CType,BiasType> mm;
    REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&td.cubeTilingData);
    TBuf<> scratch;pipe.InitBuffer(scratch,td.localMemSize);
    mm.SetLocalWorkspace(scratch.Get<uint8_t>(td.localMemSize));
    GlobalTensor<int8_t> aGm,bGm;
    GlobalTensor<OutputType> yGm;
    aGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(a),16*128);
    bGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(b),128*64);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutputType*>(y),16*64);
    mm.SetTensorA(aGm);mm.SetTensorB(bGm,W8A8_B_NZ!=0);
#if !W8A8_RAW_OUTPUT
    GlobalTensor<uint64_t> scaleGm;
    scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(scale),64);
    mm.SetQuantVector(scaleGm);
#endif
    mm.IterateAll(yGm);
    PipeBarrier<PIPE_ALL>();mm.End();
}
