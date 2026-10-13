#pragma once
#include "kernel_operator.h"
__aicore__ inline void RunVectorProbe(GM_ADDR values,GM_ADDR vector_fp32,GM_ADDR vector_fp16) {
    using namespace AscendC;
    TPipe pipe; TBuf<TPosition::VECCALC> source,rounded;
    pipe.InitBuffer(source,64*sizeof(float)); pipe.InitBuffer(rounded,64*sizeof(half));
    auto v=source.Get<float>(64); auto h=rounded.Get<half>(64);
    GlobalTensor<float> input,stages; GlobalTensor<half> result;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(values),64);
    stages.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(vector_fp32),3*64);
    result.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(vector_fp16),3*64);
    DataCopy(v,input,64); PipeBarrier<PIPE_ALL>();
    // Stage 0 standalone CAST_NONE; stage 1 Mins; stage 2 Mins -> Maxs.
    for(uint32_t stage=0;stage<3;++stage) {
        if(stage==1) Mins(v,v,65504.0f,64);
        if(stage==2) Maxs(v,v,-65504.0f,64);
        PipeBarrier<PIPE_ALL>();
        DataCopy(stages[stage*64],v,64);
        Cast(h,v,RoundMode::CAST_NONE,64); PipeBarrier<PIPE_ALL>();
        DataCopy(result[stage*64],h,64); PipeBarrier<PIPE_ALL>();
    }
}
