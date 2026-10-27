#include "kernel_operator.h"
using namespace AscendC;
#if defined(__NPU_ARCH__)
static_assert(__NPU_ARCH__ == 2002, "Stage3B-1 targets Ascend310P3/dav_2002");
#endif
extern "C" __global__ __aicore__ void d_flash_w8_a8_partial_reduce(
    GM_ADDR partials,GM_ADDR o,GM_ADDR workspace,GM_ADDR tiling) {
    GET_TILING_DATA(td,tiling);
    constexpr uint32_t count=1024;
    if(td.groupCount!=20||td.elements!=count)return; // fail by output poison, never read outside fixed ABI
    TPipe pipe;
    TQue<TPosition::VECIN,1> input;
    TQue<TPosition::VECOUT,1> output;
    TBuf<TPosition::VECCALC> converted,accumulator;
    pipe.InitBuffer(input,1,count*sizeof(half));
    pipe.InitBuffer(converted,count*sizeof(float));
    pipe.InitBuffer(accumulator,count*sizeof(float));
    pipe.InitBuffer(output,1,count*sizeof(half));
    GlobalTensor<half> x,y;
    x.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(partials),20*count);
    y.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(o),count);
    auto temp=converted.Get<float>(count);auto sum=accumulator.Get<float>(count);
    Duplicate(sum,0.0f,count);PipeBarrier<PIPE_V>();
    for(uint32_t group=0;group<20;++group) {
        auto part=input.AllocTensor<half>();
        DataCopy(part,x[group*count],count);
        input.EnQue(part);part=input.DeQue<half>(); // MTE2 -> V through VECIN queue
        Cast(temp,part,RoundMode::CAST_NONE,count);
        PipeBarrier<PIPE_V>();
        // Full elementwise overlap only; no cross-repeat address dependence.
        Add(sum,sum,temp,count);
        PipeBarrier<PIPE_V>(); // preserve FP32 group order and scratch lifetime
        input.FreeTensor(part);
    }
    auto result=output.AllocTensor<half>();
    Cast(result,sum,RoundMode::CAST_NONE,count); // the sole FP16 rounding boundary
    output.EnQue(result);result=output.DeQue<half>(); // V -> MTE3
    DataCopy(y,result,count);output.FreeTensor(result);
    PipeBarrier<PIPE_ALL>();
}
