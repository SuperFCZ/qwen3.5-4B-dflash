#include "head_vector_top1.h"
extern "C" __global__ __aicore__ void d_flash_top1_probe(GM_ADDR logits,GM_ADDR ids,GM_ADDR workspace,GM_ADDR tiling) {
    using namespace AscendC;
    TPipe pipe; TBuf<TPosition::VECIN> halfBuf,floatBuf;
    pipe.InitBuffer(halfBuf,16*64*2); pipe.InitBuffer(floatBuf,16*64*4);
    auto h=halfBuf.Get<half>(16*64); auto f=floatBuf.Get<float>(16*64);
    GlobalTensor<half> input; GlobalTensor<int64_t> output;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(logits),15*64);
    output.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ids),15);
    Duplicate(h,static_cast<half>(0),16*64); PipeBarrier<PIPE_ALL>();
    DataCopy(h,input,15*64); PipeBarrier<PIPE_ALL>();
    Cast(f,h,RoundMode::CAST_NONE,16*64); PipeBarrier<PIPE_V>();
    for(uint32_t r=0;r<15;++r) {
        const auto index=HeadVectorWinner(f[r*64],f.ReinterpretCast<uint8_t>()[15*64*4],pipe);
        output.SetValue(r,index<64?int64_t(index):-1);
        DataCacheCleanAndInvalid<int64_t,CacheLine::SINGLE_CACHE_LINE>(output[r]);
    }
}
