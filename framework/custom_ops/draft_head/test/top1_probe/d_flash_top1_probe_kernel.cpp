#include "head_vector_top1.h"
extern "C" __global__ __aicore__ void d_flash_top1_probe(GM_ADDR logits,GM_ADDR ids,GM_ADDR trace,GM_ADDR workspace,GM_ADDR tiling) {
    using namespace AscendC;
    TPipe pipe; TBuf<TPosition::VECIN> halfBuf,floatBuf,legacyBuf;
    pipe.InitBuffer(halfBuf,16*64*2); pipe.InitBuffer(floatBuf,16*64*4); pipe.InitBuffer(legacyBuf,16*64*4);
    auto h=halfBuf.Get<half>(16*64); auto f=floatBuf.Get<float>(16*64); auto legacy=legacyBuf.Get<float>(16*64);
    GlobalTensor<half> input; GlobalTensor<int64_t> output; GlobalTensor<uint32_t> debug;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(logits),15*64);
    output.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ids),15);
    debug.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(trace),15*240);
    auto localTrace=f.ReinterpretCast<uint32_t>()[256]; // offset 1024 B, outside row/scratch
    Duplicate(h,static_cast<half>(0),16*64); PipeBarrier<PIPE_ALL>();
    DataCopy(h,input,15*64); PipeBarrier<PIPE_ALL>();
    // Preserve the OLD full-tile Cast path separately for faithful NE/EQ diagnosis.
    Cast(legacy,h,RoundMode::CAST_NONE,16*64); PipeBarrier<PIPE_V>();
    for(uint32_t r=0;r<15;++r) {
        Duplicate(localTrace,uint32_t(0),240); PipeBarrier<PIPE_V>();
        Cast(f,h[r*64],RoundMode::CAST_NONE,64); PipeBarrier<PIPE_V>();
        const auto index=HeadVectorWinner<true>(h[r*64],f,f.ReinterpretCast<uint8_t>()[256],pipe,localTrace,legacy[r*64]);
        PipeBarrier<PIPE_ALL>();
        const auto e=pipe.FetchEventID(HardEvent::S_MTE3);
        SetFlag<HardEvent::S_MTE3>(e); WaitFlag<HardEvent::S_MTE3>(e);
        DataCopy(debug[r*240],localTrace,240); PipeBarrier<PIPE_ALL>();
        output.SetValue(r,index<64?int64_t(index):-1);
        DataCacheCleanAndInvalid<int64_t,CacheLine::SINGLE_CACHE_LINE>(output[r]);
    }
}
