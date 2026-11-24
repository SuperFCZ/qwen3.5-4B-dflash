#include "cpu_vor.h"
#include "kv_prologue.h"
#include <cstring>
#include <vector>

int main() {
    using namespace AscendC;
    using namespace kv_panel;
    constexpr unsigned M=80;
    std::vector<uint16_t> input(M*K);
    for(size_t i=0;i<input.size();++i) input[i]=static_cast<uint16_t>(i*541+17);
    TPipe pipe;
    TBuf<TPosition::A1> dest;
    TBuf<TPosition::VECOUT> work;
    pipe.InitBuffer(dest,2*M*K);
    pipe.InitBuffer(work,WeightBytes);
    GlobalTensor<half> x;
    x.SetGlobalBuffer(reinterpret_cast<half*>(input.data()),M*K);
    auto a=dest.Get<half>(M*K);
    if(!PackActivation<M>(pipe,a,x,work.Get<half>(WeightBytes/2)))return 1;
    for(unsigned r=0;r<M;++r)
        for(unsigned k=0;k<K;++k)
            if(std::memcmp(a.data+((k/16)*M+r)*16+k%16, &input[r*K+k],2))return 2;
    return 0;
}
