#include "d_flash_kv_panel.cpp"
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

template<class T> std::vector<T> read(const std::string& name, size_t count) {
    std::vector<T> result(count);
    std::ifstream input(name, std::ios::binary);
    if (!input.read(reinterpret_cast<char*>(result.data()), count*sizeof(T)) ||
        input.peek()!=std::char_traits<char>::eof()) throw std::runtime_error(name);
    return result;
}

int main(int argc,char** argv) {
    if(argc!=4) return 1;
    const std::string dir=std::string(argv[1])+"/";
    const uint32_t m=std::stoi(argv[2]);
    const int fail=std::stoi(argv[3]);
    auto x=read<half>(dir+(DFLASH_KV_INPUT==2 ? "packed":"nd"), m*K);
    auto w=read<int8_t>(dir+"w", K*N);
    auto s=read<half>(dir+"s", (K/Group)*N);
    auto expected=read<half>(dir+"gold",m*N);
    std::vector<half> y(m*N,half(-127));
    CpuTiling td{m,K,N,PanelK,262144-UserBytes,{N,PanelK,m,Columns,PanelK}};
    cpuOutputBase=y.data();
    cpuOutputWrites.assign(m*N,0);
    cpuOutputOwners.assign(m*N,-1);
    cpuBlockNum=8;
    matmul::cpuFailIteration=fail;
    for(cpuBlockIdx=0;cpuBlockIdx<(fail>=0?1u:cpuBlockNum);++cpuBlockIdx)
        d_flash_kv_panel(x.data(),w.data(),s.data(),y.data(),nullptr,&td);
    for(uint32_t i=0;i<m*N;++i) {
        const bool written=fail<0 || (i%N/Columns%8==0 && int((i%N/Columns/8+1)*Panels)<=fail);
        if(cpuOutputWrites[i]!=int(written)) return 2;
        if(written && (std::memcmp(&y[i],&expected[i],2) || cpuOutputOwners[i]!=int(i%N/Columns%8))) return 3;
        if(!written && y[i]!=half(-127)) return 4;
    }
    if(fail>=0) return cpuMetrics.ends==1?0:5;
#ifdef DFLASH_KV_WIDTH
    const auto calls=Tiles*Panels;
    return cpuMetrics.ends==8 && cpuMetrics.iterations==calls && cpuMetrics.fullBarriers==3*calls &&
           cpuMetrics.partials==Tiles*(Panels-1) && cpuMetrics.outputs==Tiles &&
           cpuMetrics.castCalls==Pieces*calls && cpuMetrics.mulCalls==Pieces*Groups*calls ? 0:7;
#else
    const unsigned stageFences=DFLASH_KV_INPUT==1?2*((m+StagingRows-1)/StagingRows)*8:0;
    return cpuMetrics.fullBarriers==3*Tiles+stageFences && cpuMetrics.ends==8 &&
           cpuMetrics.iterations==Tiles && cpuMetrics.partials==0 && cpuMetrics.castCalls==8*Tiles &&
           cpuMetrics.mulCalls==4*Groups*Tiles && AscendC::storeWaits==2*Tiles ? 0:6;
#endif
}
