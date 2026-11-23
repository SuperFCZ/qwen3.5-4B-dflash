// Executes the actual candidate with the repository's delayed-DMA CPU model.
#include "panel_spec.h"
#if DFLASH_PANEL_K == 2560
#include "full_l1.cpp"
#else
#include "d_flash_gate_panel.cpp"
#endif
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

template<class T>
std::vector<T> read(const std::string& p, size_t count) {
    std::vector<T> x(count);
    std::ifstream f(p,std::ios::binary);
    if (!f.read(reinterpret_cast<char*>(x.data()),count*sizeof(T)) || f.peek()!=std::char_traits<char>::eof())
        throw std::runtime_error(p);
    return x;
}

int main(int argc,char** argv) {
    using namespace gate_panel;
    if(argc!=2)return 1;
    const std::string root=std::string(argv[1])+"/";
    auto x=read<half>(root+"x", M*K);
    auto w=read<int8_t>(root+"codes", N*K);
    auto s=read<half>(root+"scales", N*K/Group);
    auto expected=read<half>(root+"reference", M*N);
    std::vector<half> y(M*N,half(-123));
    CpuTiling td{M,K,N,PanelK,262144-UserBytes,{N,K,M,Columns,PanelK}};
    cpuOutputBase=y.data();
    cpuOutputWrites.assign(M*N,0);
    cpuOutputOwners.assign(M*N,-1);
    cpuBlockNum=8;
    for(cpuBlockIdx=0;cpuBlockIdx<cpuBlockNum;++cpuBlockIdx)
        d_flash_gate_panel(x.data(),w.data(),s.data(),y.data(),nullptr,&td);
    if(std::memcmp(y.data(),expected.data(),y.size()*2)!=0)return 2;
    for(unsigned i=0;i<M*N;++i)
        if(cpuOutputWrites[i]!=1||cpuOutputOwners[i]!=int(i%N/Columns%cpuBlockNum))return 3;
    const unsigned fullFences = FullL1 ? 3*Tiles : Tiles*(Panels+1)+(SplitRaw?0:8);
    const unsigned dmaCalls = FullL1 ? 8*(K/16)+9*Tiles : Tiles*Panels*(SplitRaw?3:2);
    if(cpuMetrics.iterations!=Tiles*Panels || cpuMetrics.partials!=Tiles*(Panels-1) ||
       cpuMetrics.outputs!=Tiles || cpuMetrics.fullBarriers!=fullFences ||
       cpuMetrics.copies!=dmaCalls || cpuMetrics.brcbCalls!=0 ||
       cpuMetrics.mulCalls!=Tiles*Panels*4*Groups || cpuMetrics.ends!=8)return 4;
    std::cout<<"all outputs and tile owners match\n";
}
