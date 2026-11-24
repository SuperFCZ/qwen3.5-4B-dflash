#include "d_flash_o_panel.cpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

template <class T>
std::vector<T> read(const std::string& path, size_t count)
{
    std::vector<T> values(count);
    std::ifstream file(path, std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(values.data()), count*sizeof(T)) ||
        file.peek()!=std::char_traits<char>::eof()) throw std::runtime_error(path);
    return values;
}

int main(int argc, char** argv)
{
    if (argc<2 || argc>3) return 1;
    const std::string dir=std::string(argv[1])+"/";
    auto x=read<half>(dir+"x", M*K);
    auto codes=read<int8_t>(dir+"codes", N*K);
    auto scales=read<half>(dir+"scales", N*K/Group);
    auto expected=read<half>(dir+"expected", M*N);
    std::vector<half> y(M*N,half(-127));
    CpuTiling td{M,K,N,PanelK,262144-UserBytes,{N,PanelK,M,Columns,PanelK}};
    cpuOutputBase=y.data(); cpuOutputWrites.assign(M*N,0); cpuOutputOwners.assign(M*N,-1);
    cpuBlockNum=8;
    int fail=argc==3 ? std::stoi(argv[2]) : -1;
    matmul::cpuFailIteration=fail;
    for(cpuBlockIdx=0;cpuBlockIdx<(fail>=0?1u:cpuBlockNum);++cpuBlockIdx)
        d_flash_o_panel(x.data(),codes.data(),scales.data(),y.data(),nullptr,&td);
    for(uint32_t i=0;i<M*N;++i) {
        const bool written=fail<0 || (fail==1 && i%N<Columns);
        if(cpuOutputWrites[i]!=int(written)) return 2;
        if(written) {
            if(std::memcmp(&y[i],&expected[i],2) || cpuOutputOwners[i]!=int((i%N/Columns)%cpuBlockNum)) return 3;
        } else if(y[i]!=half(-127)) return 4;
    }
    if(fail>=0)return cpuMetrics.ends==1?0:5;
    if(cpuMetrics.iterations!=Tiles || cpuMetrics.partials || cpuMetrics.outputs!=Tiles || cpuMetrics.ends!=8 ||
       cpuMetrics.fullBarriers!=3*Tiles || cpuMetrics.copies!=8*M+(1+2*Pieces)*Tiles ||
       cpuMetrics.castCalls!=2*Pieces*Tiles || cpuMetrics.mulCalls!=Pieces*GroupsPerChunk*Tiles ||
       AscendC::storeWaits!=(Pieces-2)*Tiles || !AscendC::vectorWhileStorePending) return 6;
    for(uint32_t i=0;i<Tiles;++i) {
        const bool next=i%(Tiles/8)+1<Tiles/8;
        const auto count=2*Pieces*(i+1);
        if(matmul::qCastsAtIterate.at(i)!=count || matmul::qCastsAtOutput.at(i)!=count+(next?2:0))return 7;
    }
    std::cout<<"40,960 O outputs, K2048 chunk scales, cyclic owners and next-N scheduling match\n";
}
