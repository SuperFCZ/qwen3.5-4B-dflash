#include <algorithm>
#include "d_flash_fc_packed_panel.cpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

template<class T> std::vector<T> load(const std::string& path, size_t n)
{
    std::vector<T> v(n);std::ifstream f(path,std::ios::binary);
    if(!f.read(reinterpret_cast<char*>(v.data()),n*sizeof(T)) || f.peek()!=std::char_traits<char>::eof())throw std::runtime_error(path);
    return v;
}

int main(int argc,char** argv)
{
    if(argc<2||argc>3)return 1;
    const std::string d=std::string(argv[1])+"/";constexpr uint32_t m=64;
    auto x=load<half>(d+"x",m*K);auto w=load<int8_t>(d+"w",K*N);auto s=load<half>(d+"s",K*N/Group);
    auto expected=load<half>(d+"expected",m*N);std::vector<half> y(m*N,half(-127));
    CpuTiling td{m,K,N,PanelK,262144-UserBytes,{N,PanelK,m,Columns,PanelK}};
    cpuOutputBase=y.data();cpuOutputOwners.assign(m*N,-1);cpuOutputWrites.assign(m*N,0);cpuBlockNum=8;
    const int fail=argc==3?std::stoi(argv[2]):-1;matmul::cpuFailIteration=fail;
    for(cpuBlockIdx=0;cpuBlockIdx<(fail<0?8u:1u);++cpuBlockIdx)d_flash_fc_packed_panel(x.data(),w.data(),s.data(),y.data(),nullptr,&td);
    for(uint32_t i=0;i<m*N;++i){
        const bool written=fail<0 || (fail>=int(Panels) && i%N<Columns);
        if(cpuOutputWrites[i]!=int(written))return 2;
        if(written){if(std::memcmp(&y[i],&expected[i],2) || cpuOutputOwners[i]!=int((i%N/Columns)%8))return 3;}
        else if(y[i]!=half(-127))return 4;
    }
    if(fail>=0)return cpuMetrics.ends==1?0:5;
#if FC_TEST_BANKED
    const uint64_t copies=(3+Pieces)*Panels*Tiles;
    if(cpuMetrics.eventAllocations!=3*8 || cpuMetrics.eventReleases!=3*8) return 7;
#else
    const uint64_t copies=(2+2*Pieces)*Panels*Tiles;
#endif
    if(cpuMetrics.iterations!=Tiles*Panels || cpuMetrics.partials!=Tiles*(Panels-1) || cpuMetrics.outputs!=Tiles ||
       cpuMetrics.copies!=copies || cpuMetrics.fullBarriers!=3*Tiles*Panels ||
       cpuMetrics.mulCalls!=Groups*Pieces*Panels*Tiles || AscendC::storeWaits!=(Pieces-2)*Panels*Tiles)return 6;
    std::cout<<"C64 N"<<Columns<<": 163,840 outputs, real contiguous A panel loads, partials and owners verified\n";
}
