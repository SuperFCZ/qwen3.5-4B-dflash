#include "d_flash_fc_panel.cpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

template<class T>
std::vector<T> load(const std::string& path, size_t n)
{
    std::vector<T> data(n); std::ifstream in(path, std::ios::binary);
    if (!in.read(reinterpret_cast<char*>(data.data()), n*sizeof(T)) || in.peek()!=std::char_traits<char>::eof())
        throw std::runtime_error(path);
    return data;
}

int main(int argc, char** argv)
{
    if (argc<3 || argc>4) return 1;
    const uint32_t m=std::stoul(argv[2]); const std::string d=std::string(argv[1])+"/";
    auto x=load<half>(d+"x"+argv[2],m*K);
    auto expected=load<half>(d+"expected"+argv[2],m*N);
    auto codes=load<int8_t>(d+"codes",K*N); auto scales=load<half>(d+"scales",K*N/Group);
    std::vector<half> y(m*N,half(-127));
    CpuTiling td{m,K,N,PanelK,262144-UserBytes,{N,StreamA?K:PanelK,m,Columns,PanelK}};
    cpuOutputBase=y.data();cpuOutputWrites.assign(m*N,0);cpuOutputOwners.assign(m*N,-1);
    cpuBlockNum=8;const int fail=argc==4?std::stoi(argv[3]):-1;matmul::cpuFailIteration=fail;
    for (cpuBlockIdx=0; cpuBlockIdx<(fail<0?cpuBlockNum:1u); ++cpuBlockIdx)
        d_flash_fc_panel(x.data(),codes.data(),scales.data(),y.data(),nullptr,&td);
    for (uint32_t i=0; i<m*N; ++i) {
        const bool written=fail<0 || (fail>=int(Panels) && i%N<Columns);
        if(cpuOutputWrites[i]!=int(written))return 2;
        if(written) {
            if(std::memcmp(&y[i],&expected[i],2) || cpuOutputOwners[i]!=int((i%N/Columns)%8))return 3;
        } else if(y[i]!=half(-127))return 4;
    }
    if(fail>=0)return cpuMetrics.ends==1?0:5;
    const uint32_t casts=StreamA?1:2;
    const uint32_t acopies=StreamA?0:(m==16?8*m:m*Panels*Tiles);
    if(cpuMetrics.iterations!=Tiles*Panels || cpuMetrics.partials!=Tiles*(Panels-1) ||
       cpuMetrics.outputs!=Tiles || cpuMetrics.ends!=8 || cpuMetrics.fullBarriers!=3*Tiles*Panels ||
       cpuMetrics.copies!=acopies+(1+2*Pieces)*Panels*Tiles || cpuMetrics.castCalls!=casts*Pieces*Panels*Tiles ||
       cpuMetrics.mulCalls!=GroupsPerPart*Pieces*Panels*Tiles || cpuMetrics.brcbCalls!=0 ||
       AscendC::storeWaits!=(Pieces-2)*Panels*Tiles) return 6;
    for(uint32_t i=0;i<Tiles*Panels;++i)
        if(matmul::qCastsAtIterate.at(i)!=(i+1)*Pieces*casts)return 7;
    for(uint32_t i=0;i<Tiles;++i)
        if(matmul::qCastsAtOutput.at(i)!=(i+1)*Panels*Pieces*casts+(i%(Tiles/8)+1<Tiles/8?casts:0))return 8;
    std::cout<<"C"<<m<<" K"<<PanelK<<": checked all "<<m*N<<" outputs, FP32 partials and cyclic owners\n";
}
