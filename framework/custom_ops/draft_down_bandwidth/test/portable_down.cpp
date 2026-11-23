// Exercise the actual Down kernel with the delayed-transfer CPU model.
#include "d_flash_down_panel.cpp"
#include <algorithm>
#include <fstream>
#include <iostream>

template<class T>
std::vector<T> load(const std::string& name, size_t elements) {
    std::ifstream f(name, std::ios::binary);
    std::vector<T> out(elements);
    if (!f.read(reinterpret_cast<char*>(out.data()),elements*sizeof(T)) ||
        f.peek()!=std::char_traits<char>::eof()) throw std::runtime_error(name);
    return out;
}

int main(int argc,char** argv) {
    using namespace down_panel;
    if (argc<2 || argc>3) return 1;
    const std::string dir=std::string(argv[1])+"/";
    const bool inject=argc==3;
    auto x=load<half>(dir+"x",M*K);
    auto q=load<int8_t>(dir+"codes",K*N);
    auto s=load<half>(dir+"scales",K/Group*N);
    auto golden=load<half>(dir+"golden",M*N);
    std::vector<half> y(M*N,half(-123));
    CpuTiling td{M,K,N,PanelK,262144-UserBytes,{N,PanelK,M,Columns,PanelK}};
    cpuOutputBase=y.data();
    cpuOutputWrites.assign(M*N,0);
    cpuOutputOwners.assign(M*N,-1);
    cpuBlockNum=8;
    if (inject) matmul::cpuFailIteration=1; // second partial of the first N64
    for (cpuBlockIdx=0;cpuBlockIdx<(inject?1u:cpuBlockNum);++cpuBlockIdx)
        d_flash_down_panel(x.data(),q.data(),s.data(),y.data(),nullptr,&td);
    if (inject) return std::all_of(cpuOutputWrites.begin(),cpuOutputWrites.end(),[](auto v){return v==0;}) &&
                       std::all_of(y.begin(),y.end(),[](auto v){return v==half(-123);}) ? 0 : 5;
    if (std::memcmp(y.data(),golden.data(),M*N*2)!=0) return 2;
    for (uint32_t i=0;i<M*N;++i)
        if (cpuOutputWrites[i]!=1 || cpuOutputOwners[i]!=int((i%N/Columns)%cpuBlockNum)) return 3;
    if (cpuMetrics.iterations!=Tiles*Panels || cpuMetrics.partials!=Tiles ||
        cpuMetrics.outputs!=Tiles || cpuMetrics.ends!=8 || cpuMetrics.fullBarriers!=Tiles*(2*Panels+1) ||
        cpuMetrics.copies!=8*M+Tiles*Panels*(1+2*Pieces) ||
        cpuMetrics.mulCalls!=Tiles*Panels*Pieces*(ChunkK/Group) ||
        cpuMetrics.castCalls!=2*Tiles*Panels*Pieces ||
        AscendC::storeWaits!=Tiles*Panels*(Pieces-W16Banks) ||
        (AscendC::vectorWhileStorePending>0)!=(W16Banks==2)) return 4;
    std::cout<<"all 40960 outputs matched; A1 scatter, K partials and W16 bank lifetimes checked\n";
}
