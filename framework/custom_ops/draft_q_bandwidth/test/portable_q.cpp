#include "d_flash_q_panel.cpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

template<class T>
std::vector<T> load(const std::string& file,size_t n) {
    std::vector<T> v(n);std::ifstream f(file,std::ios::binary);
    if(!f.read(reinterpret_cast<char*>(v.data()),n*sizeof(T)) || f.peek()!=std::char_traits<char>::eof())
        throw std::runtime_error(file);
    return v;
}
int main(int argc,char** argv) {
    if(argc<2 || argc>3)return 1;
    const std::string dir=std::string(argv[1])+"/";
    auto x=load<half>(dir+"x",M*K);
    auto codes=load<int8_t>(dir+"codes",N*K);
    auto scales=load<half>(dir+"scales",N*K/Group);
    auto expected=load<half>(dir+"expected",M*N);
    std::vector<half> y(M*N,half(-127));
    CpuTiling td{M,K,N,K,262144-UserBytes,{N,K,M,Columns,K}};
    cpuOutputBase=y.data();cpuOutputWrites.assign(M*N,0);cpuOutputOwners.assign(M*N,-1);
    cpuBlockNum=8;
    const bool fail=argc==3;
    if(fail)matmul::cpuFailIteration=0;
    for(cpuBlockIdx=0;cpuBlockIdx<(fail?1u:cpuBlockNum);++cpuBlockIdx)
        d_flash_q_panel(x.data(),codes.data(),scales.data(),y.data(),nullptr,&td);
    if(fail)return std::all_of(cpuOutputWrites.begin(),cpuOutputWrites.end(),[](int n){return n==0;}) &&
                   std::all_of(y.begin(),y.end(),[](half v){return v==half(-127);}) && cpuMetrics.ends==1 ? 0:4;
    if(std::memcmp(expected.data(),y.data(),2*M*N)!=0)return 2;
    for(uint32_t i=0;i<M*N;++i)if(cpuOutputWrites[i]!=1 || cpuOutputOwners[i]!=int(i%N/Columns%8))return 3;
    if(cpuMetrics.iterations!=Tiles || cpuMetrics.partials!=0 || cpuMetrics.outputs!=Tiles || cpuMetrics.ends!=8 ||
       cpuMetrics.copies!=8*M+9*Tiles || cpuMetrics.fullBarriers!=3*Tiles ||
       cpuMetrics.castCalls!=8*Tiles || cpuMetrics.mulCalls!=4*Groups*Tiles || cpuMetrics.brcbCalls!=0 ||
       AscendC::storeWaits!=2*Tiles || !AscendC::vectorWhileStorePending ||
       cpuMetrics.castsWithPendingDma!=(Lookahead?6*Tiles:0))return 5;
    std::cout<<"65,536 Q outputs, owners, W16 lifetimes and raw prefetch ordering checked\n";
}
