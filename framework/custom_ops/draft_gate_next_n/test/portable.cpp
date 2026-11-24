#include "gate_impl.cpp"
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

template<class T> std::vector<T> read(const std::string& f,size_t count) {
    std::vector<T> v(count);
    std::ifstream stream(f,std::ios::binary);
    if(!stream.read(reinterpret_cast<char*>(v.data()),count*sizeof(T)) ||
       stream.peek()!=std::char_traits<char>::eof()) throw std::runtime_error(f);
    return v;
}

int main(int argc,char** argv) {
    if(argc!=3) return 1;
    const std::string d=std::string(argv[1])+"/";
    auto x=read<half>(d+"x",M*K);
    auto w=read<int8_t>(d+"w",N*K);
    auto s=read<half>(d+"s",N*K/Group);
    auto expected=read<half>(d+"gold",M*N);
    std::vector<half> y(M*N,half(-127));
    CpuTiling td{M,K,N,K,262144-UserBytes,{N,K,M,Columns,K}};
    cpuOutputBase=y.data();
    cpuOutputWrites.assign(M*N,0);cpuOutputOwners.assign(M*N,-1);
    const int fail=std::stoi(argv[2]);
    matmul::cpuFailIteration=fail;
    cpuBlockNum=8;
    for(cpuBlockIdx=0;cpuBlockIdx<(fail<0?8u:1u);++cpuBlockIdx)
        d_flash_gate_panel(x.data(),w.data(),s.data(),y.data(),nullptr,&td);
    for(unsigned i=0;i<M*N;++i) {
        const auto tile=i%N/Columns;
        const bool written=fail<0 || (tile%8==0 && int(tile/8)<fail);
        if(cpuOutputWrites[i]!=int(written))return 2;
        if(written && (std::memcmp(&y[i],&expected[i],2) || cpuOutputOwners[i]!=int(tile%8)))return 3;
        if(!written && y[i]!=half(-127))return 4;
    }
    if(fail>=0)return cpuMetrics.ends==1 && int(cpuMetrics.iterations)==fail?0:5;
    if(cpuMetrics.iterations!=Tiles || cpuMetrics.partials || cpuMetrics.outputs!=Tiles ||
       cpuMetrics.ends!=8 || cpuMetrics.fullBarriers!=3*Tiles || cpuMetrics.copies!=8*(K/16)+9*Tiles ||
       cpuMetrics.castCalls!=8*Tiles || cpuMetrics.mulCalls!=4*Groups*Tiles || AscendC::storeWaits!=2*Tiles)
        return 6;
    for(unsigned i=0;i<Tiles;++i) {
        const bool next=i%(Tiles/8)+1<Tiles/8;
        if(matmul::qCastsAtIterate.at(i)!=8*(i+1) ||
           matmul::qCastsAtOutput.at(i)!=8*(i+1)+(TEST_SCHEDULE && next?2:0))return 7;
        if(AscendC::pendingAtFullFence.at(3*i)!=(TEST_SCHEDULE==2 && next?2u:0u))return 8;
    }
    return 0;
}
