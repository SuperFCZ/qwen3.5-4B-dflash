#include "variant.cpp"
#include <fstream>
#include <iostream>

int main(int argc,char** argv) {
    if (argc != 3) return 1;
    constexpr unsigned M=16, K=2560, N=19456;
    const bool tiled=std::string(argv[2])=="tile";
    std::vector<int8_t> w(N*K);
    std::ifstream f(argv[1],std::ios::binary);
    if (!f.read(reinterpret_cast<char*>(w.data()),w.size()) || f.peek()!=std::char_traits<char>::eof()) return 1;
    std::vector<half> x(M*K),s(20*N),y(M*N,half(-123));
    for(unsigned g=0;g<20;++g) {
        for(unsigned n=0;n<N;++n) s[g*N+n]=half(0.001013f*(1+(g*13+n*7)%101));
        for(unsigned m=0;m<M;++m) x[m*K+g*128+(m*9+g*3)%128]=half(0.125f*(int((g+m)%7)-3));
    }
    CpuTiling t{M,K,N,256,185600,{N,K,M,64,256}};
    cpuOutputBase=y.data(); cpuOutputWrites.assign(M*N,0); cpuOutputOwners.assign(M*N,-1);
    cpuBlockNum=8;
    for(cpuBlockIdx=0;cpuBlockIdx<cpuBlockNum;++cpuBlockIdx) ENTRY(x.data(),w.data(),s.data(),y.data(),nullptr,&t);
    for(unsigned m=0;m<M;++m) for(unsigned n=0;n<N;++n) {
        float acc=0;
        for(unsigned g=0;g<20;++g) {
            const unsigned k=g*128+(m*9+g*3)%128;
            const int code=int((k/32+7*n+k%32)%255)-127;
            const half dequant=half(float(code)*float(s[g*N+n]));
            acc+=float(x[m*K+k])*float(dequant);
        }
        const half want=half(acc);
        const unsigned i=m*N+n;
        if(std::memcmp(&want,&y[i],2)||cpuOutputWrites[i]!=1||cpuOutputOwners[i]!=int(n/64%cpuBlockNum)) return 2;
    }
    constexpr unsigned iters=(N/64)*(K/256);
    if(cpuMetrics.iterations!=iters||cpuMetrics.partials!=iters-N/64||
       cpuMetrics.copies!=iters*(tiled?3:10)||cpuMetrics.castCalls!=iters*(tiled?1:8)||
       cpuMetrics.mulCalls!=2*iters||cpuMetrics.brcbCalls!=0||AscendC::cpuScaleCalls!=2*iters||
       cpuMetrics.fullBarriers!=iters+N/64||cpuMetrics.eventAllocations!=2*cpuBlockNum||
       cpuMetrics.eventReleases!=2*cpuBlockNum)return 3;
    std::cout<<"311296 exact, all tiles covered\n";
}
