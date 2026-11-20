#include "variant.cpp"
#include <iostream>

int main(int argc, char** argv) {
    const bool candidate = argc==2 && std::string(argv[1])=="transpose";
    constexpr unsigned M=16, K=2560, N=19456;
    std::vector<half> x(M*K), scale(20*N), y(M*N,half(-123));
    std::vector<int8_t> w(N*K);
    auto code=[](unsigned n,unsigned k){ return int((k/32+7*n+k%32)%255)-127; };
    auto factor=[](unsigned g,unsigned n){return half(0.001013f*(1+(g*13+n*7)%101));};
    for(unsigned g=0;g<20;++g) {
        for(unsigned n=0;n<N;++n)scale[g*N+n]=factor(g,n);
        for(unsigned m=0;m<M;++m)x[m*K+g*128+(m*9+g*3)%128]=half(0.125f*(int((g+m)%7)-3));
    }
    for(unsigned p=0;p<K/32;++p)for(unsigned n=0;n<N;++n)for(unsigned lane=0;lane<32;++lane)
        w[(p*N+n)*32+lane]=int8_t(code(n,p*32+lane));
    CpuTiling t{M,K,N,256,185600,{N,K,M,64,256}};
    cpuOutputBase=y.data();cpuOutputWrites.assign(M*N,0);cpuOutputOwners.assign(M*N,-1);
    cpuBlockNum=8;
    for(cpuBlockIdx=0;cpuBlockIdx<8;++cpuBlockIdx)
        d_flash_group_quant_linear(x.data(),w.data(),scale.data(),y.data(),nullptr,&t);
    for(unsigned m=0;m<M;++m)for(unsigned n=0;n<N;++n) {
        float acc=0;
        for(unsigned g=0;g<20;++g){
            unsigned k=g*128+(m*9+g*3)%128;
            half weight=half(float(code(n,k))*float(factor(g,n)));
            acc+=float(x[m*K+k])*float(weight);
        }
        const half want=half(acc);
        const unsigned i=m*N+n;
        if(std::memcmp(&want,&y[i],2) || cpuOutputWrites[i]!=1 || cpuOutputOwners[i]!=int(n/64%8))return 2;
    }
    if(cpuMetrics.iterations!=3040 || cpuMetrics.partials!=2736 || cpuMetrics.fullBarriers!=3344 ||
       cpuMetrics.mulCalls!=6080 || cpuMetrics.castCalls!=24320 ||
       cpuMetrics.brcbCalls!=(candidate?0u:6080u) || AscendC::cpuScaleCalls!=(candidate?6080u:0u))return 3;
    std::cout<<"PORTABLE_GATE_311296_BITS_MATCH\n";
}
