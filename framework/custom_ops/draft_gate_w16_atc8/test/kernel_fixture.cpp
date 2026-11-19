// Deterministic host-only fixture: call the actual candidate translation unit.
#include "candidate.cpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
int main(int argc,char** argv) {
    constexpr unsigned m=16, k=2560, n=19456;
    bool down=argc>1 && std::string(argv[1])=="down";
    const unsigned K=down?9728:k, N=down?2560:n;
    std::vector<half> x(m*K), scales((K/128)*N), y(m*N,half(-301));
    std::vector<int8_t> weights(K*N);
    auto code=[](unsigned col,unsigned row){return int((row/32+col*3+row%32)%19)-9;};
    auto factor=[](unsigned group,unsigned col){return half(0.03125f*(1+(group+col)%5));};
    for(unsigned g=0;g<K/128;++g) {
        for(unsigned col=0;col<N;++col) scales[g*N+col]=factor(g,col);
        for(unsigned row=0;row<m;++row) x[row*K+128*g+(row*7+g)%128]=half((int(g%3)-1)*0.5f);
    }
    for(unsigned plane=0;plane<K/32;++plane) for(unsigned col=0;col<N;++col)
        for(unsigned u=0;u<32;++u) weights[(plane*N+col)*32+u]=int8_t(code(col,plane*32+u));
    cpuBlockNum=8;
    cpuBlockIdx=7;  // exercises final N tile as well as stride-8 reuse
    cpuOutputBase=y.data();
    cpuOutputWrites.assign(m*N,0);
    cpuOutputOwners.assign(m*N,-1);
    CpuTiling td{m,K,N,256,down?185600u:152832u,{N,K,m,64,256}};
    d_flash_group_quant_linear(x.data(),weights.data(),scales.data(),y.data(),nullptr,&td);
    for(unsigned row=0;row<m;++row) for(unsigned col=0;col<N;++col) {
        const unsigned index=row*N+col;
        const bool owned=(col/64)%8==7;
        if(cpuOutputWrites[index]!=unsigned(owned)) return 3;
        if(!owned) {if(y[index]!=half(-301)) return 4;continue;}
        float acc=0;
        for(unsigned g=0;g<K/128;++g) {
            unsigned j=128*g+(row*7+g)%128;
            half w=half(float(code(col,j))*float(factor(g,col)));
            acc+=float(x[row*K+j])*float(w);
        }
        if(y[index]!=half(acc)) return 5;
    }
    if(cpuMetrics.iterations!=N/64/8*(K/256) || cpuMetrics.ends!=1 ||
       cpuMetrics.eventAllocations!=2 || cpuMetrics.eventReleases!=2) return 6;
    std::cout<<"CPU_CANDIDATE_OWNERS_AND_BITS_MATCH\n";
}
