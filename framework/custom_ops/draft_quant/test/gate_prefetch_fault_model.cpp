// Actual Gate Kernel fault paths; dependency-model evidence only, never NPU.
#include <iostream>
#include "../op_kernel/d_flash_group_quant_linear.cpp"
int main() {
    constexpr uint32_t m=16,k=2560,n=19456;
    std::vector<half> x(m*k),s(20*n,static_cast<half>(1)),y(m*n,static_cast<half>(-17));
    std::vector<int8_t> q(k*n);
    CpuTiling tiling{m,k,n,256,185600,{n,k,m,64,256}};
    cpuBlockIdx=0;cpuBlockNum=1;cpuOutputBase=y.data();
    // First/middle/last K tile in first and second N tiles, with/without lookahead.
    for (int failure : {0,4,9,10,14,19}) {
        std::fill(y.begin(),y.end(),static_cast<half>(-17));
        cpuOutputWrites.assign(m*n,0);cpuOutputOwners.assign(m*n,-1);
        cpuMetrics={};matmul::cpuFailIteration=failure;
        d_flash_group_quant_linear(x.data(),q.data(),s.data(),y.data(),nullptr,&tiling);
        assert(cpuMetrics.iterations==static_cast<uint32_t>(failure));
        assert(cpuMetrics.outputs==static_cast<uint32_t>(failure/10));
        assert(cpuMetrics.ends==1 && cpuMetrics.eventAllocations==2 && cpuMetrics.eventReleases==2);
        assert(cpuMetrics.ubBytes==262144-8192); // SDK Brcb reservation is not a TPipe allocation.
        for (uint32_t row=0;row<m;++row) for(uint32_t col=0;col<n;++col) {
            const bool written=col<static_cast<uint32_t>(failure/10)*64;
            assert(cpuOutputWrites[row*n+col]==static_cast<unsigned>(written));
            assert(y[row*n+col]==static_cast<half>(written?0:-17));
        }
    }
    matmul::cpuFailIteration=-1;
    std::cout<<"Gate prefetch faults/first-last K/N/event release verified in CPU model; NPU NOT_RUN\n";
}
