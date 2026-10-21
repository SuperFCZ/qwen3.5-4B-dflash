// Executes derived source against the existing CPU dependency/API model only.
// NOT an Ascend simulator, CANN compilation, hardware timing or NPU validation.
#include <iostream>
#include DERIVED_KERNEL
static uint16_t bits(half v){uint16_t b;std::memcpy(&b,&v,2);return b;}
int main(){
    constexpr uint32_t m=16,k=2560,n=19456;
    std::vector<half> x(m*k,0),w(n*k),y(m*n+2,static_cast<half>(-17));
    for(uint32_t nt=0;nt<n/64;++nt)for(uint32_t kt=0;kt<10;++kt)
        for(uint32_t a=0;a<64;++a)for(uint32_t b=0;b<256;++b){
            const uint32_t col=nt*64+a,kk=kt*256+b;
            float value=static_cast<float>(static_cast<int>((col*7+kk*3)%127)-63)/64;
            if(kk==0)value=127;if(kk==256)value=-127;if(kk==1024)value=static_cast<float>(col%7+1)/64;
            w[((nt*10+kt)*64+a)*256+b]=static_cast<half>(value);
        }
    for(uint32_t row=0;row<m;++row){
        const float sign=row%2?-1:1;
        x[row*k]=static_cast<half>(sign*1024);x[row*k+256]=static_cast<half>(sign*1024);
        x[row*k+1024]=static_cast<half>(sign/64);x[row*k+k-1]=static_cast<half>(sign/128);
    }
    const auto savedX=x,savedW=w;
    CpuTiling tiling{m,k,n,256,202240,{n,k,m,64,256}};
    cpuBlockNum=7;cpuOutputBase=y.data()+1;cpuOutputWrites.assign(m*n,0);cpuOutputOwners.assign(m*n,-1);
    for(int core=6;core>=0;--core){
        cpuBlockIdx=core;cpuMetrics={};
        d_flash_gate_fp16_tiled(x.data(),w.data(),cpuOutputBase,nullptr,&tiling);
        uint32_t tiles=0;for(uint32_t nt=core;nt<n/64;nt+=7)++tiles;
        assert(cpuMetrics.ubBytes==253952);
        assert(cpuMetrics.iterations==tiles*10 && cpuMetrics.partials==tiles*9);
        assert(cpuMetrics.copies==tiles*10 && cpuMetrics.castCalls==0 && cpuMetrics.brcbCalls==0 && cpuMetrics.mulCalls==0);
        assert(cpuMetrics.vectorBarriers==1 && cpuMetrics.fullBarriers==tiles*31);
        assert(cpuMetrics.outputs==tiles && cpuMetrics.ends==1 && cpuMetrics.eventAllocations==0);
    }
    for(uint32_t row=0;row<m;++row)for(uint32_t col=0;col<n;++col){
        const float sign=row%2?-1.0f:1.0f;
        const float last=static_cast<float>(static_cast<int>((col*7+(k-1)*3)%127)-63)/64;
        const half expected=static_cast<half>(sign*(col%7+1)/4096+sign*last/128);
        assert(bits(cpuOutputBase[row*n+col])==bits(expected));assert(cpuOutputWrites[row*n+col]==1);
    }
    assert(y.front()==static_cast<half>(-17)&&y.back()==static_cast<half>(-17));
    assert(std::memcmp(x.data(),savedX.data(),x.size()*2)==0 && std::memcmp(w.data(),savedW.data(),w.size()*2)==0);
    std::cout<<"CPU packed full-K ownership/FP32 cancellation/dependency model passed; NPU NOT_RUN\n";
}
