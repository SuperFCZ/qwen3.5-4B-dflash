// Runs the production head kernel under an abstract CPU API, not CANN/NPU.
#include "../op_kernel/d_flash_draft_lm_head_top1.cpp"
#include "../op_kernel/d_flash_draft_lm_head_top1_audit.cpp"
#include <thread>
#include <cmath>
#include <iostream>

uint16_t Bits(half x) { uint16_t bits; std::memcpy(&bits,&x,2); return bits; }
void Check(uint32_t m,uint32_t p,int pattern,bool audit,int fail=-1) {
    CpuRun context; context.m=m; context.p=p; context.pattern=pattern; context.failCore=fail; run=&context;
    std::vector<half> h(m*2560); half virtualWeight=0;
    for(uint32_t row=0;row<m;++row) {
        if(pattern==7) h[row*2560+row]=1;
        else if(pattern==2) h[row*2560+2559]=1;
        else { h[row*2560]=pattern==5?2:1; if(pattern==3) h[row*2560+1]=half(1.0f/2048); }
    }
    const auto original=h; context.hidden=h.data(); context.weight=&virtualWeight;
    std::vector<int64_t> output(m+2,-77);
    std::vector<uint16_t> diagnostic(audit?m*248320+2:2,0x3555);
    // Poisoned workspace reused across launches must not be an implicit zero contract.
    std::vector<uint64_t> workspace((head_contract::WorkspaceBytes(p)+128)/8,0xdeadbeefdeadbeefULL);
    CpuTiling t{m,p,65536,{}};
    std::vector<std::thread> threads;
    for(uint32_t i=0;i<p;++i) threads.emplace_back([&,i]{
        core=p-1-i; // intentionally reverse launch order
        if(audit) d_flash_draft_lm_head_top1_audit(reinterpret_cast<GM_ADDR>(h.data()),reinterpret_cast<GM_ADDR>(&virtualWeight),
            reinterpret_cast<GM_ADDR>(output.data()+1),reinterpret_cast<GM_ADDR>(diagnostic.data()+1),
            reinterpret_cast<GM_ADDR>(workspace.data()+8),reinterpret_cast<GM_ADDR>(&t));
        else d_flash_draft_lm_head_top1(reinterpret_cast<GM_ADDR>(h.data()),reinterpret_cast<GM_ADDR>(&virtualWeight),
            reinterpret_cast<GM_ADDR>(output.data()+1),reinterpret_cast<GM_ADDR>(workspace.data()+8),reinterpret_cast<GM_ADDR>(&t));
        assert(ubBytes==head_contract::UserUbBytes(p)+t.matmulUbBytes);
    });
    for(auto& thread:threads) thread.join();
    assert(context.barriers==p && context.stores==m);
    if(fail<0) { assert(context.calls==3880); for(auto n:context.visits) assert(n==1); }
    assert(h==original && output.front()==-77 && output.back()==-77);
    for(uint32_t r=0;r<m;++r) {
        int64_t expected=0;
        if(pattern==1) expected=64; if(pattern==2 || pattern==4) expected=248319;
        if(pattern==3 || pattern==5 || pattern==6) expected=63; if(pattern==7) expected=r+1;
        if(fail>=0) expected=-1;
        assert(output[r+1]==expected);
        if(audit && fail<0) for(uint32_t n=0;n<248320;++n) {
            half value=0;
            if(pattern==1 && (n==64 || n==53760)) value=3;
            if(pattern==2 && n==248319) value=2;
            if(pattern==3 && (n==63 || n==64)) value=1; // larger FP32 value must round DOWN to a tie
            if(pattern==4) value=half(n==248319?-.5f:-1);
            if(pattern==5 && (n==63 || n==64)) value=half(INFINITY);
            if(pattern==6 && n==63) { assert((diagnostic[1+r*248320+n]&0x7fff)>0x7c00); continue; }
            if(pattern==6 && n==64) value=half(INFINITY);
            if(pattern==7 && n==r+1) value=2;
            assert(diagnostic[1+r*248320+n]==Bits(value));
        }
    }
    assert(diagnostic.front()==0x3555 && diagnostic.back()==0x3555);
    for(size_t i=0;i<8;++i) { assert(workspace[i]==0xdeadbeefdeadbeefULL); assert(workspace[workspace.size()-1-i]==0xdeadbeefdeadbeefULL); }
}
int main() {
    for(uint32_t m=1;m<=15;++m) Check(m,m%3==0?7:3,7,true);
    for(uint32_t p:{1U,3U,7U}) for(int pattern=0;pattern<8;++pattern) {
        Check(15,p,pattern,true); Check(1,p,pattern,false);
    }
    Check(15,7,7,false,2);
    // Ordering tests independent of Matmul: signed zero, NaNs, negative values.
    assert(head_contract::Better(0x8000,1,0,2));
    assert(!head_contract::Better(0,2,0x8000,1));
    assert(head_contract::Better(0xbc00,2,0xc000,1));
    assert(head_contract::Better(0x7e00,10,0x7c00,0));
    assert(!head_contract::Better(0x7e01,11,0x7e00,10));
    for(uint32_t m:{0U,16U,80U}) assert(!head_contract::Rows(m));
    std::cout<<"CPU model: 63 full-vocabulary normal/audit runs + failed-core barrier test; NPU NOT_RUN\n";
}
