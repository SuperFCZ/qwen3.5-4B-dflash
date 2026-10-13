// Runs the production head kernel under an abstract CPU API, not CANN/NPU.
#include "../op_kernel/d_flash_draft_lm_head_top1.cpp"
#include "../op_kernel/d_flash_draft_lm_head_top1_audit.cpp"
#include <thread>
#include <cmath>
#include <iostream>

uint16_t Bits(half x) { uint16_t bits; std::memcpy(&bits,&x,2); return bits; }
void Check(uint32_t m,uint32_t p,int pattern,bool audit,int fail=-1,bool probe=false) {
    CpuRun context; context.m=m; context.p=p; context.pattern=pattern; context.failCore=fail; run=&context;
    std::vector<half> h(m*2560); half virtualWeight=0;
    for(uint32_t row=0;row<m;++row) {
        if(pattern==7) h[row*2560+row]=1;
        else if(pattern==2) h[row*2560+2559]=1;
        else { h[row*2560]=(pattern==5 || pattern==8)?2:1; if(pattern==3) h[row*2560+1]=half(1.0f/2048); }
    }
    const auto original=h; context.hidden=h.data(); context.weight=&virtualWeight;
    std::vector<int64_t> output(m+2,-77);
    std::vector<uint16_t> diagnostic(audit?m*248320+2:2,0x3555);
    // Poisoned workspace reused across launches must not be an implicit zero contract.
    std::vector<uint64_t> workspace((head_contract::WorkspaceBytes(p)+128)/8,0xdeadbeefdeadbeefULL);
    std::vector<float> raw(2*16*64+2,-12345.f);
    CpuTiling t{m,p,65536,{}};
    std::vector<std::thread> threads;
    for(uint32_t i=0;i<p;++i) threads.emplace_back([&,i]{
        core=p-1-i; // intentionally reverse launch order
        if(probe) RunHead<true,true>(reinterpret_cast<GM_ADDR>(h.data()),reinterpret_cast<GM_ADDR>(&virtualWeight),
            reinterpret_cast<GM_ADDR>(output.data()+1),reinterpret_cast<GM_ADDR>(diagnostic.data()+1),
            reinterpret_cast<GM_ADDR>(workspace.data()+8),t,reinterpret_cast<GM_ADDR>(raw.data()+1));
        else if(audit) d_flash_draft_lm_head_top1_audit(reinterpret_cast<GM_ADDR>(h.data()),reinterpret_cast<GM_ADDR>(&virtualWeight),
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
        if(pattern==3 || pattern==5) expected=63; if(pattern==6) expected=63; if(pattern==7) expected=r+1;
        if(fail>=0) expected=-1;
        assert(output[r+1]==expected);
        if(audit && fail<0) for(uint32_t n=0;n<248320;++n) {
            half value=0;
            if(pattern==1 && (n==64 || n==53760)) value=3;
            if(pattern==2 && n==248319) value=2;
            if(pattern==3 && (n==63 || n==64)) value=1; // larger FP32 value must round DOWN to a tie
            if(pattern==4) value=half(n==248319?-.5f:-1);
            if(pattern==5 && (n==63 || n==64)) value=DFLASH_HEAD_SATURATION?half(65504):half(INFINITY);
            if(pattern==6 && n==64) value=DFLASH_HEAD_SATURATION?half(65504):half(INFINITY);
            if(pattern==8 && (n==63 || n==64)) value=DFLASH_HEAD_SATURATION?half(-65504):half(-INFINITY);
            if(pattern==7 && n==r+1) value=2;
            if(pattern==6 && n==63) { assert((diagnostic[1+r*248320+n]&0x7fff)>0x7c00); continue; }
            assert(diagnostic[1+r*248320+n]==Bits(value));
        }
    }
    if(probe) {
        assert(raw.front()==-12345.f && raw.back()==-12345.f);
        for(uint32_t r=0;r<m;++r) {
            assert(raw[1+r*64+63]==131008.f);
            assert(raw[1+16*64+r*64]==131008.f);
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
    Check(15,7,5,true,-1,true);
    Check(15,7,8,true); Check(1,3,8,false);
    // Ordering tests independent of Matmul: signed zero, NaNs, negative values.
    assert(head_contract::Better(0x8000,1,0,2));
    assert(!head_contract::Better(0,2,0x8000,1));
    assert(head_contract::Better(0xbc00,2,0xc000,1));
    assert(head_contract::Better(0x7e00,10,0x7c00,0));
    assert(!head_contract::Better(0x7e01,11,0x7e00,10));
    for(uint32_t m:{0U,16U,80U}) assert(!head_contract::Rows(m));
    // Independent expected bit patterns at and around the FP16 limit, both
    // signs; include RNE ties, subnormals and zero sign. These are CPU oracles.
    const std::pair<float,uint16_t> boundary[] = {
        {65472.f,0x7bfe},{65488.f,0x7bfe},{65488.00390625f,0x7bff},
        {65503.f,0x7bff},{65504.f,0x7bff},{65519.f,0x7bff},{65520.f,0x7bff},
        {65536.f,0x7bff},{131008.f,0x7bff},
        {1.f+1.f/2048,0x3c00},{1.f+3.f/2048,0x3c02},
        {1.f/16777216,0x0001},{1.f/33554432,0x0000},{0.f,0x0000}};
    for(auto test:boundary) for(bool negative:{false,true}) {
        float input=negative?-test.first:test.first; half output=0;
        if constexpr(DFLASH_HEAD_SATURATION==1) HeadFiniteClamp({&input,1},1);
        if constexpr(DFLASH_HEAD_SATURATION==2) {
            alignas(32) float values[64]{}; values[0]=input;
            alignas(32) uint8_t scratch[288]{};
            HeadCompareSelectClamp({values,64},{scratch,288},64); input=values[0];
        }
        AscendC::Cast({&output,1},{&input,1},AscendC::RoundMode::CAST_NONE,1);
        auto expected=test.second;
        if(!DFLASH_HEAD_SATURATION && test.first>=65520.f) expected=0x7c00;
        assert(Bits(output)==static_cast<uint16_t>(expected|(negative?0x8000:0)));
    }
    // A different lane pattern in every block detects accidentally repeating
    // mode-0's first 64 mask bits across the full 1024-element tile.
    alignas(32) float payload[1024]; alignas(32) uint8_t scratch[2048];
    std::memset(scratch,0xa5,sizeof(scratch));
    const uint32_t patterns[]={0x7fc01234,0xffc12345,0x7f800001,0xff800001,
        0x00000000,0x80000000,0x00000001,0x80000001,0x47ffe000,0xc7ffe000,
        0x7f800000,0xff800000,0x477fe000,0xc77fe000,0x3f801000};
    uint32_t wanted[1024];
    for(uint32_t i=0;i<1024;++i) {
        auto raw=patterns[(i+i/64)%15]; std::memcpy(payload+i,&raw,4);
        wanted[i]=payload[i]>65504.f?0x477fe000:(payload[i]<-65504.f?0xc77fe000:raw);
    }
    HeadCompareSelectClamp({payload,1024},{scratch,2048},1024);
    for(uint32_t i=0;i<1024;++i) { uint32_t raw; std::memcpy(&raw,payload+i,4); assert(raw==wanted[i]); }
    for(uint32_t i=288;i<2048;++i) assert(scratch[i]==0xa5);
    for(uint32_t p:{1U,3U,7U,64U}) {
        assert(head_contract::IdOffset(p)==32*p && head_contract::SyncOffset(p)==160*p);
        assert(head_contract::WorkspaceBytes(p)==192*p); // GM workspace unchanged
        assert(head_contract::UserUbBytes(p)==16*2560*2+16*64*6+160+32*p);
    }
    std::cout<<"CPU model: 66 full-vocabulary normal/audit runs + failed-core barrier + saturation boundary tests; NPU NOT_RUN\n";
}
