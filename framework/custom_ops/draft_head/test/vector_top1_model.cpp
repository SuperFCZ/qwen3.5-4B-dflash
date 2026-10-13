// Semantic reference only: this does NOT emulate vcmax/vcmp behavior on NPU.
#include "cpu_include/kernel_operator.h"
#include "../op_kernel/head_vector_top1.h"
#include "../op_host/head_contract.h"
#include <random>
#include <iostream>
void Check(const std::array<uint16_t,64>& bits) {
    alignas(32) half source[64]; alignas(32) float widened[64]; alignas(32) uint8_t scratch[736];
    std::memcpy(source,bits.data(),128); std::memset(scratch,0xa5,sizeof(scratch));
    AscendC::TPipe pipe;
    AscendC::Cast({widened,64},{source,64},AscendC::RoundMode::CAST_NONE,64);
    HeadRawNanMask({source,64},{scratch,704});
    uint64_t mask=0,expectedMask=0; std::memcpy(&mask,scratch,8);
    for(uint32_t i=0;i<64;++i) if(head_contract::IsNan(bits[i])) expectedMask|=1ULL<<i;
    assert(mask==expectedMask);
    auto actual=HeadVectorWinner({source,64},{widened,64},{scratch,704},pipe);
    uint16_t best=0; int64_t expected=-1;
    for(uint32_t i=0;i<64;++i) if(head_contract::Better(bits[i],i,best,expected)) { expected=i; best=bits[i]; }
    assert(actual==uint32_t(expected)); assert(!std::memcmp(source,bits.data(),128));
    for(int i=704;i<736;++i) assert(scratch[i]==0xa5);
}
int main() {
    assert(HeadFirstBit(0)==64);
    for(int i=0;i<64;++i) assert(HeadFirstBit(1ULL<<i)==uint32_t(i));
    AscendC::cpuOrderedNanNE=true; // simulate the failure hypothesis; new classifier must not depend on NE
    std::array<uint16_t,64> x{};
    for(uint16_t value:{uint16_t(0),uint16_t(0x8000),uint16_t(1),uint16_t(0x8001),uint16_t(0x7c00),uint16_t(0xfc00),uint16_t(0x7e01),uint16_t(0xfc01)}) { x.fill(value); Check(x); }
    for(uint32_t v=0;v<65536;++v) {
        x.fill(0); x[v%64]=uint16_t(v); Check(x);
        x.fill(uint16_t(v)); x[(v+17)%64]=0x8000; Check(x);
    }
    std::mt19937 rng(391);
    for(int i=0;i<4096;++i) { for(auto& v:x) v=uint16_t(rng()); Check(x); }
    {
        alignas(32) half raw[64]; alignas(32) float widened[64];
        alignas(32) uint8_t scratch[704]; alignas(32) uint32_t trace[240]{};
        const uint16_t nan=0x7e01; for(auto& v:raw) std::memcpy(&v,&nan,2);
        AscendC::TPipe pipe; AscendC::Cast({widened,64},{raw,64},AscendC::RoundMode::CAST_NONE,64);
        const auto index=HeadVectorWinner<true>({raw,64},{widened,64},{scratch,704},pipe,{trace,240},{widened,64});
        assert(index==0 && trace[106]==64 && trace[110]==64 && trace[111]==64);
        assert(trace[96]==0 && trace[97]==0 && trace[98]==0 && trace[99]==0);
        assert(trace[100]==0xffffffffU && trace[101]==0xffffffffU && trace[107]==0 && trace[108]==0);
        assert(trace[109]==0x48543102);
    }
    std::cout<<"Vector Top1 CPU oracle: all FP16 patterns, NaN/zero/ties; NPU NOT_RUN\n";
}
