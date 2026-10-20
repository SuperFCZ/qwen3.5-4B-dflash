// CPU layout/synchronization test of the real CopyRawTile; NOT NPU evidence.
#include <iostream>
#include <algorithm>
#include "../op_kernel/d_flash_group_quant_linear.cpp"
int main() {
    constexpr uint32_t K=2560,N=19456;
    std::vector<int8_t> q(K*N),original;
    std::vector<half> s((K/128)*N),saved;
    uint32_t state=0x391817;
    auto random=[&]() { state^=state<<13;state^=state>>17;state^=state<<5;return state; };
    for (auto &v:q) v=static_cast<int8_t>(random());
    for (auto &v:s) { uint16_t bits=static_cast<uint16_t>(0x3000+(random()%1024));std::memcpy(&v,&bits,2); }
    original=q;saved=s;
    AscendC::GlobalTensor<int8_t> qGm{q.data(),K*N};
    AscendC::GlobalTensor<half> sGm{s.data(),static_cast<uint32_t>(s.size())};
    std::vector<int8_t> localQ(64*256+64,42);
    std::vector<half> localS(128+32,static_cast<half>(-17));
    unsigned tiles=0;
    for (uint32_t nb=0;nb<N;nb+=64) for (uint32_t kb=0;kb<K;kb+=256) {
        for (bool candidate:{false,true}) {
            AscendC::TPipe pipe;cpuMetrics={};
            CopyRawTile({localQ.data()+32,64*256},{localS.data()+16,128},qGm,sGm,N,256,nb,kb,candidate);
            assert(cpuMetrics.copies==(candidate?2:10));
            AscendC::PipeBarrier<PIPE_ALL>();
            for (uint32_t k1=0;k1<8;++k1) for (uint32_t n=0;n<64;++n) for (uint32_t k0=0;k0<32;++k0) {
                // Independent logical-index NZ formula; compare every moved byte.
                uint32_t k=kb+k1*32+k0,c=nb+n;
                uint32_t offset=((k/32*(N/16)+c/16)*16+c%16)*32+k%32;
                assert(localQ[32+(k1*64+n)*32+k0]==q[offset]);
            }
            for (uint32_t g=0;g<2;++g) for(uint32_t n=0;n<64;++n)
                assert(std::memcmp(&localS[16+g*64+n],&s[(kb/128+g)*N+nb+n],2)==0);
            for(uint32_t i=0;i<32;++i) assert(localQ[i]==42 && localQ[32+64*256+i]==42);
            for(uint32_t i=0;i<16;++i) assert(localS[i]==static_cast<half>(-17) && localS[16+128+i]==static_cast<half>(-17));
        }
        ++tiles;
    }
    assert(tiles==3040 && q==original && s==saved);
    std::cout<<"3040 Gate tiles: 10->2 raw copy instructions, exact bytes/guards/readonly; NPU NOT_RUN\n";
}
