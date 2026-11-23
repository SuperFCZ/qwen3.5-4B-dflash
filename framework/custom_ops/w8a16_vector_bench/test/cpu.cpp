// Functional test of the actual kernel with CANN's CPU model. No timings.
#include "tikicpulib.h"
#include "workload.h"
#include "../vector_kernel.cpp"
#include <iostream>

uint16_t bits(float x) {
    half h(x);
    static_assert(sizeof(h) == sizeof(uint16_t));
    uint16_t word;
    std::memcpy(&word, &h, sizeof(word));
    return word;
}

int main() {
    using namespace vb;
    using namespace AscendC;
    constexpr uint32_t blocks = 2;
    vb::Shape s{3*NTile, 3*KTile}; // one block owns two N tiles
    size_t outBytes = std::max(size_t(blocks)*Elements*2, s.tapBytes());
    auto* q = static_cast<int8_t*>(GmAlloc(s.weights()));
    auto* scales = static_cast<uint16_t*>(GmAlloc(s.scaleWords()*2));
    auto* sink = static_cast<uint8_t*>(GmAlloc(outBytes));
    if (!q || !scales || !sink) return 1;
    for (uint32_t g = 0; g < s.k/Group; ++g)
        for (uint32_t n = 0; n < s.n; ++n) scales[g*s.n+n] = bits(scaleAt(n,g));
    SetKernelMode(KernelMode::AIC_MODE);
    for (uint32_t m = Fp16Fma; m <= StreamFlatB16; ++m) {
        const auto mode = static_cast<vb::Mode>(m);
        std::memcpy(q, packCodes(s,flat(mode)).data(), s.weights());
        for (uint32_t iters : {2u,3u}) {
            std::memset(sink,0xa5,outBytes);
            ICPU_RUN_KF(dflash_vector_probe, blocks, reinterpret_cast<uint8_t*>(q),
                        reinterpret_cast<uint8_t*>(scales), sink, m, s.n, s.kTiles(), iters);
            try { checkOutput(s,mode,blocks,iters,std::vector<uint8_t>(sink,sink+outBytes),bits); }
            catch (const std::exception& e) {
                std::cerr << "mode " << m << " iterations " << iters << ": " << e.what() << "\n"; return 2;
            }
            if (streamMode(mode)) break;
        }
    }
    GmFree(q); GmFree(scales); GmFree(sink);
    const vb::Shape gate{19456,2560}, down{2560,9728};
    if (gate.inputBytes()!=50585600 || down.inputBytes()!=25292800 ||
        work(Fp16Fma,s,8,100)!=13107200 || fmaOrMulTF(Fp16Fma,1e8,100.0)!=2.0 ||
        vectorBudgetMs(gate,8,100.0)!=0.49807360 ||
        !(vectorBudgetMs(down,7,100.0)>down.weights()/1.0e8)) return 3;
    std::cout << "All 12 device paths and rate units checked on the CPU model.\n";
}
