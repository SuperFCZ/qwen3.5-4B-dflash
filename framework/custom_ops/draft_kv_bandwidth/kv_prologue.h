#pragma once
#include "kernel_operator.h"
#include "panel_vector.h"
#include "kv_spec.h"

namespace kv_panel {
// Pack [M,K] into A1 [K/16,M,16] using the two as-yet unused W16 banks.
// uint16 bitwise identity also preserves signed zeros and NaN payloads.
template<uint32_t M>
__aicore__ inline bool PackActivation(AscendC::TPipe& pipe, AscendC::LocalTensor<half> a,
    AscendC::GlobalTensor<half> x, AscendC::LocalTensor<half> weights)
{
    using namespace AscendC;
    using gate_panel::Pair;
    constexpr uint32_t slab = 256, blocks = slab / 16;
    static_assert(K % slab == 0 && blocks == 16 && M * slab <= Elements,
                  "each activation scratch slice must fit one W16 bank");
    auto nd = weights;
    auto nz = weights[Elements];
    auto src = nd.template ReinterpretCast<uint16_t>();
    auto dst = nz.template ReinterpretCast<uint16_t>();
    const DataCopyParams rows{M, blocks, (K - slab) / 16, 0};
    const BinaryRepeatParams perm{M, 1, 1, 1, blocks, blocks};
    const auto ready = pipe.FetchEventID(HardEvent::MTE2_V);
    const auto inputFree = pipe.FetchEventID(HardEvent::V_MTE2);
    const auto outputReady = pipe.FetchEventID(HardEvent::V_MTE3);
    const auto outputFree = pipe.FetchEventID(HardEvent::MTE3_V);
    if (static_cast<uint32_t>(ready) >= QUE_MAX_EVENT || static_cast<uint32_t>(inputFree) >= QUE_MAX_EVENT ||
        static_cast<uint32_t>(outputReady) >= QUE_MAX_EVENT || static_cast<uint32_t>(outputFree) >= QUE_MAX_EVENT)
        return false;
    DataCopy(nd, x, rows);
    for (uint32_t k0 = 0; k0 < K; k0 += slab) {
        Pair<HardEvent::MTE2_V>(ready);
        if (k0) Pair<HardEvent::MTE3_V>(outputFree);
        // Eight adjacent K16 blocks in the source; consecutive repeats visit adjacent rows.
        // The corresponding destination blocks are M blocks apart, consecutive rows one block apart.
        for (uint32_t col = 0; col < slab; col += 128) {
            auto bits = src[col];
            Or(dst[col * M], bits, bits, uint64_t(128), uint8_t(M), perm);
        }
        if (k0 + slab < K) {
            Pair<HardEvent::V_MTE2>(inputFree);
            DataCopy(nd, x[k0 + slab], rows);
        }
        Pair<HardEvent::V_MTE3>(outputReady);
        DataCopy(a[k0 * M], nz, M * slab);
    }
    PipeBarrier<PIPE_ALL>(); // A1 is ready and both W16 banks are free for the first weight stripe.
    return true;
}
} // namespace kv_panel
