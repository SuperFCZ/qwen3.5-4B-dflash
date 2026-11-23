#pragma once
#include "kernel_operator.h"

namespace gate_panel {
using namespace AscendC;
// Each 16-lane vector block contains one K row and 16 consecutive output channels.
// All chunks have an even number of 128-lane repeats. Split at <= 255 repeats.
template<uint32_t Count>
__aicore__ inline void CastCodes(LocalTensor<half> w, LocalTensor<int8_t> q)
{
    constexpr uint32_t calls = (Count + 128 * 255 - 1) / (128 * 255);
    static_assert(Count % (128 * calls) == 0, "complete Cast repeats required");
    constexpr uint32_t count = Count / calls;
    constexpr uint32_t repeats = count / 128;
    static_assert(repeats > 0 && repeats <= 255, "Cast repeat overflow");
    const UnaryRepeatParams dense{1, 1, 8, 4};
    for (uint32_t i = 0; i < calls; ++i) {
        Cast(w[i * count], q[i * count], RoundMode::CAST_NONE, uint64_t(128), uint8_t(repeats), dense);
    }
}

template<HardEvent Direction>
__aicore__ inline void Pair(TEventID id)
{
    SetFlag<Direction>(id);
    WaitFlag<Direction>(id);
}

template<HardEvent Direction>
__aicore__ inline void Handoff(TPipe& pipe)
{
    Pair<Direction>(pipe.FetchEventID(Direction));
}

}  // namespace gate_panel
