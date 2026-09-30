// Prove the CPU dependency checker actually rejects the hazards A3.2 avoids.
#include "kernel_operator.h"
#include <string>

using namespace AscendC;
template <typename F> void Reject(F function, const char *message) {
    bool rejected = false;
    try { function(); }
    catch (const std::logic_error &error) { rejected = std::string(error.what()).find(message) != std::string::npos; }
    assert(rejected);
}

int main()
{
    TPipe pipe;
    int8_t input[32] = {}, raw[32] = {};
    half weight[32] = {};
    Tensor<int8_t> gm{input, 32}, q{raw, 32};
    Tensor<half> w{weight, 32};
    input[0] = -127;
    DataCopy(q, gm, 32);
    assert(raw[0] == 0);  // the transfer is deferred, not an eager memcpy
    Reject([&] { (void)q.GetValue(0); }, "DMA completion");
    const auto scalar = pipe.AllocEventID<HardEvent::MTE2_S>();
    SetFlag<HardEvent::MTE2_S>(scalar);
    Reject([&] { pipe.ReleaseEventID<HardEvent::MTE2_S>(scalar); }, "live/unreserved");
    const auto libraryId = pipe.FetchEventID(HardEvent::MTE2_S);
    assert(libraryId != scalar);
    SetFlag<HardEvent::MTE2_S>(libraryId);
    Reject([&] { SetFlag<HardEvent::MTE2_S>(pipe.FetchEventID(HardEvent::MTE2_S)); }, "collision");
    WaitFlag<HardEvent::MTE2_S>(libraryId);
    WaitFlag<HardEvent::MTE2_S>(scalar);
    pipe.ReleaseEventID<HardEvent::MTE2_S>(scalar);
    assert(q.GetValue(0) == -127);
    Reject([&] { Cast(w, q, RoundMode::CAST_NONE, 32); }, "consumer dependency");
    const auto vector = pipe.FetchEventID(HardEvent::MTE2_V);
    SetFlag<HardEvent::MTE2_V>(vector);
    WaitFlag<HardEvent::MTE2_V>(vector);
    Cast(w, q, RoundMode::CAST_NONE, 32);
    Reject([&] { DataCopy(q, gm, 32); }, "raw bank reused");
    PipeBarrier<PIPE_V>();
    Reject([&] { BeforeMatmul(Bytes(weight, sizeof(weight))); }, "vector writes");
    const auto ready = pipe.FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(ready);
    WaitFlag<HardEvent::V_S>(ready);
    BeforeMatmul(Bytes(weight, sizeof(weight)));
    Reject([&] { Muls(w, w, static_cast<half>(1), 32); }, "Cube consumer");
    PipeBarrier<PIPE_ALL>();
    Muls(w, w, static_cast<half>(1), 32);
    PipeBarrier<PIPE_ALL>();
    return 0;
}
