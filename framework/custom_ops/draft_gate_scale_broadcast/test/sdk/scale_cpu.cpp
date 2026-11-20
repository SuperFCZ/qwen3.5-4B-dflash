// This executable uses the installed CANN CPU debugging library only.
#include "tikicpulib.h"
#include "kernel_operator.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include "../../kernel/scale_expand.inc"

namespace {
constexpr unsigned channels = 64;
constexpr unsigned guard = 16;       // one 32 B block
constexpr unsigned inputWords = guard + channels + guard;
constexpr unsigned expandedWords = guard + 1024 + 128 + guard;
constexpr unsigned outputWords = expandedWords * 2 + inputWords;
constexpr uint16_t poison = 0xa55a;
}

extern "C" __global__ __aicore__ void check_scale(GM_ADDR input, GM_ADDR output, uint32_t cases)
{
    using namespace AscendC;
    TPipe pipe;
    TBuf<TPosition::VECIN> sBuf;
    TBuf<TPosition::VECCALC> baseBuf, newBuf;
    pipe.InitBuffer(sBuf, inputWords * sizeof(uint16_t));
    pipe.InitBuffer(baseBuf, expandedWords * sizeof(uint16_t));
    pipe.InitBuffer(newBuf, expandedWords * sizeof(uint16_t));
    auto src = sBuf.Get<uint16_t>();
    auto base = baseBuf.Get<uint16_t>();
    auto dest = newBuf.Get<uint16_t>();
    GlobalTensor<uint16_t> in, out;
    in.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(input), cases * inputWords);
    out.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(output), cases * outputWords);
    for (uint32_t i = 0; i < cases; ++i) {
        DataCopy(src, in[i * inputWords], inputWords);
        Duplicate(base, poison, expandedWords);
        Duplicate(dest, poison, expandedWords);
        PipeBarrier<PIPE_ALL>();
        auto scales = src[guard].ReinterpretCast<half>();
        Brcb(base[guard].ReinterpretCast<half>(), scales, 8, {1, 8});
        PipeBarrier<PIPE_V>();
        ExpandGateScale(dest[guard].ReinterpretCast<half>(), scales);
        PipeBarrier<PIPE_ALL>();
        DataCopy(out[i * outputWords], base, expandedWords);
        DataCopy(out[i * outputWords + expandedWords], dest, expandedWords);
        DataCopy(out[i * outputWords + 2 * expandedWords], src, inputWords);
        PipeBarrier<PIPE_ALL>();
    }
}

int main(int argc, char** argv)
{
    if (argc != 2) return 2;
    std::ifstream f(argv[1], std::ios::binary | std::ios::ate);
    if (!f) return 2;
    const auto bytes = f.tellg();
    if (bytes <= 0 || bytes % (channels * 2) != 0 || bytes > 778240) return 2;
    std::vector<uint16_t> scales(static_cast<size_t>(bytes) / 2);
    f.seekg(0);
    if (!f.read(reinterpret_cast<char*>(scales.data()), bytes)) return 2;
    const uint32_t cases = scales.size() / channels;
    auto* in = static_cast<uint16_t*>(AscendC::GmAlloc(cases * inputWords * 2));
    auto* out = static_cast<uint16_t*>(AscendC::GmAlloc(cases * outputWords * 2));
    if (!in || !out) return 2;
    std::fill_n(in, cases * inputWords, poison);
    for (unsigned i = 0; i < cases; ++i)
        std::copy_n(scales.data() + i * channels, channels, in + i * inputWords + guard);
    std::fill_n(out, cases * outputWords, uint16_t(0x1234));
    AscendC::SetKernelMode(KernelMode::AIC_MODE);
    ICPU_RUN_KF(check_scale, 1, reinterpret_cast<uint8_t*>(in), reinterpret_cast<uint8_t*>(out), cases);
    size_t bad = 0;
    for (unsigned i = 0; i < cases; ++i) {
        const uint16_t* row = out + i * outputWords;
        for (unsigned j = 0; j < expandedWords; ++j) {
            const bool active = j >= guard && j < guard + 1024;
            const uint16_t want = active ? scales[i * channels + (j - guard) / 16] : poison;
            bad += (row[j] != want) + (row[expandedWords + j] != want);
        }
        for (unsigned j = 0; j < inputWords; ++j) {
            const uint16_t want = j >= guard && j < guard + channels ? scales[i * channels + j - guard] : poison;
            bad += (row[2 * expandedWords + j] != want) + (in[i * inputWords + j] != want);
        }
    }
    AscendC::GmFree(in);
    AscendC::GmFree(out);
    std::cout << "CPU_SCALE_WORDS " << scales.size() << " EXPANDED_PER_VARIANT " << cases * 1024
              << " PAYLOAD_GUARD_READONLY_ERRORS " << bad << "\n";
    return bad != 0;
}
