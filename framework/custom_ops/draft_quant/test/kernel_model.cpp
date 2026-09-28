// Execute the actual kernel body against a CPU API model, compare to an
// independent dense oracle. This never loads CANN or claims NPU verification.
#include <algorithm>
#include <iostream>
#include "../op_kernel/d_flash_group_quant_linear.cpp"
#include "../op_host/d_flash_group_quant_linear_contract.h"

static uint16_t Bits(half value)
{
    uint16_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static void Check(uint32_t k, uint32_t n, int pattern)
{
    constexpr uint32_t m = 16;
    const uint32_t tileK = static_cast<uint32_t>(draft_quant_contract::KTile(k));
    std::vector<half> x(m * k), scales((k / 128) * n);
    std::vector<int8_t> q(n * k), nz;
    for (uint32_t c = 0; c < n; ++c) {
        for (uint32_t i = 0; i < k; ++i) q[c * k + i] = (c * 31 + i * 7 + (i / 128) * 11) % 256 - 128;
    }
    for (uint32_t g = 0; g < k / 128; ++g) {
        for (uint32_t c = 0; c < n; ++c) scales[g * n + c] = static_cast<half>((1 + (g * 5 + c) % 13) / 64.0f);
    }
    for (uint32_t row = 0; row < m; ++row) {
        if (pattern == 0) {
            const uint32_t group = row % (k / 128);
            const uint32_t pos = group * 128 + (row / (k / 128) % 2 ? 127 : 0);
            x[row * k + pos] = static_cast<half>((row % 2 ? -1.0f : 1.0f) / 128);
        } else if (pattern == 1) {
            for (uint32_t i = 0; i < k; ++i) x[row * k + i] = static_cast<half>((i + row) % 3 ? 1.0f / 128 : -1.0f / 128);
        } else {
            const float sign = row % 2 ? -1.0f : 1.0f;
            x[row * k] = static_cast<half>(sign * 1024);
            x[row * k + 1] = static_cast<half>(sign / 64);
            x[row * k + 128] = static_cast<half>(sign * 1024);
        }
    }
    if (pattern == 2) {
        std::fill(q.begin(), q.end(), 0);
        std::fill(scales.begin(), scales.end(), static_cast<half>(1));
        for (uint32_t c = 0; c < n; ++c) {
            q[c * k] = 127;
            q[c * k + 1] = c % 7 + 1;
            q[c * k + 128] = -127;
        }
    }
    // Offline packing in physical traversal order, independent of tile offsets.
    for (uint32_t k1 = 0; k1 < k / 32; ++k1)
        for (uint32_t n1 = 0; n1 < n / 16; ++n1)
            for (uint32_t n0 = 0; n0 < 16; ++n0)
                for (uint32_t k0 = 0; k0 < 32; ++k0)
                    nz.push_back(q[(n1 * 16 + n0) * k + k1 * 32 + k0]);
    const auto savedX = x, savedScale = scales;
    const auto savedNz = nz;
    std::vector<half> output(m * n + 2, static_cast<half>(-17));
    CpuTiling tiling{k, n, tileK, 65536, {n, k, m, 64, tileK}};
    cpuMetrics = {};
    d_flash_group_quant_linear(x.data(), nz.data(), scales.data(), output.data() + 1, nullptr, &tiling);
    assert(Bits(output.front()) == Bits(static_cast<half>(-17)) && Bits(output.back()) == Bits(static_cast<half>(-17)));
    assert(x == savedX && scales == savedScale && nz == savedNz);
    assert(cpuMetrics.ubBytes == draft_quant_contract::UserUbBytes(k) + tiling.matmulUbBytes);
    assert(cpuMetrics.iterations == (n / 64) * (k / tileK));
    assert(cpuMetrics.partials == (n / 64) * (k / tileK - 1));
    assert(cpuMetrics.outputs == n / 64 && cpuMetrics.stores == m * n && cpuMetrics.ends == 1);
    for (uint32_t row = 0; row < m; ++row) {
        for (uint32_t c = 0; c < n; ++c) {
            double sum = 0;
            for (uint32_t i = 0; i < k; ++i) {
                const half w = static_cast<half>(static_cast<float>(q[c * k + i]) *
                                                static_cast<float>(scales[(i / 128) * n + c]));
                sum += static_cast<double>(x[row * k + i]) * static_cast<double>(w);
            }
            assert(Bits(output[1 + row * n + c]) == Bits(static_cast<half>(sum)));
        }
    }
}

int main()
{
    for (uint32_t k : {256U, 512U, 1024U})
        for (uint32_t n : {64U, 128U, 256U})
            for (int pattern = 0; pattern < 3; ++pattern) Check(k, n, pattern);
    std::cout << "CPU kernel model: 27 indexing/accumulation cases passed; NPU NOT_RUN\n";
}
