#pragma once
#include "bench_shape.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace vb {
struct Shape {
    uint32_t n, k;
    uint32_t nTiles() const { return n / NTile; }
    uint32_t kTiles() const { return k / KTile; }
    size_t weights() const { return size_t(n) * k; }
    size_t scaleWords() const { return weights() / Group; }
    size_t inputBytes() const { return weights() + 2 * scaleWords(); }
    size_t tapBytes() const { return size_t(nTiles()) * kTiles() * TapBytes; }
};

inline bool flat(Mode mode) {
    return mode == CastFlat || mode == FlatB16 || mode == CopyFlat || mode == StreamFlatB16;
}
inline bool copyOnly(Mode mode) { return mode == CopyNz || mode == CopyFlat; }
inline bool streamMode(Mode mode) { return mode >= CopyNz; }
inline int8_t codeAt(uint32_t n, uint32_t k) {
    return static_cast<int8_t>(int((13*n + 7*k + 11*(k/Plane) + 3*(n/NTile)) % 251) - 125);
}
inline float scaleAt(uint32_t n, uint32_t g) {
    return float(1 + (5*n + 3*g + n/NTile) % 8) * (1.0f/16.0f);
}

inline std::vector<int8_t> packCodes(Shape s, bool tileMajor) {
    std::vector<int8_t> codes(s.weights());
    for (uint32_t nt = 0; nt < s.nTiles(); ++nt) {
        for (uint32_t kt = 0; kt < s.kTiles(); ++kt) {
            for (uint32_t row = 0; row < NTile; ++row) {
                for (uint32_t j = 0; j < KTile; ++j) {
                    uint32_t n = nt * NTile + row, k = kt * KTile + j;
                    size_t off = tileMajor ? ((size_t(nt) * s.kTiles() + kt) * NTile + row) * KTile + j
                                           : (size_t(k / Plane) * s.n + n) * Plane + k % Plane;
                    codes[off] = codeAt(n, k);
                }
            }
        }
    }
    return codes;
}

inline void mismatch(size_t offset, unsigned expected, unsigned actual) {
    throw std::runtime_error("output check failed at byte " + std::to_string(offset) +
                             ", expected " + std::to_string(expected) + ", got " + std::to_string(actual));
}

template<class EncodeHalf>
void checkOutput(Shape s, Mode mode, uint32_t blocks, uint32_t iterations,
                 const std::vector<uint8_t>& output, EncodeHalf encode) {
    const size_t needed = streamMode(mode) ? s.tapBytes() : size_t(blocks) * Elements * 2;
    if (output.size() < needed) throw std::runtime_error("short device output");
    const auto word = [&](size_t off, float value) {
        uint16_t actual;
        std::memcpy(&actual, &output.at(off), 2);
        const uint16_t expected = encode(value);
        if (actual != expected) mismatch(off, expected, actual);
    };
    if (!streamMode(mode)) {
        const uint32_t kt = (iterations - 1) % 2;
        for (uint32_t b = 0; b < blocks; ++b) {
            for (uint32_t row = 0; row < NTile; ++row) {
                for (uint32_t j = 0; j < KTile; ++j) {
                    uint32_t n = b * NTile + row, k = kt * KTile + j;
                    float value = float(codeAt(n, k));
                    if (mode == Fp16Fma) value = iterations * 0.125f;
                    else if (mode == Fp16Mul) value = iterations % 2 ? -0.5f : 0.5f;
                    else if (mode != CastNz && mode != CastFlat) value *= scaleAt(n, k / Group);
                    word((size_t(b) * Elements + row * KTile + j)*2, value);
                }
            }
        }
        return;
    }
    for (uint32_t nt = 0; nt < s.nTiles(); ++nt) {
        for (uint32_t kt = 0; kt < s.kTiles(); ++kt) {
            size_t base = (size_t(nt) * s.kTiles() + kt) * TapBytes;
            uint32_t n = nt * NTile, k = kt * KTile;
            for (uint32_t p = 0; p < KTile / Plane; ++p) {
                if (copyOnly(mode)) {
                    for (uint32_t j = 0; j < 32; ++j) {
                        auto expected = uint8_t(flat(mode) ? codeAt(n + 8*p, k + j) : codeAt(n, k + p*Plane + j));
                        auto off = base + p*32 + j;
                        if (output.at(off) != expected) mismatch(off, expected, output[off]);
                    }
                } else {
                    for (uint32_t j = 0; j < 16; ++j) {
                        uint32_t kk = k + p*Plane + j;
                        word(base + p*32 + j*2, float(codeAt(n, kk)) * scaleAt(n, kk/Group));
                    }
                }
            }
            if (copyOnly(mode)) {
                for (uint32_t g = 0; g < KTile / Group; ++g)
                    for (uint32_t j = 0; j < 16; ++j)
                        word(base + 256 + g*32 + j*2, scaleAt(n+j, kt*(KTile/Group)+g));
            }
        }
    }
}

// Durations are in microseconds, GB and GF are decimal. One converted code is one weight.
inline double work(Mode mode, Shape s, uint32_t blocks, uint32_t iterations) {
    return streamMode(mode) ? double(s.weights()) : double(blocks) * Elements * iterations;
}
inline double gigaPerSecond(double units, double us) { return units / (us * 1000.0); }
inline double fmaOrMulTF(Mode mode, double lanes, double us) {
    return gigaPerSecond(lanes, us) * (mode == Fp16Fma ? 2.0 : 1.0) / 1000.0;
}
inline double vectorBudgetMs(Shape s, uint32_t blocks, double gweightsPerSec) {
    uint32_t longest = (s.nTiles() + blocks - 1) / blocks;
    return double(longest) * blocks * NTile * s.k / (1.0e6 * gweightsPerSec);
}
}  // namespace vb
