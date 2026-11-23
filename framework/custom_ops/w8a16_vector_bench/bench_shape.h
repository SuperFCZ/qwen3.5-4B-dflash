#pragma once
#include <stdint.h>

namespace vb {
constexpr uint32_t NTile = 64;
constexpr uint32_t KTile = 256;
constexpr uint32_t Group = 128;
constexpr uint32_t Plane = 32;
constexpr uint32_t Elements = NTile * KTile;
constexpr uint32_t ScaleWords = NTile * (KTile / Group);
constexpr uint32_t ExpandedBytes = 2304;  // 2048 active + the m200 Brcb rear area.
constexpr uint32_t TapBytes = (KTile / Plane + KTile / Group) * 32;
constexpr uint32_t UbIterations = 2047;  // odd; 0.125 * i stays exactly representable in FP16.

enum Mode : uint32_t {
    Fp16Fma, Fp16Mul, CastFlat, CastNz, NzBrcb, NzB16, FlatB16,
    CopyNz, CopyFlat, StreamNzBrcb, StreamNzB16, StreamFlatB16
};
}  // namespace vb
