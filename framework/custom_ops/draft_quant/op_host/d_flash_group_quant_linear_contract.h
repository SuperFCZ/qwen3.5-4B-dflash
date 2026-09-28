#ifndef D_FLASH_GROUP_QUANT_LINEAR_CONTRACT_H
#define D_FLASH_GROUP_QUANT_LINEAR_CONTRACT_H

#include <cstddef>
#include <cstdint>
#include <initializer_list>

// SDK-independent shape predicates, also exercised by the CPU metadata test.
namespace draft_quant_contract {
constexpr int64_t kRows = 16;
constexpr int64_t kGroupSize = 128;
constexpr int64_t kColumnTile = 64;

inline bool IsSupportedShape(int64_t m, int64_t k, int64_t n)
{
    return m == kRows && (k == 256 || k == 512 || k == 1024) &&
           (n == 64 || n == 128 || n == 256);
}

inline int64_t KTile(int64_t k)
{
    // Retain the proven whole-K path at K=256. Larger K streams one group.
    return k == 256 ? 256 : kGroupSize;
}

inline uint64_t UserUbBytes(int64_t k)
{
    const uint64_t tileK = static_cast<uint64_t>(KTile(k));
    return kColumnTile * tileK * 3 + (tileK / kGroupSize) * kColumnTile * 2;
}

template <typename Shape>
bool IsShape(const Shape *shape, std::initializer_list<int64_t> dims)
{
    if (shape == nullptr || shape->GetDimNum() != dims.size()) return false;
    size_t index = 0;
    for (int64_t dim : dims) {
        if (shape->GetDim(index++) != dim) return false;
    }
    return true;
}

template <typename Shape, typename Format>
bool IsNzWeightDescriptor(const Shape *origin, const Shape *storage,
                          Format originFormat, Format storageFormat,
                          Format nd, Format nz, int64_t n, int64_t k)
{
    // Both transports must describe the exact physical INT8 NZ allocation.
    if (!IsSupportedShape(kRows, k, n)) return false;
    if (storageFormat != nz || !IsShape(storage, {k / 32, n / 16, 16, 32})) return false;
    // GE carries logical [N,K] separately from the physical NZ storage.
    const bool geLogical = originFormat == nd && IsShape(origin, {n, k});
    // CANN 9.0 individual ACLNN copies ViewShape into BOTH tiling shapes.
    // Its caller therefore passes a physical NZ view, not a logical ND view.
    const bool aclnnPhysical = originFormat == nz && IsShape(origin, {k / 32, n / 16, 16, 32});
    return geLogical || aclnnPhysical;
}
}  // namespace draft_quant_contract

#endif
