#ifndef D_FLASH_GROUP_QUANT_LINEAR_CONTRACT_H
#define D_FLASH_GROUP_QUANT_LINEAR_CONTRACT_H

#include <cstddef>
#include <cstdint>
#include <initializer_list>

// SDK-independent shape predicates, also exercised by the CPU metadata test.
namespace draft_quant_contract {
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
