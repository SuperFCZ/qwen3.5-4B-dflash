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

inline bool IsSupportedRows(int64_t m)
{
    return m == 16 || m == 32 || m == 64 || m == 80;
}

inline bool IsSupportedShape(int64_t m, int64_t k, int64_t n)
{
    const bool synthetic = IsSupportedRows(m) && (k == 256 || k == 512 || k == 1024) &&
                    (n == 64 || n == 128 || n == 256);
    const bool m16 = m == 16 && ((k == 2560 && (n == 19456 || n == 4096)) ||
                                ((k == 9728 || k == 4096) && n == 2560));
    const bool kv = (m == 32 || m == 80) && k == 2560 && n == 2048;
    const bool fc = (m == 16 || m == 64) && k == 12800 && n == 2560;
    return synthetic || m16 || kv || fc;
}

inline int64_t KTile(int64_t k)
{
    // Retain the proven whole-K path at K=256. Larger K streams one group.
    return k == 256 ? 256 : kGroupSize;
}

inline uint32_t LaunchBlockCount(int64_t n, uint32_t availableCores, uint32_t coreLimit = 0)
{
    if (n <= 0 || n % kColumnTile != 0 || availableCores == 0) return 0;
    const uint64_t tiles = static_cast<uint64_t>(n / kColumnTile);
    const uint32_t permitted = coreLimit != 0 && coreLimit < availableCores ? coreLimit : availableCores;
    return tiles < permitted ? static_cast<uint32_t>(tiles) : permitted;
}

inline uint64_t UserUbBytes(int64_t k, uint32_t rawBanks = 1)
{
    const uint64_t tileK = static_cast<uint64_t>(KTile(k));
    // W16 remains single-buffered. Only raw INT8 codes and FP16 GN scales
    // get a second bank in A3.2 (8320 additional bytes for K128).
    return kColumnTile * tileK * (2 + rawBanks) +
           (tileK / kGroupSize) * kColumnTile * 2 * rawBanks;
}

struct CubePlan {
    int64_t singleM, singleN, singleK;
    int64_t baseM, baseN, baseK;
};

inline bool IsCompatibleCubePlan(int64_t globalK, const CubePlan &plan, int64_t m = kRows)
{
    // Global (M,K,N) pairs are validated separately. The Cube sees a local
    // 64-column tile, covering all M rows and the current K group.
    if (globalK != 256 && globalK != 512 && globalK != 1024 &&
        globalK != 2560 && globalK != 9728 && globalK != 4096 && globalK != 12800) return false;
    if (!IsSupportedRows(m)) return false;
    const int64_t tileK = KTile(globalK);
    // One output block covers ALL M rows. This reuses each raw/dequantized B
    // tile without an outer M loop or separate per-M-block FP16 partial sums.
    if (plan.singleM != m || plan.singleN != kColumnTile || plan.singleK != tileK ||
        plan.baseM != m || plan.baseN != kColumnTile) return false;

    if (tileK == globalK) {
        // IterateAll owns the entire K reduction. CANN's final Cube baseK can
        // differ from SetFixSplit's requested 128; in particular 256 is valid.
        // Quantization groups were already applied while building FP16 B.
        return plan.baseK > 0 && plan.baseK <= tileK && plan.baseK % 16 == 0;
    }
    // Keep the already-validated streamed partial-sum contract unchanged.
    return plan.baseK == kGroupSize;
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
                          Format nd, Format nz, int64_t n, int64_t k, int64_t m = kRows)
{
    // Both transports must describe the exact physical INT8 NZ allocation.
    if (!IsSupportedShape(m, k, n)) return false;
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
