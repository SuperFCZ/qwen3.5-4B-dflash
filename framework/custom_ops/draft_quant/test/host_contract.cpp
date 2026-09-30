// CPU-only metadata regression. This does not compile or execute an Ascend op.
#include <cassert>
#include <vector>

#include "../op_host/d_flash_group_quant_linear_contract.h"

namespace {
enum class Format { ND, NZ, NCHW };

struct Shape {
    std::vector<int64_t> dims;
    size_t GetDimNum() const { return dims.size(); }
    int64_t GetDim(size_t index) const { return dims[index]; }
};

bool Accepted(const Shape &origin, const Shape &storage, Format originFormat, Format storageFormat)
{
    return draft_quant_contract::IsNzWeightDescriptor(
        &origin, &storage, originFormat, storageFormat, Format::ND, Format::NZ, 64, 256);
}
}  // namespace

int main()
{
    assert(draft_quant_contract::LaunchBlockCount(64, 8) == 1);
    assert(draft_quant_contract::LaunchBlockCount(128, 8) == 2);
    assert(draft_quant_contract::LaunchBlockCount(256, 8) == 4);
    assert(draft_quant_contract::LaunchBlockCount(19456, 8) == 8);
    assert(draft_quant_contract::LaunchBlockCount(2560, 8) == 8);
    assert(draft_quant_contract::LaunchBlockCount(19456, 8, 1) == 1);
    assert(draft_quant_contract::LaunchBlockCount(19456, 8, 3) == 3);
    assert(draft_quant_contract::LaunchBlockCount(19456, 8, 64) == 8);
    assert(draft_quant_contract::LaunchBlockCount(19456, 512) == 304);
    assert(draft_quant_contract::LaunchBlockCount(2560, 64) == 40);
    assert(draft_quant_contract::LaunchBlockCount(2560, 0) == 0);
    for (int64_t n : {-64, 0, 65}) assert(draft_quant_contract::LaunchBlockCount(n, 8) == 0);
    for (int64_t n : {64, 128, 256, 2560, 19456}) {
        for (uint32_t cores : {1U, 2U, 3U, 6U, 8U, 32U, 512U}) {
            for (uint32_t cap : {0U, 1U, 3U, 8U, 64U}) {
                const auto count = draft_quant_contract::LaunchBlockCount(n, cores, cap);
                assert(count > 0 && count <= cores && count <= static_cast<uint64_t>(n / 64));
                assert(cap == 0 || count <= cap);
                std::vector<uint32_t> owned(count, 0);
                for (uint32_t tile = 0; tile < static_cast<uint32_t>(n / 64); ++tile) ++owned[tile % count];
                for (auto tiles : owned) assert(tiles > 0 && (tiles == owned[0] || tiles + 1 == owned[0]));
            }
        }
    }
    for (int64_t k : {256, 512, 1024}) {
        for (int64_t n : {64, 128, 256}) {
            assert(draft_quant_contract::IsSupportedShape(16, k, n));
            const Shape logicalShape{{n, k}}, physicalShape{{k / 32, n / 16, 16, 32}};
            assert(draft_quant_contract::IsNzWeightDescriptor(&logicalShape, &physicalShape,
                Format::ND, Format::NZ, Format::ND, Format::NZ, n, k));
            assert(draft_quant_contract::IsNzWeightDescriptor(&physicalShape, &physicalShape,
                Format::NZ, Format::NZ, Format::ND, Format::NZ, n, k));
            assert(draft_quant_contract::KTile(k) == (k == 256 ? 256 : 128));
            assert(draft_quant_contract::UserUbBytes(k) == (k == 256 ? 49408 : 24704));
        }
    }
    assert(!draft_quant_contract::IsSupportedShape(32, 256, 64));
    assert(!draft_quant_contract::IsSupportedShape(16, 384, 64));
    assert(!draft_quant_contract::IsSupportedShape(16, 512, 80));
    assert(!draft_quant_contract::IsSupportedShape(16, -256, 64));
    for (const auto &pair : {std::vector<int64_t>{2560, 19456}, std::vector<int64_t>{9728, 2560}}) {
        const auto k = pair[0], n = pair[1];
        assert(draft_quant_contract::IsSupportedShape(16, k, n));
        assert(!draft_quant_contract::IsSupportedShape(64, k, n));
        const Shape logical{{n, k}}, physical{{k / 32, n / 16, 16, 32}};
        assert(draft_quant_contract::IsNzWeightDescriptor(&logical, &physical,
            Format::ND, Format::NZ, Format::ND, Format::NZ, n, k));
        assert(draft_quant_contract::IsNzWeightDescriptor(&physical, &physical,
            Format::NZ, Format::NZ, Format::ND, Format::NZ, n, k));
        assert(draft_quant_contract::UserUbBytes(k) == 24704);
    }
    assert(!draft_quant_contract::IsSupportedShape(16, 2560, 2560));
    assert(!draft_quant_contract::IsSupportedShape(16, 9728, 19456));
    assert(!draft_quant_contract::IsSupportedShape(16, 2560, 64));

    // Regression: K=256 uses IterateAll, so a returned baseK=256 is valid.
    // Neither the dequantization group nor the caller's buffer size changes.
    for (int64_t baseK : {16, 32, 64, 128, 256}) {
        assert(draft_quant_contract::IsCompatibleCubePlan(256, {16, 64, 256, 16, 64, baseK}));
    }
    for (int64_t badK : {-16, 0, 129, 512}) {
        assert(!draft_quant_contract::IsCompatibleCubePlan(256, {16, 64, 256, 16, 64, badK}));
    }
    // Keep all restrictions on the receiver's passing K=512/1024 path.
    for (int64_t k : {512, 1024, 2560, 9728}) {
        assert(draft_quant_contract::UserUbBytes(k, 2) == 33024);
        assert(draft_quant_contract::UserUbBytes(k, 2) - draft_quant_contract::UserUbBytes(k) == 8320);
        assert(draft_quant_contract::IsCompatibleCubePlan(k, {16, 64, 128, 16, 64, 128}));
        assert(!draft_quant_contract::IsCompatibleCubePlan(k, {16, 64, 128, 16, 64, 256}));
        assert(!draft_quant_contract::IsCompatibleCubePlan(k, {16, 64, 128, 16, 64, 64}));
    }
    for (int64_t k : {256, 512, 1024, 2560, 9728}) {
        const int64_t tileK = draft_quant_contract::KTile(k);
        assert(!draft_quant_contract::IsCompatibleCubePlan(k, {32, 64, tileK, 16, 64, 128}));
        assert(!draft_quant_contract::IsCompatibleCubePlan(k, {16, 128, tileK, 16, 64, 128}));
        assert(!draft_quant_contract::IsCompatibleCubePlan(k, {16, 64, tileK * 2, 16, 64, 128}));
        assert(!draft_quant_contract::IsCompatibleCubePlan(k, {16, 64, tileK, 32, 64, 128}));
        assert(!draft_quant_contract::IsCompatibleCubePlan(k, {16, 64, tileK, 16, 32, 128}));
    }
    assert(!draft_quant_contract::IsCompatibleCubePlan(384, {16, 64, 128, 16, 64, 128}));
    const Shape logical{{64, 256}};
    const Shape physical{{8, 4, 16, 32}};

    // GE preserves logical origin metadata; direct ACLNN forwards physical view.
    assert(Accepted(logical, physical, Format::ND, Format::NZ));
    assert(Accepted(physical, physical, Format::NZ, Format::NZ));

    // Regression: CANN 9.0 forwards the old caller's logical ViewShape into
    // both tiling shapes. Do not hide that transport bug by accepting 2-D NZ.
    assert(!Accepted(logical, logical, Format::NZ, Format::NZ));
    assert(!Accepted(physical, physical, Format::ND, Format::NZ));
    assert(!Accepted(logical, physical, Format::NZ, Format::NZ));
    assert(!Accepted(logical, physical, Format::NCHW, Format::NZ));
    assert(!Accepted(physical, physical, Format::NZ, Format::ND));
    assert(!Accepted(logical, physical, Format::ND, Format::ND));

    // Equal byte counts alone cannot establish the layout or axis order.
    for (const Shape &wrong : {Shape{{4, 8, 16, 32}}, Shape{{8, 4, 32, 16}},
                              Shape{{16, 4, 16, 16}}, Shape{{16384}},
                              Shape{{8, 4, 16, 31}}, Shape{{8, 4, 16, 33}}}) {
        assert(!Accepted(logical, wrong, Format::ND, Format::NZ));
        assert(!Accepted(wrong, physical, Format::NZ, Format::NZ));
    }
    for (const Shape &wrong : {Shape{{256, 64}}, Shape{{63, 256}}, Shape{{64, 255}}}) {
        assert(!Accepted(wrong, physical, Format::ND, Format::NZ));
    }
    const Shape *missing = nullptr;
    assert(!draft_quant_contract::IsNzWeightDescriptor(
        missing, &physical, Format::ND, Format::NZ, Format::ND, Format::NZ, 64, 256));
    assert(!draft_quant_contract::IsNzWeightDescriptor(
        &logical, missing, Format::ND, Format::NZ, Format::ND, Format::NZ, 64, 256));
    return 0;
}
