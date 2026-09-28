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
