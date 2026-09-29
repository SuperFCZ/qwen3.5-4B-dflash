#include <cstdio>
#include <initializer_list>
#include <limits>

#include "d_flash_group_quant_linear_contract.h"
#include "d_flash_group_quant_linear_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace {
constexpr int64_t kM = 16;
constexpr int64_t kTileN = 64;
constexpr int64_t kGroup = 128;

using draft_quant_contract::IsShape;

bool ValidInput(gert::TilingContext *context, size_t index, ge::DataType dtype,
                ge::Format format, std::initializer_list<int64_t> origin,
                std::initializer_list<int64_t> storage)
{
    const auto *shape = context->GetInputShape(index);
    const auto *desc = context->GetInputDesc(index);
    return shape != nullptr && desc != nullptr && desc->GetDataType() == dtype &&
           desc->GetStorageFormat() == format && desc->GetOriginFormat() == ge::FORMAT_ND &&
           IsShape(&shape->GetOriginShape(), origin) &&
           IsShape(&shape->GetStorageShape(), storage);
}

bool ValidWeightInput(gert::TilingContext *context, int64_t n, int64_t k)
{
    const auto *shape = context->GetInputShape(1);
    const auto *desc = context->GetInputDesc(1);
    return shape != nullptr && desc != nullptr && desc->GetDataType() == ge::DT_INT8 &&
           draft_quant_contract::IsNzWeightDescriptor(
               &shape->GetOriginShape(), &shape->GetStorageShape(),
               desc->GetOriginFormat(), desc->GetStorageFormat(),
               ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, n, k);
}

bool ReadDimensions(const gert::Shape *x, const gert::Shape *s, int64_t &k, int64_t &n)
{
    if (x == nullptr || s == nullptr || x->GetDimNum() != 2 || s->GetDimNum() != 2) return false;
    k = x->GetDim(1);
    n = s->GetDim(1);
    return draft_quant_contract::IsSupportedShape(x->GetDim(0), k, n) && s->GetDim(0) == k / kGroup;
}

void PrintShape(const gert::Shape *shape)
{
    if (shape == nullptr) {
        std::fprintf(stderr, "<null>");
        return;
    }
    std::fprintf(stderr, "[");
    for (size_t i = 0; i < shape->GetDimNum(); ++i) {
        std::fprintf(stderr, "%s%lld", i == 0 ? "" : ",", static_cast<long long>(shape->GetDim(i)));
    }
    std::fprintf(stderr, "]");
}

void DumpTensor(gert::TilingContext *context, size_t index, bool output)
{
    const auto *shape = output ? context->GetOutputShape(index) : context->GetInputShape(index);
    const auto *desc = output ? context->GetOutputDesc(index) : context->GetInputDesc(index);
    std::fprintf(stderr, "DFlashGroupQuantLinear: %s[%zu]", output ? "output" : "input", index);
    if (desc == nullptr) {
        std::fprintf(stderr, " desc=<null>");
    } else {
        std::fprintf(stderr, " dtype=%d origin_format=%d storage_format=%d",
                     static_cast<int>(desc->GetDataType()), static_cast<int>(desc->GetOriginFormat()),
                     static_cast<int>(desc->GetStorageFormat()));
    }
    std::fprintf(stderr, " origin_shape=");
    PrintShape(shape == nullptr ? nullptr : &shape->GetOriginShape());
    std::fprintf(stderr, " storage_shape=");
    PrintShape(shape == nullptr ? nullptr : &shape->GetStorageShape());
    std::fprintf(stderr, "\n");
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const auto *xShape = context->GetInputShape(0);
    const auto *sShape = context->GetInputShape(2);
    int64_t k = 0, n = 0;
    if (!ReadDimensions(xShape == nullptr ? nullptr : &xShape->GetOriginShape(),
                        sShape == nullptr ? nullptr : &sShape->GetOriginShape(), k, n) ||
        !ValidInput(context, 0, ge::DT_FLOAT16, ge::FORMAT_ND, {kM, k}, {kM, k}) ||
        !ValidWeightInput(context, n, k) ||
        !ValidInput(context, 2, ge::DT_FLOAT16, ge::FORMAT_ND,
                    {k / kGroup, n}, {k / kGroup, n})) {
        std::fprintf(stderr, "DFlashGroupQuantLinear: requires M=16 with A1 K={256,512,1024} "
                             "N={64,128,256} or A2 (K,N)={(2560,19456),(9728,2560)}, "
                             "INT8 NZ storage=[K/32,N/16,16,32], "
                             "origin=ND [N,K] (GE) or physical NZ (ACLNN), FP16 GN=[K/128,N]; "
                             "formats ND=%d NZ=%d, dtypes FP16=%d INT8=%d\n",
                     static_cast<int>(ge::FORMAT_ND), static_cast<int>(ge::FORMAT_FRACTAL_NZ),
                     static_cast<int>(ge::DT_FLOAT16), static_cast<int>(ge::DT_INT8));
        for (size_t i = 0; i < 3; ++i) DumpTensor(context, i, false);
        DumpTensor(context, 0, true);
        return ge::GRAPH_FAILED;
    }
    const auto *yShape = context->GetOutputShape(0);
    const auto *yDesc = context->GetOutputDesc(0);
    if (yShape == nullptr || yDesc == nullptr ||
        yDesc->GetDataType() != ge::DT_FLOAT16 || yDesc->GetStorageFormat() != ge::FORMAT_ND ||
        !IsShape(&yShape->GetOriginShape(), {kM, n}) ||
        !IsShape(&yShape->GetStorageShape(), {kM, n})) {
        std::fprintf(stderr, "DFlashGroupQuantLinear: requires FP16 ND output [16,N]\n");
        DumpTensor(context, 0, true);
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int64_t tileK = draft_quant_contract::KTile(k);
    const uint64_t codesBytes = kTileN * tileK;
    const uint64_t scaleBytes = (tileK / kGroup) * kTileN * 2;
    const uint64_t weightBytes = kTileN * tileK * 2;
    const uint64_t userUbBytes = draft_quant_contract::UserUbBytes(k);
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    if (ubBytes <= userUbBytes) return ge::GRAPH_FAILED;
    const uint64_t matmulUbBytes = (ubBytes - userUbBytes) / 32 * 32;
    if (matmulUbBytes > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        return ge::GRAPH_FAILED;
    }

    matmul_tiling::MatmulApiTiling cubeTiling(platform);
    cubeTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT16);
    cubeTiling.SetBType(matmul_tiling::TPosition::VECOUT, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT16, true);
    cubeTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT16);
    cubeTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                           matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBias(false);
    cubeTiling.SetShape(kM, kTileN, tileK);
    // A uses the full input row stride; C uses the full output row stride.
    // B is a local [64,tileK] tensor, so its row stride is tileK, not global K.
    cubeTiling.SetOrgShape(kM, n, k, tileK);
    cubeTiling.SetFixSplit(kM, kTileN, kGroup);
    cubeTiling.SetBufferSpace(-1, -1, static_cast<int32_t>(matmulUbBytes));

    DFlashGroupQuantLinearTilingData tiling;
    if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) {
        std::fprintf(stderr, "DFlashGroupQuantLinear: MatmulApiTiling failed for K=%lld N=%lld tileK=%lld\n",
                     static_cast<long long>(k), static_cast<long long>(n), static_cast<long long>(tileK));
        return ge::GRAPH_FAILED;
    }
    // Whole-K IterateAll and streamed Iterate(enPartialSum) have different
    // baseK contracts; group_size is not the Cube's internal reduction tile.
    auto &cube = tiling.cubeTilingData;  // CANN 9.0 getters are non-const.
    const draft_quant_contract::CubePlan plan{
        cube.get_singleCoreM(), cube.get_singleCoreN(), cube.get_singleCoreK(),
        cube.get_baseM(), cube.get_baseN(), cube.get_baseK()};
    if (!draft_quant_contract::IsCompatibleCubePlan(k, plan)) {
        std::fprintf(stderr, "DFlashGroupQuantLinear: incompatible %s tiling for K=%lld N=%lld; "
                             "singleCore M/N/K actual=%lld/%lld/%lld expected=16/64/%lld; "
                             "base M/N/K actual=%lld/%lld/%lld; expected M/N=16/64, baseK %s\n",
                     tileK == k ? "whole-K IterateAll" : "streamed partial-sum",
                     static_cast<long long>(k), static_cast<long long>(n),
                     static_cast<long long>(plan.singleM), static_cast<long long>(plan.singleN),
                     static_cast<long long>(plan.singleK), static_cast<long long>(tileK),
                     static_cast<long long>(plan.baseM), static_cast<long long>(plan.baseN),
                     static_cast<long long>(plan.baseK),
                     tileK == k ? "must be positive, 16-aligned and <=256" : "must equal 128");
        return ge::GRAPH_FAILED;
    }
    tiling.set_globalK(k);
    tiling.set_globalN(n);
    tiling.set_tileK(tileK);
    tiling.set_matmulUbBytes(matmulUbBytes);
    context->SetTilingKey(0);
    context->SetBlockDim(1);
    if (context->GetRawTilingData()->GetCapacity() < tiling.GetDataSize()) return ge::GRAPH_FAILED;
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->GetWorkspaceSizes(1)[0] = static_cast<size_t>(platform.GetLibApiWorkSpaceSize());
    const int64_t aTileBytes = static_cast<int64_t>(cube.get_baseM()) * cube.get_baseK() * 2;
    const int64_t bTileBytes = static_cast<int64_t>(cube.get_baseN()) * cube.get_baseK() * 2;
    const int64_t cTileBytes = static_cast<int64_t>(cube.get_baseM()) * cube.get_baseN() * 4;
    // Account using real tile sizes/depths, not TCubeTiling's reserved share*
    // fields. These are planned API buffers, not measured device traffic.
    std::fprintf(stdout, "DFlashGroupQuantLinear A1: M=16 K=%lld N=%lld tileN=64 tileK=%lld "
                         "Ntiles=%lld Ktiles=%lld; UB codes=%llu scales=%llu W16=%llu "
                         "MatMul=%llu bytes; base M/N/K=%d/%d/%d; "
                         "planned L1/L0A/L0B/L0C=%lld/%lld/%lld/%lld bytes; "
                         "transLength=%d; GM workspace=%zu bytes\n",
                 static_cast<long long>(k), static_cast<long long>(n), static_cast<long long>(tileK),
                 static_cast<long long>(n / kTileN), static_cast<long long>(k / tileK),
                 static_cast<unsigned long long>(codesBytes),
                 static_cast<unsigned long long>(scaleBytes),
                 static_cast<unsigned long long>(weightBytes),
                 static_cast<unsigned long long>(matmulUbBytes),
                 cube.get_baseM(), cube.get_baseN(), cube.get_baseK(),
                 static_cast<long long>(aTileBytes * cube.get_depthA1() + bTileBytes * cube.get_depthB1()),
                 static_cast<long long>(aTileBytes * cube.get_dbL0A()),
                 static_cast<long long>(bTileBytes * cube.get_dbL0B()),
                 static_cast<long long>(cTileBytes * cube.get_dbL0C()), cube.get_transLength(),
                 context->GetWorkspaceSizes(1)[0]);
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    // Accept the GE logical origin and the direct ACLNN physical NZ view.
    // Tiling additionally enforces the matching format/storage combination.
    const auto *w = context->GetInputShape(1);
    int64_t k = 0, n = 0;
    if (!ReadDimensions(context->GetInputShape(0), context->GetInputShape(2), k, n) ||
        (!IsShape(w, {n, k}) && !IsShape(w, {k / 32, n / 16, 16, 32}))) return GRAPH_FAILED;
    auto *y = context->GetOutputShape(0);
    if (y == nullptr) return GRAPH_FAILED;
    y->SetDimNum(2);
    y->SetDim(0, kM);
    y->SetDim(1, n);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    if (context->GetInputDataType(0) != DT_FLOAT16 ||
        context->GetInputDataType(1) != DT_INT8 ||
        context->GetInputDataType(2) != DT_FLOAT16) return GRAPH_FAILED;
    context->SetOutputDataType(0, DT_FLOAT16);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class DFlashGroupQuantLinear : public OpDef {
public:
    explicit DFlashGroupQuantLinear(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("w_nz").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_FRACTAL_NZ});
        this->Input("s").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend310p");
    }
};
OP_ADD(DFlashGroupQuantLinear);
}  // namespace ops
