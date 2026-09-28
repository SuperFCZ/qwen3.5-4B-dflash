#include <cstdio>
#include <initializer_list>
#include <limits>

#include "d_flash_group_quant_linear_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace {
constexpr int64_t kM = 16;
constexpr int64_t kK = 256;
constexpr int64_t kN = 64;
constexpr int64_t kGroup = 128;
// These allocations must match the kernel, including their lifetimes. All are
// 32-byte aligned; MatMul gets a separate UB region, never the full UB again.
constexpr uint64_t kCodesBytes = kN * kK;
constexpr uint64_t kScaleBytes = (kK / kGroup) * kN * 2;
constexpr uint64_t kWeightBytes = kN * kK * 2;
constexpr uint64_t kUserUbBytes = kCodesBytes + kScaleBytes + kWeightBytes;

bool IsShape(const gert::Shape *shape, std::initializer_list<int64_t> dims)
{
    if (shape == nullptr || shape->GetDimNum() != dims.size()) return false;
    size_t index = 0;
    for (int64_t dim : dims) {
        if (shape->GetDim(index++) != dim) return false;
    }
    return true;
}

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
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    // FRACTAL_NZ must carry BOTH logical [N,K] and physical [K1,N1,16,32].
    // A rank-4 ND carrier or an FP16-NZ descriptor is a different ABI.
    if (!ValidInput(context, 0, ge::DT_FLOAT16, ge::FORMAT_ND, {kM, kK}, {kM, kK}) ||
        !ValidInput(context, 1, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ,
                    {kN, kK}, {kK / 32, kN / 16, 16, 32}) ||
        !ValidInput(context, 2, ge::DT_FLOAT16, ge::FORMAT_ND,
                    {kK / kGroup, kN}, {kK / kGroup, kN})) {
        std::fprintf(stderr, "DFlashGroupQuantLinear: requires tiny M16 K256 N64, "
                             "INT8 NZ origin=[64,256] storage=[8,4,16,32], FP16 GN=[2,64]\n");
        return ge::GRAPH_FAILED;
    }
    const auto *yShape = context->GetOutputShape(0);
    const auto *yDesc = context->GetOutputDesc(0);
    if (yShape == nullptr || yDesc == nullptr ||
        yDesc->GetDataType() != ge::DT_FLOAT16 || yDesc->GetStorageFormat() != ge::FORMAT_ND ||
        !IsShape(&yShape->GetOriginShape(), {kM, kN}) ||
        !IsShape(&yShape->GetStorageShape(), {kM, kN})) return ge::GRAPH_FAILED;

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    if (ubBytes <= kUserUbBytes) return ge::GRAPH_FAILED;
    const uint64_t matmulUbBytes = (ubBytes - kUserUbBytes) / 32 * 32;
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
    cubeTiling.SetShape(kM, kN, kK);
    cubeTiling.SetOrgShape(kM, kN, kK);
    cubeTiling.SetFixSplit(kM, kN, kGroup);
    cubeTiling.SetBufferSpace(-1, -1, static_cast<int32_t>(matmulUbBytes));

    DFlashGroupQuantLinearTilingData tiling;
    if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) return ge::GRAPH_FAILED;
    tiling.set_matmulUbBytes(matmulUbBytes);
    context->SetTilingKey(0);
    context->SetBlockDim(1);
    if (context->GetRawTilingData()->GetCapacity() < tiling.GetDataSize()) return ge::GRAPH_FAILED;
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->GetWorkspaceSizes(1)[0] = static_cast<size_t>(platform.GetLibApiWorkSpaceSize());
    auto &cube = tiling.cubeTilingData;
    const int64_t aTileBytes = static_cast<int64_t>(cube.get_baseM()) * cube.get_baseK() * 2;
    const int64_t bTileBytes = static_cast<int64_t>(cube.get_baseN()) * cube.get_baseK() * 2;
    const int64_t cTileBytes = static_cast<int64_t>(cube.get_baseM()) * cube.get_baseN() * 4;
    // Account using real tile sizes/depths, not TCubeTiling's reserved share*
    // fields. These are planned API buffers, not measured device traffic.
    std::fprintf(stdout, "DFlashGroupQuantLinear tiny: UB codes=%llu scales=%llu W16=%llu "
                         "MatMul=%llu bytes; base M/N/K=%d/%d/%d; "
                         "planned L1/L0A/L0B/L0C=%lld/%lld/%lld/%lld bytes; "
                         "transLength=%d; GM workspace=%zu bytes\n",
                 static_cast<unsigned long long>(kCodesBytes),
                 static_cast<unsigned long long>(kScaleBytes),
                 static_cast<unsigned long long>(kWeightBytes),
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
    if (!IsShape(context->GetInputShape(0), {kM, kK}) ||
        !IsShape(context->GetInputShape(1), {kN, kK}) ||
        !IsShape(context->GetInputShape(2), {kK / kGroup, kN})) return GRAPH_FAILED;
    auto *y = context->GetOutputShape(0);
    if (y == nullptr) return GRAPH_FAILED;
    y->SetDimNum(2);
    y->SetDim(0, kM);
    y->SetDim(1, kN);
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
