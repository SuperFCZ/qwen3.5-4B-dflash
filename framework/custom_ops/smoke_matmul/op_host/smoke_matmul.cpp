#include "smoke_matmul_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace {
constexpr int64_t kM = 16;
constexpr int64_t kK = 256;
constexpr int64_t kN = 64;

bool IsShape(const gert::Shape *shape, int64_t first, int64_t second)
{
    return shape != nullptr && shape->GetDimNum() == 2 &&
           shape->GetDim(0) == first && shape->GetDim(1) == second;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const auto *x = context->GetInputShape(0);
    const auto *w = context->GetInputShape(1);
    if (x == nullptr || w == nullptr ||
        !IsShape(&x->GetOriginShape(), kM, kK) ||
        !IsShape(&w->GetOriginShape(), kN, kK)) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    matmul_tiling::MatmulApiTiling cubeTiling(platform);
    cubeTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT16);
    // W is physically [N, K]; the logical right operand is W.T, [K, N].
    cubeTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT16, true);
    cubeTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT16);
    cubeTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                           matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBias(false);
    cubeTiling.SetShape(kM, kN, kK);
    cubeTiling.SetOrgShape(kM, kN, kK);
    cubeTiling.SetBufferSpace(-1, -1, -1);

    SmokeMatmulTilingData tiling;
    if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tiling.set_localMemSize(ubSize);
    context->SetTilingKey(2);  // 310P path with local UB workspace.
    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->GetWorkspaceSizes(1)[0] =
        static_cast<size_t>(platform.GetLibApiWorkSpaceSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *x = context->GetInputShape(0);
    const auto *w = context->GetInputShape(1);
    if (!IsShape(x, kM, kK) || !IsShape(w, kN, kK)) {
        return GRAPH_FAILED;
    }
    auto *y = context->GetOutputShape(0);
    y->SetDimNum(2);
    y->SetDim(0, kM);
    y->SetDim(1, kN);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT16);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class SmokeMatmul : public OpDef {
public:
    explicit SmokeMatmul(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("w").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend310p");
    }
};
OP_ADD(SmokeMatmul);
}  // namespace ops
