#include "target_gdr_prefix_state_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <initializer_list>
#include <utility>

namespace {
bool Matches(const gert::StorageShape* shape, std::initializer_list<int64_t> dims) {
    if (!shape || shape->GetStorageShape().GetDimNum() != dims.size()) return false;
    size_t i = 0;
    for (const auto size : dims) {
        if (shape->GetStorageShape().GetDim(i++) != size) return false;
    }
    return true;
}

ge::graphStatus Infer(gert::InferShapeContext* ctx) {
    auto* out = ctx->GetOutputShape(0);
    const auto* initial = ctx->GetInputShape(4);
    if (!out || !initial) return ge::GRAPH_FAILED;
    *out = *initial;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Types(gert::InferDataTypeContext* ctx) {
    ctx->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tile(gert::TilingContext* ctx) {
    if (!ctx->GetPlatformInfo() || !ctx->GetAttrs() ||
        !Matches(ctx->GetInputShape(0), {1,16,32,128}) ||
        !Matches(ctx->GetInputShape(1), {1,16,32,128}) ||
        !Matches(ctx->GetInputShape(2), {1,16,32}) ||
        !Matches(ctx->GetInputShape(3), {1,16,32}) ||
        !Matches(ctx->GetInputShape(4), {1,32,128,128}) ||
        !Matches(ctx->GetInputShape(5), {1})) return ge::GRAPH_FAILED;
    auto platform = platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
    if (platform.GetCoreNumAic() < 8) return ge::GRAPH_FAILED;
    const auto* level = ctx->GetAttrs()->GetAttrPointer<int64_t>(0);
    const int64_t mode = level ? *level : 2;
    if (mode < 0 || mode > 3) return ge::GRAPH_FAILED;
    auto* raw = ctx->GetRawTilingData();
    auto* ws = ctx->GetWorkspaceSizes(1);
    optiling::TargetGdrPrefixStateTiling td;
    if (!raw || !ws || raw->GetCapacity() < td.GetDataSize()) return ge::GRAPH_FAILED;
    td.set_pruneLevel(static_cast<uint32_t>(mode));
    td.SaveToBuffer(raw->GetData(), raw->GetCapacity());
    raw->SetDataSize(td.GetDataSize());
    // Three 64x32 working planes fit in the first 32 KiB. The clone can emit
    // at most 16x32x128 FP16 readout values into a disjoint private sink.
    ws[0] = platform.GetLibApiWorkSpaceSize() + 32768 + 16*32*128*sizeof(uint16_t);
    ctx->SetBlockDim(8);
    ctx->SetTilingKey(1);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class TargetGdrPrefixState : public OpDef {
public:
    explicit TargetGdrPrefixState(const char* name) : OpDef(name) {
        for (const auto& p : {std::pair<const char*,ge::DataType>{"key",ge::DT_FLOAT16},
                              {"value",ge::DT_FLOAT16},{"g",ge::DT_FLOAT},{"beta",ge::DT_FLOAT16},
                              {"initial",ge::DT_FLOAT},{"length",ge::DT_INT16}}) {
            this->Input(p.first).ParamType(REQUIRED).DataType({p.second}).Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
        }
        this->Output("state").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("prune_level").AttrType(OPTIONAL).Int(2);
        this->SetInferShape(Infer).SetInferDataType(Types);
        this->AICore().SetTiling(Tile);
        this->AICore().AddConfig("ascend310p");
    }
};
OP_ADD(TargetGdrPrefixState);
}
