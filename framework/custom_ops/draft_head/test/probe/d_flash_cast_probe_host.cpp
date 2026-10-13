#include "register/op_def_registry.h"
#include "d_flash_cast_probe_tiling.h"
namespace {
ge::graphStatus Tiling(gert::TilingContext* c) {
    const auto* x=c->GetInputShape(0); const auto* d=c->GetInputDesc(0);
    if(!x || !d || d->GetDataType()!=ge::DT_FLOAT || d->GetStorageFormat()!=ge::FORMAT_ND ||
       x->GetStorageShape().GetDimNum()!=1 || x->GetStorageShape().GetDim(0)!=64) return ge::GRAPH_FAILED;
    for(int i=0;i<2;++i) {
        const auto* s=c->GetOutputShape(i); const auto* o=c->GetOutputDesc(i);
        if(!s || !o || o->GetDataType()!=(i?ge::DT_FLOAT16:ge::DT_FLOAT) || o->GetStorageFormat()!=ge::FORMAT_ND ||
           s->GetStorageShape().GetDimNum()!=2 || s->GetStorageShape().GetDim(0)!=5 || s->GetStorageShape().GetDim(1)!=64)
            return ge::GRAPH_FAILED;
    }
    c->SetBlockDim(1); c->SetTilingKey(0); c->GetWorkspaceSizes(1)[0]=0;
    optiling::DFlashCastProbeTilingData t; t.set_count(64);
    if(c->GetRawTilingData()->GetCapacity()<t.GetDataSize()) return ge::GRAPH_FAILED;
    t.SaveToBuffer(c->GetRawTilingData()->GetData(),c->GetRawTilingData()->GetCapacity());
    c->GetRawTilingData()->SetDataSize(t.GetDataSize()); return ge::GRAPH_SUCCESS;
}
ge::graphStatus Shape(gert::InferShapeContext* c) {
    for(int i=0;i<2;++i) { auto* s=c->GetOutputShape(i); s->SetDimNum(2); s->SetDim(0,5); s->SetDim(1,64); }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus Type(gert::InferDataTypeContext* c) {
    c->SetOutputDataType(0,ge::DT_FLOAT); c->SetOutputDataType(1,ge::DT_FLOAT16); return ge::GRAPH_SUCCESS;
}
}
namespace ops {
class DFlashCastProbe:public OpDef {
public:
    explicit DFlashCastProbe(const char* name):OpDef(name) {
        this->Input("values").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("vector_fp32").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("vector_fp16").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->SetInferShape(Shape).SetInferDataType(Type); this->AICore().SetTiling(Tiling).AddConfig("ascend310p");
    }
}; OP_ADD(DFlashCastProbe);
}
