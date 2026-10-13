#include "register/op_def_registry.h"
#include "d_flash_top1_probe_tiling.h"
namespace {
ge::graphStatus Tiling(gert::TilingContext* c) {
    const auto* x=c->GetInputShape(0); const auto* y=c->GetOutputShape(0);
    if(!x || !y || x->GetStorageShape().GetDimNum()!=2 || x->GetStorageShape().GetDim(0)!=15 ||
       x->GetStorageShape().GetDim(1)!=64 || y->GetStorageShape().GetDimNum()!=1 || y->GetStorageShape().GetDim(0)!=15 ||
       c->GetInputDesc(0)->GetDataType()!=ge::DT_FLOAT16 || c->GetOutputDesc(0)->GetDataType()!=ge::DT_INT64)
        return ge::GRAPH_FAILED;
    const auto* d=c->GetOutputShape(1);
    if(!d || d->GetStorageShape().GetDimNum()!=2 || d->GetStorageShape().GetDim(0)!=15 ||
       d->GetStorageShape().GetDim(1)!=240 || c->GetOutputDesc(1)->GetDataType()!=ge::DT_UINT32) return ge::GRAPH_FAILED;
    c->SetBlockDim(1); c->SetTilingKey(0); c->GetWorkspaceSizes(1)[0]=0;
    optiling::DFlashTop1ProbeTilingData t; t.set_count(15);
    if(c->GetRawTilingData()->GetCapacity()<t.GetDataSize()) return ge::GRAPH_FAILED;
    t.SaveToBuffer(c->GetRawTilingData()->GetData(),c->GetRawTilingData()->GetCapacity());
    c->GetRawTilingData()->SetDataSize(t.GetDataSize()); return ge::GRAPH_SUCCESS;
}
ge::graphStatus Shape(gert::InferShapeContext* c) {
    auto* s=c->GetOutputShape(0); s->SetDimNum(1); s->SetDim(0,15);
    auto* d=c->GetOutputShape(1); d->SetDimNum(2); d->SetDim(0,15); d->SetDim(1,240); return ge::GRAPH_SUCCESS;
}
ge::graphStatus Type(gert::InferDataTypeContext* c) { c->SetOutputDataType(0,ge::DT_INT64); c->SetOutputDataType(1,ge::DT_UINT32); return ge::GRAPH_SUCCESS; }
}
namespace ops {
class DFlashTop1Probe:public OpDef {
public:
    explicit DFlashTop1Probe(const char* name):OpDef(name) {
        this->Input("logits").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("ids").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Output("trace").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->SetInferShape(Shape).SetInferDataType(Type); this->AICore().SetTiling(Tiling).AddConfig("ascend310p");
    }
}; OP_ADD(DFlashTop1Probe);
}
