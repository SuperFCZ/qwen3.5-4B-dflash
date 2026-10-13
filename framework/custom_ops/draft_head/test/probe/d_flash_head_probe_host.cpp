#include "d_flash_head_probe_tiling.h"
#include "head_host.h"
namespace {
template<class S,class D> bool Extra(const S* s,const D* d,
           ge::DataType type,std::initializer_list<int64_t> shape) {
    return s && d && d->GetDataType()==type && d->GetStorageFormat()==ge::FORMAT_ND &&
        head_host::Shape(&s->GetOriginShape(),shape) && head_host::Shape(&s->GetStorageShape(),shape);
}
ge::graphStatus ProbeTiling(gert::TilingContext* c) {
    if(!Extra(c->GetInputShape(2),c->GetInputDesc(2),ge::DT_FLOAT,{64}) ||
       !Extra(c->GetOutputShape(2),c->GetOutputDesc(2),ge::DT_FLOAT,{2,16,64}) ||
       !Extra(c->GetOutputShape(3),c->GetOutputDesc(3),ge::DT_FLOAT,{3,64}) ||
       !Extra(c->GetOutputShape(4),c->GetOutputDesc(4),ge::DT_FLOAT16,{3,64})) return ge::GRAPH_FAILED;
    return head_host::Tiling<optiling::DFlashHeadProbeTilingData,true>(c);
}
ge::graphStatus ProbeShape(gert::InferShapeContext* c) {
    if(head_host::InferShape<true>(c)!=ge::GRAPH_SUCCESS || !head_host::Shape(c->GetInputShape(2),{64})) return ge::GRAPH_FAILED;
    auto* s=c->GetOutputShape(2); s->SetDimNum(3); s->SetDim(0,2); s->SetDim(1,16); s->SetDim(2,64);
    for(int i=3;i<5;++i) { auto* v=c->GetOutputShape(i); v->SetDimNum(2); v->SetDim(0,3); v->SetDim(1,64); }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus ProbeType(gert::InferDataTypeContext* c) {
    if(head_host::InferType<true>(c)!=ge::GRAPH_SUCCESS || c->GetInputDataType(2)!=ge::DT_FLOAT) return ge::GRAPH_FAILED;
    c->SetOutputDataType(2,ge::DT_FLOAT); c->SetOutputDataType(3,ge::DT_FLOAT); c->SetOutputDataType(4,ge::DT_FLOAT16);
    return ge::GRAPH_SUCCESS;
}
}
namespace ops {
class DFlashHeadProbe : public OpDef {
public:
    explicit DFlashHeadProbe(const char* name):OpDef(name) {
        this->Input("hidden").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("values").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("token_id").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Output("logits").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("precast").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("vector_fp32").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("vector_fp16").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->SetInferShape(ProbeShape).SetInferDataType(ProbeType);
        this->AICore().SetTiling(ProbeTiling).AddConfig("ascend310p");
    }
}; OP_ADD(DFlashHeadProbe);
}
