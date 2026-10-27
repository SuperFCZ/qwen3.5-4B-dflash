#include "d_flash_w8_a8_partial_reduce_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstdio>
#include <initializer_list>
namespace {
bool Dims(const gert::Shape *s,std::initializer_list<int64_t> dims) {
    if(!s||s->GetDimNum()!=dims.size())return false;
    size_t i=0;for(auto v:dims)if(s->GetDim(i++)!=v)return false;return true;
}
}
namespace optiling {
static ge::graphStatus Tiling(gert::TilingContext *c) {
    auto x=c->GetInputShape(0);auto xd=c->GetInputDesc(0);
    auto y=c->GetOutputShape(0);auto yd=c->GetOutputDesc(0);
    if(!x||!xd||!y||!yd||xd->GetDataType()!=ge::DT_FLOAT16||yd->GetDataType()!=ge::DT_FLOAT16||
       xd->GetStorageFormat()!=ge::FORMAT_ND||yd->GetStorageFormat()!=ge::FORMAT_ND||
       !Dims(&x->GetOriginShape(),{20,16,64})||!Dims(&x->GetStorageShape(),{20,16,64})||
       !Dims(&y->GetOriginShape(),{16,64})||!Dims(&y->GetStorageShape(),{16,64}))return ge::GRAPH_FAILED;
    auto platform=platform_ascendc::PlatformAscendC(c->GetPlatformInfo());uint64_t ub=0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
    constexpr uint64_t explicitUb=2*1024+4*1024+4*1024+2*1024;
    constexpr uint64_t reserve=8*1024;
    if(ub<explicitUb+reserve) { std::fprintf(stderr,"STAGE3B_REJECT UB=%llu need>=20480\n",(unsigned long long)ub);return ge::GRAPH_FAILED; }
    DFlashW8A8PartialReduceTilingData data;data.set_groupCount(20);data.set_elements(1024);
    c->SetBlockDim(1);c->SetTilingKey(0);c->GetWorkspaceSizes(1)[0]=0;
    data.SaveToBuffer(c->GetRawTilingData()->GetData(),c->GetRawTilingData()->GetCapacity());
    c->GetRawTilingData()->SetDataSize(data.GetDataSize());
    std::fprintf(stderr,"STAGE3B_REDUCE_PLAN groups=20 elements=1024 blockDim=1 explicitUB=12288 reserve=8192 actualUB=%llu workspace=0 order=0..19\n",(unsigned long long)ub);
    return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus Infer(gert::InferShapeContext *c) {
    if(!Dims(c->GetInputShape(0),{20,16,64}))return GRAPH_FAILED;
    auto y=c->GetOutputShape(0);y->SetDimNum(2);y->SetDim(0,16);y->SetDim(1,64);return GRAPH_SUCCESS;
}
static graphStatus Dtype(gert::InferDataTypeContext *c) { c->SetOutputDataType(0,DT_FLOAT16);return GRAPH_SUCCESS; }
}
namespace ops {
class DFlashW8A8PartialReduce: public OpDef {
public:
    explicit DFlashW8A8PartialReduce(const char *name):OpDef(name) {
        Input("partials").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        Output("o").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        SetInferShape(ge::Infer).SetInferDataType(ge::Dtype);
        AICore().SetTiling(optiling::Tiling).AddConfig("ascend310p");
    }
};
OP_ADD(DFlashW8A8PartialReduce);
}
