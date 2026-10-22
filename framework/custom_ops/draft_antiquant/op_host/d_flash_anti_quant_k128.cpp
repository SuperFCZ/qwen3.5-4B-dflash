#include <cstdio>
#include <initializer_list>
#include "d_flash_anti_quant_k128_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
namespace {
bool Shape(const gert::Shape *s, std::initializer_list<int64_t> dims) {
    if (!s || s->GetDimNum()!=dims.size()) return false;
    size_t i=0; for (auto d:dims) if(s->GetDim(i++)!=d) return false;
    return true;
}
bool Input(gert::TilingContext *c, size_t i, ge::DataType dt,
           std::initializer_list<int64_t> dims) {
    auto s=c->GetInputShape(i); auto d=c->GetInputDesc(i);
    return s && d && d->GetDataType()==dt && d->GetStorageFormat()==ge::FORMAT_ND &&
        Shape(&s->GetOriginShape(),dims) && Shape(&s->GetStorageShape(),dims);
}
}
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *c) {
    const auto w=c->GetInputShape(1); const auto wd=c->GetInputDesc(1);
    // ACLNN can expose physical NZ as origin; GE exposes logical ND origin.
    if (!Input(c,0,ge::DT_FLOAT16,{16,128}) || !Input(c,2,ge::DT_FLOAT16,{1,64}) ||
        !Input(c,3,ge::DT_FLOAT16,{1,64}) || !w || !wd || wd->GetDataType()!=ge::DT_INT8 ||
        wd->GetStorageFormat()!=ge::FORMAT_FRACTAL_NZ ||
        !Shape(&w->GetStorageShape(),{4,4,16,32}) ||
        !((wd->GetOriginFormat()==ge::FORMAT_ND && Shape(&w->GetOriginShape(),{64,128})) ||
          (wd->GetOriginFormat()==ge::FORMAT_FRACTAL_NZ && Shape(&w->GetOriginShape(),{4,4,16,32}))))
        return ge::GRAPH_FAILED;
    const auto y=c->GetOutputShape(0); const auto yd=c->GetOutputDesc(0);
    if(!y || !yd || yd->GetDataType()!=ge::DT_FLOAT16 || yd->GetStorageFormat()!=ge::FORMAT_ND ||
       !Shape(&y->GetOriginShape(),{16,64}) || !Shape(&y->GetStorageShape(),{16,64})) return ge::GRAPH_FAILED;
    auto platform=platform_ascendc::PlatformAscendC(c->GetPlatformInfo());
    uint64_t ub=0,l1=0,l0a=0,l0b=0,l0c=0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1,l1);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A,l0a);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B,l0b);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C,l0c);
    // 256 B user vectors; 8192 B conservative SDK reserve, separate from workspace.
    if(ub!=256*1024) return ge::GRAPH_FAILED;
    const uint64_t temp=(ub-256-8192)/32*32;
    matmul_tiling::MatmulApiTiling t(platform);
    using namespace matmul_tiling;
    t.SetAType(TPosition::GM,CubeFormat::ND,DataType::DT_FLOAT16);
    t.SetBType(TPosition::GM,CubeFormat::NZ,DataType::DT_INT8,true);
    t.SetCType(TPosition::GM,CubeFormat::ND,DataType::DT_FLOAT16);
    t.SetBiasType(TPosition::GM,CubeFormat::ND,DataType::DT_FLOAT);
    t.SetBias(false); t.SetShape(16,64,128); t.SetOrgShape(16,64,128);
    t.SetFixSplit(16,64,128); t.SetBufferSpace(-1,-1,static_cast<int32_t>(temp));
    DFlashAntiQuantK128TilingData data;
    if(t.GetTiling(data.cubeTilingData)==-1) return ge::GRAPH_FAILED;
    auto &p=data.cubeTilingData;
    if(p.get_singleCoreM()!=16 || p.get_singleCoreN()!=64 || p.get_singleCoreK()!=128 ||
       p.get_baseM()!=16 || p.get_baseN()!=64 || p.get_baseK()!=128) {
        std::fprintf(stderr,"ANTIQUANT incompatible single=%d/%d/%d base=%d/%d/%d\n",
            p.get_singleCoreM(),p.get_singleCoreN(),p.get_singleCoreK(),p.get_baseM(),p.get_baseN(),p.get_baseK());
        return ge::GRAPH_FAILED;
    }
    const uint64_t a=16*128*2,b=64*128*2,acc=16*64*4;
    if(p.get_depthA1()<1 || p.get_depthB1()<1 || p.get_dbL0A()<1 || p.get_dbL0B()<1 || p.get_dbL0C()<1 ||
       a*p.get_depthA1()+b*p.get_depthB1()>l1 || a*p.get_dbL0A()>l0a ||
       b*p.get_dbL0B()>l0b || acc*p.get_dbL0C()>l0c) return ge::GRAPH_FAILED;
    std::fprintf(stderr,"ANTIQUANT_K128 blockDim=1 single=16/64/128 base=16/64/128 "
        "user_ub=256 sdk_ub=%llu sdk_reserved=8192 L1=%llu/%llu L0A=%llu/%llu L0B=%llu/%llu L0C=%llu/%llu "
        "depthA1=%d depthB1=%d dbL0A=%d dbL0B=%d dbL0C=%d\n",
        (unsigned long long)temp,(unsigned long long)(a*p.get_depthA1()+b*p.get_depthB1()),(unsigned long long)l1,
        (unsigned long long)(a*p.get_dbL0A()),(unsigned long long)l0a,(unsigned long long)(b*p.get_dbL0B()),(unsigned long long)l0b,
        (unsigned long long)(acc*p.get_dbL0C()),(unsigned long long)l0c,p.get_depthA1(),p.get_depthB1(),p.get_dbL0A(),p.get_dbL0B(),p.get_dbL0C());
    data.set_localMemSize(temp); c->SetBlockDim(1); c->SetTilingKey(0);
    data.SaveToBuffer(c->GetRawTilingData()->GetData(),c->GetRawTilingData()->GetCapacity());
    c->GetRawTilingData()->SetDataSize(data.GetDataSize());
    c->GetWorkspaceSizes(1)[0]=platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext *c) {
    if(!Shape(c->GetInputShape(0),{16,128}) || !Shape(c->GetInputShape(2),{1,64}) ||
       !Shape(c->GetInputShape(3),{1,64})) return GRAPH_FAILED;
    auto y=c->GetOutputShape(0); y->SetDimNum(2); y->SetDim(0,16); y->SetDim(1,64); return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *c) {c->SetOutputDataType(0,DT_FLOAT16);return GRAPH_SUCCESS;}
}
namespace ops {
class DFlashAntiQuantK128: public OpDef {
public:
    explicit DFlashAntiQuantK128(const char *name):OpDef(name) {
        Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        Input("w_nz").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_FRACTAL_NZ});
        Input("scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        Input("offset").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend310p");
    }
};
OP_ADD(DFlashAntiQuantK128);
}
