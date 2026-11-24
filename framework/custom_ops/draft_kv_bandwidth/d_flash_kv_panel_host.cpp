#include "kv_spec.h"
#include "d_flash_kv_panel_tiling.h"
#include "d_flash_group_quant_linear_contract.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#define DFLASH_SCHEDULER_POLICY 0
#include "scheduler_evidence.h"

namespace {
using namespace kv_panel;
using draft_quant_contract::IsShape;

template<class S> uint32_t Rows(const S* x)
{
    if (!x) return 0;
    constexpr bool packed = DFLASH_KV_INPUT == 2;
    if (x->GetDimNum() != (packed ? 3 : 2)) return 0;
    const int64_t m = x->GetDim(packed ? 1 : 0);
    if (m != 32 && m != 80) return 0;
    return (packed ? IsShape(x,{K/16,m,16}) : IsShape(x,{m,K})) ? m : 0;
}

template<class Shape, class Desc>
bool ND(const Shape* s, const Desc* d, ge::DataType dt,
        std::initializer_list<int64_t> dims)
{
    return s && d && d->GetDataType()==dt && d->GetOriginFormat()==ge::FORMAT_ND &&
           d->GetStorageFormat()==ge::FORMAT_ND && IsShape(&s->GetStorageShape(), dims) &&
           IsShape(&s->GetOriginShape(), dims);
}
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* ctx)
{
    using namespace kv_panel;
    const auto* x = ctx->GetInputShape(0);
    if (!x) return ge::GRAPH_FAILED;
    const auto m = Rows(&x->GetStorageShape());
    if (!m || Rows(&x->GetOriginShape()) != m ||
        !ND(x, ctx->GetInputDesc(0), ge::DT_FLOAT16,
            DFLASH_KV_INPUT==2 ? std::initializer_list<int64_t>{K/16,m,16} : std::initializer_list<int64_t>{m,K}) ||
        !ND(ctx->GetInputShape(1),ctx->GetInputDesc(1),ge::DT_INT8,{Tiles,1,K*Columns}) ||
        !ND(ctx->GetInputShape(2),ctx->GetInputDesc(2),ge::DT_FLOAT16,{Tiles,Groups,Columns}) ||
        !ND(ctx->GetOutputShape(0),ctx->GetOutputDesc(0),ge::DT_FLOAT16,{m,N})) return ge::GRAPH_FAILED;
    platform_ascendc::PlatformAscendC platform(ctx->GetPlatformInfo());
    const uint32_t cores = platform.GetCoreNum();
    dflash_scheduler::Plan sched;
    if (!dflash_scheduler::Make(platform, cores, draft_quant_contract::LaunchBlockCount(N,cores,0),
                                Tiles, 0, sched)) return ge::GRAPH_FAILED;
    using Mem = platform_ascendc::CoreMemType;
    auto cap = [&](Mem which) { uint64_t b=0; platform.GetCoreMemSize(which,b); return b; };
    if (cap(Mem::UB)!=262144 || cap(Mem::L1)<=ResidentL1(m)) return ge::GRAPH_FAILED;
    const uint64_t scratch = cap(Mem::UB)-UserBytes;
    using namespace matmul_tiling;
    MatmulApiTiling mm(platform);
    mm.SetAType(TPosition::A1,CubeFormat::NZ,DataType::DT_FLOAT16);
    mm.SetBType(TPosition::B1,CubeFormat::NZ,DataType::DT_FLOAT16,false);
    mm.SetCType(TPosition::GM,CubeFormat::ND,DataType::DT_FLOAT16);
    mm.SetBiasType(TPosition::GM,CubeFormat::ND,DataType::DT_FLOAT);
    mm.SetBias(false);
    mm.SetShape(m,Columns,K);
    mm.SetOrgShape(m,N,K,K);
    mm.SetFixSplit(m,Columns,BaseK(m));
    mm.SetBufferSpace(static_cast<int>(cap(Mem::L1)-ResidentL1(m)), -1, static_cast<int>(scratch));
    DFlashKvPanelTilingData td;
    auto& cube = td.cubeTilingData;
    if (mm.GetTiling(cube)!=0 || cube.get_singleCoreM()!=m || cube.get_singleCoreN()!=Columns ||
        cube.get_singleCoreK()!=K || cube.get_baseM()!=m || cube.get_baseN()!=Columns ||
        cube.get_baseK()!=BaseK(m) || cube.get_stepKa()<1 || cube.get_stepKb()<1 ||
        cube.get_depthA1()<1 || cube.get_depthB1()<1 || cube.get_dbL0A()<1 ||
        cube.get_dbL0B()<1 || cube.get_dbL0C()!=1 || cube.get_transLength()<=0 ||
        static_cast<uint64_t>(cube.get_transLength())>scratch) return ge::GRAPH_FAILED;
    const uint64_t a=2*m*BaseK(m), b=2*Columns*BaseK(m);
    const uint64_t l1=ResidentL1(m)+a*cube.get_depthA1()+b*cube.get_depthB1();
    const uint64_t l0a=a*cube.get_dbL0A(), l0b=b*cube.get_dbL0B(), l0c=4*m*Columns;
    if (l1>cap(Mem::L1) || l0a>cap(Mem::L0_A) || l0b>cap(Mem::L0_B) || l0c>cap(Mem::L0_C))
        return ge::GRAPH_FAILED;
    td.set_globalM(m); td.set_globalK(K); td.set_globalN(N); td.set_tileK(K); td.set_matmulUbBytes(scratch);
    ctx->SetTilingKey(0); ctx->SetBlockDim(sched.selected);
    auto* raw=ctx->GetRawTilingData();
    if (!raw || raw->GetCapacity()<td.GetDataSize()) return ge::GRAPH_FAILED;
    td.SaveToBuffer(raw->GetData(),raw->GetCapacity()); raw->SetDataSize(td.GetDataSize());
    const size_t workspace=platform.GetLibApiWorkSpaceSize();
    ctx->GetWorkspaceSizes(1)[0]=workspace;
    if (!dflash_platform_diag::Write(ctx->GetPlatformInfo(),platform,m,K,N,cores,sched.selected,0,workspace) ||
        !dflash_scheduler::Write(ctx,m,K,N,sched,cores,0,UserBytes,scratch,l1,l0a,l0b,l0c,
                                 CodeBytes,ScaleBytes,WeightBytes,RawBanks,K,td.GetDataSize()))
        return ge::GRAPH_FAILED;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* ctx)
{
    using namespace kv_panel;
    const auto m=Rows(ctx->GetInputShape(0));
    auto* y=ctx->GetOutputShape(0);
    if (!m || !y || !IsShape(ctx->GetInputShape(1),{Tiles,1,K*Columns}) ||
        !IsShape(ctx->GetInputShape(2),{Tiles,Groups,Columns})) return GRAPH_FAILED;
    y->SetDimNum(2); y->SetDim(0,m); y->SetDim(1,N);
    return GRAPH_SUCCESS;
}
static graphStatus InferType(gert::InferDataTypeContext* ctx)
{
    if (ctx->GetInputDataType(0)!=DT_FLOAT16 || ctx->GetInputDataType(1)!=DT_INT8 ||
        ctx->GetInputDataType(2)!=DT_FLOAT16) return GRAPH_FAILED;
    ctx->SetOutputDataType(0,DT_FLOAT16); return GRAPH_SUCCESS;
}
}
namespace ops {
class DFlashKvPanel : public OpDef {
public:
    explicit DFlashKvPanel(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("w_tiles").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND});
        this->Input("s").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend310p");
    }
};
OP_ADD(DFlashKvPanel);
}
