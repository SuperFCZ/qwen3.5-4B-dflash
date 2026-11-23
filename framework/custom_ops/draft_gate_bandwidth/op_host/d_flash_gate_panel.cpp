#include "panel_spec.h"
#include "d_flash_gate_panel_tiling.h"
#include "d_flash_group_quant_linear_contract.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#define DFLASH_SCHEDULER_POLICY 0
#include "scheduler_evidence.h"

namespace {
using namespace gate_panel;
using draft_quant_contract::IsShape;

bool Input(gert::TilingContext* ctx, uint32_t slot, ge::DataType type, std::initializer_list<int64_t> dims)
{
    const auto* shape = ctx->GetInputShape(slot);
    const auto* desc = ctx->GetInputDesc(slot);
    return shape && desc && desc->GetDataType() == type && desc->GetOriginFormat() == ge::FORMAT_ND &&
        desc->GetStorageFormat() == ge::FORMAT_ND && IsShape(&shape->GetOriginShape(), dims) &&
        IsShape(&shape->GetStorageShape(), dims);
}
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* ctx)
{
    using namespace gate_panel;
    const auto* out = ctx->GetOutputShape(0);
    const auto* dst = ctx->GetOutputDesc(0);
    if (!Input(ctx, 0, ge::DT_FLOAT16, {M,K}) ||
        !Input(ctx, 1, ge::DT_INT8, {Tiles,Panels,PanelK*Columns}) ||
        !Input(ctx, 2, ge::DT_FLOAT16, {Tiles,K/Group,Columns}) ||
        !out || !dst || dst->GetDataType()!=ge::DT_FLOAT16 || dst->GetStorageFormat()!=ge::FORMAT_ND ||
        !IsShape(&out->GetStorageShape(), {M,N}) || !IsShape(&out->GetOriginShape(), {M,N}))
        return ge::GRAPH_FAILED;
    platform_ascendc::PlatformAscendC platform(ctx->GetPlatformInfo());
    const uint32_t cores = platform.GetCoreNum();
    const uint32_t launch = draft_quant_contract::LaunchBlockCount(N, cores, 0);
    dflash_scheduler::Plan sched;
    if (!dflash_scheduler::Make(platform, cores, launch, Tiles, 0, sched)) return ge::GRAPH_FAILED;
    using Mem = platform_ascendc::CoreMemType;
    auto capacity = [&](Mem type) {
        uint64_t bytes=0;
        platform.GetCoreMemSize(type, bytes);
        return bytes;
    };
    const uint64_t ub = capacity(Mem::UB);
    if (ub != 262144 || ub <= UserBytes) return ge::GRAPH_FAILED;
    const uint64_t scratch = ub - UserBytes;
    using namespace matmul_tiling;
    MatmulApiTiling mm(platform);
    mm.SetAType(FullL1 ? TPosition::A1 : TPosition::GM, FullL1 ? CubeFormat::NZ : CubeFormat::ND, DataType::DT_FLOAT16);
    mm.SetBType(FullL1 ? TPosition::B1 : TPosition::VECOUT, CubeFormat::NZ, DataType::DT_FLOAT16, false);
    mm.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    mm.SetBiasType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    mm.SetBias(false);
    mm.SetShape(M, Columns, PanelK);
    mm.SetOrgShape(M, N, K, PanelK);
    mm.SetFixSplit(M, Columns, 256);
    const auto l1Capacity = capacity(Mem::L1);
    if (l1Capacity <= ResidentL1) return ge::GRAPH_FAILED;
    mm.SetBufferSpace(FullL1 ? static_cast<int>(l1Capacity-ResidentL1) : -1, -1, static_cast<int>(scratch));
    DFlashGatePanelTilingData td;
    auto& cube = td.cubeTilingData;
    if (mm.GetTiling(cube) != 0 || cube.get_singleCoreM() != M || cube.get_singleCoreN() != Columns ||
        cube.get_singleCoreK() != PanelK || cube.get_baseM() != M || cube.get_baseN() != Columns ||
        cube.get_baseK() != 256 || cube.get_stepKa() != PanelK/256 || cube.get_stepKb() != PanelK/256 ||
        cube.get_depthA1() < 1 || cube.get_depthB1() < 1 ||
        cube.get_dbL0A() < 1 || cube.get_dbL0B() < 1 || cube.get_dbL0C() != 1 ||
        cube.get_transLength() <= 0 || static_cast<uint64_t>(cube.get_transLength()) > scratch) {
        std::fprintf(stderr, "Gate panel SDK plan rejected: panel=%u base=%d/%d/%d scratch=%lu trans=%d\n",
                     PanelK,cube.get_baseM(),cube.get_baseN(),cube.get_baseK(),scratch,cube.get_transLength());
        return ge::GRAPH_FAILED;
    }
    const uint64_t a = M*256*2, b = Columns*256*2, c = M*Columns*4;
    const uint64_t l1 = ResidentL1 + a*cube.get_depthA1() + b*cube.get_depthB1();
    const uint64_t l0a = a*cube.get_dbL0A(), l0b = b*cube.get_dbL0B(), l0c = c*cube.get_dbL0C();
    if (l1>l1Capacity || l0a>capacity(Mem::L0_A) || l0b>capacity(Mem::L0_B) ||
        l0c>capacity(Mem::L0_C)) return ge::GRAPH_FAILED;
    td.set_globalM(M); td.set_globalK(K); td.set_globalN(N); td.set_tileK(PanelK); td.set_matmulUbBytes(scratch);
    ctx->SetTilingKey(0);
    ctx->SetBlockDim(sched.selected);
    auto* raw = ctx->GetRawTilingData();
    if (!raw || raw->GetCapacity() < td.GetDataSize()) return ge::GRAPH_FAILED;
    td.SaveToBuffer(raw->GetData(),raw->GetCapacity());
    raw->SetDataSize(td.GetDataSize());
    const size_t workspace = platform.GetLibApiWorkSpaceSize();
    ctx->GetWorkspaceSizes(1)[0] = workspace;
    if (!dflash_platform_diag::Write(ctx->GetPlatformInfo(),platform,M,K,N,cores,sched.selected,0,workspace) ||
        !dflash_scheduler::Write(ctx,M,K,N,sched,cores,0,UserBytes,scratch,l1,l0a,l0b,l0c,
                                 CodeBytes,ScaleBytes,WeightBytes,RawBanks,PanelK,td.GetDataSize()))
        return ge::GRAPH_FAILED;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* ctx)
{
    using namespace gate_panel;
    auto* y=ctx->GetOutputShape(0);
    if (!y || !IsShape(ctx->GetInputShape(0),{M,K}) ||
        !IsShape(ctx->GetInputShape(1),{Tiles,Panels,PanelK*Columns}) ||
        !IsShape(ctx->GetInputShape(2),{Tiles,K/Group,Columns})) return GRAPH_FAILED;
    y->SetDimNum(2); y->SetDim(0,M); y->SetDim(1,N);
    return GRAPH_SUCCESS;
}
static graphStatus InferType(gert::InferDataTypeContext* ctx)
{
    if (ctx->GetInputDataType(0)!=DT_FLOAT16 || ctx->GetInputDataType(1)!=DT_INT8 ||
        ctx->GetInputDataType(2)!=DT_FLOAT16) return GRAPH_FAILED;
    ctx->SetOutputDataType(0,DT_FLOAT16);
    return GRAPH_SUCCESS;
}
}
namespace ops {
class DFlashGatePanel : public OpDef {
public:
    explicit DFlashGatePanel(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("w_tiles").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND});
        this->Input("s").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend310p");
    }
};
OP_ADD(DFlashGatePanel);
}
