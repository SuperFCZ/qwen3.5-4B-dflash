#include "target_gate_up_panel_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <cstdio>

namespace {
constexpr int K = 2560, N = 9216;
constexpr int Alignment = 32;
uint32_t Align(uint32_t n) { return (n + Alignment - 1) / Alignment * Alignment; }

ge::graphStatus Infer(gert::InferShapeContext* ctx) {
    const auto* x = ctx->GetInputShape(0);
    if (!x || x->GetDimNum() != 3) return ge::GRAPH_FAILED;
    for (int i = 0; i < 2; ++i) {
        auto* y = ctx->GetOutputShape(i);
        if (!y) return ge::GRAPH_FAILED;
        y->SetDimNum(3);
        y->SetDim(0, 1); y->SetDim(1, x->GetDim(1)); y->SetDim(2, N);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Types(gert::InferDataTypeContext* ctx) {
    ctx->SetOutputDataType(0, ge::DT_FLOAT16);
    ctx->SetOutputDataType(1, ge::DT_FLOAT16);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tile(gert::TilingContext* ctx) {
    using namespace matmul_tiling;
    if (!ctx->GetPlatformInfo() || !ctx->GetAttrs()) return ge::GRAPH_FAILED;
    for (size_t i = 0; i < 5; ++i) if (!ctx->GetInputShape(i)) return ge::GRAPH_FAILED;
    const auto& x = ctx->GetInputShape(0)->GetStorageShape();
    const auto& b = ctx->GetInputShape(1)->GetStorageShape();
    if (x.GetDimNum()!=3 || b.GetDimNum()!=5) return ge::GRAPH_FAILED;
    const int m = x.GetDim(1), n = b.GetDim(1)*32;
    const bool packedA = x.GetDim(0) == K/32 && x.GetDim(2) == 32;
    if ((!packedA && (x.GetDim(0)!=1 || x.GetDim(2)!=K)) || (m!=16 && m!=64) ||
        (n!=128 && n!=256) || b.GetDim(0)!=2*N/n || b.GetDim(2)!=K/16 ||
        b.GetDim(3)!=16 || b.GetDim(4)!=32) return ge::GRAPH_FAILED;
    for (size_t i = 2; i < 5; ++i) {
        const auto& s = ctx->GetInputShape(i)->GetStorageShape();
        if (s.GetDimNum()!=1 || s.GetDim(0)!=(i==2 ? m : 1)) return ge::GRAPH_FAILED;
    }
    const auto* attr = ctx->GetAttrs();
    const auto* dq = attr->GetAttrPointer<int64_t>(0);
    const auto* pf = attr->GetAttrPointer<int64_t>(1);
    const int cubeDequant = dq ? *dq : 1, prefetch = pf ? *pf : 0;
    if (cubeDequant<0 || cubeDequant>1 || prefetch<0 || prefetch>1 || (prefetch && (n!=128 || !cubeDequant))) return ge::GRAPH_FAILED;
    auto platform = platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
    uint64_t ub=0,l1=0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
    const uint32_t cores=std::min<uint32_t>(8,platform.GetCoreNumAic());
    const uint32_t userUB = (cubeDequant ? 6 : 12)*m*n + m*(4+6*16) + 1024;
    const uint32_t userL1 = m*K + (prefetch ? 2 : 1)*n*K;
    if (!cores || ub<=userUB || l1<=userL1) return ge::GRAPH_FAILED;
    optiling::TargetGateUpPanelTiling data;
    MatmulApiTiling t(platform);
    t.SetAType(TPosition::A1, CubeFormat::NZ, DataType::DT_INT8);
    t.SetBType(TPosition::B1, CubeFormat::NZ, DataType::DT_INT8);
    t.SetCType(TPosition::VECIN, CubeFormat::NZ, cubeDequant ? DataType::DT_FLOAT16 : DataType::DT_INT32);
    if (cubeDequant) t.SetDequantType(DequantType::SCALAR);
    t.SetBiasType(TPosition::GM, CubeFormat::ND, DataType::DT_INT32);
    t.SetBias(false);
    t.SetOrgShape(m,n,K);
    t.SetShape(m,n,K);
    t.SetFixSplit(m,n,128);
    // The full A and B panels are already resident. An unnecessarily deep SDK
    // K schedule can demand more transformation scratch than fits in UB.
    const int sdkL1 = static_cast<int>(std::min<uint64_t>(l1-userL1,128*1024));
    t.SetBufferSpace(sdkL1, -1, static_cast<int>(ub-userUB));
    if (t.GetTiling(data.cube)!=0 || static_cast<int>(data.cube.get_baseM())!=m ||
        static_cast<int>(data.cube.get_baseN())!=n || static_cast<int>(data.cube.get_singleCoreK())!=K) {
        std::fprintf(stderr,"TargetGateUpPanel unsupported Cube tiling M=%d N=%d prefetch=%d\n",m,n,prefetch);
        return ge::GRAPH_FAILED;
    }
    SysTilingTempBufSize tmp{};
    if (MatmulGetTmpBufSize(data.cube,tmp)!=0 || tmp.ubSize<0 || tmp.l1Size<0 ||
        tmp.l1Size>sdkL1 || userUB+Align(tmp.ubSize)>ub)
        return ge::GRAPH_FAILED;
    data.set_rows(m); data.set_tileN(n); data.set_cores(cores); data.set_packedA(packedA);
    data.set_prefetchB(prefetch); data.set_cubeDequant(cubeDequant); data.set_localBytes(Align(tmp.ubSize));
    ctx->SetBlockDim(cores); ctx->SetTilingKey(1);
    auto* work=ctx->GetWorkspaceSizes(1);
    auto* raw=ctx->GetRawTilingData();
    if (!work || !raw || raw->GetCapacity()<data.GetDataSize()) return ge::GRAPH_FAILED;
    work[0]=platform.GetLibApiWorkSpaceSize();
    data.SaveToBuffer(raw->GetData(),raw->GetCapacity());
    raw->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class TargetGateUpPanel : public OpDef {
public:
    explicit TargetGateUpPanel(const char* name) : OpDef(name) {
        for (const char* p : {"x","panels"})
            this->Input(p).ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        for (const char* p : {"row_scale","gate_scale","up_scale"})
            this->Input(p).ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        for (const char* p : {"gate","up"})
            this->Output(p).ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("cube_dequant").AttrType(OPTIONAL).Int(1);
        this->Attr("prefetch_b").AttrType(OPTIONAL).Int(0);
        this->SetInferShape(Infer).SetInferDataType(Types);
        this->AICore().SetTiling(Tile);
        this->AICore().AddConfig("ascend310p");
    }
};
OP_ADD(TargetGateUpPanel);
}
