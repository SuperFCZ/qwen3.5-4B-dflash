#pragma once
#include <cstdio>
#include <initializer_list>
#include "head_contract.h"
#include "head_config.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace head_host {
using namespace head_contract;
template<class S> bool Shape(const S* s, std::initializer_list<int64_t> dims) {
    if(!s || s->GetDimNum()!=dims.size()) return false;
    size_t i=0; for(auto d:dims) if(s->GetDim(i++)!=d) return false;
    return true;
}
inline bool Input(gert::TilingContext* c, size_t i, std::initializer_list<int64_t> dims) {
    const auto* s=c->GetInputShape(i); const auto* d=c->GetInputDesc(i);
    return s && d && d->GetDataType()==ge::DT_FLOAT16 && d->GetOriginFormat()==ge::FORMAT_ND &&
        d->GetStorageFormat()==ge::FORMAT_ND && Shape(&s->GetOriginShape(),dims) && Shape(&s->GetStorageShape(),dims);
}
template<class T, bool Audit> ge::graphStatus Tiling(gert::TilingContext* c) {
    const auto* x=c->GetInputShape(0);
    if(!x || x->GetOriginShape().GetDimNum()!=2) return ge::GRAPH_FAILED;
    const int64_t m=x->GetOriginShape().GetDim(0);
    if(m<1 || m>15 || !Input(c,0,{m,kK}) || !Input(c,1,{kN,kK})) return ge::GRAPH_FAILED;
    const auto* y=c->GetOutputShape(0); const auto* yd=c->GetOutputDesc(0);
    if(!y || !yd || yd->GetDataType()!=ge::DT_INT64 || yd->GetStorageFormat()!=ge::FORMAT_ND ||
       !Shape(&y->GetOriginShape(),{m}) || !Shape(&y->GetStorageShape(),{m})) return ge::GRAPH_FAILED;
    if constexpr(Audit) {
        const auto* s=c->GetOutputShape(1); const auto* d=c->GetOutputDesc(1);
        if(!s || !d || d->GetDataType()!=ge::DT_FLOAT16 || d->GetStorageFormat()!=ge::FORMAT_ND ||
           !Shape(&s->GetOriginShape(),{m,kN}) || !Shape(&s->GetStorageShape(),{m,kN})) return ge::GRAPH_FAILED;
    }
    auto platform=platform_ascendc::PlatformAscendC(c->GetPlatformInfo());
    const uint32_t p=Partitions(platform.GetCoreNum(),DFLASH_HEAD_CORE_LIMIT);
    if(!p || p>platform.GetCoreNum()) return ge::GRAPH_FAILED;
    // Soft SyncAll must run all participating cores concurrently, even across streams.
    if(c->SetScheduleMode(1)!=ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    uint64_t ub=0; platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
    if(ub<=UserUbBytes(p)) return ge::GRAPH_FAILED;
    const uint64_t mmUb=(ub-UserUbBytes(p))/32*32;
    matmul_tiling::MatmulApiTiling mm(platform);
    mm.SetAType(matmul_tiling::TPosition::VECOUT,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_FLOAT16);
    mm.SetBType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_FLOAT16,true);
    mm.SetCType(matmul_tiling::TPosition::VECIN,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_FLOAT16);
    mm.SetBiasType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_FLOAT);
    mm.SetBias(false);
    mm.SetShape(16,64,kK); mm.SetOrgShape(16,64,kK,kK); mm.SetFixSplit(16,64,128);
    mm.SetBufferSpace(-1,-1,static_cast<int32_t>(mmUb));
    T t;
    if(mm.GetTiling(t.cubeTilingData)==-1) return ge::GRAPH_FAILED;
    auto& cube=t.cubeTilingData; // CANN 9.0 getters are non-const.
    if(cube.get_singleCoreM()!=16 || cube.get_singleCoreN()!=64 || cube.get_singleCoreK()!=kK ||
       cube.get_baseM()!=16 || cube.get_baseN()!=64 || cube.get_baseK()<=0 ||
       cube.get_baseK()%16 || cube.get_baseK()>kK) {
        std::fprintf(stderr,"HeadTop1: incompatible full-K single-output-block Matmul tiling\n");
        return ge::GRAPH_FAILED;
    }
    t.set_m(m); t.set_partitions(p); t.set_matmulUbBytes(mmUb);
    c->SetTilingKey(0); c->SetBlockDim(p);
    if(c->GetRawTilingData()->GetCapacity()<t.GetDataSize()) return ge::GRAPH_FAILED;
    t.SaveToBuffer(c->GetRawTilingData()->GetData(),c->GetRawTilingData()->GetCapacity());
    c->GetRawTilingData()->SetDataSize(t.GetDataSize());
    const uint64_t system=platform.GetLibApiWorkSpaceSize();
    c->GetWorkspaceSizes(1)[0]=system+WorkspaceBytes(p);
    std::fprintf(stdout,"DFLASH_HEAD_LAUNCH {\"version\":1,\"m\":%lld,\"k\":2560,\"n\":248320,"
        "\"tile_n\":64,\"padded_m\":16,\"partitions\":%u,\"available_cores\":%u,\"core_limit\":%u,"
        "\"audit\":%s,\"user_workspace_bytes\":%llu,\"system_workspace_bytes\":%llu,"
        "\"user_ub_bytes\":%llu,\"matmul_ub_bytes\":%llu,\"base_k\":%d,\"single_k\":2560,\"batchmode\":1}\n",
        static_cast<long long>(m),p,platform.GetCoreNum(),DFLASH_HEAD_CORE_LIMIT,Audit?"true":"false",
        static_cast<unsigned long long>(WorkspaceBytes(p)),static_cast<unsigned long long>(system),
        static_cast<unsigned long long>(UserUbBytes(p)),static_cast<unsigned long long>(mmUb),cube.get_baseK());
    return ge::GRAPH_SUCCESS;
}
template<bool Audit> ge::graphStatus InferShape(gert::InferShapeContext* c) {
    const auto* x=c->GetInputShape(0);
    if(!x || x->GetDimNum()!=2 || x->GetDim(0)<1 || x->GetDim(0)>15 ||
       !Shape(x,{x->GetDim(0),kK}) || !Shape(c->GetInputShape(1),{kN,kK})) return ge::GRAPH_FAILED;
    auto* y=c->GetOutputShape(0); y->SetDimNum(1); y->SetDim(0,x->GetDim(0));
    if constexpr(Audit) { auto* z=c->GetOutputShape(1); z->SetDimNum(2); z->SetDim(0,x->GetDim(0)); z->SetDim(1,kN); }
    return ge::GRAPH_SUCCESS;
}
template<bool Audit> ge::graphStatus InferType(gert::InferDataTypeContext* c) {
    if(c->GetInputDataType(0)!=ge::DT_FLOAT16 || c->GetInputDataType(1)!=ge::DT_FLOAT16) return ge::GRAPH_FAILED;
    c->SetOutputDataType(0,ge::DT_INT64);
    if constexpr(Audit) c->SetOutputDataType(1,ge::DT_FLOAT16);
    return ge::GRAPH_SUCCESS;
}
}
