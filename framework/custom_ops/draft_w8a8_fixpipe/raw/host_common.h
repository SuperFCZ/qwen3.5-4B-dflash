#pragma once
#include <cstdio>
#include <initializer_list>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
inline bool W8Dims(const gert::Shape *s,std::initializer_list<int64_t> dims) {
    if(!s||s->GetDimNum()!=dims.size())return false;
    size_t i=0;for(auto v:dims)if(s->GetDim(i++)!=v)return false;return true;
}
inline bool W8Input(gert::TilingContext *c,size_t i,ge::DataType dtype,std::initializer_list<int64_t> dims) {
    const auto s=c->GetInputShape(i);const auto d=c->GetInputDesc(i);
    return s&&d&&d->GetDataType()==dtype&&d->GetStorageFormat()==ge::FORMAT_ND&&
        W8Dims(&s->GetOriginShape(),dims)&&W8Dims(&s->GetStorageShape(),dims);
}
inline ge::graphStatus W8Shape(gert::InferShapeContext *c) {
    if(!W8Dims(c->GetInputShape(0),{16,128})||!W8Dims(c->GetInputShape(1),{128,64}))return ge::GRAPH_FAILED;
    auto y=c->GetOutputShape(0);y->SetDimNum(2);y->SetDim(0,16);y->SetDim(1,64);return ge::GRAPH_SUCCESS;
}
template<class Tiling>
ge::graphStatus W8Tiling(gert::TilingContext *c,bool raw) {
    if(!W8Input(c,0,ge::DT_INT8,{16,128}))return ge::GRAPH_FAILED;
    const auto b=c->GetInputShape(1);const auto bd=c->GetInputDesc(1);
    if(!b||!bd||bd->GetDataType()!=ge::DT_INT8)return ge::GRAPH_FAILED;
    if(!W8Input(c,1,ge::DT_INT8,{128,64}))return ge::GRAPH_FAILED;
    const auto y=c->GetOutputShape(0);const auto yd=c->GetOutputDesc(0);
    if(!y||!yd||yd->GetDataType()!=ge::DT_INT32||yd->GetStorageFormat()!=ge::FORMAT_ND||
       !W8Dims(&y->GetOriginShape(),{16,64})||!W8Dims(&y->GetStorageShape(),{16,64}))return ge::GRAPH_FAILED;
    auto platform=platform_ascendc::PlatformAscendC(c->GetPlatformInfo());
    uint64_t ub=0,l1=0,l0a=0,l0b=0,l0c=0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1,l1);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A,l0a);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B,l0b);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C,l0c);
    if(ub!=256*1024)return ge::GRAPH_FAILED;
    uint64_t scratch=(ub-8192)/32*32;
    using namespace matmul_tiling;
    MatmulApiTiling mm(platform);
    if(mm.SetAType(TPosition::GM,CubeFormat::ND,DataType::DT_INT8)==-1||
       mm.SetBType(TPosition::GM,CubeFormat::ND,DataType::DT_INT8,false)==-1||
       mm.SetCType(TPosition::GM,CubeFormat::ND,DataType::DT_INT32)==-1)return ge::GRAPH_FAILED;
    mm.SetBiasType(TPosition::GM,CubeFormat::ND,DataType::DT_INT32);mm.SetBias(false);
    mm.SetShape(16,64,128);mm.SetOrgShape(16,64,128);mm.SetFixSplit(16,64,128);
    mm.SetBufferSpace(-1,-1,static_cast<int32_t>(scratch));
    Tiling td;if(mm.GetTiling(td.cubeTilingData)==-1)return ge::GRAPH_FAILED;
    auto &p=td.cubeTilingData;
    if(p.get_singleCoreM()!=16||p.get_singleCoreN()!=64||p.get_singleCoreK()!=128||
       p.get_baseM()!=16||p.get_baseN()!=64||p.get_baseK()!=128)return ge::GRAPH_FAILED;
    const uint64_t a=16*128,bbytes=64*128,acc=16*64*4;
    if(p.get_depthA1()<1||p.get_depthB1()<1||p.get_dbL0A()<1||p.get_dbL0B()<1||p.get_dbL0C()<1||
       a*p.get_depthA1()+bbytes*p.get_depthB1()>l1||a*p.get_dbL0A()>l0a||
       bbytes*p.get_dbL0B()>l0b||acc*p.get_dbL0C()>l0c)return ge::GRAPH_FAILED;
    std::fprintf(stderr,"W8A8_PLAN raw=%d nz=%d blockDim=1 single=16/64/128 base=16/64/128 "
        "scratch=%llu UB=%llu L1=%llu L0A=%llu L0B=%llu L0C=%llu depthA1=%d depthB1=%d dbL0A=%d dbL0B=%d dbL0C=%d cType=%d\n",
        raw,0,(unsigned long long)scratch,(unsigned long long)ub,(unsigned long long)l1,
        (unsigned long long)l0a,(unsigned long long)l0b,(unsigned long long)l0c,
        p.get_depthA1(),p.get_depthB1(),p.get_dbL0A(),p.get_dbL0B(),p.get_dbL0C(),static_cast<int>(DataType::DT_INT32));
    td.set_localMemSize(scratch);c->SetBlockDim(1);c->SetTilingKey(0);
    td.SaveToBuffer(c->GetRawTilingData()->GetData(),c->GetRawTilingData()->GetCapacity());
    c->GetRawTilingData()->SetDataSize(td.GetDataSize());c->GetWorkspaceSizes(1)[0]=platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
