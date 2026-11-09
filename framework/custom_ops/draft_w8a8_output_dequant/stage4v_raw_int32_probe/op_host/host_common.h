#pragma once
#include <cstdio>
#include <initializer_list>
#include <algorithm>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "core_probe.h"
#include "t1_contract.h"
inline ge::graphStatus S4RAWReject(const char *why) {
    std::fprintf(stderr,"STAGE4RAW_TILING_REJECT: %s\n",why);return ge::GRAPH_FAILED;
}
inline bool S4RAWDims(const gert::Shape *s,std::initializer_list<int64_t> dims) {
    if(!s||s->GetDimNum()!=dims.size())return false;
    size_t i=0;for(auto v:dims)if(s->GetDim(i++)!=v)return false;return true;
}
inline bool S4RAWInput(gert::TilingContext *c,size_t i,ge::DataType dtype,std::initializer_list<int64_t> dims) {
    const auto s=c->GetInputShape(i);const auto d=c->GetInputDesc(i);
    return s&&d&&d->GetDataType()==dtype&&d->GetStorageFormat()==ge::FORMAT_ND&&
        S4RAWDims(&s->GetOriginShape(),dims)&&S4RAWDims(&s->GetStorageShape(),dims);
}
inline ge::graphStatus S4RAWShape(gert::InferShapeContext *c) {
    if(!S4RAWDims(c->GetInputShape(0),{16,2560})||!S4RAWDims(c->GetInputShape(1),{2560,256}))return ge::GRAPH_FAILED;
    auto y=c->GetOutputShape(0);y->SetDimNum(2);y->SetDim(0,16);y->SetDim(1,256);return ge::GRAPH_SUCCESS;
}
template<class Tiling>
ge::graphStatus S4RAWTiling(gert::TilingContext *c) {
    if(!S4RAWInput(c,0,ge::DT_INT8,{16,2560}))return ge::GRAPH_FAILED;
    const auto b=c->GetInputShape(1);const auto bd=c->GetInputDesc(1);
    if(!b||!bd||bd->GetDataType()!=ge::DT_INT8)return ge::GRAPH_FAILED;
    if(!S4RAWInput(c,1,ge::DT_INT8,{2560,256}))return ge::GRAPH_FAILED;
    const int64_t modeValue=0,limitValue=1,tileValue=0;
    const auto mode=&modeValue;const auto limit=&limitValue;const auto tile=&tileValue;
    const auto y=c->GetOutputShape(0);const auto yd=c->GetOutputDesc(0);
    if(!y||!yd||yd->GetDataType()!=ge::DT_INT32||yd->GetStorageFormat()!=ge::FORMAT_ND||
       !S4RAWDims(&y->GetOriginShape(),{16,256})||!S4RAWDims(&y->GetStorageShape(),{16,256}))return ge::GRAPH_FAILED;
    if(!c->GetPlatformInfo())return S4RAWReject("missing formal TilingContext platform");
    auto platform=platform_ascendc::PlatformAscendC(c->GetPlatformInfo());
    const auto coreProbe=S4RAWProbeCores(platform);
    const uint32_t cores=S4RAWSelectSchedulingLimit(coreProbe);
    if(!cores)return S4RAWReject("requires dav_2002, nonzero aic/aiv/all, aic==aiv and all<=aic");
    const uint32_t blocks=1;
    if(blocks>cores)return S4RAWReject("requested blocks exceed available AICores");
    const uint64_t system=platform.GetLibApiWorkSpaceSize();
    // A full platform-sized arena per core; never infer an undocumented per-core fraction.
    const uint64_t arena=(system+511)/512*512,stride=arena+1024;
    const uint64_t userWorkspace=blocks*stride,totalWorkspace=system+userWorkspace;
    if(!system||totalWorkspace>512ULL*1024*1024)return S4RAWReject("invalid/over-budget system arena reservation");
    std::fprintf(stderr,"STAGE4RAW_SCHEDULE coreLimit=%lld blocks=%u cores=%u system=%llu arena=%llu stride=%llu user=%llu total=%llu mode=%lld\n",
       (long long)*limit,blocks,cores,(unsigned long long)system,(unsigned long long)arena,(unsigned long long)stride,
       (unsigned long long)userWorkspace,(unsigned long long)totalWorkspace,(long long)*mode);
    uint64_t ub=0,l1=0,l0a=0,l0b=0,l0c=0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1,l1);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A,l0a);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B,l0b);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C,l0c);
    if(ub!=256*1024) { std::fprintf(stderr,"STAGE4RAW actual UB=%llu\n",(unsigned long long)ub);return S4RAWReject("unexpected UB capacity"); }
    // Partial16384 + Arena temporary512; three redzone pairs; retain8192 Reserve.
    constexpr uint64_t userBuffers=16896, redzones=1536, reserve=8192;
    uint64_t scratch=202240; // fixed ceiling, unused space is NOT allocated
    if(scratch!=202240 || scratch+userBuffers+redzones+reserve>ub)return S4RAWReject("UB budget");
    using namespace matmul_tiling;
    MatmulApiTiling mm(platform);
    if(mm.SetAType(TPosition::GM,CubeFormat::ND,DataType::DT_INT8)==-1)return S4RAWReject("SetAType INT8 ND");
    if(mm.SetBType(TPosition::GM,CubeFormat::ND,DataType::DT_INT8,false)==-1)return S4RAWReject("SetBType INT8 ND non-transposed");
    if(mm.SetCType(TPosition::VECIN,CubeFormat::NZ,DataType::DT_INT32)==-1)return S4RAWReject("SetCType DT_INT32 candidate");
    mm.SetBiasType(TPosition::GM,CubeFormat::ND,DataType::DT_INT32);mm.SetBias(false);
    mm.SetShape(16,256,2560);mm.SetOrgShape(16,256,2560);mm.SetFixSplit(16,256,128);
    mm.SetBufferSpace(-1,-1,static_cast<int32_t>(scratch));
    Tiling td;if(mm.GetTiling(td.cubeTilingData)!=0) {
        std::fprintf(stderr,"STAGE4RAW GetTiling failed: Host C=DT_INT32, dequant=NONE; no automatic fallback\n");
        return ge::GRAPH_FAILED;
    }
    auto &p=td.cubeTilingData;
    if(!S4RAWAcceptedT1(p,ub,l1,l0a,l0b,l0c,system,scratch))return S4RAWReject("actual TilingContext differs from accepted T1 resource contract");
    if(p.get_singleCoreM()!=16||p.get_singleCoreN()!=256||p.get_singleCoreK()!=2560||
       p.get_baseM()!=16||p.get_baseN()!=256||p.get_baseK()!=128) {
        std::fprintf(stderr,"STAGE4RAW rejected tiling single=%d/%d/%d base=%d/%d/%d\n",
            p.get_singleCoreM(),p.get_singleCoreN(),p.get_singleCoreK(),p.get_baseM(),p.get_baseN(),p.get_baseK());
        return ge::GRAPH_FAILED;
    }
    const uint64_t a=16*128,bbytes=256*128,acc=16*256*4;
    if(p.get_depthA1()<1||p.get_depthB1()<1||p.get_dbL0A()<1||p.get_dbL0B()<1||p.get_dbL0C()<1||
       a*p.get_depthA1()+bbytes*p.get_depthB1()>l1||a*p.get_dbL0A()>l0a||
       bbytes*p.get_dbL0B()>l0b||acc*p.get_dbL0C()+p.get_shareL0CSize()>l0c)return S4RAWReject("L1/L0 depth or capacity check");
    // Raw INT32: two B conversion views and the INT32 C view; no Scale storage.
    if(p.get_transLength()<0 || p.get_transLength()%32!=0 || static_cast<uint64_t>(p.get_transLength())>scratch ||
       2*bbytes+16*256*4>scratch) {
        std::fprintf(stderr,"STAGE4RAW transLength=%d scratch=%llu\n",p.get_transLength(),(unsigned long long)scratch);
        return S4RAWReject("Raw UB workspace capacity");
    }
    std::fprintf(stderr,"STAGE4RAW_PLAN arch=2002 slots=1 logicalMatmuls=1 M=16 K=2560 baseK=128 tileN=256 outputN=256 HostC=VECIN_NZ_DT_INT32 dequant=NONE "
        "single=%d/%d/%d base=%d/%d/%d userUB=16896 ubGuards=1536 reserve=8192 scratch=%llu UB=%llu L1=%llu L0A=%llu L0B=%llu L0C=%llu "
        "depthA1=%d depthB1=%d dbL0A=%d dbL0B=%d dbL0C=%d transLength=%d\n",
        p.get_singleCoreM(),p.get_singleCoreN(),p.get_singleCoreK(),p.get_baseM(),p.get_baseN(),p.get_baseK(),
        (unsigned long long)scratch,(unsigned long long)ub,(unsigned long long)l1,
        (unsigned long long)l0a,(unsigned long long)l0b,(unsigned long long)l0c,
        p.get_depthA1(),p.get_depthB1(),p.get_dbL0A(),p.get_dbL0B(),p.get_dbL0C(),p.get_transLength());
    td.set_launchBlocks(blocks);td.set_systemBytes(system);td.set_arenaBytes(arena);td.set_arenaStride(stride);
    td.set_tileId(*tile);td.set_outputMode(*mode);td.set_localMemSize(scratch);c->SetBlockDim(blocks);c->SetTilingKey(0);
    if(!c->GetRawTilingData()||!c->GetRawTilingData()->GetData()||c->GetRawTilingData()->GetCapacity()<td.GetDataSize())return S4RAWReject("Tiling serialization capacity");
    td.SaveToBuffer(c->GetRawTilingData()->GetData(),c->GetRawTilingData()->GetCapacity());
    c->GetRawTilingData()->SetDataSize(td.GetDataSize());c->GetWorkspaceSizes(1)[0]=totalWorkspace;
    return ge::GRAPH_SUCCESS;
}
