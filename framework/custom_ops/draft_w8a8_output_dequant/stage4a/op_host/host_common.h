#pragma once
#include <cstdio>
#include <initializer_list>
#include <algorithm>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
inline ge::graphStatus S4AReject(const char *why) {
    std::fprintf(stderr,"STAGE4A_TILING_REJECT: %s\n",why);return ge::GRAPH_FAILED;
}
inline bool S4ADims(const gert::Shape *s,std::initializer_list<int64_t> dims) {
    if(!s||s->GetDimNum()!=dims.size())return false;
    size_t i=0;for(auto v:dims)if(s->GetDim(i++)!=v)return false;return true;
}
inline bool S4AInput(gert::TilingContext *c,size_t i,ge::DataType dtype,std::initializer_list<int64_t> dims) {
    const auto s=c->GetInputShape(i);const auto d=c->GetInputDesc(i);
    return s&&d&&d->GetDataType()==dtype&&d->GetStorageFormat()==ge::FORMAT_ND&&
        S4ADims(&s->GetOriginShape(),dims)&&S4ADims(&s->GetStorageShape(),dims);
}
inline ge::graphStatus S4AShape(gert::InferShapeContext *c) {
    if(!S4ADims(c->GetInputShape(0),{20,16,128})||!S4ADims(c->GetInputShape(1),{8,20,128,64})||!S4ADims(c->GetInputShape(2),{8,20,64}))return ge::GRAPH_FAILED;
    auto y=c->GetOutputShape(0);y->SetDimNum(2);y->SetDim(0,16);y->SetDim(1,512);return ge::GRAPH_SUCCESS;
}
template<class Tiling>
ge::graphStatus S4ATiling(gert::TilingContext *c) {
    if(!S4AInput(c,0,ge::DT_INT8,{20,16,128})||!S4AInput(c,2,ge::DT_UINT64,{8,20,64}))return ge::GRAPH_FAILED;
    const auto b=c->GetInputShape(1);const auto bd=c->GetInputDesc(1);
    if(!b||!bd||bd->GetDataType()!=ge::DT_INT8)return ge::GRAPH_FAILED;
    if(!S4AInput(c,1,ge::DT_INT8,{8,20,128,64}))return ge::GRAPH_FAILED;
    const auto attrs=c->GetAttrs();
    const auto mode=attrs?attrs->GetAttrPointer<int64_t>(0):nullptr;
    if(!mode||*mode<0||*mode>21)return S4AReject("output_mode must be 0..21");
    std::fprintf(stderr,"STAGE4A_OUTPUT_MODE=%lld\n",(long long)*mode);
    const auto limit=attrs->GetAttrPointer<int64_t>(1);
    if(!limit||*limit<0||*limit>8)return S4AReject("core_limit must be 0..8");
    const auto y=c->GetOutputShape(0);const auto yd=c->GetOutputDesc(0);
    if(!y||!yd||yd->GetDataType()!=ge::DT_FLOAT16||yd->GetStorageFormat()!=ge::FORMAT_ND||
       !S4ADims(&y->GetOriginShape(),{16,512})||!S4ADims(&y->GetStorageShape(),{16,512}))return ge::GRAPH_FAILED;
    auto platform=platform_ascendc::PlatformAscendC(c->GetPlatformInfo());
    const uint32_t cores=platform.GetCoreNumAic();
    if(!cores || cores!=platform.GetCoreNum())return S4AReject("requires real coupled AI Core count");
    const uint32_t blocks=*limit?static_cast<uint32_t>(*limit):std::min(8U,cores);
    if(blocks>cores)return S4AReject("requested blocks exceed available AICores");
    const uint64_t system=platform.GetLibApiWorkSpaceSize();
    // A full platform-sized arena per core; never infer an undocumented per-core fraction.
    const uint64_t arena=(system+511)/512*512,stride=arena+1024;
    const uint64_t userWorkspace=blocks*stride,totalWorkspace=system+userWorkspace;
    if(!system||totalWorkspace>512ULL*1024*1024)return S4AReject("invalid/over-budget system arena reservation");
    std::fprintf(stderr,"STAGE4A_SCHEDULE coreLimit=%lld blocks=%u cores=%u system=%llu arena=%llu stride=%llu user=%llu total=%llu mode=%lld\n",
       (long long)*limit,blocks,cores,(unsigned long long)system,(unsigned long long)arena,(unsigned long long)stride,
       (unsigned long long)userWorkspace,(unsigned long long)totalWorkspace,(long long)*mode);
    uint64_t ub=0,l1=0,l0a=0,l0b=0,l0c=0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1,l1);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A,l0a);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B,l0b);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C,l0c);
    if(ub!=256*1024) { std::fprintf(stderr,"STAGE4A actual UB=%llu\n",(unsigned long long)ub);return S4AReject("unexpected UB capacity"); }
    // User buffers 12 KiB; five 256B prefix/suffix redzones; retain 8 KiB SDK headroom.
    constexpr uint64_t userBuffers=12288, redzones=2560, reserve=8192;
    uint64_t scratch=(ub-userBuffers-redzones-reserve)/256*256;
    if(scratch>241664 || scratch+userBuffers+redzones+reserve>ub)return S4AReject("UB budget");
    using namespace matmul_tiling;
    MatmulApiTiling mm(platform);
    if(mm.SetAType(TPosition::GM,CubeFormat::ND,DataType::DT_INT8)==-1)return S4AReject("SetAType INT8 ND");
    if(mm.SetBType(TPosition::GM,CubeFormat::ND,DataType::DT_INT8,false)==-1)return S4AReject("SetBType INT8 ND non-transposed");
    if(mm.SetCType(TPosition::VECIN,CubeFormat::NZ,DataType::DT_FLOAT16)==-1)return S4AReject("SetCType DT_FLOAT16 candidate");
    mm.SetBiasType(TPosition::GM,CubeFormat::ND,DataType::DT_INT32);mm.SetBias(false);
    if(mm.SetDequantType(DequantType::TENSOR)==-1)return S4AReject("SetDequantType TENSOR");
    mm.SetShape(16,64,128);mm.SetOrgShape(16,64,128);mm.SetFixSplit(16,64,128);
    mm.SetBufferSpace(-1,-1,static_cast<int32_t>(scratch));
    Tiling td;if(mm.GetTiling(td.cubeTilingData)==-1) {
        std::fprintf(stderr,"STAGE4A GetTiling failed: Host C=DT_FLOAT16, dequant=TENSOR; no automatic fallback\n");
        return ge::GRAPH_FAILED;
    }
    auto &p=td.cubeTilingData;
    if(p.get_singleCoreM()!=16||p.get_singleCoreN()!=64||p.get_singleCoreK()!=128||
       p.get_baseM()!=16||p.get_baseN()!=64||p.get_baseK()!=128) {
        std::fprintf(stderr,"STAGE4A rejected tiling single=%d/%d/%d base=%d/%d/%d\n",
            p.get_singleCoreM(),p.get_singleCoreN(),p.get_singleCoreK(),p.get_baseM(),p.get_baseN(),p.get_baseK());
        return ge::GRAPH_FAILED;
    }
    const uint64_t a=16*128,bbytes=64*128,acc=16*64*4;
    if(p.get_depthA1()<1||p.get_depthB1()<1||p.get_dbL0A()<1||p.get_dbL0B()<1||p.get_dbL0C()<1||
       a*p.get_depthA1()+bbytes*p.get_depthB1()>l1||a*p.get_dbL0A()>l0a||
       bbytes*p.get_dbL0B()>l0b||acc*p.get_dbL0C()>l0c)return S4AReject("L1/L0 depth or capacity check");
    // CFG_NORM: non-transposed B ND uses two layout-conversion tiles; C workspace is half.
    // SDK vector scale storage starts at transLength; guard its actual bound.
    if(p.get_transLength()<0 || p.get_transLength()%32!=0 || static_cast<uint64_t>(p.get_transLength())+64*8>scratch ||
       2*bbytes+16*64*2>scratch) {
        std::fprintf(stderr,"STAGE4A transLength=%d scratch=%llu\n",p.get_transLength(),(unsigned long long)scratch);
        return S4AReject("UB workspace/scale vector capacity");
    }
    std::fprintf(stderr,"STAGE4A_PLAN arch=2002 slots=8 groups=20 M=16 K=2560 groupK=128 tileN=64 outputN=512 HostC=VECIN_NZ_DT_FLOAT16 dequant=TENSOR "
        "single=%d/%d/%d base=%d/%d/%d userUB=12288 ubGuards=2560 reserve=8192 scratch=%llu UB=%llu L1=%llu L0A=%llu L0B=%llu L0C=%llu "
        "depthA1=%d depthB1=%d dbL0A=%d dbL0B=%d dbL0C=%d transLength=%d\n",
        p.get_singleCoreM(),p.get_singleCoreN(),p.get_singleCoreK(),p.get_baseM(),p.get_baseN(),p.get_baseK(),
        (unsigned long long)scratch,(unsigned long long)ub,(unsigned long long)l1,
        (unsigned long long)l0a,(unsigned long long)l0b,(unsigned long long)l0c,
        p.get_depthA1(),p.get_depthB1(),p.get_dbL0A(),p.get_dbL0B(),p.get_dbL0C(),p.get_transLength());
    td.set_launchBlocks(blocks);td.set_systemBytes(system);td.set_arenaBytes(arena);td.set_arenaStride(stride);
    td.set_outputMode(*mode);td.set_localMemSize(scratch);c->SetBlockDim(blocks);c->SetTilingKey(0);
    td.SaveToBuffer(c->GetRawTilingData()->GetData(),c->GetRawTilingData()->GetCapacity());
    c->GetRawTilingData()->SetDataSize(td.GetDataSize());c->GetWorkspaceSizes(1)[0]=totalWorkspace;
    return ge::GRAPH_SUCCESS;
}
