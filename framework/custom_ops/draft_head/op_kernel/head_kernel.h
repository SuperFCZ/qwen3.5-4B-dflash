#pragma once
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#define DFLASH_HEAD_DEVICE 1
#include "head_contract.h"

using HeadA=matmul::MatmulType<AscendC::TPosition::VECOUT,matmul::CubeFormat::ND,half>;
using HeadB=matmul::MatmulType<AscendC::TPosition::GM,matmul::CubeFormat::ND,half,true>;
using HeadC=matmul::MatmulType<AscendC::TPosition::VECIN,matmul::CubeFormat::ND,half>;
using HeadBias=matmul::MatmulType<AscendC::TPosition::GM,matmul::CubeFormat::ND,float>;

__aicore__ inline void ScalarToMte3(AscendC::TPipe& pipe) {
    const auto event=pipe.FetchEventID(AscendC::HardEvent::S_MTE3);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(event);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(event);
}
template<bool Audit, class Tiling>
__aicore__ inline void RunHead(GM_ADDR hidden, GM_ADDR weight, GM_ADDR tokenId, GM_ADDR auditLogits,
                              GM_ADDR workspace, Tiling& t) {
    using namespace AscendC;
    using namespace head_contract;
    const uint32_t core=GetBlockIdx(), p=t.partitions, m=t.m;
    // Host binds p to the actual launch and physical core count, with batchmode.
    TPipe pipe;
    matmul::Matmul<HeadA,HeadB,HeadC,HeadBias> mm;
    REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&t.cubeTilingData);
    TBuf<TPosition::VECOUT> xBuf,maxBuf,idBuf;
    TBuf<TPosition::VECIN> cBuf;
    TBuf<> syncBuf,mmBuf;
    pipe.InitBuffer(xBuf,16*kK*2); pipe.InitBuffer(cBuf,16*kTileN*2);
    pipe.InitBuffer(maxBuf,32); pipe.InitBuffer(idBuf,128); pipe.InitBuffer(syncBuf,p*32);
    pipe.InitBuffer(mmBuf,t.matmulUbBytes);
    mm.SetLocalWorkspace(mmBuf.Get<uint8_t>(t.matmulUbBytes));
    auto x=xBuf.Get<half>(16*kK); auto c=cBuf.Get<half>(16*kTileN);
    auto maxLocal=maxBuf.Get<half>(16); auto idLocal=idBuf.Get<int64_t>(16);
    auto syncLocal=syncBuf.Get<int32_t>(p*8);
    auto maxBits=maxLocal.ReinterpretCast<uint16_t>();
    auto cBits=c.ReinterpretCast<uint16_t>();
    GlobalTensor<half> xGm,wGm,partialMax,debugGm;
    GlobalTensor<int64_t> ids,partialId;
    GlobalTensor<int32_t> syncGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(hidden),m*kK);
    wGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(weight),uint64_t(kN)*kK);
    ids.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(tokenId),m);
    // msopgen operator projects pass USER workspace here; system space is
    // separate via GetSysWorkSpacePtr. Do not apply GetUserWorkspace twice.
    partialMax.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(workspace+MaxOffset(p)),p*16);
    partialId.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(workspace+IdOffset(p)),p*16);
    syncGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace+SyncOffset(p)),p*8);
    if constexpr(Audit) debugGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(auditLogits),uint64_t(m)*kN);
    // Official soft SyncAll kernel initialization: EVERY core zeros ALL flags.
    // Fresh each invocation; correctness never relies on caller workspace zeroing.
    for(uint32_t i=0;i<p*8;++i) syncLocal.SetValue(i,0);
    ScalarToMte3(pipe); DataCopy(syncGm,syncLocal,p*8); PipeBarrier<PIPE_ALL>();
    Duplicate(x,static_cast<half>(0),16*kK); PipeBarrier<PIPE_ALL>();
    DataCopy(x,xGm,m*kK); PipeBarrier<PIPE_ALL>();
    for(uint32_t r=0;r<16;++r) { maxBits.SetValue(r,uint16_t(0)); idLocal.SetValue(r,int64_t(-1)); }
    bool healthy=true;
    for(uint32_t tile=core;tile<kTiles;tile+=p) {
        const uint32_t begin=tile*kTileN;
        mm.SetTensorA(x); mm.SetTensorB(wGm[uint64_t(begin)*kK],true);
        // One Iterate owns ALL K=2560; no split-K or intermediate FP16 sums.
        if(!mm.Iterate(false)) { healthy=false; break; }
        mm.GetTensorC(c,0,true); // compact ND [16,64], and the FP16 rounding boundary
        PipeBarrier<PIPE_ALL>();
        const auto ready=pipe.FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(ready); WaitFlag<HardEvent::V_S>(ready);
        for(uint32_t r=0;r<m;++r) for(uint32_t n=0;n<kTileN;++n) {
            const auto bits=cBits.GetValue(r*kTileN+n);
            const int64_t id=begin+n;
            if(Better(bits,id,maxBits.GetValue(r),idLocal.GetValue(r))) { maxBits.SetValue(r,bits); idLocal.SetValue(r,id); }
        }
        if constexpr(Audit) {
            const DataCopyParams copy{static_cast<uint16_t>(m),4,0,static_cast<uint16_t>((kN-kTileN)*2/32)};
            DataCopy(debugGm[begin],c,copy);
        }
        PipeBarrier<PIPE_ALL>();
    }
    mm.End(); PipeBarrier<PIPE_ALL>();
    if(!healthy) for(uint32_t r=0;r<16;++r) idLocal.SetValue(r,int64_t(-1));
    ScalarToMte3(pipe);
    DataCopy(partialMax[core*16],maxLocal,16); DataCopy(partialId[core*16],idLocal,16);
    PipeBarrier<PIPE_ALL>();
    SyncAll(syncGm,syncLocal,p); // coupled 310P software barrier, not unsupported hardware SyncAll()
    if(core==0) {
        for(uint32_t r=0;r<16;++r) idLocal.SetValue(r,int64_t(-1));
        // Matmul has ended: reuse its external logit tile as incoming scratch.
        // Running best remains in the explicitly budgeted maxBuf/idBuf.
        auto incomingIds=c.ReinterpretCast<int64_t>()[4]; // offset 32 B, size >=128 B
        bool valid=true;
        for(uint32_t part=0;part<p;++part) {
            DataCopy(c,partialMax[part*16],16); DataCopy(incomingIds,partialId[part*16],16);
            const auto ready=pipe.FetchEventID(HardEvent::MTE2_S);
            SetFlag<HardEvent::MTE2_S>(ready); WaitFlag<HardEvent::MTE2_S>(ready);
            for(uint32_t r=0;r<m;++r) {
                const auto id=incomingIds.GetValue(r); const auto bits=cBits.GetValue(r);
                if(id<0 || id>=kN) valid=false;
                else if(Better(bits,id,maxBits.GetValue(r),idLocal.GetValue(r))) { maxBits.SetValue(r,bits); idLocal.SetValue(r,id); }
            }
            PipeBarrier<PIPE_ALL>();
        }
        // Scalar exact-length writes avoid overrunning an INT64 [1..15] tail.
        for(uint32_t r=0;r<m;++r) {
            ids.SetValue(r,valid?idLocal.GetValue(r):int64_t(-1));
            DataCacheCleanAndInvalid<int64_t,CacheLine::SINGLE_CACHE_LINE>(ids[r]);
        }
    }
}
