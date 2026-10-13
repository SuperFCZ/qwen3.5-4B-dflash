// Abstract CPU API model only. Does not emulate CANN's soft barrier or caches.
#pragma once
#include <algorithm>
#include <atomic>
#include <array>
#include <cassert>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>
#include <type_traits>
#define __aicore__
#define __global__
#define __gm__
using half=_Float16;
using GM_ADDR=uint8_t*;
constexpr int PIPE_ALL=0, PIPE_V=1;
struct CpuCube { int singleM=16,singleN=64,singleK=2560; };
struct CpuTiling { uint32_t m,partitions; uint64_t matmulUbBytes=65536; CpuCube cubeTilingData; };
#define GET_TILING_DATA(name,addr) auto& name=*reinterpret_cast<CpuTiling*>(addr)
inline void* GetSysWorkSpacePtr() { return nullptr; }
struct CpuRun {
    uint32_t m=15,p=7; int pattern=0,failCore=-1;
    std::vector<uint32_t> visits=std::vector<uint32_t>(3880);
    std::atomic<uint32_t> calls{0},barriers{0},stores{0};
    std::mutex mutex; std::condition_variable cv; uint32_t arrived=0;
    std::array<bool,64> initialized{},maxWritten{},idWritten{};
    const half* hidden=nullptr; const half* weight=nullptr;
};
inline CpuRun* run=nullptr;
inline thread_local uint32_t core=0;
inline thread_local uint64_t ubBytes=0;

namespace AscendC {
inline thread_local bool cpuOrderedNanNE=false;
enum class TPosition { GM,VECIN,VECOUT,VECCALC };
enum class CacheLine { SINGLE_CACHE_LINE };
enum class HardEvent { S_MTE3,V_S,MTE2_S };
enum class CMPMODE { GT,LT,EQ,NE };
enum class ReduceOrder { ORDER_VALUE_INDEX };
enum class SELMODE { VSEL_CMPMASK_SPR };
enum class RoundMode { CAST_NONE };
inline uint32_t GetBlockIdx() { return core; }
inline uint32_t GetBlockNum() { return run->p; }
template<HardEvent E> void SetFlag(int) {}
template<HardEvent E> void WaitFlag(int) {}
template<int P> void PipeBarrier() {}

template<typename T> struct LocalTensor {
    T* data=nullptr; uint64_t count=0;
    LocalTensor operator[](uint64_t n) const { assert(n<count); return {data+n,count-n}; }
    T GetValue(uint64_t n) const { assert(n<count); return data[n]; }
    template<typename U> void SetValue(uint64_t n,U value) { assert(n<count); data[n]=static_cast<T>(value); }
    template<typename U> LocalTensor<U> ReinterpretCast() const { return {reinterpret_cast<U*>(data),count*sizeof(T)/sizeof(U)}; }
};
template<typename T> struct GlobalTensor {
    T* base=nullptr; uint64_t offset=0,count=0;
    void SetGlobalBuffer(T* ptr,uint64_t n) { base=ptr; offset=0; count=n; }
    GlobalTensor operator[](uint64_t n) const { assert(n<count); return {base,offset+n,count-n}; }
    T* Data() const { return base+offset; }
    void SetValue(uint64_t n,T value) { assert(n<count); Data()[n]=value; ++run->stores; }
};
template<TPosition P=TPosition::VECCALC> class TBuf {
    std::vector<uint64_t> values; size_t bytes=0;
public:
    void Allocate(uint64_t n) { bytes=n; values.resize((n+7)/8); }
    template<typename T> LocalTensor<T> Get(uint64_t n) { assert(n*sizeof(T)<=bytes); return {reinterpret_cast<T*>(values.data()),n}; }
};
class TPipe {
public:
    TPipe() { ubBytes=0; }
    template<TPosition P> void InitBuffer(TBuf<P>& b,uint64_t bytes) { assert(bytes%32==0); b.Allocate(bytes); ubBytes+=bytes; }
    int FetchEventID(HardEvent) { return 0; }
};
template<typename T> void Duplicate(LocalTensor<T> out,T value,uint32_t count) {
    static_assert(std::is_same_v<T,half> || std::is_same_v<T,float> || std::is_same_v<T,int16_t> ||
                  std::is_same_v<T,uint16_t> || std::is_same_v<T,int32_t> || std::is_same_v<T,uint32_t>);
    assert(count<=out.count); std::fill(out.data,out.data+count,value);
}
// CPU control model uses IEEE conversion (including overflow to Inf). This is
// NOT a device emulator. Candidate Mins/Maxs below model the REQUIRED semantics;
// independent NPU diagnostics must prove them, especially NaN and signed zero.
inline void Mins(LocalTensor<float> dst,LocalTensor<float> src,float bound,uint32_t count) {
    for(uint32_t i=0;i<count;++i) dst.data[i]=src.data[i]>bound?bound:src.data[i];
}
inline void Maxs(LocalTensor<float> dst,LocalTensor<float> src,float bound,uint32_t count) {
    for(uint32_t i=0;i<count;++i) dst.data[i]=src.data[i]<bound?bound:src.data[i];
}
// CPU reference for required Compare/Select semantics, NOT a device emulator.
inline void CompareScalar(LocalTensor<uint8_t> mask,LocalTensor<float> src,float limit,CMPMODE mode,uint32_t count) {
    assert(count==64 && src.count>=count && mask.count>=32);
    for(uint32_t i=0;i<count/8;++i) mask.data[i]=0;
    for(uint32_t i=0;i<count;++i) {
        const bool selected=mode==CMPMODE::GT?src.data[i]>limit:mode==CMPMODE::LT?src.data[i]<limit:
            mode==CMPMODE::EQ?src.data[i]==limit:src.data[i]!=limit;
        if(selected) mask.data[i/8]|=uint8_t(1U<<(i%8));
    }
}
inline void Compare(LocalTensor<uint8_t> mask,LocalTensor<float> a,LocalTensor<float> b,CMPMODE mode,uint32_t count) {
    assert(count==64 && mode==CMPMODE::NE && mask.count>=32);
    std::memset(mask.data,0,8);
    for(uint32_t i=0;i<count;++i) if(a.data[i]!=b.data[i] && !(cpuOrderedNanNE && (std::isnan(a.data[i]) || std::isnan(b.data[i])))) mask.data[i/8]|=uint8_t(1U<<(i%8));
}
inline void WholeReduceMax(LocalTensor<float> dst,LocalTensor<float> src,int mask,int repeat,int dstStride,int blockStride,int srcStride,ReduceOrder order) {
    assert(mask==64 && repeat==1 && dstStride==1 && blockStride==1 && srcStride==8 && order==ReduceOrder::ORDER_VALUE_INDEX);
    float value=src.data[0]; uint32_t index=0;
    for(uint32_t i=0;i<64;++i) { if(std::isnan(src.data[i])) { value=src.data[i]; index=i; } else if(src.data[i]>=value) { value=src.data[i]; index=i; } }
    dst.data[0]=value; std::memcpy(dst.data+1,&index,4); // deliberately choose LAST tie: kernel must ignore it
}
inline void Cast(LocalTensor<float> dst,LocalTensor<half> src,RoundMode mode,uint32_t count) {
    assert(mode==RoundMode::CAST_NONE && count<=dst.count && count<=src.count);
    for(uint32_t i=0;i<count;++i) dst.data[i]=static_cast<float>(src.data[i]);
}
inline void Select(LocalTensor<float> dst,LocalTensor<uint8_t> mask,LocalTensor<float> yes,
                   LocalTensor<float> no,SELMODE mode,uint32_t count) {
    assert(mode==SELMODE::VSEL_CMPMASK_SPR && count==64 && dst.count>=count && yes.count>=count && no.count>=count);
    for(uint32_t i=0;i<count;++i) {
        const auto* source=(mask.data[i/8]&(1U<<(i%8)))?yes.data+i:no.data+i;
        // Bit copy explicitly preserves NaN payload/sign and zero sign.
        uint32_t raw; std::memcpy(&raw,source,4); std::memcpy(dst.data+i,&raw,4);
    }
}
inline void And(LocalTensor<uint16_t> dst,LocalTensor<uint16_t> a,LocalTensor<uint16_t> b,uint32_t count) {
    assert(count<=dst.count && count<=a.count && count<=b.count);
    for(uint32_t i=0;i<count;++i) dst.data[i]=a.data[i]&b.data[i];
}
inline void Adds(LocalTensor<int16_t> dst,LocalTensor<int16_t> src,int16_t value,uint32_t count) {
    assert(count<=dst.count && count<=src.count);
    for(uint32_t i=0;i<count;++i) { int v=int(src.data[i])+value; assert(v>=-32768 && v<=32767); dst.data[i]=int16_t(v); }
}
inline void Cast(LocalTensor<half> dst,LocalTensor<int16_t> src,RoundMode mode,uint32_t count) {
    assert(mode==RoundMode::CAST_NONE && count<=dst.count && count<=src.count);
    for(uint32_t i=0;i<count;++i) dst.data[i]=static_cast<half>(src.data[i]);
}
template<typename T> void DataCopy(LocalTensor<T> dst,LocalTensor<T> src,uint32_t count) {
    assert(count<=dst.count && count<=src.count && count*sizeof(T)%32==0);
    std::memcpy(dst.data,src.data,count*sizeof(T));
}
inline void Cast(LocalTensor<half> dst,LocalTensor<float> src,RoundMode mode,uint32_t count) {
    assert(mode==RoundMode::CAST_NONE && count<=dst.count && count<=src.count);
    for(uint32_t i=0;i<count;++i) dst.data[i]=static_cast<half>(src.data[i]);
}
template<typename T> void DataCopy(LocalTensor<T> dst,GlobalTensor<T> src,uint32_t count) {
    assert(count<=dst.count && count<=src.count && count*sizeof(T)%32==0);
    std::lock_guard<std::mutex> guard(run->mutex);
    std::memcpy(dst.data,src.Data(),count*sizeof(T));
}
template<typename T> void DataCopy(GlobalTensor<T> dst,LocalTensor<T> src,uint32_t count) {
    assert(count<=dst.count && count<=src.count && count*sizeof(T)%32==0);
    std::lock_guard<std::mutex> guard(run->mutex);
    if constexpr(std::is_same_v<T,int32_t>) {
        assert(count==run->p*8);
        for(uint32_t i=0;i<count;++i) assert(src.data[i]==0);
        run->initialized[core]=true;
    } else if constexpr(std::is_same_v<T,int64_t>) {
        assert(count==16 && dst.offset==core*16); run->idWritten[core]=true;
    } else if(count==16) {
        assert(dst.offset==core*16); run->maxWritten[core]=true;
    }
    std::memcpy(dst.Data(),src.data,count*sizeof(T));
}
struct DataCopyParams { uint16_t blockCount,blockLen,srcStride,dstStride; };
template<typename T> void DataCopy(GlobalTensor<T> dst,LocalTensor<T> src,const DataCopyParams& p) {
    assert(p.blockCount==run->m && p.blockLen==4 && p.srcStride==0 && p.dstStride==15516);
    for(uint32_t row=0;row<p.blockCount;++row) {
        const auto a=row*(p.blockLen+p.dstStride)*32/sizeof(T),b=row*(p.blockLen+p.srcStride)*32/sizeof(T);
        DataCopy(dst[a],src[b],p.blockLen*32/sizeof(T));
    }
}
template<typename T,CacheLine C> void DataCacheCleanAndInvalid(GlobalTensor<T>) {}
inline void SyncAll(GlobalTensor<int32_t> flags,LocalTensor<int32_t> local,int32_t used) {
    assert(used==int(run->p) && flags.count==run->p*8 && local.count==run->p*8);
    std::unique_lock<std::mutex> lock(run->mutex);
    assert(run->initialized[core] && run->maxWritten[core] && run->idWritten[core]);
    ++run->arrived; ++run->barriers;
    if(run->arrived==run->p) run->cv.notify_all();
    else run->cv.wait(lock,[]{return run->arrived==run->p;});
}
}
