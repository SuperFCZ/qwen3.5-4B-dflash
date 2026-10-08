// Abstract CPU API model only. Does not emulate CANN's soft barrier or caches.
#pragma once
#include <algorithm>
#include <atomic>
#include <array>
#include <cassert>
#include <condition_variable>
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
constexpr int PIPE_ALL=0;
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
enum class TPosition { GM,VECIN,VECOUT,VECCALC };
enum class CacheLine { SINGLE_CACHE_LINE };
enum class HardEvent { S_MTE3,V_S,MTE2_S };
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
    assert(count<=out.count); std::fill(out.data,out.data+count,value);
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
