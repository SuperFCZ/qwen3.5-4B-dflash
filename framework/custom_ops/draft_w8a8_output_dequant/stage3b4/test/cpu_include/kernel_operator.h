// CPU test double only; no claim about actual CANN events, MMAD or dequant instructions.
#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#define __global__
#define __aicore__
#define __gm__
using GM_ADDR=uint8_t*;
struct TestTiling {uint64_t localMemSize;uint32_t outputMode;int cubeTilingData;};
#define GET_TILING_DATA(name,ptr) TestTiling name=*reinterpret_cast<TestTiling*>(ptr)
namespace AscendC {
using half=_Float16;
enum class TPosition {GM,VECIN,VECCALC};
enum class CubeFormat {ND,NZ};
enum class RoundMode {CAST_NONE};
enum class HardEvent {V_S};
enum Pipe {PIPE_ALL,PIPE_V};
inline size_t ubBytes=0,adds=0,casts=0,copies=0,ends=0,iterations=0;
inline int fault=0,targetGroup=19;
inline bool firstDmaAfterTwo=false;
inline void Require(bool ok){if(!ok)throw std::runtime_error("CPU protocol/bounds violation");}
template<class T>struct LocalTensor {
 T* data=nullptr;size_t size=0;
 LocalTensor operator[](size_t i)const{Require(i<=size);return {data+i,size-i};}
 void SetSize(uint32_t n){Require(n<=size);size=n;}
 T GetValue(size_t i)const{Require(i<size);return data[i];}
};
template<class T>struct GlobalTensor {
 T* data=nullptr;size_t size=0;
 void SetGlobalBuffer(T*p,size_t n){data=p;size=n;}
 GlobalTensor operator[](size_t i)const{Require(i<=size);return {data+i,size-i};}
};
template<TPosition pos=TPosition::VECCALC>struct TBuf {
 std::vector<uint8_t> bytes;
 template<class T>LocalTensor<T> Get(){return {reinterpret_cast<T*>(bytes.data()),bytes.size()/sizeof(T)};}
};
struct TPipe {
 template<TPosition p>void InitBuffer(TBuf<p>&b,size_t bytes){b.bytes.resize(bytes);ubBytes+=bytes;}
 int FetchEventID(HardEvent){return 0;}
};
inline void* GetSysWorkSpacePtr(){return nullptr;}
template<Pipe>void PipeBarrier(){}
template<HardEvent>void SetFlag(int){}
template<HardEvent>void WaitFlag(int){}
template<class T>void Duplicate(LocalTensor<T> d,T v,size_t n){Require(n<=d.size);for(size_t i=0;i<n;++i)d.data[i]=v;}
template<class D,class S>void Cast(LocalTensor<D>d,LocalTensor<S>s,RoundMode,size_t n){Require(n<=d.size&&n<=s.size);for(size_t i=0;i<n;++i)d.data[i]=static_cast<D>(s.data[i]);++casts;}
inline void Add(LocalTensor<float>d,LocalTensor<float>a,LocalTensor<float>b,size_t n){Require(n<=d.size&&n<=a.size&&n<=b.size);for(size_t i=0;i<n;++i)d.data[i]=a.data[i]+b.data[i];++adds;}
template<class T>void DataCopy(GlobalTensor<T>d,LocalTensor<T>s,size_t n){Require(n<=d.size&&n<=s.size&&iterations==20&&ends==20&&adds==20);std::memcpy(d.data,s.data,n*sizeof(T));++copies;firstDmaAfterTwo=true;}
}
