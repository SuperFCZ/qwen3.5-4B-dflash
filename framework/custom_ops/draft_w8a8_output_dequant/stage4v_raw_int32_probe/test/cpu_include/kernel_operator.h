// CPU bounds/lifecycle double. Does not emulate SDK scheduling or hardware events.
#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <type_traits>
#define __global__
#define __aicore__
#define __gm__
using GM_ADDR=uint8_t*;
struct TestTiling {uint64_t localMemSize;uint32_t outputMode;int32_t tileId;uint32_t launchBlocks;uint64_t systemBytes,arenaBytes,arenaStride;int cubeTilingData;};
#define GET_TILING_DATA(name,ptr) TestTiling name=*reinterpret_cast<TestTiling*>(ptr)
namespace AscendC {
enum class TPosition{GM,VECIN,VECCALC};enum class CubeFormat{ND,NZ};enum class HardEvent{V_S,MTE2_S};enum Pipe{PIPE_ALL};
inline size_t ubBytes=0,iterations=0,ends=0,copies=0;inline int fault=0;inline uint32_t blockNum=1;
inline int32_t* outputBase=nullptr;inline uint8_t* workspaceBase=nullptr;inline std::vector<int> writes(4096);
inline void Require(bool ok){if(!ok)throw std::runtime_error("CPU bounds/lifetime failure");}
inline uint32_t GetBlockIdx(){return 0;}inline uint32_t GetBlockNum(){return blockNum;}
template<class T>struct LocalTensor{T*data;size_t size;LocalTensor operator[](size_t i)const{Require(i<=size);return{data+i,size-i};}void SetSize(size_t n){Require(n<=size);size=n;}template<class U>LocalTensor<U>ReinterpretCast(){return{reinterpret_cast<U*>(data),size*sizeof(T)/sizeof(U)};}T GetValue(size_t i){Require(i<size);return data[i];}};
template<class T>struct GlobalTensor{T*data;size_t size;void SetGlobalBuffer(T*p,size_t n){data=p;size=n;}GlobalTensor operator[](size_t i)const{Require(i<=size);return{data+i,size-i};}};
template<TPosition pos=TPosition::VECCALC>struct TBuf{std::vector<uint8_t> bytes;template<class T>LocalTensor<T>Get(){return{reinterpret_cast<T*>(bytes.data()),bytes.size()/sizeof(T)};}};
struct TPipe{template<TPosition p>void InitBuffer(TBuf<p>&b,size_t n){b.bytes.resize(n);ubBytes+=n;}int FetchEventID(HardEvent){return 0;}};
template<Pipe>void PipeBarrier(){}template<HardEvent>void SetFlag(int){}template<HardEvent>void WaitFlag(int){}
template<class T>void Duplicate(LocalTensor<T>d,T v,size_t n){Require(n<=d.size);for(size_t i=0;i<n;++i)d.data[i]=v;}
template<class T>void DataCopy(GlobalTensor<T>d,LocalTensor<T>s,size_t n){Require(n<=d.size&&n<=s.size&&n*sizeof(T)%32==0);if constexpr(std::is_same_v<T,int32_t>){Require(iterations==1&&ends==1);auto offset=d.data-outputBase;Require(offset>=0&&offset+n<=4096);for(size_t i=0;i<n;++i)Require(++writes[offset+i]==1);++copies;}std::memcpy(d.data,s.data,n*sizeof(T));}
template<class T>void DataCopy(LocalTensor<T>d,GlobalTensor<T>s,size_t n){Require(n<=d.size&&n<=s.size&&n*sizeof(T)%32==0);std::memcpy(d.data,s.data,n*sizeof(T));}
inline void RegisterWorkspace(uint8_t*p){Require(p==workspaceBase+512);}
}
