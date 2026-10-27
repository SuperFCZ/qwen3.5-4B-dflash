// Test-only scalar execution model. Does not model CANN instructions or async events.
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <stdexcept>
#define __global__
#define __aicore__
#define __gm__
using GM_ADDR=uint8_t*;
struct TestTiling {uint32_t groupCount,elements;};
#define GET_TILING_DATA(name,ptr) TestTiling name=*reinterpret_cast<TestTiling*>(ptr)
namespace AscendC {
using half=_Float16;
enum class TPosition {VECIN,VECOUT,VECCALC};
enum class RoundMode {CAST_NONE};
enum Pipe {PIPE_V,PIPE_ALL};
inline size_t ubBytes=0,addCalls=0,casts=0,copiesIn=0,copiesOut=0;
inline void Require(bool v) {if(!v)throw std::runtime_error("CPU model bounds/lifetime violation");}
template<class T> struct LocalTensor {T *data;size_t size;};
template<class T> struct GlobalTensor {
 T* data=nullptr;size_t size=0;
 void SetGlobalBuffer(T*p,size_t n){data=p;size=n;}
 GlobalTensor operator[](size_t n)const{Require(n<=size);return {data+n,size-n};}
};
template<TPosition pos> struct TBuf {
 std::vector<uint8_t> bytes;
 template<class T> LocalTensor<T> Get(size_t count){Require(count*sizeof(T)<=bytes.size());return {reinterpret_cast<T*>(bytes.data()),count};}
};
template<TPosition pos,int depth> struct TQue {
 std::vector<uint8_t> bytes;int state=0;
 template<class T> LocalTensor<T> AllocTensor(){Require(state==0);state=1;return {reinterpret_cast<T*>(bytes.data()),bytes.size()/sizeof(T)};}
 template<class T> void EnQue(LocalTensor<T>){Require(state==1);state=2;}
 template<class T> LocalTensor<T> DeQue(){Require(state==2);state=1;return {reinterpret_cast<T*>(bytes.data()),bytes.size()/sizeof(T)};}
 template<class T> void FreeTensor(LocalTensor<T>){Require(state==1);state=0;}
};
struct TPipe {
 template<TPosition p>void InitBuffer(TBuf<p>&b,size_t n){b.bytes.resize(n);ubBytes+=n;}
 template<TPosition p,int d>void InitBuffer(TQue<p,d>&q,int depth,size_t n){Require(depth==1);q.bytes.resize(n);ubBytes+=n;}
};
template<Pipe p>void PipeBarrier(){}
template<class T>void Duplicate(LocalTensor<T> dst,T value,size_t n){Require(n<=dst.size);for(size_t i=0;i<n;++i)dst.data[i]=value;}
template<class T>void DataCopy(LocalTensor<T> dst,GlobalTensor<T> src,size_t n){Require(n<=dst.size&&n<=src.size);std::memcpy(dst.data,src.data,n*sizeof(T));++copiesIn;}
template<class T>void DataCopy(GlobalTensor<T> dst,LocalTensor<T> src,size_t n){Require(n<=dst.size&&n<=src.size);std::memcpy(dst.data,src.data,n*sizeof(T));++copiesOut;}
template<class D,class S>void Cast(LocalTensor<D> dst,LocalTensor<S> src,RoundMode,size_t n){Require(n<=dst.size&&n<=src.size);for(size_t i=0;i<n;++i)dst.data[i]=static_cast<D>(src.data[i]);++casts;}
template<class T>void Add(LocalTensor<T> dst,LocalTensor<T> a,LocalTensor<T>b,size_t n){Require(n<=dst.size&&n<=a.size&&n<=b.size);for(size_t i=0;i<n;++i)dst.data[i]=a.data[i]+b.data[i];++addCalls;}
}
