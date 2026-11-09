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
enum class TPosition{GM,VECIN,VECCALC,CO1};enum class CubeFormat{ND,NZ};enum class HardEvent{M_V,V_M,V_S,MTE2_V,V_MTE3,MTE3_MTE2,MTE2_S,S_MTE3};enum Pipe{PIPE_ALL,PIPE_V};
enum class BlockMode{BLOCK_MODE_MATRIX};enum class DeqScale{DEQ_NONE};struct DataCopyParams{int blockCount,blockLen,srcStride,dstStride;};struct DataCopyEnhancedParams{BlockMode blockMode;DeqScale deqScale;bool isRelu=false;};
inline size_t ubBytes=0,iterations=0,ends=0,copies=0;inline int fault=0;inline uint32_t blockNum=1;inline size_t l0bytes=0;inline bool mready=false,vready=false;inline int evstate[8]={};inline std::vector<uint8_t*> buffers;
inline int32_t* outputBase=nullptr;inline uint8_t* workspaceBase=nullptr;inline std::vector<int> writes(4096);
inline void Require(bool ok){if(!ok)throw std::runtime_error("CPU bounds/lifetime failure");}
inline uint32_t GetBlockIdx(){return 0;}inline uint32_t GetBlockNum(){return blockNum;}
template<class T>struct LocalTensor{T*data;size_t size;uint64_t address=0;TPosition pos=TPosition::VECCALC;LocalTensor operator[](size_t i)const{Require(i<=size);return{data+i,size-i,address+i*sizeof(T),pos};}void SetSize(size_t n){Require(n<=size);size=n;}template<class U>LocalTensor<U>ReinterpretCast(){return{reinterpret_cast<U*>(data),size*sizeof(T)/sizeof(U),address,pos};}uint64_t GetPhyAddr()const{return address+(fault==9&&address==222208?32:0);}size_t GetSize()const{return size;}int GetPosition()const{return int(pos);}void SetValue(size_t i,T v){Require(i<size);data[i]=v;}T GetValue(size_t i){Require(i<size);return data[i];}};
template<class T>struct GlobalTensor{T*data;size_t size;uint64_t address=0;TPosition pos=TPosition::VECCALC;void SetGlobalBuffer(T*p,size_t n){data=p;size=n;}GlobalTensor operator[](size_t i)const{Require(i<=size);return{data+i,size-i};}};
template<TPosition pos=TPosition::VECCALC>struct TBuf{std::vector<uint8_t> bytes;uint64_t address=0;template<class T>LocalTensor<T>Get(){return{reinterpret_cast<T*>(bytes.data()),bytes.size()/sizeof(T),address,pos};}};
struct TPipe{template<TPosition p>void InitBuffer(TBuf<p>&b,size_t n){b.bytes.resize(n);if constexpr(p==TPosition::CO1){b.address=l0bytes;l0bytes+=n;}else{b.address=ubBytes;ubBytes+=n;buffers.push_back(b.bytes.data());}}int FetchEventID(HardEvent){return 0;}template<HardEvent E>int AllocEventID(){Require(evstate[int(E)]==0);evstate[int(E)]=1;return 0;}template<HardEvent E>void ReleaseEventID(int i){Require(i==0&&evstate[int(E)]==3);evstate[int(E)]=0;}template<TPosition>uint64_t GetQueueEndAddress(){return ubBytes+(fault==9?32:0);}};
template<Pipe>void PipeBarrier(){}template<HardEvent E>void SetFlag(int i){Require(i==0&&evstate[int(E)]==1);evstate[int(E)]=2;}template<HardEvent E>void WaitFlag(int i){Require(i==0&&evstate[int(E)]==2);evstate[int(E)]=3;if(E==HardEvent::M_V)mready=true;if(E==HardEvent::V_M)vready=true;}
template<class T>void Duplicate(LocalTensor<T>d,T v,size_t n){Require(n<=d.size);for(size_t i=0;i<n;++i)d.data[i]=v;}
template<class T>void DataCopy(GlobalTensor<T>d,LocalTensor<T>s,size_t n){Require(n<=d.size&&n<=s.size&&n*sizeof(T)%32==0);if constexpr(std::is_same_v<T,int32_t>){Require(iterations==1&&ends==1&&mready&&vready);auto offset=d.data-outputBase;Require(offset>=0&&offset+n<=4096);for(size_t i=0;i<n;++i)Require(++writes[offset+i]==1);++copies;}std::memcpy(d.data,s.data,n*sizeof(T));}
template<class T>void DataCopy(LocalTensor<T>d,GlobalTensor<T>s,size_t n){Require(n<=d.size&&n<=s.size&&n*sizeof(T)%32==0);std::memcpy(d.data,s.data,n*sizeof(T));}
inline void RegisterWorkspace(uint8_t*p){Require(p==workspaceBase+512);}
}

namespace AscendC {template<class T>void DataCopy(LocalTensor<T>d,LocalTensor<T>s,DataCopyParams p,DataCopyEnhancedParams e){Require(mready&&s.pos==TPosition::CO1&&d.pos!=TPosition::CO1&&p.blockCount==1&&p.blockLen==1&&p.srcStride==0&&p.dstStride==0&&!e.isRelu&&p.blockLen*256<=int(d.size)&&p.blockLen*256<=int(s.size));std::memcpy(d.data,s.data,1024);}}
