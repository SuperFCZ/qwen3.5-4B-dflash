#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include <stdexcept>
// A wire-layout assertion alone is NOT SDK evidence. Co1ProveLayout checks all
// bytes against actual parent and standalone SDK serializers below.
struct Co1WirePrefix {
 uint64_t localMemSize;uint32_t outputMode;int32_t tileId;uint32_t launchBlocks;
 uint64_t systemBytes,arenaBytes,arenaStride;
};
static_assert(sizeof(Co1WirePrefix)==48 && offsetof(Co1WirePrefix,localMemSize)==0 &&
 offsetof(Co1WirePrefix,outputMode)==8 && offsetof(Co1WirePrefix,tileId)==12 &&
 offsetof(Co1WirePrefix,launchBlocks)==16 && offsetof(Co1WirePrefix,systemBytes)==24 &&
 offsetof(Co1WirePrefix,arenaBytes)==32 && offsetof(Co1WirePrefix,arenaStride)==40,"wire prefix ABI changed");
struct Co1SerializationLayout{size_t offset,bytes,total;bool childSaveLeftDestinationUnchanged;};
template<class Parent>Co1SerializationLayout Co1ProveLayout(){
 Parent probe;optiling::TCubeTiling standalone;
 Co1MarkerFields(probe.cubeTilingData);Co1MarkerFields(standalone);
 Co1WirePrefix prefix;std::memset(&prefix,0,sizeof(prefix));
 prefix.localMemSize=0x0102030405060708ULL;prefix.outputMode=0x11223344;prefix.tileId=0x12345678;
 prefix.launchBlocks=0x22334455;prefix.systemBytes=0x3142536475869708ULL;
 prefix.arenaBytes=0x4152637485960718ULL;prefix.arenaStride=0x5162738495061728ULL;
 probe.set_localMemSize(prefix.localMemSize);probe.set_outputMode(prefix.outputMode);probe.set_tileId(prefix.tileId);
 probe.set_launchBlocks(prefix.launchBlocks);probe.set_systemBytes(prefix.systemBytes);probe.set_arenaBytes(prefix.arenaBytes);probe.set_arenaStride(prefix.arenaStride);
 if(standalone.GetDataSize()!=200||probe.cubeTilingData.GetDataSize()!=200||probe.GetDataSize()!=248)throw std::runtime_error("SDK nested sizes changed");
 std::vector<uint8_t> child(200,0xa5),inner(200,0),parent(248,0);
 probe.cubeTilingData.SaveToBuffer(child.data(),child.size());
 standalone.SaveToBuffer(inner.data(),inner.size());probe.SaveToBuffer(parent.data(),parent.size());
 size_t matches=0,offset=0;for(size_t i=0;i+inner.size()<=parent.size();++i)if(std::memcmp(parent.data()+i,inner.data(),inner.size())==0){++matches;offset=i;}
 if(matches!=1||offset!=sizeof(prefix)||offset+inner.size()!=parent.size()||std::memcmp(parent.data(),&prefix,sizeof(prefix))!=0)throw std::runtime_error("SDK nested layout proof failed");
 // The marker pattern must actually have been serialized; an all-zero match is not evidence.
 for(size_t i=0;i<50;++i){uint32_t x=0;std::memcpy(&x,inner.data()+4*i,4);if(x!=0x13570000U+i)throw std::runtime_error("SDK scalar schema order changed");}
 bool unchanged=true;for(auto b:child)unchanged=unchanged&&(b==0xa5);
 return {offset,inner.size(),parent.size(),unchanged};
}
