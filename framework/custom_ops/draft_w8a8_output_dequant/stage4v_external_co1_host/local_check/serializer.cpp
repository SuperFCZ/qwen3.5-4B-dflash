// Ordinary Host CPU serializer test; no ACL, PlatformInfo, GetTiling or Kernel.
#include "d_flash_external_co1_host_probe_tiling.h"
#include <iostream>
#include <fstream>
#include "co1_fields.h"
#include "nested_serialization.h"
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("output directory required");std::string d=argv[1];
 using T=optiling::DFlashExternalCo1HostProbeTilingData;
 auto layout=Co1ProveLayout<T>();T td;
 td.set_localMemSize(202240);td.set_outputMode(0);td.set_tileId(0);td.set_launchBlocks(1);
 td.set_systemBytes(2097152);td.set_arenaBytes(2097152);td.set_arenaStride(2098176);
 Co1MarkerFields(td.cubeTilingData);
 for(unsigned fill:{0U,0xa5U,0xffU}){
  std::vector<uint8_t> child(200,fill);td.cubeTilingData.SaveToBuffer(child.data(),child.size());
  for(auto b:child)if(b!=fill)throw std::runtime_error("local nested child behavior changed; review, do not generalize");
 }
 std::vector<uint8_t> parent(248,0);td.SaveToBuffer(parent.data(),parent.size());
 std::ofstream f(d+"/serializer-parent.bin",std::ios::binary);f.write(reinterpret_cast<char*>(parent.data()),parent.size());if(!f)throw std::runtime_error("write parent");f.close();
 std::ofstream j(d+"/serializer-observation.json");
 j<<"{\"status\":\"LOCAL_SERIALIZER_SELF_CHECK_ONLY\",\"cube_payload_offset\":"<<layout.offset<<",\"cube_payload_bytes\":"<<layout.bytes<<",\"op_serialization_bytes\":"<<layout.total<<",\"cube_payload_source\":\"SDK_PARENT_SERIALIZER_CPU_TEST\",\"schema_sha256\":\""<<Co1SchemaSha<<"\",\"layout_proof\":\"SDK_PARENT_AND_STANDALONE_MARKERS\",\"standalone_cube_role\":\"DIAGNOSTIC_ONLY\",\"system_workspace_bytes\":2097152,\"arena_bytes\":2097152,\"fields\":";
 Co1Fields(j,td.cubeTilingData);j<<",\"child_target_unchanged_for_fills\":[0,165,255],\"npu\":\"NOT_RUN\"}\n";
 if(!j)throw std::runtime_error("write JSON");return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
