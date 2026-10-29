// Host-only: no kernel registration, stream, device buffers or launch API.
#include <acl/acl.h>
#include <tiling/platform/platform_ascendc.h>
#include <tiling/tiling_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>
#include <stdexcept>
using namespace matmul_tiling;
static std::ofstream json_out;
static void check(aclError e,const char* name){if(e!=ACL_SUCCESS)throw std::runtime_error(std::string(name)+" returned "+std::to_string(e));}
#define FIELD(name) template<class T> auto field_##name(T& t,int)->decltype(t.get_##name(),void()){json_out<<",\"" #name "\":"<<t.get_##name();} template<class T> void field_##name(T&,long){json_out<<",\"" #name "\":null";}
FIELD(usedCoreNum) FIELD(singleCoreM) FIELD(singleCoreN) FIELD(singleCoreK)
FIELD(baseM) FIELD(baseN) FIELD(baseK) FIELD(depthA1) FIELD(depthB1)
FIELD(dbL0A) FIELD(dbL0B) FIELD(dbL0C) FIELD(transLength)
FIELD(stepM) FIELD(stepN) FIELD(stepKa) FIELD(stepKb) FIELD(iterateOrder)
FIELD(shareMode) FIELD(shareL1Size) FIELD(shareL0CSize) FIELD(shareUbSize)
FIELD(isBias) FIELD(Ka) FIELD(Kb) FIELD(M) FIELD(N)
#undef FIELD
// Preserve actual return values including APIs whose installed signature is void.
template<class F> bool call(const char* name,F fn,bool& first){
 if(!first)json_out<<",";first=false;json_out<<"\""<<name<<"\":";
 if constexpr(std::is_void_v<decltype(fn())>){fn();json_out<<"null";return true;}
 else {auto rc=fn();json_out<<static_cast<long long>(rc);return rc>=0;}
}
static void query(platform_ascendc::PlatformAscendC& platform,const std::string& folder,const char* name,int n,int bn,int scratch){
 MatmulApiTiling mm(platform);bool first=true,ok=true;
 json_out<<"{\"name\":\""<<name<<"\",\"requested_n\":"<<n<<",\"requested_base_n\":"<<bn<<",\"scratch\":"<<scratch<<",\"setters\":{";
#define SET(label,expr) if(ok)ok=call(label,[&](){return expr;},first)
 SET("SetAType",mm.SetAType(TPosition::GM,CubeFormat::ND,DataType::DT_INT8));
 SET("SetBType",mm.SetBType(TPosition::GM,CubeFormat::ND,DataType::DT_INT8,false));
 SET("SetCType",mm.SetCType(TPosition::VECIN,CubeFormat::NZ,DataType::DT_FLOAT16));
 SET("SetBiasType",mm.SetBiasType(TPosition::GM,CubeFormat::ND,DataType::DT_INT32));
 SET("SetBias",mm.SetBias(false));
 SET("SetDequantType",mm.SetDequantType(DequantType::TENSOR));
 SET("SetShape",mm.SetShape(16,n,128));SET("SetOrgShape",mm.SetOrgShape(16,n,128));
 SET("SetFixSplit",mm.SetFixSplit(16,bn,128));
 SET("SetBufferSpace",mm.SetBufferSpace(-1,-1,scratch));
#undef SET
 json_out<<"},\"setup_accepted\":"<<(ok?"true":"false");
 if(!ok){json_out<<",\"get_tiling_rc\":null,\"fields\":null}";return;}
 optiling::TCubeTiling td;auto rc=mm.GetTiling(td);json_out<<",\"get_tiling_rc\":"<<rc;
 if(rc<0){json_out<<",\"fields\":null}";return;}
 const auto bytes=td.GetDataSize();if(!bytes||bytes>1024*1024)throw std::runtime_error("Tiling serialized size invalid");
 std::vector<uint8_t> data(bytes);td.SaveToBuffer(data.data(),bytes);
 std::ofstream file(folder+"/"+name+"-tiling.bin",std::ios::binary);file.write(reinterpret_cast<char*>(data.data()),data.size());if(!file)throw std::runtime_error("Tiling save failed");
 json_out<<",\"serialized_bytes\":"<<bytes<<",\"fields\":{\"schema\":1";
#define OUT(name) field_##name(td,0)
 OUT(usedCoreNum);OUT(singleCoreM);OUT(singleCoreN);OUT(singleCoreK);OUT(baseM);OUT(baseN);OUT(baseK);
 OUT(depthA1);OUT(depthB1);OUT(dbL0A);OUT(dbL0B);OUT(dbL0C);OUT(transLength);
 OUT(stepM);OUT(stepN);OUT(stepKa);OUT(stepKb);OUT(iterateOrder);OUT(shareMode);OUT(shareL1Size);OUT(shareL0CSize);OUT(shareUbSize);
 OUT(isBias);OUT(Ka);OUT(Kb);OUT(M);OUT(N);
#undef OUT
 json_out<<"}}";
}
int main(int argc,char**argv){
 bool initialized=false,selected=false;int device=0;
 try{
  if(argc!=3)throw std::runtime_error("usage: stage4i_host DEVICE OUTPUT_DIR");
  size_t end=0;device=std::stoi(argv[1],&end);if(end!=std::string(argv[1]).size()||device<0)throw std::runtime_error("invalid device");
  json_out.open(std::string(argv[2])+"/host-query.json");if(!json_out)throw std::runtime_error("cannot open report");
  check(aclInit(nullptr),"aclInit");initialized=true;check(aclrtSetDevice(device),"aclrtSetDevice");selected=true;
  const char* soc=aclrtGetSocName();if(!soc||std::string(soc)!="Ascend310P3")throw std::runtime_error("actual SoC is not Ascend310P3");
  auto* p=platform_ascendc::PlatformAscendCManager::GetInstance();
  if(!p)throw std::runtime_error("no-argument PlatformAscendCManager initialization returned null");
  auto arch=static_cast<long long>(p->GetCurNpuArch());if(arch!=2002)throw std::runtime_error("platform arch is not 2002");
  uint64_t ub=0,l1=0,l0a=0,l0b=0,l0c=0;
  using C=platform_ascendc::CoreMemType;
  p->GetCoreMemSize(C::UB,ub);p->GetCoreMemSize(C::L1,l1);p->GetCoreMemSize(C::L0_A,l0a);p->GetCoreMemSize(C::L0_B,l0b);p->GetCoreMemSize(C::L0_C,l0c);
  if(!ub||!l1||!l0a||!l0b||!l0c)throw std::runtime_error("zero platform capacity");
  json_out<<"{\"schema\":1,\"platform_source\":\"NO_ARGUMENT_MANAGER_ON_SELECTED_DEVICE\",\"device\":"<<device<<",\"soc\":\""<<soc<<"\",\"arch\":"<<arch
    <<",\"soc_enum\":"<<static_cast<long long>(p->GetSocVersion())<<",\"aic\":"<<p->GetCoreNumAic()<<",\"aiv\":"<<p->GetCoreNumAiv()<<",\"all\":"<<p->GetCoreNum()
    <<",\"system_workspace_bytes\":"<<p->GetLibApiWorkSpaceSize()<<",\"capacity\":{\"UB\":"<<ub<<",\"L1\":"<<l1<<",\"L0A\":"<<l0a<<",\"L0B\":"<<l0b<<",\"L0C\":"<<l0c<<"},\"cases\":[";
  query(*p,argv[2],"N64",64,64,239104);json_out<<",";
  query(*p,argv[2],"T1",128,128,226816);json_out<<",";
  query(*p,argv[2],"T2",128,64,226816);json_out<<"]}\n";
  json_out.flush();if(!json_out)throw std::runtime_error("report write failed");
  check(aclrtResetDevice(device),"aclrtResetDevice");selected=false;check(aclFinalize(),"aclFinalize");initialized=false;return 0;
 }catch(const std::exception& e){std::cerr<<"STAGE4I_HOST_BLOCKED: "<<e.what()<<"\n";if(selected)aclrtResetDevice(device);if(initialized)aclFinalize();return 1;}
}
