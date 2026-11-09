#pragma once
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <fcntl.h>
#include <unistd.h>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "core_probe.h"
#include <cstring>
#include "co1_fields.h"
#include "nested_serialization.h"
inline void Co1Write(const std::string&p,const void*data,size_t bytes){
 int fd=open(p.c_str(),O_CREAT|O_EXCL|O_WRONLY|O_NOFOLLOW,0600);
 if(fd<0)throw std::runtime_error("capture open failed");
 const char*s=static_cast<const char*>(data);size_t done=0;
 while(done<bytes){auto n=write(fd,s+done,bytes-done);if(n<=0){close(fd);throw std::runtime_error("capture write failed");}done+=n;}
 if(close(fd))throw std::runtime_error("capture close failed");
}
struct Co1Capture{
 std::string stem,fields="null",error="",status="BLOCKED";
 std::ostringstream setters;bool first=true;long long rc=-999;bool called=false;
 uint64_t ub=0,l1=0,l0a=0,l0b=0,l0c=0,system=0,arena=0,total=0;
 Co1SerializationLayout layout{0,0,0,false};
 S4CO1CoreSnapshot core{0,0,0,-1};uint32_t selected=0;std::map<std::string,bool> checks;
 Co1Capture(){static std::atomic<unsigned>count{0};const char*d=std::getenv("DFLASH_CO1_CAPTURE_DIR");if(!d||d[0]!='/')throw std::runtime_error("absolute capture directory required");stem=std::string(d)+"/callback-"+std::to_string(count++);}
 template<class F>auto call(const char*,F f)->typename std::enable_if<std::is_void<decltype(f())>::value,void>::type{f();setters<<"null";}
 template<class F>auto call(const char*n,F f)->typename std::enable_if<!std::is_void<decltype(f())>::value,void>::type{auto value=f();setters<<value;if(value<0)throw std::runtime_error(n);}
 template<class F>void set(const char*n,F f){if(!first)setters<<",";first=false;setters<<"\""<<n<<"\":";call(n,f);}
 void check(const char*n,bool ok){checks[n]=ok;}
 bool ok(){for(auto&x:checks)if(!x.second)return false;return true;}
 void finish(){std::ostringstream o;o<<"{\"status\":\""<<status<<"\",\"context\":\"REGISTERED_FORMAL_TILING_CALLBACK\",\"kernel\":\"ECHO_ONLY\",\"npu\":\"NOT_RUN\",\"error\":\""<<error<<"\",\"setters\":{"<<setters.str()<<"},\"get_tiling_rc\":";
 if(called)o<<rc;else o<<"null";
 o<<",\"fields\":"<<fields<<",\"cube_payload_offset\":"<<layout.offset<<",\"cube_payload_bytes\":"<<layout.bytes<<",\"cube_payload_source\":\"FORMAL_RAW_TILING_DATA_PARENT_SLICE\",\"op_serialization_bytes\":"<<layout.total<<",\"schema_sha256\":\""<<Co1SchemaSha<<"\",\"layout_proof\":\"SDK_PARENT_AND_STANDALONE_MARKERS\",\"child_save_unchanged_in_selftest\":"<<(layout.childSaveLeftDestinationUnchanged?"true":"false")<<",\"standalone_cube_role\":\"DIAGNOSTIC_ONLY\",\"capacity\":{\"UB\":"<<ub<<",\"L1\":"<<l1<<",\"L0A\":"<<l0a<<",\"L0B\":"<<l0b<<",\"L0C\":"<<l0c<<"},\"core\":{\"aic\":"<<core.aic<<",\"aiv\":"<<core.aiv<<",\"all\":"<<core.all<<",\"selected\":"<<selected<<",\"arch\":"<<core.arch<<"},\"block_dim\":1,\"system_workspace_bytes\":"<<system<<",\"arena_bytes\":"<<arena<<",\"requested_workspace_bytes\":"<<total<<",\"checks\":{";
 bool f=true;for(auto&x:checks){if(!f)o<<",";f=false;o<<"\""<<x.first<<"\":"<<(x.second?"true":"false");}
 o<<"},\"external_l0c\":{\"bytes\":16384,\"start\":0,\"alignment\":1024,\"effective_shared\":0,\"allocation\":\"PLANNED_NOT_EXECUTED\"},\"ub\":{\"scratch\":202240,\"echo\":16384,\"scale\":2048,\"fp16\":8192,\"guards\":2048,\"reserve\":8192,\"total\":239104,\"effective_shared\":0}}\n";
 auto text=o.str();Co1Write(stem+".json",text.data(),text.size());}
};
inline bool S4CO1Dims(const gert::Shape*s,std::initializer_list<int64_t>d){if(!s||s->GetDimNum()!=d.size())return false;size_t i=0;for(auto v:d)if(s->GetDim(i++)!=v)return false;return true;}
inline bool Co1Input(gert::TilingContext*c,size_t i,ge::DataType dt,std::initializer_list<int64_t>d){auto s=c->GetInputShape(i);auto t=c->GetInputDesc(i);return s&&t&&t->GetDataType()==dt&&t->GetStorageFormat()==ge::FORMAT_ND&&S4CO1Dims(&s->GetOriginShape(),d)&&S4CO1Dims(&s->GetStorageShape(),d);}
inline ge::graphStatus S4CO1Shape(gert::InferShapeContext*c){if(!S4CO1Dims(c->GetInputShape(0),{16,2560})||!S4CO1Dims(c->GetInputShape(1),{2560,256})||!S4CO1Dims(c->GetInputShape(2),{256}))return ge::GRAPH_FAILED;auto y=c->GetOutputShape(0);if(!y)return ge::GRAPH_FAILED;y->SetDimNum(2);y->SetDim(0,16);y->SetDim(1,256);auto w=c->GetOutputShape(1);if(!w)return ge::GRAPH_FAILED;w->SetDimNum(1);w->SetDim(0,32);return ge::GRAPH_SUCCESS;}
template<class T> bool Co1Aux(T&p); // generated with all remaining schema fields

template<class Tiling>ge::graphStatus S4CO1Tiling(gert::TilingContext*c){
 try{Co1Capture r;
  try{
   if(!Co1Input(c,0,ge::DT_INT8,{16,2560})||!Co1Input(c,1,ge::DT_INT8,{2560,256})||!Co1Input(c,2,ge::DT_UINT64,{256}))throw std::runtime_error("input ABI");
   auto y=c->GetOutputShape(0);auto yd=c->GetOutputDesc(0);
   if(!y||!yd||yd->GetDataType()!=ge::DT_INT32||yd->GetStorageFormat()!=ge::FORMAT_ND||!S4CO1Dims(&y->GetStorageShape(),{16,256})||!S4CO1Dims(&y->GetOriginShape(),{16,256}))throw std::runtime_error("output ABI");
   auto w=c->GetOutputShape(1);auto wd=c->GetOutputDesc(1);if(!w||!wd||wd->GetDataType()!=ge::DT_UINT64||wd->GetStorageFormat()!=ge::FORMAT_ND||!S4CO1Dims(&w->GetOriginShape(),{32})||!S4CO1Dims(&w->GetStorageShape(),{32}))throw std::runtime_error("witness ABI");
   if(!c->GetPlatformInfo())throw std::runtime_error("formal PlatformInfo absent");
   platform_ascendc::PlatformAscendC platform(c->GetPlatformInfo());r.core=S4CO1ProbeCores(platform);r.selected=S4CO1SelectSchedulingLimit(r.core);
   if(!r.selected)throw std::runtime_error("invalid coupled core policy");
   using P=platform_ascendc::CoreMemType;
   platform.GetCoreMemSize(P::UB,r.ub);platform.GetCoreMemSize(P::L1,r.l1);platform.GetCoreMemSize(P::L0_A,r.l0a);platform.GetCoreMemSize(P::L0_B,r.l0b);platform.GetCoreMemSize(P::L0_C,r.l0c);
   r.system=platform.GetLibApiWorkSpaceSize();if(!r.system||r.system>512ULL*1024*1024)throw std::runtime_error("system Workspace range");r.arena=(r.system+511)/512*512;r.total=r.system+r.arena+1024;
   using namespace matmul_tiling;MatmulApiTiling mm(platform);
   r.set("SetAType",[&]{return mm.SetAType(TPosition::GM,CubeFormat::ND,DataType::DT_INT8);});
   r.set("SetBType",[&]{return mm.SetBType(TPosition::GM,CubeFormat::ND,DataType::DT_INT8,false);});
   r.set("SetCType",[&]{return mm.SetCType(TPosition::CO1,CubeFormat::NZ,DataType::DT_INT32);});
   r.set("SetBiasType",[&]{return mm.SetBiasType(TPosition::GM,CubeFormat::ND,DataType::DT_INT32);});
   r.set("SetBias",[&]{return mm.SetBias(false);});
   r.set("SetShape",[&]{return mm.SetShape(16,256,2560);});r.set("SetOrgShape",[&]{return mm.SetOrgShape(16,256,2560);});
   r.set("SetFixSplit",[&]{return mm.SetFixSplit(16,256,128);});r.set("SetBufferSpace",[&]{return mm.SetBufferSpace(-1,-1,202240);});
   r.layout=Co1ProveLayout<Tiling>();
   Tiling td;r.called=true;r.rc=mm.GetTiling(td.cubeTilingData);
   if(r.rc!=0)throw std::runtime_error("GetTiling rejected CO1");auto&p=td.cubeTilingData;
   std::ostringstream fields;Co1Fields(fields,p);r.fields=fields.str();
   if(p.GetDataSize()!=200)throw std::runtime_error("TCube serialized size changed");std::vector<uint8_t> cube(p.GetDataSize());p.SaveToBuffer(cube.data(),cube.size());Co1Write(r.stem+"-cube.bin",cube.data(),cube.size());
   r.check("admitted_formal_contract",EchoAccepted(p,r.core,r.selected,r.ub,r.l1,r.l0a,r.l0b,r.l0c,r.system));
   r.check("shape",p.get_singleCoreM()==16&&p.get_singleCoreN()==256&&p.get_singleCoreK()==2560&&p.get_baseM()==16&&p.get_baseN()==256&&p.get_baseK()==128&&p.get_M()==16&&p.get_N()==256&&p.get_Ka()==2560&&p.get_Kb()==2560);
   r.check("scope",p.get_usedCoreNum()==1&&p.get_isBias()==0&&p.get_shareMode()==0);
   auto ka=p.get_stepKa(),kb=p.get_stepKb(),da=p.get_depthA1(),db=p.get_depthB1();
   bool steps=ka>0&&kb>0&&ka<=20&&kb<=20&&20%ka==0&&20%kb==0&&(ka%kb==0||kb%ka==0);
   r.check("steps",steps&&p.get_stepM()==1&&p.get_stepN()==1);
   r.check("depths",steps&&da>0&&db>0&&da<=20&&db<=20&&da%ka==0&&db%kb==0&&(da/ka==1||da/ka==2)&&(db/kb==1||db/kb==2));
   uint64_t l1=(da>0?uint64_t(da)*2048:0)+(db>0?uint64_t(db)*32768:0);
   r.check("L1",r.l1>0&&l1>0&&l1<=r.l1&&p.get_shareL1Size()>=0&&l1==uint64_t(p.get_shareL1Size()));
   auto a0=p.get_dbL0A(),b0=p.get_dbL0B(),c0=p.get_dbL0C();
   r.check("db",(a0==1||a0==2)&&(b0==1||b0==2)&&(c0==1||c0==2));
   r.check("L0AB",r.l0a>=uint64_t(a0)*2048&&r.l0b==65536&&r.l0b>=uint64_t(b0)*32768);
   r.check("external_L0C",r.l0c>=16384&&p.get_shareL0CSize()>=0);
   r.check("UB",r.ub==262144&&239104<=r.ub&&p.get_shareUbSize()>=0);
   r.check("scratch",p.get_transLength()>=0&&p.get_transLength()%32==0&&p.get_transLength()<=202240);
   r.check("auxiliary_consumers",Co1Aux(p));r.check("Workspace",r.total<=512ULL*1024*1024);
   if(!r.ok())throw std::runtime_error("resource checks rejected");
   td.set_localMemSize(202240);td.set_outputMode(0);td.set_tileId(0);td.set_launchBlocks(1);td.set_systemBytes(r.system);td.set_arenaBytes(r.arena);td.set_arenaStride(r.arena+1024);
   c->SetBlockDim(1);c->SetTilingKey(0);auto raw=c->GetRawTilingData();
   if(!raw||!raw->GetData()||raw->GetCapacity()<td.GetDataSize())throw std::runtime_error("Op Tiling capacity");
   td.SaveToBuffer(raw->GetData(),raw->GetCapacity());raw->SetDataSize(td.GetDataSize());c->GetWorkspaceSizes(1)[0]=r.total;
   if(td.GetDataSize()!=r.layout.total||r.layout.offset+r.layout.bytes!=r.layout.total)throw std::runtime_error("actual parent serialization size changed");
   Co1WirePrefix expected;std::memset(&expected,0,sizeof(expected));expected.localMemSize=202240;expected.launchBlocks=1;expected.systemBytes=r.system;expected.arenaBytes=r.arena;expected.arenaStride=r.arena+1024;
   if(std::memcmp(raw->GetData(),&expected,sizeof(expected))||!EchoWireMatches(p,static_cast<const unsigned char*>(raw->GetData())))throw std::runtime_error("actual serialized execution contract mismatch");
   Co1Write(r.stem+"-op.bin",raw->GetData(),td.GetDataSize());
   Co1Write(r.stem+"-cube-payload.bin",static_cast<const uint8_t*>(raw->GetData())+r.layout.offset,r.layout.bytes);r.status="FORMAL_CALLBACK_ACCEPTED";
  }catch(const std::exception&e){r.error=e.what();std::fprintf(stderr,"CO1_HOST_REJECT: %s\n",e.what());}
  r.finish();return r.status=="FORMAL_CALLBACK_ACCEPTED"?ge::GRAPH_SUCCESS:ge::GRAPH_FAILED;
 }catch(const std::exception&e){std::fprintf(stderr,"CO1_CAPTURE_REJECT: %s\n",e.what());return ge::GRAPH_FAILED;}
}
