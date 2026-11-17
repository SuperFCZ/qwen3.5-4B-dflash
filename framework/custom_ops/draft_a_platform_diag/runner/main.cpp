// Host-only ACLNN metadata diagnostic. No aclnn Execute call exists here.
#include "runner_common.h"
#include "aclnn_d_flash_group_quant_linear.h"
#include <fstream>
#include <iostream>
#include <sstream>
using namespace draft_quant_test;
struct Tensor { GuardedBuffer data;aclTensor* desc=nullptr;~Tensor(){if(desc)aclDestroyTensor(desc);data.Release();} };
void Bind(Tensor&t,std::vector<int64_t>shape,aclDataType type,aclFormat fmt,const std::vector<uint8_t>&bytes){
 t.data.Allocate(bytes.size());t.data.CopyIn(bytes);std::vector<int64_t>strides(shape.size(),1);
 for(size_t i=shape.size();i>1;--i)strides[i-2]=strides[i-1]*shape[i-1];
 t.desc=aclCreateTensor(shape.data(),shape.size(),type,strides.data(),0,fmt,shape.data(),shape.size(),t.data.data);
 if(!t.desc)throw std::runtime_error("aclCreateTensor");
}
struct Session {int device=0;bool init=false,set=false;aclrtStream stream=nullptr;
 ~Session(){if(stream)aclrtSynchronizeStream(stream);if(stream)aclrtDestroyStream(stream);if(set)aclrtResetDevice(device);if(init)aclFinalize();}};
int main(int argc,char**argv){
 if(argc!=3)return 2;std::string root=argv[2];Session session;
 try{
  session.device=std::stoi(argv[1]);
  auto journal=[&](const char*event,aclError rc,const std::string&extra=""){
   std::ofstream f(root+"/runtime-events.jsonl",std::ios::app);f<<"{\"event\":\""<<event<<"\",\"returncode\":"<<rc<<extra<<"}\n";
   if(!f)throw std::runtime_error("runtime journal write");Check(rc,event);
  };
  aclError rc=aclInit(nullptr);session.init=rc==ACL_SUCCESS;journal("aclInit",rc);
  rc=aclrtSetDevice(session.device);session.set=rc==ACL_SUCCESS;journal("aclrtSetDevice",rc,",\"requested_device\":"+std::to_string(session.device));
  int32_t actual=-1;rc=aclrtGetDevice(&actual);journal("aclrtGetDevice",rc,",\"device\":"+std::to_string(actual));
  if(actual!=session.device)throw std::runtime_error("wrong current device");
  aclrtContext before=nullptr;rc=aclrtGetCurrentContext(&before);journal("context_after_set_device",rc,std::string(",\"present\":")+(before?"true":"false"));
  if(!before)throw std::runtime_error("default Context absent");
  rc=aclrtCreateStream(&session.stream);journal("aclrtCreateStream",rc,std::string(",\"present\":")+(session.stream?"true":"false"));
  aclrtContext after=nullptr;rc=aclrtGetCurrentContext(&after);journal("context_after_stream",rc,std::string(",\"same_context\":")+(after==before?"true":"false"));
  if(!session.stream||after!=before)throw std::runtime_error("Context/Stream mismatch");
  auto x=ReadBytes(root+"/x.bin",16*12800*2),w=ReadBytes(root+"/w_nz.bin",12800*2560),s=ReadBytes(root+"/s_gn.bin",100*2560*2);
  struct Workspace : GuardedBuffer { ~Workspace(){Release();} };
  Tensor a,b,scale,y;Workspace workspace;
  Bind(a,{16,12800},ACL_FLOAT16,ACL_FORMAT_ND,x);Bind(b,{400,160,16,32},ACL_INT8,ACL_FORMAT_FRACTAL_NZ,w);
  Bind(scale,{100,2560},ACL_FLOAT16,ACL_FORMAT_ND,s);std::vector<uint8_t>poison(16*2560*2,0x7f);Bind(y,{16,2560},ACL_FLOAT16,ACL_FORMAT_ND,poison);
  uint64_t first=0;
  for(int i=0;i<2;++i){uint64_t size=0;aclOpExecutor*executor=nullptr;
   rc=aclnnDFlashGroupQuantLinearGetWorkspaceSize(a.desc,b.desc,scale.desc,y.desc,&size,&executor);
   journal("GetWorkspaceSize",rc,",\"call\":"+std::to_string(i)+",\"workspace\":"+std::to_string(size)+",\"executor_present\":"+(executor?"true":"false"));
   if(!executor)throw std::runtime_error("missing executor");if(i==0){first=size;if(size)workspace.Allocate(size);}else if(size!=first)throw std::runtime_error("Workspace drift");
   int32_t device=-1;Check(aclrtGetDevice(&device),"recheck device");aclrtContext ctx=nullptr;Check(aclrtGetCurrentContext(&ctx),"recheck Context");
   if(device!=actual||ctx!=before)throw std::runtime_error("runtime selection drift");
  }
  journal("aclrtSynchronizeStream",aclrtSynchronizeStream(session.stream));
  a.data.CheckGuards();b.data.CheckGuards();scale.data.CheckGuards();y.data.CheckGuards();workspace.CheckGuards();
  if(a.data.CopyOut()!=x||b.data.CopyOut()!=w||scale.data.CopyOut()!=s||y.data.CopyOut()!=poison)throw std::runtime_error("readonly/Poison changed without Execute");
  {std::ifstream maps("/proc/self/maps",std::ios::binary);std::ofstream snapshot(root+"/maps.txt",std::ios::binary);
   if(!maps||!snapshot)throw std::runtime_error("maps snapshot open");snapshot<<maps.rdbuf();
   if(!snapshot)throw std::runtime_error("maps snapshot write");}
  std::ofstream report(root+"/execution.json");report<<"{\"status\":\"HOST_TILING_ONLY\",\"m\":16,\"k\":12800,\"n\":2560,\"device\":"<<actual
   <<",\"workspace\":"<<first<<",\"acl_init\":true,\"set_device\":true,\"context_present\":true,\"context_stable\":true,\"stream_created\":true,\"guards\":true,\"readonly\":true,\"poison_unchanged\":true,\"execute_calls\":0,\"explicit_context_created\":false}\n";
  if(!report)throw std::runtime_error("execution report write");return 0;
 }catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<'\n';std::ofstream(root+"/execution.json")<<"{\"status\":\"BLOCKED\",\"execute_calls\":0}\n";return 1;}
}
