// HOST ONLY. No execute API reference, stream, launch or tensor upload exists.
#include "runner_common.h"
#include "aclnn_d_flash_external_co1_host_probe.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdlib>
using namespace draft_quant_test;
struct Tensor{
 void*data=nullptr;aclTensor*desc=nullptr;
 void make(std::vector<int64_t> dims,aclDataType dtype,size_t bytes){
  Check(aclrtMalloc(&data,bytes,ACL_MEM_MALLOC_HUGE_FIRST),"descriptor storage");
  std::vector<int64_t> strides(dims.size(),1);for(size_t i=dims.size()-1;i>0;--i)strides[i-1]=strides[i]*dims[i];
  desc=aclCreateTensor(dims.data(),dims.size(),dtype,strides.data(),0,ACL_FORMAT_ND,dims.data(),dims.size(),data);
  if(!desc)throw std::runtime_error("aclCreateTensor");
 }
 ~Tensor(){if(desc)aclDestroyTensor(desc);if(data)aclrtFree(data);}
};
int main(int argc,char**argv){
 bool init=false,selected=false;int device=0;std::string folder;
 try{
  if(argc!=3)throw std::runtime_error("usage: co1_host_runner DEVICE NEW_CAPTURE_DIR");
  size_t end=0;device=std::stoi(argv[1],&end);if(end!=std::string(argv[1]).size()||device<0)throw std::runtime_error("device");
  folder=argv[2];if(folder.empty()||folder[0]!='/')throw std::runtime_error("absolute capture dir");
  if(setenv("DFLASH_CO1_CAPTURE_DIR",folder.c_str(),1))throw std::runtime_error("capture env");
  Check(aclInit(nullptr),"aclInit");init=true;Check(aclrtSetDevice(device),"aclrtSetDevice");selected=true;
  auto soc=aclrtGetSocName();if(!soc||std::string(soc)!="Ascend310P3")throw std::runtime_error("requires Ascend310P3");
  uint64_t bytes=0;aclnnStatus rc;
  {
   Tensor a,b,s,o;a.make({16,2560},ACL_INT8,40960);b.make({2560,256},ACL_INT8,655360);s.make({256},ACL_UINT64,2048);o.make({16,256},ACL_INT32,16384);
   aclOpExecutor*executor=nullptr;
   rc=aclnnDFlashExternalCo1HostProbeGetWorkspaceSize(a.desc,b.desc,s.desc,o.desc,&bytes,&executor);
   // Executor is intentionally never executed; this bounded Host-only process exits.
   std::ofstream out(folder+"/execution.json");
   out<<"{\"status\":\""<<(rc==0?"HOST_QUERY_RETURNED_ZERO":"HOST_QUERY_REJECTED")<<"\",\"returncode\":"<<rc<<",\"workspace_bytes\":"<<bytes<<",\"device\":"<<device<<",\"soc\":\"Ascend310P3\",\"executor_returned\":"<<(executor?"true":"false")<<",\"execute_api_calls\":0,\"npu\":\"NOT_RUN\",\"input_payload\":\"NOT_UPLOADED_NOT_READ\"}\n";
   if(!out)throw std::runtime_error("execution report");
   if(rc!=0||!executor)throw std::runtime_error("GetWorkspaceSize failed");
  }
  Check(aclrtResetDevice(device),"reset");selected=false;Check(aclFinalize(),"finalize");init=false;return 0;
 }catch(const std::exception&e){std::cerr<<"CO1_HOST_ONLY_BLOCKED: "<<e.what()<<"\n";if(selected)aclrtResetDevice(device);if(init)aclFinalize();return 1;}
}
