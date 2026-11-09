#include "runner_common.h"
#include "aclnn_d_flash_external_co1_echo_probe.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include "co1_fields.h"
using namespace draft_quant_test;
struct Tensor {
    GuardedBuffer buffer; aclTensor *desc=nullptr;
    void Make(const std::vector<int64_t>& logical, const std::vector<int64_t>& storage,
              aclDataType dt, aclFormat fmt, size_t bytes, const std::vector<uint8_t>* data=nullptr) {
        buffer.Allocate(bytes); if(data)buffer.CopyIn(*data);
        std::vector<int64_t> stride(logical.size(),1);
        for(size_t i=logical.size()-1;i>0;--i)stride[i-1]=stride[i]*logical[i];
        desc=aclCreateTensor(logical.data(),logical.size(),dt,stride.data(),0,fmt,
                             storage.data(),storage.size(),buffer.data);
        if(!desc)throw std::runtime_error("aclCreateTensor failed");
    }
    void Release(){if(desc)aclDestroyTensor(desc);desc=nullptr;buffer.Release();}
};
struct Session {
    bool initialized=false,selected=false;int device=0;aclrtStream stream=nullptr;
    Tensor x,w,s,y,z;GuardedBuffer workspace;
    ~Session(){if(stream)aclrtSynchronizeStream(stream);z.Release();y.Release();s.Release();w.Release();x.Release();workspace.Release();
        if(stream)aclrtDestroyStream(stream);if(selected)aclrtResetDevice(device);if(initialized)aclFinalize();}
};

int main(int argc,char**argv){std::string folder;try{
 if(argc!=4)throw std::runtime_error("usage: echo_runner DEVICE CASE_DIR host|echo");
 bool host=std::string(argv[3])=="host";if(!host&&std::string(argv[3])!="echo")throw std::runtime_error("VDEQ16 not implemented or admitted");
 folder=argv[2];if(folder.empty()||folder[0]!='/')throw std::runtime_error("absolute case directory");
 if(setenv("DFLASH_CO1_CAPTURE_DIR",folder.c_str(),1))throw std::runtime_error("capture env");
 Session r;size_t end=0;std::string device=argv[1];r.device=std::stoi(device,&end);if(end!=device.size()||r.device<0)throw std::runtime_error("device");
 auto a=ReadBytes(folder+"/a.bin",40960),b=ReadBytes(folder+"/b.bin",655360),scale=ReadBytes(folder+"/scale.bin",2048),expected=ReadBytes(folder+"/expected.bin",16384);
 Check(aclInit(nullptr),"init");r.initialized=true;Check(aclrtSetDevice(r.device),"device");r.selected=true;
 const char*soc=aclrtGetSocName();if(!soc||std::string(soc)!="Ascend310P3")throw std::runtime_error("SoC");
 r.x.Make({16,2560},{16,2560},ACL_INT8,ACL_FORMAT_ND,40960,&a);r.w.Make({2560,256},{2560,256},ACL_INT8,ACL_FORMAT_ND,655360,&b);r.s.Make({256},{256},ACL_UINT64,ACL_FORMAT_ND,2048,&scale);
 r.y.Make({16,256},{16,256},ACL_INT32,ACL_FORMAT_ND,16384);r.z.Make({32},{32},ACL_UINT64,ACL_FORMAT_ND,256);
 if(!host)Check(aclrtCreateStream(&r.stream),"stream");
 uint64_t previous=0;std::vector<uint8_t> first;
 std::ofstream(folder+"/execution.json")<<"{\"status\":\"RUNNING\",\"npu\":\"NOT_RUN\"}\n";
 for(int repeat=0;repeat<(host?1:2);++repeat){uint64_t bytes=0;aclOpExecutor*executor=nullptr;
  auto rc=aclnnDFlashExternalCo1EchoProbeGetWorkspaceSize(r.x.desc,r.w.desc,r.s.desc,r.y.desc,r.z.desc,&bytes,&executor);
  if(rc!=0||!executor)throw std::runtime_error("GetWorkspaceSize rejected");
  if(bytes!=EchoExpectedWorkspace||repeat&&bytes!=previous)throw std::runtime_error("Workspace differs from admitted Formal contract");
  previous=bytes;
  if(host){std::ofstream(folder+"/execution.json")<<"{\"status\":\"HOST_ONLY\",\"npu\":\"NOT_RUN\",\"workspace_bytes\":"<<bytes<<",\"device\":"<<r.device<<"}\n";return 0;}
  if(!repeat)r.workspace.Allocate(bytes);
  const uint8_t poison=repeat?0x80:0x7f;
  Check(aclrtMemset(r.y.buffer.data,16384,poison,16384),"poison echo");Check(aclrtMemset(r.z.buffer.data,256,poison,256),"poison witness");
  if(r.y.buffer.CopyOut()!=std::vector<uint8_t>(16384,poison)||r.z.buffer.CopyOut()!=std::vector<uint8_t>(256,poison))throw std::runtime_error("Poison verification");
  CheckNamedGuards({{"a",&r.x.buffer},{"b",&r.w.buffer},{"scale",&r.s.buffer},{"echo",&r.y.buffer},{"witness",&r.z.buffer},{"workspace",&r.workspace}},std::cout);
  std::ofstream(folder+"/execution.json")<<"{\"status\":\"RUNNING\",\"npu\":\"ATTEMPTED\"}\n";
  Check(aclnnDFlashExternalCo1EchoProbe(r.workspace.data,bytes,executor,r.stream),"Echo execute");Check(aclrtSynchronizeStream(r.stream),"sync");
  auto y=r.y.buffer.CopyOut(),w=r.z.buffer.CopyOut();WriteBytes(folder+"/actual-"+std::to_string(repeat)+".bin",y);WriteBytes(folder+"/witness-"+std::to_string(repeat)+".bin",w);
  CheckNamedGuards({{"a",&r.x.buffer},{"b",&r.w.buffer},{"scale",&r.s.buffer},{"echo",&r.y.buffer},{"witness",&r.z.buffer},{"workspace",&r.workspace}},std::cout);
  if(r.x.buffer.CopyOut()!=a||r.w.buffer.CopyOut()!=b||r.s.buffer.CopyOut()!=scale)throw std::runtime_error("Readonly input changed");
  uint64_t meta[32];std::memcpy(meta,w.data(),256);
  if(meta[0]!=0x434f314543484f31ULL||meta[1]!=1||meta[2]!=0||meta[3]!=0||meta[4]!=16384||meta[5]!=230912||meta[6]!=1||meta[7]!=0)throw std::runtime_error("Device witness/resource/guard failure");
  for(int i=0;i<8;++i)if(meta[8+i]>=8||!meta[16+i])throw std::runtime_error("Event witness missing");
  if(y!=expected)throw std::runtime_error("INT32 Echo mismatch");if(!repeat)first=y;else if(y!=first)throw std::runtime_error("Repeat drift");
 }
 std::ofstream out(folder+"/execution.json");out<<"{\"status\":\"ECHO_EXECUTION_MATCH\",\"npu\":\"COMPLETED\",\"device\":"<<r.device<<",\"workspace_bytes\":"<<previous<<",\"repetitions\":2,\"guards\":true,\"readonly\":true,\"poison\":true,\"repeat_equal\":true,\"l0c_guard\":\"RANGE_ONLY_NO_CANARIES\",\"vdeq16\":\"NOT_IMPLEMENTED_PENDING_ECHO_PASS\",\"cpu_fallback\":false}\n";
 if(!out)throw std::runtime_error("report write");return 0;
 }catch(const std::exception&e){std::cerr<<"ECHO_BLOCKED: "<<e.what()<<"\n";return 1;}}
