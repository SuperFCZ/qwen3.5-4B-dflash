// Raw-only ND INT8 Cube. Two inputs, INT32 output, no scale or timing.
#include "runner_common.h"
#include <iostream>
#include "aclnn_d_flash_w8_a8_accumulator.h"
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
    Tensor x,w,y;GuardedBuffer workspace;
    ~Session(){if(stream)aclrtSynchronizeStream(stream);y.Release();w.Release();x.Release();workspace.Release();
        if(stream)aclrtDestroyStream(stream);if(selected)aclrtResetDevice(device);if(initialized)aclFinalize();}
};
int main(int argc,char **argv) {
    std::string folder;
    try {
        if(argc!=5)throw std::runtime_error("usage: runner DEVICE CASE_DIR nd raw");
        folder=argv[2];std::ofstream(folder+"/execution.json")<<"{\"status\":\"RUNNING\"}\n";
        Session r;size_t end=0;std::string id=argv[1];r.device=std::stoi(id,&end);
        if(end!=id.size()||r.device<0)throw std::runtime_error("invalid device id");
        if(std::string(argv[3])!="nd"||std::string(argv[4])!="raw")throw std::runtime_error("raw-only requires nd raw");
        auto x=ReadBytes(folder+"/a.bin",2048),w=ReadBytes(folder+"/b_nd.bin",8192);
        const char *op="DFlashW8A8Accumulator";
        const size_t outputBytes=4096;
        Check(aclInit(nullptr),"aclInit");r.initialized=true;
        Check(aclrtSetDevice(r.device),"aclrtSetDevice");r.selected=true;
        auto soc=aclrtGetSocName();if(!soc||std::string(soc)!="Ascend310P3")throw std::runtime_error("actual SoC must be Ascend310P3");
        Check(aclrtCreateStream(&r.stream),"aclrtCreateStream");
        r.x.Make({16,128},{16,128},ACL_INT8,ACL_FORMAT_ND,2048,&x);
        r.y.Make({16,64},{16,64},ACL_INT32,ACL_FORMAT_ND,outputBytes);
        r.w.Make({128,64},{128,64},ACL_INT8,ACL_FORMAT_ND,8192,&w);
        uint64_t size=0,previous=0;aclOpExecutor *executor=nullptr;std::vector<uint8_t> first;
        Timing timing;  // no performance stage, not even warmup
        auto prepare=[&](){
            executor=nullptr;size=0;
            Check(aclnnDFlashW8A8AccumulatorGetWorkspaceSize(r.x.desc,r.w.desc,r.y.desc,&size,&executor),"raw GetWorkspaceSize");
            if(!first.empty()&&previous!=size)throw std::runtime_error("workspace drift");
            if(size&&r.workspace.bytes!=size)r.workspace.Allocate(size);previous=size;
        };
        auto execute=[&](){
            Check(aclnnDFlashW8A8Accumulator(r.workspace.data,size,executor,r.stream),"raw execute");
            Check(aclrtSynchronizeStream(r.stream),"synchronize");
        };
        auto poison=[&](int i){Check(aclrtMemset(r.y.buffer.data,outputBytes,i?0xff:0x7f,outputBytes),"poison");};
        auto check=[&](const std::string &file){
            CheckNamedGuards({{"x",&r.x.buffer},{"w",&r.w.buffer},
                              {"y",&r.y.buffer},{"workspace",&r.workspace}},std::cout);
            if(r.x.buffer.CopyOut()!=x||r.w.buffer.CopyOut()!=w)throw std::runtime_error("read-only x/w changed");
            auto out=r.y.buffer.CopyOut();WriteBytes(folder+"/"+file,out);
            if(first.empty())first=out;else if(out!=first)throw std::runtime_error("repeat output drift");
        };
        RunMeasuredCalls(timing,prepare,execute,poison,check);
        std::ofstream report(folder+"/execution.json");
        report<<"{\"status\":\"EXECUTION_COMPLETED_NUMERICS_UNCHECKED\",\"op\":\""<<op<<"\",\"soc\":\"Ascend310P3\","
            "\"shape\":[16,128,64],\"guards\":true,\"readonly\":true,\"repeat_equal\":true,\"repetitions\":2,"
            "\"device\":"<<r.device<<",\"workspace_bytes\":"<<size<<",\"cpu_fallback\":false";
        report<<",\"layout\":\"nd\",\"raw_accumulator\":true,\"raw_only\":true,\"input_count\":2";
        timing.Write(report);report<<"}\n";if(!report)throw std::runtime_error("report write failed");return 0;
    } catch(const std::exception &e){
        if(!folder.empty())std::ofstream(folder+"/execution.json")<<"{\"status\":\"FAIL\"}\n";
        std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;
    }
}
