// Stage3B-1 ordered partial reduction only. No timing or earlier-kernel execution.
#include "runner_common.h"
#include <iostream>
#include "aclnn_d_flash_w8_a8_partial_reduce.h"
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
    Tensor x,y;GuardedBuffer workspace;
    ~Session(){if(stream)aclrtSynchronizeStream(stream);y.Release();x.Release();workspace.Release();
        if(stream)aclrtDestroyStream(stream);if(selected)aclrtResetDevice(device);if(initialized)aclFinalize();}
};
int main(int argc,char **argv) {
    std::string folder;
    try {
        if(argc!=3)throw std::runtime_error("usage: runner DEVICE CASE_DIR");
        folder=argv[2];std::ofstream(folder+"/execution.json")<<"{\"status\":\"RUNNING\"}\n";
        Session r;size_t end=0;std::string id=argv[1];r.device=std::stoi(id,&end);
        if(end!=id.size()||r.device<0)throw std::runtime_error("invalid device id");
        auto x=ReadBytes(folder+"/partials_fp16.bin",40960);
        const char *op="DFlashW8A8PartialReduce";
        const size_t outputBytes=2048;
        Check(aclInit(nullptr),"aclInit");r.initialized=true;
        Check(aclrtSetDevice(r.device),"aclrtSetDevice");r.selected=true;
        auto soc=aclrtGetSocName();if(!soc||std::string(soc)!="Ascend310P3")throw std::runtime_error("actual SoC must be Ascend310P3");
        Check(aclrtCreateStream(&r.stream),"aclrtCreateStream");
        r.x.Make({20,16,64},{20,16,64},ACL_FLOAT16,ACL_FORMAT_ND,40960,&x);
        r.y.Make({16,64},{16,64},ACL_FLOAT16,ACL_FORMAT_ND,outputBytes);
        uint64_t size=0,previous=0;aclOpExecutor *executor=nullptr;std::vector<uint8_t> first;
        Timing timing;  // no performance stage, not even warmup
        auto prepare=[&](){
            executor=nullptr;size=0;
            Check(aclnnDFlashW8A8PartialReduceGetWorkspaceSize(r.x.desc,r.y.desc,&size,&executor),"Stage3B GetWorkspaceSize");
            if(!first.empty()&&previous!=size)throw std::runtime_error("workspace drift");
            if(size&&r.workspace.bytes!=size)r.workspace.Allocate(size);previous=size;
            CheckNamedGuards({{"partials",&r.x.buffer},{"o",&r.y.buffer},{"workspace",&r.workspace}},std::cout);
        };
        auto execute=[&](){
            Check(aclnnDFlashW8A8PartialReduce(r.workspace.data,size,executor,r.stream),"Stage3B execute");
            Check(aclrtSynchronizeStream(r.stream),"synchronize");
        };
        auto poison=[&](int i){
            uint8_t pattern=i?0xff:0x7f;
            Check(aclrtMemset(r.y.buffer.data,outputBytes,pattern,outputBytes),"poison");
            if(r.y.buffer.CopyOut()!=std::vector<uint8_t>(outputBytes,pattern))throw std::runtime_error("poison readback mismatch");
        };
        auto check=[&](const std::string &file){
            CheckNamedGuards({{"partials",&r.x.buffer},
                              {"o",&r.y.buffer},{"workspace",&r.workspace}},std::cout);
            if(r.x.buffer.CopyOut()!=x)throw std::runtime_error("read-only partials changed");
            auto out=r.y.buffer.CopyOut();WriteBytes(folder+"/"+file,out);
            if(first.empty())first=out;else if(out!=first)throw std::runtime_error("repeat output drift");
        };
        RunMeasuredCalls(timing,prepare,execute,poison,check);
        std::ofstream report(folder+"/execution.json");
        report<<"{\"status\":\"EXECUTION_COMPLETED_NUMERICS_UNCHECKED\",\"op\":\""<<op<<"\",\"soc\":\"Ascend310P3\","
            "\"input_shape\":[20,16,64],\"output_shape\":[16,64],\"group_count\":20,\"guards\":true,\"readonly\":true,\"repeat_equal\":true,\"repetitions\":2,"
            "\"device\":"<<r.device<<",\"workspace_bytes\":"<<size<<",\"cpu_fallback\":false";
        report<<",\"layout\":\"nd\",\"input_count\":1,\"poison_verified\":true,\"poison_patterns\":[127,255]";
        timing.Write(report);report<<"}\n";if(!report)throw std::runtime_error("report write failed");return 0;
    } catch(const std::exception &e){
        if(!folder.empty())std::ofstream(folder+"/execution.json")<<"{\"status\":\"FAIL\"}\n";
        std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;
    }
}
