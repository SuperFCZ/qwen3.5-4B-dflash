// Independent K128 executable; native and custom are separate processes/links.
#include "runner_common.h"
#include <iostream>
#ifdef ANTIQUANT_NATIVE_REFERENCE
#include ANTIQUANT_MM_HEADER
#else
#include "aclnn_d_flash_anti_quant_k128.h"
#endif
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
    Tensor x,w,s,o,y;GuardedBuffer workspace;
    ~Session(){if(stream)aclrtSynchronizeStream(stream);y.Release();o.Release();s.Release();w.Release();x.Release();workspace.Release();
        if(stream)aclrtDestroyStream(stream);if(selected)aclrtResetDevice(device);if(initialized)aclFinalize();}
};
int main(int argc,char **argv) {
    std::string folder;
    try {
        if(argc!=3)throw std::runtime_error("usage: runner DEVICE CASE_DIR (K128 untimed only)");
        folder=argv[2];std::ofstream(folder+"/execution.json")<<"{\"status\":\"RUNNING\"}\n";
        Session r;size_t end=0;std::string id=argv[1];r.device=std::stoi(id,&end);
        if(end!=id.size()||r.device<0)throw std::runtime_error("invalid device id");
        auto x=ReadBytes(folder+"/x.bin",4096);
#ifdef ANTIQUANT_NATIVE_REFERENCE
        auto w=ReadBytes(folder+"/w_fp16_kn.bin",16384);
        const char *op="aclnnMm";
#else
        auto w=ReadBytes(folder+"/w_nz.bin",8192),s=ReadBytes(folder+"/scale.bin",128),o=ReadBytes(folder+"/offset.bin",128);
        if(std::any_of(o.begin(),o.end(),[](uint8_t v){return v!=0;}))throw std::runtime_error("symmetric offset must be positive zero bits");
        const char *op="DFlashAntiQuantK128";
#endif
        Check(aclInit(nullptr),"aclInit");r.initialized=true;
        Check(aclrtSetDevice(r.device),"aclrtSetDevice");r.selected=true;
        auto soc=aclrtGetSocName();if(!soc||std::string(soc)!="Ascend310P3")throw std::runtime_error("actual SoC must be Ascend310P3");
        Check(aclrtCreateStream(&r.stream),"aclrtCreateStream");
        r.x.Make({16,128},{16,128},ACL_FLOAT16,ACL_FORMAT_ND,4096,&x);
        r.y.Make({16,64},{16,64},ACL_FLOAT16,ACL_FORMAT_ND,2048);
#ifdef ANTIQUANT_NATIVE_REFERENCE
        r.w.Make({128,64},{128,64},ACL_FLOAT16,ACL_FORMAT_ND,16384,&w);
#else
        r.w.Make({64,128},{4,4,16,32},ACL_INT8,ACL_FORMAT_FRACTAL_NZ,8192,&w);
        r.s.Make({1,64},{1,64},ACL_FLOAT16,ACL_FORMAT_ND,128,&s);
        r.o.Make({1,64},{1,64},ACL_FLOAT16,ACL_FORMAT_ND,128,&o);
#endif
        uint64_t size=0,previous=0;aclOpExecutor *executor=nullptr;std::vector<uint8_t> first;
        Timing timing;  // no performance stage, not even warmup
        auto prepare=[&](){
            executor=nullptr;size=0;
#ifdef ANTIQUANT_NATIVE_REFERENCE
            Check(aclnnMmGetWorkspaceSize(r.x.desc,r.w.desc,r.y.desc,0,&size,&executor),"aclnnMmGetWorkspaceSize");
#else
            Check(aclnnDFlashAntiQuantK128GetWorkspaceSize(r.x.desc,r.w.desc,r.s.desc,r.o.desc,r.y.desc,&size,&executor),"aclnnDFlashAntiQuantK128GetWorkspaceSize");
#endif
            if(!first.empty()&&previous!=size)throw std::runtime_error("workspace drift");
            if(size&&r.workspace.bytes!=size)r.workspace.Allocate(size);previous=size;
        };
        auto execute=[&](){
#ifdef ANTIQUANT_NATIVE_REFERENCE
            Check(aclnnMm(r.workspace.data,size,executor,r.stream),"aclnnMm");
#else
            Check(aclnnDFlashAntiQuantK128(r.workspace.data,size,executor,r.stream),"aclnnDFlashAntiQuantK128");
#endif
            Check(aclrtSynchronizeStream(r.stream),"synchronize");
        };
        auto poison=[&](int i){Check(aclrtMemset(r.y.buffer.data,2048,i?0xff:0x7f,2048),"poison");};
        auto check=[&](const std::string &file){
            CheckNamedGuards({{"x",&r.x.buffer},{"w",&r.w.buffer},{"scale",&r.s.buffer},{"offset",&r.o.buffer},
                              {"y",&r.y.buffer},{"workspace",&r.workspace}},std::cout);
            if(r.x.buffer.CopyOut()!=x||r.w.buffer.CopyOut()!=w)throw std::runtime_error("read-only x/w changed");
#ifndef ANTIQUANT_NATIVE_REFERENCE
            if(r.s.buffer.CopyOut()!=s||r.o.buffer.CopyOut()!=o)throw std::runtime_error("read-only scale/offset changed");
#endif
            auto out=r.y.buffer.CopyOut();WriteBytes(folder+"/"+file,out);
            if(first.empty())first=out;else if(out!=first)throw std::runtime_error("repeat output drift");
        };
        RunMeasuredCalls(timing,prepare,execute,poison,check);
        std::ofstream report(folder+"/execution.json");
        report<<"{\"status\":\"EXECUTION_COMPLETED_NUMERICS_UNCHECKED\",\"op\":\""<<op<<"\",\"soc\":\"Ascend310P3\","
            "\"shape\":[16,128,64],\"guards\":true,\"readonly\":true,\"repeat_equal\":true,\"repetitions\":2,"
            "\"device\":"<<r.device<<",\"workspace_bytes\":"<<size<<",\"cpu_fallback\":false";
        timing.Write(report);report<<"}\n";if(!report)throw std::runtime_error("report write failed");return 0;
    } catch(const std::exception &e){
        if(!folder.empty())std::ofstream(folder+"/execution.json")<<"{\"status\":\"FAIL\"}\n";
        std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;
    }
}
