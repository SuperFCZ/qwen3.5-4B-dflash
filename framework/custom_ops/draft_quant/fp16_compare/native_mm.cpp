// Independent native FP16 ACLNN baseline; never linked to a custom OPP.
#include "runner_common.h"
#include DFLASH_NATIVE_MM_HEADER
#include <iostream>
using namespace draft_quant_test;
namespace {
constexpr int64_t M=16,K=2560,N=19456;
struct Tensor {
    GuardedBuffer buffer; aclTensor *desc=nullptr;
    void Make(int64_t rows,int64_t cols,const std::vector<uint8_t>* values=nullptr) {
        buffer.Allocate(rows*cols*2);if(values)buffer.CopyIn(*values);
        int64_t shape[]={rows,cols},stride[]={cols,1};
        desc=aclCreateTensor(shape,2,ACL_FLOAT16,stride,0,ACL_FORMAT_ND,shape,2,buffer.data);
        if(!desc)throw std::runtime_error("aclCreateTensor failed");
    }
    void Release(){if(desc)aclDestroyTensor(desc);desc=nullptr;buffer.Release();}
};
struct Session {
    bool initialized=false,deviceSet=false;int device=0;aclrtStream stream=nullptr;
    Tensor x,w,y;GuardedBuffer workspace;
    ~Session(){
        if(stream)aclrtSynchronizeStream(stream);
        y.Release();w.Release();x.Release();workspace.Release();
        if(stream)aclrtDestroyStream(stream);
        if(deviceSet)aclrtResetDevice(device);
        if(initialized)aclFinalize();
    }
};
int Integer(const char* text){size_t end=0;std::string s(text);int n=std::stoi(s,&end);if(end!=s.size())throw std::runtime_error("invalid integer");return n;}
}
int main(int argc,char**argv){
    std::string folder;
    try {
        if(argc!=3 && argc!=6)throw std::runtime_error("usage: fp16_direct_mm DEVICE DIR [5 30 --continuous]");
        folder=argv[2];std::ofstream(folder+"/execution.json")<<"{\"status\":\"RUNNING\"}\n";
        Session r;r.device=Integer(argv[1]);if(r.device<0)throw std::runtime_error("negative device");
        Timing timing;
        if(argc==6){
            if(Integer(argv[3])!=5||Integer(argv[4])!=30||std::string(argv[5])!="--continuous")throw std::runtime_error("require continuous-v1 5/30");
            timing.warmup=5;timing.repetitions=30;timing.continuous=true;
        }
        auto x=ReadBytes(folder+"/x.bin",M*K*2),w=ReadBytes(folder+"/w_fp16_kn.bin",K*N*2);
        Check(aclInit(nullptr),"aclInit");r.initialized=true;
        Check(aclrtSetDevice(r.device),"aclrtSetDevice");r.deviceSet=true;
        const char* soc=aclrtGetSocName();
        if(!soc || std::string(soc)!="Ascend310P3")throw std::runtime_error("requires actual Ascend310P3 device");
        Check(aclrtCreateStream(&r.stream),"aclrtCreateStream");
        r.x.Make(M,K,&x);r.w.Make(K,N,&w);r.y.Make(M,N);
        uint64_t workspace=0,peak=0;aclOpExecutor*executor=nullptr;std::vector<uint8_t> first;
        auto prepare=[&](){
            workspace=0;executor=nullptr;
            // KEEP_DTYPE=0, FP16 inputs/output. No dtype downgrade or CPU path.
            Check(aclnnMmGetWorkspaceSize(r.x.desc,r.w.desc,r.y.desc,0,&workspace,&executor),"aclnnMmGetWorkspaceSize");
            if(!first.empty() && workspace!=peak)throw std::runtime_error("workspace changed");
            if(workspace && r.workspace.bytes!=workspace)r.workspace.Allocate(workspace);
            peak=std::max(peak,workspace);
        };
        auto execute=[&](){Check(aclnnMm(r.workspace.data,workspace,executor,r.stream),"aclnnMm");Check(aclrtSynchronizeStream(r.stream),"synchronize");};
        auto poison=[&](int repeat){Check(aclrtMemset(r.y.buffer.data,r.y.buffer.bytes,0x7f+(repeat%2)*128,r.y.buffer.bytes),"poison");};
        auto verify=[&](const std::string&file){
            r.x.buffer.CheckGuards("x");r.w.buffer.CheckGuards("w_fp16_kn");r.y.buffer.CheckGuards("y");r.workspace.CheckGuards("workspace");
            if(r.x.buffer.CopyOut()!=x||r.w.buffer.CopyOut()!=w)throw std::runtime_error("FP16 read-only input changed");
            auto output=r.y.buffer.CopyOut();if(first.empty())first=output;
            if(output!=first)throw std::runtime_error("FP16 repeat/timed-tail/postcheck drift");
            if(!file.empty())WriteBytes(folder+"/"+file,output);
        };
        RunMeasuredCalls(timing,prepare,execute,poison,verify);
        std::ofstream report(folder+"/execution.json");
        report<<"{\"status\":\"PASS\",\"runtime\":\"AscendCL native ACLNN\",\"op\":\"aclnnMm\","
                 "\"cpu_fallback\":false,\"input_readonly\":true,\"guards_intact\":true,\"repetitions\":2,"
                 "\"soc_name\":\"Ascend310P3\",\"m\":16,\"k\":2560,\"n\":19456,\"dtype\":\"float16\",\"weight_layout\":\"ND_KN\",\"cube_math_type\":0,"
                 "\"device_id\":"<<r.device<<",\"workspace_bytes\":"<<peak<<",\"tracked_device_allocation_bytes\":"
              <<(x.size()+w.size()+M*N*2+peak+(3+(peak?1:0))*2*kGuard);
        timing.Write(report);report<<"}\n";if(!report)throw std::runtime_error("cannot save execution");
        return 0;
    }catch(const std::exception&e){if(!folder.empty())std::ofstream(folder+"/execution.json")<<"{\"status\":\"FAIL\"}\n";std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
