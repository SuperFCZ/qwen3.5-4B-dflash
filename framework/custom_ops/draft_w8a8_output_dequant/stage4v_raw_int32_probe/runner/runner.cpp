// Untimed independent INT32 diagnostic. No FP16 or dequantization ABI.
#include "runner_common.h"
#include <iostream>
#include <regex>
#include <cstdio>
#include <iterator>
#include <cstdlib>
#include "aclnn_d_flash_full_k_int32_probe.h"
using namespace draft_quant_test;
static void AdmitWorkspace(const std::string& folder,uint64_t bytes,int64_t mode,int64_t limit,bool hostOnly) {
    // Python driver redirects this process's stderr into runner.log. Host Tiling
    // prints its actual platform plan during GetWorkspaceSize, before any kernel launch.
    std::fflush(nullptr);
    std::ifstream in(folder+"/runner.log");if(!in)throw std::runtime_error("missing Host plan log before execution");
    std::string text((std::istreambuf_iterator<char>(in)),std::istreambuf_iterator<char>());
    auto unique=[&](const char* pattern,int count) {
        std::regex expr(pattern);std::vector<uint64_t> values;
        for(auto it=std::sregex_iterator(text.begin(),text.end(),expr);it!=std::sregex_iterator();++it) {
            std::vector<uint64_t> row;for(int i=1;i<=count;++i)row.push_back(std::stoull((*it)[i].str()));
            if(!values.empty()&&values!=row)throw std::runtime_error("Core probe/policy drift");values=row;
        }
        return values;
    };
    const auto probe=unique(R"(STAGE4RAW_CORE_PROBE aic=(\d+) aiv=(\d+) all=(\d+))",3);
    const auto policy=unique(R"(STAGE4RAW_CORE_POLICY aic=(\d+) aiv=(\d+) all=(\d+) selected=(\d+))",4);
    std::regex archExpr(R"(STAGE4RAW_ARCH_PROBE arch=(\S+))");bool archSeen=false;
    for(auto it=std::sregex_iterator(text.begin(),text.end(),archExpr);it!=std::sregex_iterator();++it) {
        if((*it)[1].str()!="2002")throw std::runtime_error("requires confirmed dav_2002");archSeen=true;
    }
    if(!archSeen||probe.size()!=3||policy.size()!=4||!probe[0]||!probe[1]||!probe[2]||probe[0]!=probe[1]||probe[2]>probe[0]||
       probe[0]!=policy[0]||probe[1]!=policy[1]||probe[2]!=policy[2]||policy[3]!=probe[2])
        throw std::runtime_error("invalid Core policy BEFORE device execution");
    { // bind the accepted formal policy even for Host-only
        const char* expected=std::getenv("DFLASH_STAGE4RAW_EXPECTED_CORE_POLICY");
        const std::string actual=std::to_string(policy[0])+","+std::to_string(policy[1])+","+std::to_string(policy[2])+","+std::to_string(policy[3])+",2002";
        if(!expected||actual!=expected)throw std::runtime_error("Core policy differs from required Host-only proof");
    }
    std::regex expression(R"(STAGE4RAW_SCHEDULE coreLimit=(\d+) blocks=(\d+) cores=(\d+) system=(\d+) arena=(\d+) stride=(\d+) user=(\d+) total=(\d+) mode=(\d+))");
    std::vector<uint64_t> accepted;
    for(auto it=std::sregex_iterator(text.begin(),text.end(),expression);it!=std::sregex_iterator();++it) {
        std::vector<uint64_t> p;for(int i=1;i<=9;++i)p.push_back(std::stoull((*it)[i].str()));
        if(p[0]!=uint64_t(limit)||p[8]!=uint64_t(mode))throw std::runtime_error("Host plan scope mismatch");
        if(!accepted.empty()&&p!=accepted)throw std::runtime_error("Host resource plan drift");accepted=p;
    }
    const auto &p=accepted;
    if(p.size()!=9||p[2]!=policy[3]||p[1]<1||p[1]!=1||p[1]>p[2]||p[1]!=(limit?uint64_t(limit):uint64_t(1))||
       !p[3]||p[3]>512ULL*1024*1024||p[4]!=(p[3]+511)/512*512||p[5]!=p[4]+1024||
       p[6]!=p[1]*p[5]||p[7]!=p[3]+p[6]||p[7]>512ULL*1024*1024||bytes<p[7])
        throw std::runtime_error("workspace/core admission failed BEFORE device execution");
    std::ofstream f(folder+"/resource-admission.json");
    f<<"{\"status\":\"ADMITTED_BEFORE_EXECUTION\",\"workspace_bytes\":"<<bytes<<",\"declared_total\":"<<p[7]<<",\"blocks\":"<<p[1]<<",\"cores\":"<<p[2]
     <<",\"core_policy\":{\"aic\":"<<policy[0]<<",\"aiv\":"<<policy[1]<<",\"all\":"<<policy[2]<<",\"selected\":"<<policy[3]<<",\"arch\":2002},\"host_proof_checked\":"<<(hostOnly?"false":"true")<<"}\n";
    if(!f)throw std::runtime_error("resource admission write failed");
}
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
        if(argc!=3&&argc!=4)throw std::runtime_error("usage: raw_probe_runner DEVICE CASE_DIR [--host-only]");
        const bool hostOnly=argc==4&&std::string(argv[3])=="--host-only";
        if(argc==4&&!hostOnly)throw std::runtime_error("unknown option; no benchmark/fault mode");
        folder=argv[2];std::ofstream(folder+"/execution.json")<<"{\"status\":\"RUNNING\",\"npu\":\"NOT_RUN\"}\n";
        std::vector<uint8_t> expected;
        if(!hostOnly)expected=ReadBytes(folder+"/oracle_i32.bin",16384);
        Session r;size_t end=0;std::string id=argv[1];r.device=std::stoi(id,&end);
        if(end!=id.size()||r.device<0)throw std::runtime_error("invalid device id");
        auto x=ReadBytes(folder+"/a_i8.bin",40960),w=ReadBytes(folder+"/b_i8.bin",655360);
        Check(aclInit(nullptr),"aclInit");r.initialized=true;
        Check(aclrtSetDevice(r.device),"aclrtSetDevice");r.selected=true;
        auto soc=aclrtGetSocName();if(!soc||std::string(soc)!="Ascend310P3")throw std::runtime_error("actual SoC must be Ascend310P3");
        Check(aclrtCreateStream(&r.stream),"aclrtCreateStream");
        r.x.Make({16,2560},{16,2560},ACL_INT8,ACL_FORMAT_ND,40960,&x);
        r.w.Make({2560,256},{2560,256},ACL_INT8,ACL_FORMAT_ND,655360,&w);
        r.y.Make({16,256},{16,256},ACL_INT32,ACL_FORMAT_ND,16384);
        uint64_t previous=0;std::vector<uint8_t> first;
        for(int repeat=0;repeat<(hostOnly?1:2);++repeat){
            uint64_t size=0;aclOpExecutor *executor=nullptr;
            Check(aclnnDFlashFullKInt32ProbeGetWorkspaceSize(r.x.desc,r.w.desc,r.y.desc,&size,&executor),"Raw INT32 GetWorkspaceSize");
            AdmitWorkspace(folder,size,0,1,hostOnly);
            if(hostOnly){std::ofstream(folder+"/execution.json")<<"{\"status\":\"HOST_TILING_ONLY\",\"npu\":\"NOT_RUN\",\"device\":"<<r.device<<",\"workspace_bytes\":"<<size<<"}\n";return 0;}
            const char *expectedWorkspace=std::getenv("DFLASH_STAGE4RAW_EXPECTED_WORKSPACE");
            if(!expectedWorkspace||std::to_string(size)!=expectedWorkspace)throw std::runtime_error("workspace differs from formal Host admission");
            if(repeat&&previous!=size)throw std::runtime_error("workspace drift");
            if(!repeat)r.workspace.Allocate(size);previous=size;
            uint8_t pattern=repeat?0x80:0x7f;
            Check(aclrtMemset(r.y.buffer.data,16384,pattern,16384),"poison");
            if(r.y.buffer.CopyOut()!=std::vector<uint8_t>(16384,pattern))throw std::runtime_error("poison readback mismatch");
            CheckNamedGuards({{"a",&r.x.buffer},{"b",&r.w.buffer},{"o",&r.y.buffer},{"workspace",&r.workspace}},std::cout);
            std::ofstream(folder+"/execution.json")<<"{\"status\":\"RUNNING\",\"npu\":\"ATTEMPTED\"}\n";
            Check(aclnnDFlashFullKInt32Probe(r.workspace.data,size,executor,r.stream),"Raw INT32 execute");
            Check(aclrtSynchronizeStream(r.stream),"synchronize");
            auto out=r.y.buffer.CopyOut();WriteBytes(folder+"/actual-"+std::to_string(repeat)+".bin",out);
            CheckNamedGuards({{"a",&r.x.buffer},{"b",&r.w.buffer},{"o",&r.y.buffer},{"workspace",&r.workspace}},std::cout);
            if(r.x.buffer.CopyOut()!=x||r.w.buffer.CopyOut()!=w)throw std::runtime_error("readonly A/B changed");
            if(out!=expected)throw std::runtime_error("strict INT32 Golden mismatch");
            if(!repeat)first=out;else if(out!=first)throw std::runtime_error("repeat drift");
        }
        std::ofstream report(folder+"/execution.json");
        report<<"{\"status\":\"RAW_INT32_EXECUTION_MATCH\",\"npu\":\"COMPLETED\",\"device\":"<<r.device<<",\"workspace_bytes\":"<<previous
          <<",\"shape\":[16,2560,256],\"elements\":4096,\"repetitions\":2,\"guards\":true,\"readonly\":true,\"poison\":true,\"poison_patterns\":[127,128],\"repeat_equal\":true,\"cpu_fallback\":false,\"benchmark\":\"NOT_RUN\"}\n";
        if(!report)throw std::runtime_error("report write failed");return 0;
    }catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
