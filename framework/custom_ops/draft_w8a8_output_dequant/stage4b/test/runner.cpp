// Stage4B output tensor-dequant checks only. No timing and no Raw execution.
#include "runner_common.h"
#include <iostream>
#include <regex>
#include <cstdio>
#include <iterator>
#include <cstdlib>
#include "aclnn_d_flash_w8_a8_full_gate.h"
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
    const auto probe=unique(R"(STAGE4B_CORE_PROBE aic=(\d+) aiv=(\d+) all=(\d+))",3);
    const auto policy=unique(R"(STAGE4B_CORE_POLICY aic=(\d+) aiv=(\d+) all=(\d+) selected=(\d+))",4);
    std::regex archExpr(R"(STAGE4B_ARCH_PROBE arch=(\S+))");bool archSeen=false;
    for(auto it=std::sregex_iterator(text.begin(),text.end(),archExpr);it!=std::sregex_iterator();++it) {
        if((*it)[1].str()!="2002")throw std::runtime_error("requires confirmed dav_2002");archSeen=true;
    }
    if(!archSeen||probe.size()!=3||policy.size()!=4||!probe[0]||!probe[1]||!probe[2]||probe[0]!=probe[1]||probe[2]>probe[0]||
       probe[0]!=policy[0]||probe[1]!=policy[1]||probe[2]!=policy[2]||policy[3]!=probe[2])
        throw std::runtime_error("invalid Core policy BEFORE device execution");
    if(!hostOnly) {
        const char* expected=std::getenv("DFLASH_STAGE4B_EXPECTED_CORE_POLICY");
        const std::string actual=std::to_string(policy[0])+","+std::to_string(policy[1])+","+std::to_string(policy[2])+","+std::to_string(policy[3])+",2002";
        if(!expected||actual!=expected)throw std::runtime_error("Core policy differs from required Host-only proof");
    }
    std::regex expression(R"(STAGE4B_SCHEDULE coreLimit=(\d+) blocks=(\d+) cores=(\d+) system=(\d+) arena=(\d+) stride=(\d+) user=(\d+) total=(\d+) mode=(\d+))");
    std::vector<uint64_t> accepted;
    for(auto it=std::sregex_iterator(text.begin(),text.end(),expression);it!=std::sregex_iterator();++it) {
        std::vector<uint64_t> p;for(int i=1;i<=9;++i)p.push_back(std::stoull((*it)[i].str()));
        if(p[0]!=uint64_t(limit)||p[8]!=uint64_t(mode))throw std::runtime_error("Host plan scope mismatch");
        if(!accepted.empty()&&p!=accepted)throw std::runtime_error("Host resource plan drift");accepted=p;
    }
    const auto &p=accepted;
    if(p.size()!=9||p[2]!=policy[3]||p[1]<1||p[1]>304||p[1]>p[2]||p[1]!=(limit?uint64_t(limit):std::min(uint64_t(304),p[2]))||
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
    Tensor x,w,s,y;GuardedBuffer workspace;
    ~Session(){if(stream)aclrtSynchronizeStream(stream);y.Release();s.Release();w.Release();x.Release();workspace.Release();
        if(stream)aclrtDestroyStream(stream);if(selected)aclrtResetDevice(device);if(initialized)aclFinalize();}
};
int main(int argc,char **argv) {
    std::string folder;
    try {
        if(argc!=5&&argc!=6)throw std::runtime_error("usage: runner DEVICE CASE_DIR OUTPUT_MODE CORE_LIMIT [--host-only]");
        const bool hostOnly=argc==6;
        if(hostOnly&&std::string(argv[5])!="--host-only")throw std::runtime_error("unknown diagnostic option");
        folder=argv[2];size_t modeEnd=0;std::string modeArg=argv[3];int64_t mode=std::stoll(modeArg,&modeEnd);
        if(modeEnd!=modeArg.size()||mode<0||mode>21)throw std::runtime_error("invalid output mode");std::ofstream(folder+"/execution.json")<<"{\"status\":\"RUNNING\"}\n";
        size_t limitEnd=0;std::string limitArg=argv[4];int64_t limit=std::stoll(limitArg,&limitEnd);
        if(limitEnd!=limitArg.size()||limit<0||limit>304)throw std::runtime_error("invalid core limit");
        if(hostOnly&&(mode!=21||limit!=1))throw std::runtime_error("Host-only diagnostic requires mode21/core_limit1");
        Session r;size_t end=0;std::string id=argv[1];r.device=std::stoi(id,&end);
        if(end!=id.size()||r.device<0)throw std::runtime_error("invalid device id");
        auto x=ReadBytes(folder+"/a_i8.bin",40960),w=ReadBytes(folder+"/b_i8.bin",49807360),s=ReadBytes(folder+"/quant_u64_doc_candidate.bin",3112960);
        const char *op="DFlashW8A8FullGate";
        const size_t outputBytes=622592;
        Check(aclInit(nullptr),"aclInit");r.initialized=true;
        Check(aclrtSetDevice(r.device),"aclrtSetDevice");r.selected=true;
        auto soc=aclrtGetSocName();if(!soc||std::string(soc)!="Ascend310P3")throw std::runtime_error("actual SoC must be Ascend310P3");
        Check(aclrtCreateStream(&r.stream),"aclrtCreateStream");
        r.x.Make({20,16,128},{20,16,128},ACL_INT8,ACL_FORMAT_ND,40960,&x);
        r.y.Make({16,19456},{16,19456},ACL_FLOAT16,ACL_FORMAT_ND,outputBytes);
        r.w.Make({304,20,128,64},{304,20,128,64},ACL_INT8,ACL_FORMAT_ND,49807360,&w);
        r.s.Make({304,20,64},{304,20,64},ACL_UINT64,ACL_FORMAT_ND,3112960,&s);
        uint64_t size=0,previous=0;aclOpExecutor *executor=nullptr;std::vector<uint8_t> first;
        Timing timing;  // no performance stage, not even warmup
        auto prepare=[&](){
            executor=nullptr;size=0;
            Check(aclnnDFlashW8A8FullGateGetWorkspaceSize(r.x.desc,r.w.desc,r.s.desc,mode,limit,r.y.desc,&size,&executor),"Stage4B GetWorkspaceSize");
            AdmitWorkspace(folder,size,mode,limit,hostOnly);
            if(hostOnly)return; // no workspace allocation or operator submission in this mode
            if(!first.empty()&&previous!=size)throw std::runtime_error("workspace drift");
            if(size&&r.workspace.bytes!=size)r.workspace.Allocate(size);previous=size;
            CheckNamedGuards({{"a",&r.x.buffer},{"b",&r.w.buffer},{"scale",&r.s.buffer},{"o",&r.y.buffer},{"workspace",&r.workspace}},std::cout);
        };
        auto execute=[&](){
            Check(aclnnDFlashW8A8FullGate(r.workspace.data,size,executor,r.stream),"Stage4B execute");
            Check(aclrtSynchronizeStream(r.stream),"synchronize");
        };
        auto poison=[&](int i){
            uint8_t pattern=i?0xff:0x7f;
            Check(aclrtMemset(r.y.buffer.data,outputBytes,pattern,outputBytes),"poison");
            if(r.y.buffer.CopyOut()!=std::vector<uint8_t>(outputBytes,pattern))throw std::runtime_error("poison readback mismatch");
        };
        auto check=[&](const std::string &file){
            CheckNamedGuards({{"a",&r.x.buffer},{"b",&r.w.buffer},{"scale",&r.s.buffer},
                              {"o",&r.y.buffer},{"workspace",&r.workspace}},std::cout);
            if(r.x.buffer.CopyOut()!=x||r.w.buffer.CopyOut()!=w)throw std::runtime_error("read-only x/w changed");
            if(r.s.buffer.CopyOut()!=s)throw std::runtime_error("scale uint64 input changed");
            auto out=r.y.buffer.CopyOut();WriteBytes(folder+"/"+file,out);
            if(first.empty())first=out;else if(out!=first)throw std::runtime_error("repeat output drift");
        };
        if(hostOnly) {
            prepare();
            std::ofstream report(folder+"/execution.json");
            report<<"{\"status\":\"HOST_TILING_ONLY\",\"host_only\":true,\"npu\":\"NOT_RUN\",\"device\":"<<r.device
                  <<",\"output_mode\":21,\"core_limit\":1,\"workspace_bytes\":"<<size<<"}\n";
            if(!report)throw std::runtime_error("Host diagnostic report write failed");
            return 0;
        }
        RunMeasuredCalls(timing,prepare,execute,poison,check);
        std::ofstream report(folder+"/execution.json");
        report<<"{\"status\":\"EXECUTION_COMPLETED_NUMERICS_UNCHECKED\",\"op\":\""<<op<<"\",\"soc\":\"Ascend310P3\","
            "\"shape\":[16,2560,19456],\"tile_count\":304,\"group_count\":20,\"guards\":true,\"readonly\":true,\"repeat_equal\":true,\"repetitions\":2,"
            "\"device\":"<<r.device<<",\"workspace_bytes\":"<<size<<",\"cpu_fallback\":false";
        report<<",\"output_mode\":"<<mode<<",\"core_limit\":"<<limit;
        report<<",\"resource_admission\":\"BEFORE_EXECUTION\",\"ub_guard_protocol\":\"UB_AND_CORE_ARENA_POISON_ON_FAILURE\",\"layout\":\"nd\",\"input_count\":3,\"poison_verified\":true,\"poison_patterns\":[127,255]";
        timing.Write(report);report<<"}\n";if(!report)throw std::runtime_error("report write failed");return 0;
    } catch(const std::exception &e){
        if(!folder.empty())std::ofstream(folder+"/execution.json")<<"{\"status\":\"FAIL\"}\n";
        std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;
    }
}
