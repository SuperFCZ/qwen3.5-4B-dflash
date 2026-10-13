// Diagnostic only: no host numerical fallback, no replacement of the 41-case runner.
#include "../../../draft_quant/test/runner_common.h"
#include "aclnn_d_flash_head_probe.h"
#include "aclnn_d_flash_cast_probe.h"
#include <array>
#include <iostream>
using namespace draft_quant_test;
struct Session {
    // values, x, w, ids, logits, precast, fp32 stages, fp16 stages, workspace
    std::array<GuardedBuffer,9> memory;
    std::vector<aclTensor*> tensors;
    aclrtStream stream=nullptr; int device=0; bool initialized=false,ready=false;
    ~Session() {
        if(stream) aclrtSynchronizeStream(stream);
        for(auto* t:tensors) aclDestroyTensor(t);
        for(auto& m:memory) m.Release();
        if(stream) aclrtDestroyStream(stream);
        if(ready) aclrtResetDevice(device); if(initialized) aclFinalize();
    }
    aclTensor* Tensor(int i,std::vector<int64_t> shape,aclDataType type) {
        std::vector<int64_t> stride(shape.size(),1);
        for(size_t j=shape.size();j>1;--j) stride[j-2]=stride[j-1]*shape[j-1];
        auto* t=aclCreateTensor(shape.data(),shape.size(),type,stride.data(),0,ACL_FORMAT_ND,shape.data(),shape.size(),memory[i].data);
        if(!t) throw std::runtime_error("create tensor"); tensors.push_back(t); return t;
    }
    void Guards() {
        CheckNamedGuards({{"values",&memory[0]},{"x",&memory[1]},{"w",&memory[2]},
            {"ids",&memory[3]},{"logits",&memory[4]},{"precast",&memory[5]},
            {"stages",&memory[6]},{"casts",&memory[7]},{"workspace",&memory[8]}},std::cerr);
    }
};
int main(int argc,char** argv) {
    try {
        if(argc!=5) throw std::runtime_error("usage: probe cast|head DEVICE DIRECTORY WEIGHT_OR_DASH");
        const std::string mode=argv[1],dir=argv[3]; const bool head=mode=="head";
        if(!head && mode!="cast") throw std::runtime_error("invalid probe mode");
        Session s; s.device=std::stoi(argv[2]);
        Check(aclInit(nullptr),"init"); s.initialized=true; Check(aclrtSetDevice(s.device),"device"); s.ready=true;
        Check(aclrtCreateStream(&s.stream),"stream");
        std::array<std::vector<uint8_t>,3> input;
        input[0]=ReadBytes(dir+"/values.bin",256);
        if(head) { input[1]=ReadBytes(dir+"/hidden.bin",15*2560*2); input[2]=ReadBytes(argv[4],uint64_t(248320)*2560*2); }
        for(int i=0;i<(head?3:1);++i) { s.memory[i].Allocate(input[i].size()); s.memory[i].CopyIn(input[i]); }
        auto* values=s.Tensor(0,{64},ACL_FLOAT);
        s.memory[6].Allocate(3*64*4); s.memory[7].Allocate(3*64*2);
        auto* stages=s.Tensor(6,{3,64},ACL_FLOAT); auto* casts=s.Tensor(7,{3,64},ACL_FLOAT16);
        aclTensor *x=nullptr,*w=nullptr,*ids=nullptr,*logits=nullptr,*precast=nullptr;
        if(head) {
            s.memory[3].Allocate(15*8); s.memory[4].Allocate(15*248320*2); s.memory[5].Allocate(2*16*64*4);
            x=s.Tensor(1,{15,2560},ACL_FLOAT16); w=s.Tensor(2,{248320,2560},ACL_FLOAT16);
            ids=s.Tensor(3,{15},ACL_INT64); logits=s.Tensor(4,{15,248320},ACL_FLOAT16); precast=s.Tensor(5,{2,16,64},ACL_FLOAT);
        }
        std::array<std::vector<uint8_t>,9> first;
        const char* names[]={"values","x","w","ids","logits","precast","vector-fp32","vector-fp16","workspace"};
        uint64_t allocated=0;
        for(int repeat=0;repeat<3;++repeat) {
            for(int i=head?3:6;i<8;++i) Check(aclrtMemset(s.memory[i].data,s.memory[i].bytes,repeat%2?0xff:0x7f,s.memory[i].bytes),"poison");
            aclOpExecutor* executor=nullptr; uint64_t bytes=0;
            Check(head?aclnnDFlashHeadProbeGetWorkspaceSize(x,w,values,ids,logits,precast,stages,casts,&bytes,&executor):
                aclnnDFlashCastProbeGetWorkspaceSize(values,stages,casts,&bytes,&executor),"probe workspace");
            if(repeat && allocated!=bytes) throw std::runtime_error("workspace drift");
            if(bytes && !allocated) s.memory[8].Allocate(bytes); allocated=bytes;
            s.Guards();
            Check(head?aclnnDFlashHeadProbe(s.memory[8].data,bytes,executor,s.stream):
                aclnnDFlashCastProbe(s.memory[8].data,bytes,executor,s.stream),"probe execute");
            Check(aclrtSynchronizeStream(s.stream),"probe sync"); s.Guards();
            for(int i=0;i<(head?3:1);++i) {
                std::vector<uint8_t> chunk(std::min<size_t>(4*1024*1024,input[i].size()));
                for(size_t off=0;off<input[i].size();off+=chunk.size()) {
                    const size_t n=std::min(chunk.size(),input[i].size()-off);
                    Check(aclrtMemcpy(chunk.data(),chunk.size(),static_cast<uint8_t*>(s.memory[i].data)+off,n,ACL_MEMCPY_DEVICE_TO_HOST),"readonly");
                    if(!std::equal(chunk.begin(),chunk.begin()+n,input[i].begin()+off)) throw std::runtime_error("input changed");
                }
            }
            for(int i=head?3:6;i<8;++i) {
                auto data=s.memory[i].CopyOut();
                WriteBytes(dir+"/"+names[i]+"-"+std::to_string(repeat)+".bin",data);
                if(!repeat) first[i]=data; else if(first[i]!=data) throw std::runtime_error("probe repeat differs");
            }
        }
        std::ofstream(dir+"/execution.json")<<"{\"status\":\"PASS\",\"scope\":\"diagnostic execution only\","
            "\"guards_intact\":true,\"input_readonly\":true,\"repeats\":3,\"numerical_acceptance\":\"NOT_RUN\"}\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
}
