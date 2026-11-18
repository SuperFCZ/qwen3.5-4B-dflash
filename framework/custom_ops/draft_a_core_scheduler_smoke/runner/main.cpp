#include "runner_common.h"
#include "aclnn_d_flash_group_quant_linear.h"
#include <unistd.h>
#include <iostream>
using namespace draft_quant_test;

struct Tensor : GuardedBuffer {
    aclTensor *desc = nullptr;
    void bind(const std::vector<int64_t>& shape, aclDataType type, aclFormat format,
              const std::vector<uint8_t>& host) {
        Allocate(host.size()); CopyIn(host);
        std::vector<int64_t> stride(shape.size(),1);
        for (size_t dim=shape.size(); dim>1; --dim) stride[dim-2]=stride[dim-1]*shape[dim-1];
        desc=aclCreateTensor(shape.data(),shape.size(),type,stride.data(),0,format,shape.data(),shape.size(),data);
        if (!desc) throw std::runtime_error("aclCreateTensor");
    }
    void close(){ Check(aclDestroyTensor(desc),"aclDestroyTensor"); Release(); }
};

void snapshot(const std::string& path) {
    std::ifstream maps("/proc/self/maps"); std::ofstream output(path);
    if (!maps) throw std::runtime_error("maps unavailable");
    output<<maps.rdbuf(); output.close(); if (!output) throw std::runtime_error("maps write failed");
}

bool guards(const std::string& path, const std::vector<std::pair<std::string,GuardedBuffer*>>& buffers) {
    std::ofstream out(path); bool ok=true;
    for(const auto& pair:buffers){
        auto& buf=*pair.second;
        if(!buf.allocation) throw std::runtime_error("missing guarded allocation");
        for(int side=0;side<2;++side){
            std::vector<uint8_t> bytes(kGuard);
            const auto ptr=side?static_cast<uint8_t*>(buf.data)+buf.bytes:buf.allocation;
            Check(aclrtMemcpy(bytes.data(),kGuard,ptr,kGuard,ACL_MEMCPY_DEVICE_TO_HOST),"read guard");
            out<<"{\"buffer\":\""<<pair.first<<"\",\"side\":\""<<(side?"suffix":"prefix")<<"\",\"bytes\":"<<kGuard<<",\"expected\":"<<unsigned(kCanary)<<",\"changes\":[";
            bool first=true;
            for(size_t idx=0;idx<kGuard;++idx) if(bytes[idx]!=kCanary){out<<(first?"":",")<<"["<<idx<<","<<unsigned(bytes[idx])<<"]";first=false;ok=false;}
            out<<"]}\n";
        }
    }
    out.close(); if(!out) throw std::runtime_error("guard evidence write failed"); return ok;
}

int main(int argc,char** argv){
    if(argc!=5) return 2;
    const std::string root=argv[1]; std::ofstream journal(root+"/events.jsonl");
    auto event=[&](const char* name,int rc,const std::string& extra=""){
        journal<<"{\"event\":\""<<name<<"\",\"returncode\":"<<rc<<extra<<"}"<<std::endl;
        if(!journal) throw std::runtime_error("journal write failed"); Check(rc,name);
    };
    try {
        int repeat=std::stoi(argv[2]),control=std::stoi(argv[3]);uint64_t expected=std::stoull(argv[4]);
        if((repeat!=0&&repeat!=1)||!expected) throw std::runtime_error("bad invocation");
        const auto input=ReadBytes(root+"/x.bin",16*12800*2),weight=ReadBytes(root+"/w_nz.bin",12800*2560),scale=ReadBytes(root+"/s_gn.bin",100*2560*2);
        event("aclInit",aclInit(nullptr));event("aclrtSetDevice",aclrtSetDevice(0));
        int32_t device=-1;aclrtContext ctx=nullptr;aclrtStream stream=nullptr;
        auto rc=aclrtGetDevice(&device);event("aclrtGetDevice",rc,",\"device\":"+std::to_string(device));
        rc=aclrtGetCurrentContext(&ctx);event("context",rc,ctx?",\"present\":true":",\"present\":false");
        rc=aclrtCreateStream(&stream);event("stream",rc,stream?",\"present\":true":",\"present\":false");
        if(device!=0||!ctx||!stream)throw std::runtime_error("device/context/stream");
        auto checkContext=[&](){aclrtContext current=nullptr;int32_t id=-1;
            Check(aclrtGetCurrentContext(&current),"recheck context");Check(aclrtGetDevice(&id),"recheck device");
            if(current!=ctx||id!=device)throw std::runtime_error("context/device drift");};
        Tensor activation,codes,scales,output;GuardedBuffer workspace;
        activation.bind({16,12800},ACL_FLOAT16,ACL_FORMAT_ND,input);
        codes.bind({400,160,16,32},ACL_INT8,ACL_FORMAT_FRACTAL_NZ,weight);
        scales.bind({100,2560},ACL_FLOAT16,ACL_FORMAT_ND,scale);
        std::vector<uint8_t> poison(16*2560*2,repeat?0xff:0x7f);
        output.bind({16,2560},ACL_FLOAT16,ACL_FORMAT_ND,poison);
        aclOpExecutor* executor=nullptr;uint64_t size=0;
        rc=aclnnDFlashGroupQuantLinearGetWorkspaceSize(activation.desc,codes.desc,scales.desc,output.desc,&size,&executor);
        event("GetWorkspaceSize",rc,",\"workspace\":"+std::to_string(size));
        if(!executor||size!=expected)throw std::runtime_error("executor/Workspace drift");
        workspace.Allocate(size);
        const std::vector<std::pair<std::string,GuardedBuffer*>> buffers={{"x",&activation},{"w",&codes},{"s",&scales},{"y",&output},{"workspace",&workspace}};
        if(output.CopyOut()!=poison||!guards(root+"/pre-guards.jsonl",buffers)||activation.CopyOut()!=input||codes.CopyOut()!=weight||scales.CopyOut()!=scale)
            throw std::runtime_error("pre-execute poison/guards/inputs");
        event("precheck",0,",\"poison_byte\":"+std::to_string(unsigned(poison[0])));
        snapshot(root+"/pre-maps.txt");char token=0;
        if(write(control,"?",1)!=1||read(control,&token,1)!=1||token!='G')throw std::runtime_error("Host query/DSO admission refused");
        checkContext();event("execute_attempt",0);
        event("aclnnDFlashGroupQuantLinear",aclnnDFlashGroupQuantLinear(workspace.data,size,executor,stream));
        event("synchronize",aclrtSynchronizeStream(stream));checkContext();
        WriteBytes(root+"/actual.bin",output.CopyOut());
        bool intact=guards(root+"/post-guards.jsonl",buffers);
        bool rx=activation.CopyOut()==input,rw=codes.CopyOut()==weight,rs=scales.CopyOut()==scale;
        snapshot(root+"/post-maps.txt");
        event("readonly",0,",\"x\":"+std::to_string(rx)+",\"w\":"+std::to_string(rw)+",\"s\":"+std::to_string(rs));
        if(!intact||!rx||!rw||!rs)throw std::runtime_error("guard/readonly changed");
        activation.close();codes.close();scales.close();output.close();workspace.Release();
        Check(aclrtDestroyStream(stream),"aclrtDestroyStream");Check(aclrtResetDevice(device),"aclrtResetDevice");Check(aclFinalize(),"aclFinalize");
        event("complete",0);return 0;
    }catch(const std::exception& err){std::cerr<<"BLOCKED: "<<err.what()<<std::endl;return 1;}
}
