#include "../../../draft_quant/test/runner_common.h"
#include "aclnn_d_flash_top1_probe.h"
#include <iostream>
using namespace draft_quant_test;
int main(int argc,char** argv) {
    if(argc!=3) return 2;
    try {
        const int device=std::stoi(argv[1]); const std::string dir=argv[2];
        Check(aclInit(nullptr),"init"); Check(aclrtSetDevice(device),"device"); aclrtStream stream;
        Check(aclrtCreateStream(&stream),"stream");
        GuardedBuffer x,y,w; auto input=ReadBytes(dir+"/logits.bin",15*64*2);
        x.Allocate(input.size()); x.CopyIn(input); y.Allocate(15*8);
        int64_t xs[]={15,64},xt[]={64,1},ys[]={15},yt[]={1};
        auto* xd=aclCreateTensor(xs,2,ACL_FLOAT16,xt,0,ACL_FORMAT_ND,xs,2,x.data);
        auto* yd=aclCreateTensor(ys,1,ACL_INT64,yt,0,ACL_FORMAT_ND,ys,1,y.data);
        if(!xd || !yd) throw std::runtime_error("tensor creation");
        std::vector<uint8_t> first; uint64_t allocated=0;
        for(int repeat=0;repeat<3;++repeat) {
            Check(aclrtMemset(y.data,y.bytes,repeat%2?0xff:0x7f,y.bytes),"poison");
            uint64_t bytes=0; aclOpExecutor* executor=nullptr;
            Check(aclnnDFlashTop1ProbeGetWorkspaceSize(xd,yd,&bytes,&executor),"workspace");
            if(repeat && bytes!=allocated) throw std::runtime_error("workspace drift");
            if(bytes && !allocated) w.Allocate(bytes); allocated=bytes;
            CheckNamedGuards({{"x",&x},{"y",&y},{"workspace",&w}},std::cerr);
            Check(aclnnDFlashTop1Probe(w.data,allocated,executor,stream),"execute"); Check(aclrtSynchronizeStream(stream),"sync");
            CheckNamedGuards({{"x",&x},{"y",&y},{"workspace",&w}},std::cerr);
            if(x.CopyOut()!=input) throw std::runtime_error("readonly");
            const auto out=y.CopyOut(); if(!repeat) first=out; else if(out!=first) throw std::runtime_error("repeat");
            WriteBytes(dir+"/ids-"+std::to_string(repeat)+".bin",out);
        }
        aclDestroyTensor(xd); aclDestroyTensor(yd); x.Release(); y.Release(); w.Release();
        aclrtDestroyStream(stream); aclrtResetDevice(device); aclFinalize(); return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
