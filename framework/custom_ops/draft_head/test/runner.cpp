// Standalone ACLNN / native OM head runner. No host numerical fallback.
#include "../../draft_quant/test/runner_common.h"
#include "output_buffer.h"
#include "../op_host/head_contract.h"
#include <iostream>
#ifndef HEAD_NATIVE_ONLY
#include "aclnn_d_flash_draft_lm_head_top1.h"
#include "aclnn_d_flash_draft_lm_head_top1_audit.h"
#endif
using namespace draft_quant_test;
using namespace head_contract;

namespace {
struct Session {
    bool initialized=false,deviceReady=false,modelLoaded=false;
    int device=0; uint32_t model=0; aclrtStream stream=nullptr;
    aclmdlDesc* desc=nullptr; aclmdlDataset *inputs=nullptr,*outputs=nullptr;
    std::vector<aclDataBuffer*> buffers;
    GuardedBuffer x,w,y,logits,workspace,modelWeights;
    size_t nativeOutputBytes=0;
#ifndef HEAD_NATIVE_ONLY
    aclTensor *xd=nullptr,*wd=nullptr,*yd=nullptr,*ld=nullptr;
#endif
    ~Session() {
        if(stream) aclrtSynchronizeStream(stream);
        if(inputs) aclmdlDestroyDataset(inputs); if(outputs) aclmdlDestroyDataset(outputs);
        for(auto* b:buffers) aclDestroyDataBuffer(b);
        if(desc) aclmdlDestroyDesc(desc); if(modelLoaded) aclmdlUnload(model);
#ifndef HEAD_NATIVE_ONLY
        if(xd) aclDestroyTensor(xd); if(wd) aclDestroyTensor(wd); if(yd) aclDestroyTensor(yd); if(ld) aclDestroyTensor(ld);
#endif
        modelWeights.Release(); workspace.Release(); logits.Release(); y.Release(); w.Release(); x.Release();
        if(stream) aclrtDestroyStream(stream);
        if(deviceReady) aclrtResetDevice(device); if(initialized) aclFinalize();
    }
    void Add(aclmdlDataset* set,GuardedBuffer& memory) {
        auto* b=aclCreateDataBuffer(memory.data,memory.bytes);
        if(!b) throw std::runtime_error("aclCreateDataBuffer failed");
        buffers.push_back(b); Check(aclmdlAddDatasetBuffer(set,b),"aclmdlAddDatasetBuffer");
    }
    void CheckGuards(const std::string& phase) const {
        std::cerr<<"guard_check phase="<<phase<<'\n';
        CheckNamedGuards({{"x",&x},{"w",&w},{"y",&y},{"logits",&logits},
                          {"workspace",&workspace},{"modelWeights",&modelWeights}},std::cerr);
    }
};
bool Dims(const aclmdlIODims& s,std::initializer_list<int64_t> expected) {
    if(s.dimCount!=expected.size()) return false;
    size_t i=0; for(auto d:expected) if(s.dims[i++]!=d) return false;
    return true;
}
void Native(Session& r,const char* path,int m,bool wantLogits) {
    size_t work=0,weights=0; Check(aclmdlQuerySize(path,&work,&weights),"aclmdlQuerySize");
    if(work) r.workspace.Allocate(work); if(weights) r.modelWeights.Allocate(weights);
    Check(aclmdlLoadFromFileWithMem(path,&r.model,r.workspace.data,work,r.modelWeights.data,weights),"load native OM"); r.modelLoaded=true;
    r.desc=aclmdlCreateDesc(); if(!r.desc) throw std::runtime_error("aclmdlCreateDesc failed");
    Check(aclmdlGetDesc(r.desc,r.model),"aclmdlGetDesc");
    if(aclmdlGetNumInputs(r.desc)!=2 || aclmdlGetNumOutputs(r.desc)!=1) throw std::runtime_error("native OM IO count differs");
    r.inputs=aclmdlCreateDataset(); r.outputs=aclmdlCreateDataset();
    if(!r.inputs || !r.outputs) throw std::runtime_error("dataset allocation failed");
    bool sawX=false,sawW=false;
    for(size_t i=0;i<2;++i) {
        aclmdlIODims dims{}; Check(aclmdlGetInputDims(r.desc,i,&dims),"native input dims");
        if(aclmdlGetInputDataType(r.desc,i)!=ACL_FLOAT16) throw std::runtime_error("native input dtype differs");
        if(Dims(dims,{m,kK}) && !sawX && aclmdlGetInputSizeByIndex(r.desc,i)==r.x.bytes) { r.Add(r.inputs,r.x); sawX=true; }
        else if(Dims(dims,{kN,kK}) && !sawW && aclmdlGetInputSizeByIndex(r.desc,i)==r.w.bytes) { r.Add(r.inputs,r.w); sawW=true; }
        else throw std::runtime_error("native input must be full hidden/weight ND");
    }
    aclmdlIODims dims{}; Check(aclmdlGetOutputDims(r.desc,0,&dims),"native output dims");
    const size_t required=aclmdlGetOutputSizeByIndex(r.desc,0);
    r.nativeOutputBytes=required;
    std::cerr<<"native_output M="<<m<<" top1_logical_bytes="<<m*8
             <<" logical_output_bytes="<<r.y.bytes<<" aclmdlGetOutputSizeByIndex="<<required<<'\n';
    if(!sawX || !sawW || !(wantLogits?Dims(dims,{m,kN}):Dims(dims,{m})) ||
       aclmdlGetOutputDataType(r.desc,0)!=(wantLogits?ACL_FLOAT16:ACL_INT64) || required<r.y.bytes)
        throw std::runtime_error("native output ABI differs");
    // NPU evidence: M1's native INT64 output writes bytes 8..31 beyond the
    // reported 8 bytes. Pad ONLY native Top1; keep the OM's [M]/INT64 ABI.
    const size_t physicalBytes=wantLogits?required:draft_head_test::NativeTop1PayloadBytes(size_t(m)*8,required);
    if(physicalBytes!=r.y.bytes) r.y.Allocate(physicalBytes);
    // aclDataBuffer describes storage capacity, independently of model shape.
    // Add advertises precisely the physical payload, excluding both guards.
    r.Add(r.outputs,r.y);
}
#ifndef HEAD_NATIVE_ONLY
aclTensor* Tensor(GuardedBuffer& mem,std::vector<int64_t> shape,aclDataType type) {
    std::vector<int64_t> stride(shape.size(),1);
    for(size_t i=shape.size();i>1;--i) stride[i-2]=stride[i-1]*shape[i-1];
    auto* out=aclCreateTensor(shape.data(),shape.size(),type,stride.data(),0,ACL_FORMAT_ND,shape.data(),shape.size(),mem.data);
    if(!out) throw std::runtime_error("aclCreateTensor failed"); return out;
}
#endif
void Readonly(const GuardedBuffer& memory,const std::vector<uint8_t>& original) {
    // Bound host peak memory: compare the 1.27 GB head in 4 MiB chunks.
    std::vector<uint8_t> chunk(std::min<size_t>(4*1024*1024,memory.bytes));
    for(size_t start=0;start<memory.bytes;start+=chunk.size()) {
        const size_t size=std::min(chunk.size(),memory.bytes-start);
        Check(aclrtMemcpy(chunk.data(),chunk.size(),static_cast<const uint8_t*>(memory.data)+start,size,ACL_MEMCPY_DEVICE_TO_HOST),"readonly check");
        if(!std::equal(chunk.begin(),chunk.begin()+size,original.begin()+start)) throw std::runtime_error("read-only input changed");
    }
}
int Number(const char* text) {
    size_t end=0; int out=std::stoi(text,&end);
    if(end!=std::string(text).size()) throw std::runtime_error("invalid integer"); return out;
}
}

int main(int argc,char** argv) {
    std::string dir;
    try {
        if(argc!=9) throw std::runtime_error("usage: runner MODE DEVICE CASE_DIR WEIGHT_FILE M WARMUP REPETITIONS OM_OR_DASH");
        const std::string mode=argv[1]; dir=argv[3];
        std::ofstream(dir+"/execution.json")<<"{\"status\":\"RUNNING\"}\n";
        const bool native=mode=="native-logits" || mode=="native-top1", audit=mode=="audit", wantLogits=mode=="native-logits";
        if(!native && !audit && mode!="fused") throw std::runtime_error("unknown mode");
#ifdef HEAD_NATIVE_ONLY
        if(!native) throw std::runtime_error("native-only runner cannot execute custom mode");
#endif
        const int m=Number(argv[5]),warmup=Number(argv[6]),repetitions=Number(argv[7]);
        if(!Rows(m) || (audit?(warmup!=0 || repetitions!=0):(warmup<3 || warmup>100 || repetitions<10 || repetitions>1000)))
            throw std::runtime_error("invalid M/timing protocol");
        const auto x=ReadBytes(dir+"/hidden.bin",m*kK*2);
        const auto w=ReadBytes(argv[4],uint64_t(kN)*kK*2);
        Session r; r.device=Number(argv[2]); if(r.device<0) throw std::runtime_error("invalid device");
        Check(aclInit(nullptr),"aclInit"); r.initialized=true;
        Check(aclrtSetDevice(r.device),"aclrtSetDevice"); r.deviceReady=true;
        Check(aclrtCreateStream(&r.stream),"aclrtCreateStream");
        r.x.Allocate(x.size()); r.x.CopyIn(x); r.w.Allocate(w.size()); r.w.CopyIn(w);
        r.y.Allocate(wantLogits?uint64_t(m)*kN*2:m*8);
        if(audit) r.logits.Allocate(uint64_t(m)*kN*2);
        if(native) Native(r,argv[8],m,wantLogits);
#ifndef HEAD_NATIVE_ONLY
        else {
            r.xd=Tensor(r.x,{m,kK},ACL_FLOAT16); r.wd=Tensor(r.w,{kN,kK},ACL_FLOAT16);
            r.yd=Tensor(r.y,{m},ACL_INT64); if(audit) r.ld=Tensor(r.logits,{m,kN},ACL_FLOAT16);
        }
        aclOpExecutor* executor=nullptr;
#endif
        uint64_t workspaceBytes=r.workspace.bytes;
        const size_t logicalBytes=wantLogits?uint64_t(m)*kN*2:m*8;
        // Separate metadata survives execution.json being marked FAIL later.
        std::ofstream allocationReport(dir+"/output-allocation.json");
        allocationReport<<"{\"mode\":\""<<mode<<"\",\"m\":"<<m
            <<",\"top1_logical_bytes\":"<<m*8<<",\"logical_output_bytes\":"<<logicalBytes
            <<",\"model_reported_output_bytes\":";
        if(native) allocationReport<<r.nativeOutputBytes; else allocationReport<<"null";
        allocationReport<<",\"physical_payload_bytes\":"<<r.y.bytes
            <<",\"dataset_output_buffer_bytes\":";
        if(native) allocationReport<<r.y.bytes; else allocationReport<<"null";
        allocationReport
            <<",\"allocation_request_bytes\":"<<r.y.bytes+2*kGuard
            <<",\"guard_bytes_each\":"<<kGuard<<",\"suffix_data_offset\":"<<r.y.bytes<<"}\n";
        allocationReport.close();
        if(!allocationReport) throw std::runtime_error("output allocation report write failed");
        std::cerr<<"output_allocation M="<<m<<" logical_output_bytes="<<logicalBytes
                 <<" physical_payload_bytes="<<r.y.bytes<<" allocation_request_bytes="<<r.y.bytes+2*kGuard
                 <<" suffix_data_offset="<<r.y.bytes<<'\n';
        if(native) std::cerr<<"dataset_output_buffer_bytes="<<r.y.bytes<<'\n';
        r.CheckGuards("after_setup");
        std::vector<uint8_t> first,firstLogits;
        auto prepare=[&]() {
#ifndef HEAD_NATIVE_ONLY
            if(!native) {
                uint64_t bytes=0; executor=nullptr;
                Check(audit?aclnnDFlashDraftLmHeadTop1AuditGetWorkspaceSize(r.xd,r.wd,r.yd,r.ld,&bytes,&executor):
                            aclnnDFlashDraftLmHeadTop1GetWorkspaceSize(r.xd,r.wd,r.yd,&bytes,&executor),"Head GetWorkspaceSize");
                if(!first.empty() && bytes!=workspaceBytes) throw std::runtime_error("workspace drift");
                if(bytes && bytes!=r.workspace.bytes) r.workspace.Allocate(bytes);
                workspaceBytes=bytes;
            }
#endif
        };
        bool firstExecution=true;
        auto execute=[&]() {
            if(firstExecution) { r.CheckGuards("before_first_execute"); firstExecution=false; }
            if(native) Check(aclmdlExecuteAsync(r.model,r.inputs,r.outputs,r.stream),"native Head OM");
#ifndef HEAD_NATIVE_ONLY
            else Check(audit?aclnnDFlashDraftLmHeadTop1Audit(r.workspace.data,workspaceBytes,executor,r.stream):
                             aclnnDFlashDraftLmHeadTop1(r.workspace.data,workspaceBytes,executor,r.stream),"custom Head");
#endif
            Check(aclrtSynchronizeStream(r.stream),"Head synchronize");
        };
        auto poison=[&](int repeat) {
            Check(aclrtMemset(r.y.data,r.y.bytes,repeat%2?0xff:0x7f,r.y.bytes),"output poison");
            if(audit) Check(aclrtMemset(r.logits.data,r.logits.bytes,repeat%2?0xff:0x7f,r.logits.bytes),"logit poison");
        };
        auto verify=[&](const std::string& name) {
            r.CheckGuards(name.empty()?"verify":name);
            Readonly(r.x,x); Readonly(r.w,w);
            auto out=draft_head_test::ReadLogicalOutput(r.y,logicalBytes); if(first.empty()) first=out;
            if(first!=out) throw std::runtime_error("output repeat drift");
            if(!name.empty()) WriteBytes(dir+"/"+name,out);
            if(audit) {
                auto value=r.logits.CopyOut(); if(firstLogits.empty()) firstLogits=value;
                if(value!=firstLogits) throw std::runtime_error("audit logits repeat drift");
                if(!name.empty()) WriteBytes(dir+"/logits-"+name,value);
            }
        };
        Timing timing; timing.warmup=warmup; timing.repetitions=repetitions; timing.continuous=!audit; timing.collectPrepare=!native;
        RunMeasuredCalls(timing,prepare,execute,poison,verify);
        // A different request on the SAME allocation detects stale partition
        // state. Golden is a row permutation of device results, not a CPU GEMM.
        if(m>1) {
            const auto permute=[&](const std::vector<uint8_t>& bytes) {
                std::vector<uint8_t> out(bytes.size()); const size_t width=bytes.size()/m;
                for(int row=0;row<m;++row) std::copy_n(bytes.data()+((row+1)%m)*width,width,out.data()+row*width);
                return out;
            };
            const auto alternate=permute(x); r.x.CopyIn(alternate); poison(0); prepare(); execute();
            r.CheckGuards("permuted.bin");
            Readonly(r.x,alternate); Readonly(r.w,w);
            auto out=draft_head_test::ReadLogicalOutput(r.y,logicalBytes);
            if(out!=permute(first)) throw std::runtime_error("changed-request row permutation failed");
            WriteBytes(dir+"/permuted.bin",out);
            if(audit) {
                const auto value=r.logits.CopyOut();
                if(value!=permute(firstLogits)) throw std::runtime_error("changed-request logits failed");
                WriteBytes(dir+"/logits-permuted.bin",value);
            }
            r.x.CopyIn(x); poison(1); prepare(); execute(); verify("restored.bin");
        }
        std::ofstream report(dir+"/execution.json");
        report<<"{\"status\":\"PASS\",\"mode\":\""<<mode<<"\",\"runtime\":\"AscendCL NPU\",\"cpu_fallback\":false,"
              <<"\"guards_intact\":true,\"input_readonly\":true,\"repeat_equal\":true,\"io_validated\":true,"
              <<"\"m\":"<<m<<",\"k\":2560,\"n\":248320,\"device_id\":"<<r.device<<",\"workspace_bytes\":"<<workspaceBytes
              <<",\"row_permutation_checked\":"<<(m>1?"true":"false");
        timing.Write(report); report<<"}\n";
        if(!report) throw std::runtime_error("report write failed");
        std::cout<<"Head "<<mode<<" execution PASS; numerical comparison follows\n";
        return 0;
    } catch(const std::exception& error) {
        if(!dir.empty()) std::ofstream(dir+"/execution.json")<<"{\"status\":\"FAIL\"}\n";
        std::cerr<<"FAIL: "<<error.what()<<'\n'; return 1;
    }
}
