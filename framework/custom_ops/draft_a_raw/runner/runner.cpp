// Static multi-IO OM replay; no graph substitutions or numerical fallback here.
#include "runner_common.h"
#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
using namespace draft_quant_test;
struct TensorSpec { std::string name; aclDataType dtype; std::vector<int64_t> shape; size_t bytes; };
std::vector<TensorSpec> Specs(const std::string& file) {
    std::ifstream in(file); std::vector<TensorSpec> result; std::string line;
    while(std::getline(in,line)) {
        std::istringstream row(line); TensorSpec t; std::string type; int64_t d;
        if(!(row>>t.name>>type)) throw std::runtime_error("bad tensor spec");
        if(t.name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")!=std::string::npos) throw std::runtime_error("unsafe tensor name");
        if(type=="float16") { t.dtype=ACL_FLOAT16; t.bytes=2; }
        else if(type=="int64") { t.dtype=ACL_INT64; t.bytes=8; }
        else if(type=="int16") { t.dtype=ACL_INT16; t.bytes=2; }
        else if(type=="float32") { t.dtype=ACL_FLOAT; t.bytes=4; }
        else throw std::runtime_error("unsupported dtype");
        while(row>>d) { if(d<=0 || t.bytes>size_t(-1)/size_t(d)) throw std::runtime_error("bad dimension"); t.shape.push_back(d); t.bytes*=d; }
        if(t.shape.empty()) throw std::runtime_error("rank zero not supported"); result.push_back(t);
    }
    if(result.empty()) throw std::runtime_error("missing specs"); return result;
}
struct Session {
    int device=0; bool init=false,ready=false,loaded=false; uint32_t model=0;
    aclrtStream stream=nullptr; aclmdlDesc* desc=nullptr; aclmdlDataset *inputs=nullptr,*outputs=nullptr;
    GuardedBuffer workspace,weights; std::vector<std::unique_ptr<GuardedBuffer>> x,y;
    std::vector<aclDataBuffer*> buffers;
    ~Session() {
        if(stream) aclrtSynchronizeStream(stream);
        if(inputs) aclmdlDestroyDataset(inputs); if(outputs) aclmdlDestroyDataset(outputs);
        for(auto* p:buffers) aclDestroyDataBuffer(p);
        if(desc) aclmdlDestroyDesc(desc); if(loaded) aclmdlUnload(model);
        for(auto& p:x) p->Release(); for(auto& p:y) p->Release(); workspace.Release(); weights.Release();
        if(stream) aclrtDestroyStream(stream); if(ready) aclrtResetDevice(device); if(init) aclFinalize();
    }
    void Guards() {
        CheckNamedGuards({{"workspace",&workspace},{"modelWeights",&weights}},std::cerr);
        for(size_t i=0;i<x.size();++i) CheckNamedGuards({{("input-"+std::to_string(i)).c_str(),x[i].get()}},std::cerr);
        for(size_t i=0;i<y.size();++i) CheckNamedGuards({{("output-"+std::to_string(i)).c_str(),y[i].get()}},std::cerr);
    }
};
int main(int argc,char** argv) {
    try {
        if(argc!=7) throw std::runtime_error("usage: runner DEVICE OM CASE_DIR INPUT_DIR GOLDEN_DIR smoke|benchmark");
        const bool benchmark=std::string(argv[6])=="benchmark";
        if(!benchmark && std::string(argv[6])!="smoke")throw std::runtime_error("invalid phase");
        Session s; s.device=std::stoi(argv[1]); std::string dir=argv[3],source=argv[4];
        const auto inputs=Specs(dir+"/inputs.tsv"),outputs=Specs(dir+"/outputs.tsv");
        if(inputs.size()!=15||outputs.size()!=11||outputs[0].name!="draft_top1"||outputs[0].dtype!=ACL_INT64||outputs[0].shape!=std::vector<int64_t>({1,15}))throw std::runtime_error("full Draft ABI");
        for(int i=0;i<5;++i)for(int j=0;j<2;++j)if(outputs[1+2*i+j].name!="d"+std::to_string(i)+(j?"_value":"_key"))throw std::runtime_error("KV output inventory");
        std::vector<std::vector<uint8_t>> golden;for(const auto&t:outputs)golden.push_back(ReadBytes(std::string(argv[5])+"/"+t.name+".bin",t.bytes));
        Check(aclInit(nullptr),"init"); s.init=true; Check(aclrtSetDevice(s.device),"set device"); s.ready=true;
        const char* soc=aclrtGetSocName();if(!soc||std::string(soc)!="Ascend310P3")throw std::runtime_error("requires Ascend310P3");
        Check(aclrtCreateStream(&s.stream),"stream"); size_t work=0,weights=0;
        Check(aclmdlQuerySize(argv[2],&work,&weights),"query model");
        if(work) s.workspace.Allocate(work); if(weights) s.weights.Allocate(weights);
        Check(aclmdlLoadFromFileWithMem(argv[2],&s.model,s.workspace.data,work,s.weights.data,weights),"load OM"); s.loaded=true;
        s.desc=aclmdlCreateDesc(); if(!s.desc) throw std::runtime_error("desc"); Check(aclmdlGetDesc(s.desc,s.model),"get desc");
        if(aclmdlGetNumInputs(s.desc)!=inputs.size() || aclmdlGetNumOutputs(s.desc)!=outputs.size()) throw std::runtime_error("OM public IO count differs");
        s.inputs=aclmdlCreateDataset(); s.outputs=aclmdlCreateDataset(); if(!s.inputs||!s.outputs) throw std::runtime_error("datasets");
        std::vector<std::vector<uint8_t>> originals;
        for(bool out:{false,true}) {
            const auto& specs=out?outputs:inputs;
            for(size_t i=0;i<specs.size();++i) {
                const auto& t=specs[i]; aclmdlIODims dims{};
                Check(out?aclmdlGetOutputDims(s.desc,i,&dims):aclmdlGetInputDims(s.desc,i,&dims),"dims");
                if(dims.dimCount!=t.shape.size() || (out?aclmdlGetOutputDataType(s.desc,i):aclmdlGetInputDataType(s.desc,i))!=t.dtype) throw std::runtime_error("OM dtype/rank differs");
                for(size_t j=0;j<t.shape.size();++j) if(dims.dims[j]!=t.shape[j]) throw std::runtime_error("OM shape differs");
                size_t bytes=out?aclmdlGetOutputSizeByIndex(s.desc,i):aclmdlGetInputSizeByIndex(s.desc,i);
                if(bytes<t.bytes) throw std::runtime_error("OM reported size below logical bytes");
                if(bytes>size_t(-1)-31) throw std::runtime_error("allocation overflow");
                const auto physical=(std::max(bytes,t.bytes)+31)/32*32;
                auto memory=std::make_unique<GuardedBuffer>(); memory->Allocate(physical);
                if(!out) {
                    originals.push_back(ReadBytes(source+"/"+t.name+".bin",t.bytes));
                    if(physical==t.bytes) memory->CopyIn(originals.back());
                    else { auto padded=originals.back(); padded.resize(physical,0); memory->CopyIn(padded); }
                }
                auto* b=aclCreateDataBuffer(memory->data,physical); if(!b) throw std::runtime_error("buffer");
                Check(aclmdlAddDatasetBuffer(out?s.outputs:s.inputs,b),"bind"); s.buffers.push_back(b);
                (out?s.y:s.x).push_back(std::move(memory));
                std::cerr<<(out?"output ":"input ")<<t.name<<" logical="<<t.bytes<<" reported="<<bytes<<" physical="<<physical<<'\n';
            }
        }
        std::vector<std::vector<uint8_t>> first(outputs.size());
        Timing timing; timing.continuous=benchmark; timing.collectPrepare=false; timing.warmup=benchmark?5:0; timing.repetitions=benchmark?30:0;
        uint64_t beforeLog=0;int checks=0;
        auto logBytes=[&](){std::fflush(nullptr);std::ifstream f(dir+"/runner.log",std::ios::binary|std::ios::ate);if(!f||f.tellg()<0)throw std::runtime_error("runner log absent");return uint64_t(f.tellg());};
        auto verify=[&](const std::string& name) {
            if(benchmark&&name=="benchmark-last.bin"&&logBytes()!=beforeLog)throw std::runtime_error("timed Host log changed");
            s.Guards();
            for(size_t i=0;i<inputs.size();++i) {
                std::vector<uint8_t> chunk(std::min<size_t>(4*1024*1024,s.x[i]->bytes));
                for(size_t offset=0;offset<s.x[i]->bytes;offset+=chunk.size()) {
                    const size_t count=std::min(chunk.size(),s.x[i]->bytes-offset);
                    Check(aclrtMemcpy(chunk.data(),chunk.size(),static_cast<uint8_t*>(s.x[i]->data)+offset,count,ACL_MEMCPY_DEVICE_TO_HOST),"readonly");
                    for(size_t j=0;j<count;++j) {
                        const auto expected=offset+j<inputs[i].bytes?originals[i][offset+j]:uint8_t(0);
                        if(chunk[j]!=expected) throw std::runtime_error("input modified");
                    }
                }
            }
            std::string numericalFailure;
            for(size_t i=0;i<outputs.size();++i) {
                auto value=s.y[i]->CopyOut(); value.resize(outputs[i].bytes);
                if(!name.empty())WriteBytes(dir+"/"+outputs[i].name+"-"+name,value);
                if(value!=golden[i])numericalFailure="frozen Golden mismatch: "+outputs[i].name;
                if(first[i].empty()) first[i]=value; else if(first[i]!=value)numericalFailure="repeat drift: "+outputs[i].name;
            }
            if(!numericalFailure.empty())throw std::runtime_error(numericalFailure);
            if(++checks==2 && benchmark)beforeLog=logBytes();
        };
        s.Guards();
        RunMeasuredCalls(timing,[](){},[&](){ Check(aclmdlExecuteAsync(s.model,s.inputs,s.outputs,s.stream),"execute"); Check(aclrtSynchronizeStream(s.stream),"sync"); },
            [&](int repeat){ for(auto& y:s.y){Check(aclrtMemset(y->data,y->bytes,repeat%2?0xff:0x7f,y->bytes),"poison");if(y->CopyOut()!=std::vector<uint8_t>(y->bytes,repeat%2?0xff:0x7f))throw std::runtime_error("Poison readback failed");} },verify);
        const bool permutable=false; // Full frozen Draft: no permutation of state ABI.
        std::ofstream report(dir+"/execution.json"); report<<"{\"status\":\"PASS\",\"scope\":\"full Draft frozen Golden checked in process\",\"guards\":true,\"readonly\":true,\"repeat\":true,\"cpu_fallback\":false";
        report<<",\"row_permutation_checked\":"<<(permutable?"true":"false");
        report<<",\"golden_match\":true,\"poison_verified\":true,\"sync_completed\":true,\"timed_log_stable\":true,\"device\":"<<s.device<<",\"workspace_bytes\":"<<work<<",\"model_weight_bytes\":"<<weights<<",\"verification_calls\":"<<checks;
        timing.Write(report); report<<"}\n"; return 0;
    } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
}
