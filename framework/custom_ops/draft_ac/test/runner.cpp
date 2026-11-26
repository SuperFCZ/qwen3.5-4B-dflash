// Static multi-IO OM replay; no graph substitutions or numerical fallback here.
#include "../../draft_quant/test/runner_common.h"
#include <iostream>
#include <memory>
#include <sstream>
#include <limits>
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
aclrtMemMallocPolicy PagePolicy(const std::string& name) {
    if (name == "normal") return ACL_MEM_MALLOC_NORMAL_ONLY;
    if (name == "huge-first") return ACL_MEM_MALLOC_HUGE_FIRST;
    throw std::runtime_error("unknown allocation policy " + name);
}
int main(int argc,char** argv) {
    try {
        const bool capture = argc == 9 && std::string(argv[8]) == "--capture";
        const bool snapshot = argc == 8 || capture;
        if(argc!=5 && !snapshot) throw std::runtime_error("usage: runner DEVICE OM CASE_DIR INPUT_DIR [WORK_BYTES WORK_POLICY IO_POLICY [--capture]]");
        Session s; s.device=std::stoi(argv[1]); std::string dir=argv[3],source=argv[4];
        const auto inputs=Specs(dir+"/inputs.tsv"),outputs=Specs(dir+"/outputs.tsv");
        const auto work_policy = snapshot ? PagePolicy(argv[6]) : ACL_MEM_MALLOC_HUGE_FIRST;
        const auto io_policy = snapshot ? PagePolicy(argv[7]) : ACL_MEM_MALLOC_HUGE_FIRST;
        std::vector<std::vector<uint8_t>> golden;
        if (snapshot && !capture)
            for (const auto& out : outputs) golden.push_back(ReadBytes(dir+"/golden/"+out.name+".bin", out.bytes));
        Check(aclInit(nullptr),"init"); s.init=true; Check(aclrtSetDevice(s.device),"set device"); s.ready=true;
        Check(aclrtCreateStream(&s.stream),"stream"); size_t work=0,weights=0;
        Check(aclmdlQuerySize(argv[2],&work,&weights),"query model");
        if (snapshot) {
            const std::string arg(argv[5]);
            if (arg.empty() || arg.find_first_not_of("0123456789") != std::string::npos)
                throw std::runtime_error("invalid workspace byte count");
            const auto capacity = std::stoull(arg);
            if ((!capacity && !capture) || (capacity && capacity < work) ||
                capacity > std::numeric_limits<size_t>::max() - 2*kGuard)
                throw std::runtime_error("incompatible receiver workspace capacity");
            if (capacity) work = static_cast<size_t>(capacity);
        }
        if (work) s.workspace.Allocate(work, work_policy);
        if (weights && !snapshot) s.weights.Allocate(weights);
        Check(aclmdlLoadFromFileWithMem(argv[2],&s.model,s.workspace.data,work,s.weights.data,s.weights.bytes),"load OM"); s.loaded=true;
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
                auto memory=std::make_unique<GuardedBuffer>(); memory->Allocate(physical, io_policy);
                if(!out) {
                    originals.push_back(ReadBytes(source+"/"+t.name+".bin",t.bytes));
                    if(physical==t.bytes) memory->CopyIn(originals.back());
                    else { auto padded=originals.back(); padded.resize(physical,0); memory->CopyIn(padded); }
                }
                auto* b=aclCreateDataBuffer(memory->data,snapshot ? bytes : physical); if(!b) throw std::runtime_error("buffer");
                Check(aclmdlAddDatasetBuffer(out?s.outputs:s.inputs,b),"bind"); s.buffers.push_back(b);
                (out?s.y:s.x).push_back(std::move(memory));
                std::cerr<<(out?"output ":"input ")<<t.name<<" logical="<<t.bytes<<" reported="<<bytes<<" physical="<<physical<<'\n';
            }
        }
        std::vector<std::vector<uint8_t>> first(outputs.size());
        std::vector<bool> ref_input(inputs.size(), false), changed_input(inputs.size(), false);
        if (capture) {
            for (size_t i = 0; i < inputs.size(); ++i) {
                const auto& t = inputs[i];
                // Target state is a named input/output pair. CacheUpdate may write its ref operand.
                const auto suffix = t.name.find('_');
                if (t.name.size()<3 || t.name.front()!='t' || suffix==std::string::npos || suffix<=1 ||
                    t.name.find_first_not_of("0123456789",1)!=suffix) continue;
                ref_input[i] = std::any_of(outputs.begin(), outputs.end(), [&](const auto& y) {
                    return y.name==t.name && y.dtype==t.dtype && y.shape==t.shape;
                });
            }
        }
        Timing timing; timing.continuous=!capture; timing.collectPrepare=false;
        timing.warmup=capture?0:5; timing.repetitions=capture?0:30;
        auto restore_capture = [&]() {
            for (size_t i=0; capture && i<inputs.size(); ++i) {
                if (!changed_input[i]) continue;
                auto restored=originals[i];
                restored.resize(s.x[i]->bytes,0);
                s.x[i]->CopyIn(restored);
            }
        };
        auto verify=[&](const std::string& name) {
            s.Guards();
            for(size_t i=0;i<inputs.size();++i) {
                bool changed=false;
                std::vector<uint8_t> chunk(std::min<size_t>(4*1024*1024,s.x[i]->bytes));
                for(size_t offset=0;offset<s.x[i]->bytes;offset+=chunk.size()) {
                    const size_t count=std::min(chunk.size(),s.x[i]->bytes-offset);
                    Check(aclrtMemcpy(chunk.data(),chunk.size(),static_cast<uint8_t*>(s.x[i]->data)+offset,count,ACL_MEMCPY_DEVICE_TO_HOST),"readonly");
                    for(size_t j=0;j<count;++j) {
                        const auto expected=offset+j<inputs[i].bytes?originals[i][offset+j]:uint8_t(0);
                        if(chunk[j]!=expected) {
                            if (!ref_input[i] || offset+j>=inputs[i].bytes)
                                throw std::runtime_error("input modified: "+inputs[i].name+" byte "+std::to_string(offset+j));
                            changed=true;
                        }
                    }
                }
                if (capture) {
                    const bool initial = name == "actual-0.bin";
                    if (!initial && changed != changed_input[i])
                        throw std::runtime_error("different ref mutation on repeat: "+inputs[i].name);
                    if (changed) {
                        const auto path=dir+"/"+inputs[i].name+"-input-after.bin";
                        auto actual=s.x[i]->CopyOut(); actual.resize(inputs[i].bytes);
                        if (initial) WriteBytes(path,actual);
                        else if (actual!=ReadBytes(path,inputs[i].bytes))
                            throw std::runtime_error("different ref bytes on repeat: "+inputs[i].name);
                    }
                    changed_input[i]=changed;
                }
            }
            for(size_t i=0;i<outputs.size();++i) {
                auto value=s.y[i]->CopyOut(); value.resize(outputs[i].bytes);
                if(first[i].empty()) first[i]=value;
                else if(first[i]!=value) {
                    if (capture) WriteBytes(dir+"/"+outputs[i].name+"-repeat-mismatch.bin",value);
                    throw std::runtime_error("repeat drift: "+outputs[i].name);
                }
                if (snapshot && !capture && value != golden[i]) {
                    WriteBytes(dir+"/"+outputs[i].name+"-mismatch.bin", value);
                    throw std::runtime_error("receiver snapshot mismatch: "+outputs[i].name);
                }
                if((!snapshot && !name.empty()) || (capture && name=="actual-0.bin"))
                    WriteBytes(dir+"/"+outputs[i].name+"-"+name,value);
            }
        };
        s.Guards();
        RunMeasuredCalls(timing,restore_capture,[&](){ Check(aclmdlExecuteAsync(s.model,s.inputs,s.outputs,s.stream),"execute"); Check(aclrtSynchronizeStream(s.stream),"sync"); },
            [&](int repeat){ for(auto& y:s.y) Check(aclrtMemset(y->data,y->bytes,repeat%2?0xff:0x7f,y->bytes),"poison"); },verify);
        // Changed-request check on the SAME model/workspace, outside timed calls.
        const bool permutable=!snapshot && (inputs[0].name=="x" || inputs[0].name=="hidden") && inputs[0].shape[0]>1;
        if(permutable) {
            const size_t rows=inputs[0].shape[0], rowBytes=inputs[0].bytes/rows;
            auto saved=originals[0]; auto savedOutputs=first;
            std::rotate(originals[0].begin(),originals[0].begin()+rowBytes,originals[0].end());
            auto padded=originals[0]; padded.resize(s.x[0]->bytes,0); s.x[0]->CopyIn(padded);
            for(size_t i=0;i<outputs.size();++i) {
                if(outputs[i].shape[0]!=int64_t(rows)) throw std::runtime_error("permutation output shape differs");
                std::rotate(first[i].begin(),first[i].begin()+outputs[i].bytes/rows,first[i].end());
            }
            for(int repeat=0;repeat<2;++repeat) {
                for(auto& y:s.y) Check(aclrtMemset(y->data,y->bytes,repeat?0xff:0x7f,y->bytes),"permutation poison");
                Check(aclmdlExecuteAsync(s.model,s.inputs,s.outputs,s.stream),"permutation execute");
                Check(aclrtSynchronizeStream(s.stream),"permutation sync");
                verify("permutation-"+std::to_string(repeat)+".bin");
            }
            originals[0]=std::move(saved); first=std::move(savedOutputs);
            padded=originals[0]; padded.resize(s.x[0]->bytes,0); s.x[0]->CopyIn(padded);
            Check(aclmdlExecuteAsync(s.model,s.inputs,s.outputs,s.stream),"restore execute");
            Check(aclrtSynchronizeStream(s.stream),"restore sync"); verify("restored.bin");
        }
        std::ofstream report(dir+"/execution.json");
        report.exceptions(std::ios::badbit | std::ios::failbit);
        report<<"{\"status\":\"PASS\",\"scope\":\"" << (snapshot ? "single OM receiver snapshot" : "execution only; numerical comparison separate")
              << "\",\"guards\":true,\"readonly\":" << (std::count(changed_input.begin(),changed_input.end(),true) ? "false" : "true")
              << ",\"repeat\":true,\"cpu_fallback\":false";
        report<<",\"row_permutation_checked\":"<<(permutable?"true":"false");
        if (capture) {
            report << ",\"reference_capture\":true,\"mutated_inputs\":[";
            bool emitted=false;
            for (size_t i=0;i<inputs.size();++i) if (changed_input[i]) {
                if (emitted) report << ',';
                report << '"' << inputs[i].name << '"'; emitted=true;
            }
            report << ']';
        }
        if (snapshot)
            report << ",\"allocation\":{\"workspace\":\"" << argv[6] << "\",\"io\":\"" << argv[7]
                   << "\",\"workspace_bytes\":" << work << ",\"ge_weight_bytes\":" << weights
                   << ",\"golden_outputs\":" << golden.size() << "}";
        timing.Write(report); report<<"}\n"; report.close(); return 0;
    } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
}
