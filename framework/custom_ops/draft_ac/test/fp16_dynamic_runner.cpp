// Independent dynamic FP16 replay adapter. Original static A/C runner is unchanged.
// Gear API/flattened contract follows runtime/cpp/src/acl_chunk.cpp Loaded::ReadDraftGears.
#include "../../draft_quant/test/runner_common.h"
#include <iostream>
#include <memory>
#include <cstring>
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
    GuardedBuffer workspace,weights,control; std::vector<std::unique_ptr<GuardedBuffer>> x,y;
    std::vector<aclDataBuffer*> buffers;
    ~Session() {
        if(stream) aclrtSynchronizeStream(stream);
        if(inputs) aclmdlDestroyDataset(inputs); if(outputs) aclmdlDestroyDataset(outputs);
        for(auto* p:buffers) aclDestroyDataBuffer(p);
        if(desc) aclmdlDestroyDesc(desc); if(loaded) aclmdlUnload(model);
        for(auto& p:x) p->Release(); for(auto& p:y) p->Release(); workspace.Release(); weights.Release(); control.Release();
        if(stream) aclrtDestroyStream(stream); if(ready) aclrtResetDevice(device); if(init) aclFinalize();
    }
    void Guards() {
        CheckNamedGuards({{"workspace",&workspace},{"modelWeights",&weights},{"dynamicControl",&control}},std::cerr);
        for(size_t i=0;i<x.size();++i) CheckNamedGuards({{("input-"+std::to_string(i)).c_str(),x[i].get()}},std::cerr);
        for(size_t i=0;i<y.size();++i) CheckNamedGuards({{("output-"+std::to_string(i)).c_str(),y[i].get()}},std::cerr);
    }
};
int main(int argc,char** argv) {
    try {
        if(argc!=5) throw std::runtime_error("usage: runner DEVICE OM CASE_DIR INPUT_DIR");
        Session s; s.device=std::stoi(argv[1]); std::string dir=argv[3],source=argv[4];
        const auto inputs=Specs(dir+"/inputs.tsv"),outputs=Specs(dir+"/outputs.tsv");
        Check(aclInit(nullptr),"init"); s.init=true; Check(aclrtSetDevice(s.device),"set device"); s.ready=true;
        Check(aclrtCreateStream(&s.stream),"stream"); size_t work=0,weights=0;
        Check(aclmdlQuerySize(argv[2],&work,&weights),"query model");
        if(work) s.workspace.Allocate(work); if(weights) s.weights.Allocate(weights);
        Check(aclmdlLoadFromFileWithMem(argv[2],&s.model,s.workspace.data,work,s.weights.data,weights),"load OM"); s.loaded=true;
        s.desc=aclmdlCreateDesc(); if(!s.desc) throw std::runtime_error("desc"); Check(aclmdlGetDesc(s.desc,s.model),"get desc");
        if(aclmdlGetNumInputs(s.desc)!=inputs.size()+1 || aclmdlGetNumOutputs(s.desc)!=outputs.size()) throw std::runtime_error("OM public IO count differs");
        if(inputs.size()!=17 || outputs.size()!=13 || inputs[0].name!="features" || inputs[0].shape.size()!=3 || inputs[0].shape[1]!=64)
            throw std::runtime_error("dynamic FP16 requires 17/13 ABI and maximum features C64");
        int rows=0; std::ifstream(dir+"/gear.txt")>>rows;
        if(rows!=16 && rows!=64) throw std::runtime_error("missing C16/C64 gear selection");
        size_t controlIndex=0,gearCount=0;
        Check(aclmdlGetInputIndexByName(s.desc,ACL_DYNAMIC_TENSOR_NAME,&controlIndex),"dynamic control index");
        if(controlIndex>inputs.size()) throw std::runtime_error("dynamic control index outside dataset");
        Check(aclmdlGetInputDynamicGearCount(s.desc,size_t(-1),&gearCount),"gear count");
        if(gearCount!=2) throw std::runtime_error("OM must have exactly C16/C64 gears");
        aclmdlIODims actual[2]{},expected[2]{};
        Check(aclmdlGetInputDynamicDims(s.desc,size_t(-1),actual,2),"query dynamic gears");
        for(int g=0;g<2;++g) {
            for(const auto& t:inputs) for(size_t j=0;j<t.shape.size();++j) {
                auto& e=expected[g]; if(e.dimCount>=sizeof(e.dims)/sizeof(e.dims[0])) throw std::runtime_error("gear rank overflow");
                e.dims[e.dimCount++]=(t.name=="features" && j==1)?(g?64:16):t.shape[j];
            }
            int matches=0;
            for(const auto& a:actual) if(a.dimCount==expected[g].dimCount && std::equal(a.dims,a.dims+a.dimCount,expected[g].dims)) ++matches;
            if(matches!=1) throw std::runtime_error("actual OM gear differs from complete flattened ABI");
        }
        size_t controlBytes=aclmdlGetInputSizeByIndex(s.desc,controlIndex);
        if(!controlBytes || controlBytes>1024*1024) throw std::runtime_error("invalid dynamic control bytes");
        s.control.Allocate((controlBytes+31)/32*32);
        std::vector<aclDataBuffer*> inputBuffers,outputBuffers;
        s.inputs=aclmdlCreateDataset(); s.outputs=aclmdlCreateDataset(); if(!s.inputs||!s.outputs) throw std::runtime_error("datasets");
        std::vector<std::vector<uint8_t>> originals;
        for(bool out:{false,true}) {
            const auto& specs=out?outputs:inputs;
            for(size_t i=0;i<specs.size();++i) {
                const auto& t=specs[i]; const size_t index=i+(!out && i>=controlIndex); aclmdlIODims dims{};
                Check(out?aclmdlGetOutputDims(s.desc,index,&dims):aclmdlGetInputDims(s.desc,index,&dims),"dims");
                if(dims.dimCount!=t.shape.size() || (out?aclmdlGetOutputDataType(s.desc,index):aclmdlGetInputDataType(s.desc,index))!=t.dtype) throw std::runtime_error("OM dtype/rank differs");
                if(!out && t.name=="features" && dims.dimCount==3 && dims.dims[1]==-1) dims.dims[1]=64;
                for(size_t j=0;j<t.shape.size();++j) if(dims.dims[j]!=t.shape[j]) throw std::runtime_error("OM shape differs");
                size_t bytes=out?aclmdlGetOutputSizeByIndex(s.desc,index):aclmdlGetInputSizeByIndex(s.desc,index);
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
                (out?outputBuffers:inputBuffers).push_back(b); s.buffers.push_back(b);
                (out?s.y:s.x).push_back(std::move(memory));
                std::cerr<<(out?"output ":"input ")<<t.name<<" logical="<<t.bytes<<" reported="<<bytes<<" physical="<<physical<<'\n';
            }
        }
        auto* controlBuffer=aclCreateDataBuffer(s.control.data,s.control.bytes);
        if(!controlBuffer) throw std::runtime_error("control buffer"); s.buffers.push_back(controlBuffer);
        inputBuffers.insert(inputBuffers.begin()+controlIndex,controlBuffer);
        for(auto* b:inputBuffers) Check(aclmdlAddDatasetBuffer(s.inputs,b),"bind dynamic input");
        for(auto* b:outputBuffers) Check(aclmdlAddDatasetBuffer(s.outputs,b),"bind output");
        for(size_t i=0;i<inputs.size();++i) if(inputs[i].name=="valid_rows") {
            int64_t valid=0;
            if(inputs[i].dtype==ACL_INT16 && originals[i].size()==2) { int16_t v; std::memcpy(&v,originals[i].data(),2); valid=v; }
            else if(inputs[i].dtype==ACL_INT64 && originals[i].size()==8) std::memcpy(&valid,originals[i].data(),8);
            else throw std::runtime_error("valid_rows dtype/size");
            if(valid<1 || valid>64 || (valid<=16?16:64)!=rows) throw std::runtime_error("snapshot valid_rows/selected gear differ");
        }
        Check(aclmdlSetInputDynamicDims(s.model,s.inputs,controlIndex,&expected[rows==64]),"set verified dynamic gear");
        Check(aclrtSynchronizeStream(s.stream),"sync dynamic gear");
        std::ofstream gearReport(dir+"/dynamic-gear.json");
        gearReport<<"{\"status\":\"PASS\",\"selected_rows\":"<<rows<<",\"control_index\":"<<controlIndex
                  <<",\"control_bytes\":"<<controlBytes<<",\"actual_gears\":[";
        for(int g=0;g<2;++g) { if(g) gearReport<<','; gearReport<<'[';
            for(size_t j=0;j<actual[g].dimCount;++j) { if(j) gearReport<<',';gearReport<<actual[g].dims[j]; } gearReport<<']'; }
        gearReport<<"],\"selection_timing\":\"before continuous-v1 measured calls\"}\n"; gearReport.close();
        std::vector<std::vector<uint8_t>> first(outputs.size());
        Timing timing; timing.continuous=true; timing.collectPrepare=false; timing.warmup=5; timing.repetitions=30;
        auto verify=[&](const std::string& name) {
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
            for(size_t i=0;i<outputs.size();++i) {
                auto value=s.y[i]->CopyOut(); value.resize(outputs[i].bytes);
                if(first[i].empty()) first[i]=value; else if(first[i]!=value) throw std::runtime_error("repeat drift");
                if(!name.empty()) WriteBytes(dir+"/"+outputs[i].name+"-"+name,value);
            }
        };
        s.Guards();
        RunMeasuredCalls(timing,[](){},[&](){ Check(aclmdlExecuteAsync(s.model,s.inputs,s.outputs,s.stream),"execute"); Check(aclrtSynchronizeStream(s.stream),"sync"); },
            [&](int repeat){ for(auto& y:s.y) Check(aclrtMemset(y->data,y->bytes,repeat%2?0xff:0x7f,y->bytes),"poison"); },verify);
        // Changed-request check on the SAME model/workspace, outside timed calls.
        const bool permutable=(inputs[0].name=="x" || inputs[0].name=="hidden") && inputs[0].shape[0]>1;
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
        std::ofstream report(dir+"/execution.json"); report<<"{\"status\":\"PASS\",\"scope\":\"execution only; numerical comparison separate\",\"guards\":true,\"readonly\":true,\"repeat\":true,\"cpu_fallback\":false";
        report<<",\"row_permutation_checked\":"<<(permutable?"true":"false");
        timing.Write(report); report<<"}\n"; return 0;
    } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
}
