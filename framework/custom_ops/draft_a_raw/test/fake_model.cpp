// CPU test implementation only; never linked by production CMake.
#include "acl/acl.h"
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
struct aclDataBuffer{void*data;size_t size;};
struct aclmdlDataset{std::vector<aclDataBuffer*> buffers;};
struct aclmdlDesc{};
struct Spec{std::vector<int64_t> dims;aclDataType type;size_t size;};
static std::vector<Spec> inputs,outputs;static int calls=0;
static std::vector<Spec> specs(const char*side){
 std::ifstream f(std::string(std::getenv("CPU_CASE"))+"/"+side+".tsv");std::string line;std::vector<Spec> out;
 while(std::getline(f,line)){std::istringstream s(line);std::string name,type;s>>name>>type;Spec x;int64_t d;
 x.type=type=="float16"?ACL_FLOAT16:type=="int64"?ACL_INT64:ACL_INT16;x.size=type=="int64"?8:2;
 while(s>>d){x.dims.push_back(d);x.size*=size_t(d);}out.push_back(x);}return out;
}
extern "C" {
const char* aclGetRecentErrMsg(){return "CPU_FIXTURE_NOT_DEVICE";}
const char* aclrtGetSocName(){return "Ascend310P3";}
aclError aclInit(const char*){inputs=specs("inputs");outputs=specs("outputs");return 0;}
aclError aclFinalize(){std::ofstream(std::string(std::getenv("CPU_CASE"))+"/calls.txt")<<calls;return 0;}
aclError aclrtSetDevice(int){return 0;} aclError aclrtResetDevice(int){return 0;}
aclError aclrtCreateStream(aclrtStream*p){*p=reinterpret_cast<void*>(1);return 0;}
aclError aclrtDestroyStream(aclrtStream){return 0;} aclError aclrtSynchronizeStream(aclrtStream){return 0;}
aclError aclrtMalloc(void**p,size_t n,aclrtMemMallocPolicy){*p=std::malloc(n);return *p?0:1;}
aclError aclrtFree(void*p){std::free(p);return 0;}
aclError aclrtMemcpy(void*d,size_t cap,const void*s,size_t n,aclrtMemcpyKind){if(n>cap)return 1;std::memcpy(d,s,n);return 0;}
int aclrtMemset(void*d,size_t cap,int v,size_t n){if(n>cap)return 1;std::memset(d,v,n);return 0;}
aclError aclmdlQuerySize(const char*,size_t*w,size_t*b){*w=1024;*b=2048;return 0;}
aclError aclmdlLoadFromFileWithMem(const char*,uint32_t*i,void*,size_t,void*,size_t){*i=1;return 0;}
aclError aclmdlUnload(uint32_t){return 0;}
aclmdlDesc* aclmdlCreateDesc(){return new aclmdlDesc;}
aclError aclmdlDestroyDesc(aclmdlDesc*p){delete p;return 0;}
aclError aclmdlGetDesc(aclmdlDesc*,uint32_t){return 0;}
size_t aclmdlGetNumInputs(const aclmdlDesc*){return inputs.size();}
size_t aclmdlGetNumOutputs(const aclmdlDesc*){return outputs.size();}
aclDataType aclmdlGetInputDataType(const aclmdlDesc*,size_t i){return inputs.at(i).type;}
aclDataType aclmdlGetOutputDataType(const aclmdlDesc*,size_t i){return outputs.at(i).type;}
size_t aclmdlGetInputSizeByIndex(aclmdlDesc*,size_t i){return inputs.at(i).size;}
size_t aclmdlGetOutputSizeByIndex(aclmdlDesc*,size_t i){return outputs.at(i).size;}
aclError aclmdlGetInputDims(const aclmdlDesc*,size_t i,aclmdlIODims*d){auto s=inputs.at(i);d->dimCount=s.dims.size();for(size_t j=0;j<s.dims.size();++j)d->dims[j]=s.dims[j];return 0;}
aclError aclmdlGetOutputDims(const aclmdlDesc*,size_t i,aclmdlIODims*d){auto s=outputs.at(i);d->dimCount=s.dims.size();for(size_t j=0;j<s.dims.size();++j)d->dims[j]=s.dims[j];return 0;}
aclmdlDataset* aclmdlCreateDataset(){return new aclmdlDataset;}
aclError aclmdlDestroyDataset(aclmdlDataset*p){delete p;return 0;}
aclDataBuffer* aclCreateDataBuffer(void*p,size_t n){return new aclDataBuffer{p,n};}
aclError aclDestroyDataBuffer(aclDataBuffer*p){delete p;return 0;}
aclError aclmdlAddDatasetBuffer(aclmdlDataset*d,aclDataBuffer*b){d->buffers.push_back(b);return 0;}
aclError aclmdlExecuteAsync(uint32_t,const aclmdlDataset*in,aclmdlDataset*out,aclrtStream){
 ++calls;for(auto*b:out->buffers)std::memset(b->data,0,b->size);
 const char*f=std::getenv("CPU_FAULT");const int when=std::getenv("CPU_WHEN")?std::atoi(std::getenv("CPU_WHEN")):1;
 if(f&&calls==when){std::string fault(f);if(fault=="log")std::cerr<<"CPU abnormal timed log\n";
 if(fault=="execute")return 1;
 if(fault=="id")static_cast<unsigned char*>(out->buffers[0]->data)[0]=1;
 if(fault=="kv")static_cast<unsigned char*>(out->buffers.back()->data)[outputs.back().size-1]=1;
 if(fault=="readonly")static_cast<unsigned char*>(in->buffers[0]->data)[0]=1;
 if(fault=="guard")static_cast<unsigned char*>(out->buffers[0]->data)[out->buffers[0]->size]=1;}
 return 0;
}
}
