// Runtime test double rejects any ACL copy/memset during continuous warmup/samples.
#include "acl/acl.h"
#include "aclnn_d_flash_w8_a8_full_gate.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
static int prepares=0,executes=0;static bool continuousInterval=false;
static aclTensor* output=nullptr;static aclOpExecutor executor;
const char* aclGetRecentErrMsg(){return "CPU fixture only";}const char* aclrtGetSocName(){return "Ascend310P3";}
int aclrtMalloc(void**p,size_t n,int){if(continuousInterval)std::abort();*p=std::malloc(n);return *p?0:1;}
int aclrtFree(void*p){std::free(p);return 0;}
int aclrtMemset(void*p,size_t,int value,size_t n){if(continuousInterval)std::abort();std::memset(p,value,n);return 0;}
int aclrtMemcpy(void*d,size_t,const void*s,size_t n,int){if(continuousInterval)std::abort();std::memcpy(d,s,n);return 0;}
int aclInit(const char*){return 0;}int aclFinalize(){return 0;}int aclrtSetDevice(int){return 0;}int aclrtResetDevice(int){return 0;}
int aclrtCreateStream(aclrtStream*p){*p=(void*)1;return 0;}int aclrtDestroyStream(aclrtStream){return 0;}int aclrtSynchronizeStream(aclrtStream){if(executes==37)continuousInterval=false;return 0;}
aclTensor* aclCreateTensor(const int64_t*,uint64_t,int,const int64_t*,int64_t,int,const int64_t*,uint64_t,void*p){return new aclTensor{p};}
int aclDestroyTensor(const aclTensor*p){delete p;return 0;}
int aclnnDFlashW8A8FullGateGetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,int64_t mode,int64_t limit,aclTensor*out,uint64_t*size,aclOpExecutor**e){
 volatile int clockResolutionWork=0;for(int i=0;i<1000;++i)clockResolutionWork+=i; // CPU fixture only
 ++prepares;continuousInterval=prepares>=3&&prepares<=37;output=out;*e=&executor;unsigned blocks=limit?limit:7;*size=4096+blocks*5120;
 if(prepares==3&&std::getenv("TEST_TIMED_LOG"))std::fprintf(stderr,"unexpected timed Host diagnostic\n");
 if(prepares==1)std::fprintf(stderr,"STAGE4B_CORE_PROBE aic=8 aiv=8 all=7\nSTAGE4B_ARCH_PROBE arch=2002\nSTAGE4B_CORE_POLICY aic=8 aiv=8 all=7 selected=7\nSTAGE4B_SCHEDULE coreLimit=%lld blocks=%u cores=7 system=4096 arena=4096 stride=5120 user=%u total=%llu mode=%lld\nscratch=239104 userUB=12288 ubGuards=2560 reserve=8192 single=16/64/128 base=16/64/128\n",(long long)limit,blocks,blocks*5120,(unsigned long long)*size,(long long)mode);
 return 0;
}
int aclnnDFlashW8A8FullGate(void*,uint64_t,aclOpExecutor*,aclrtStream){++executes;std::memset(output->data,0,622592);if(std::getenv("TEST_BAD_GOLDEN"))((unsigned char*)output->data)[0]=1;return 0;}
