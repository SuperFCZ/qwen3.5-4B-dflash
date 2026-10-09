// Exercise the real shared guard implementation with CPU memory standing in
// for ACL allocations. Aligned stores below are a hypothesis, NOT NPU evidence.
#include <cstddef>
extern "C" const char* aclGetRecentErrMsg();
extern "C" int aclrtMemset(void*,std::size_t,int,std::size_t);
#include "../../draft_quant/test/runner_common.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>

extern "C" const char* aclGetRecentErrMsg() { return "CPU guard model"; }
extern "C" aclError aclrtMalloc(void** ptr,size_t bytes,aclrtMemMallocPolicy) {
    *ptr=std::malloc(bytes); return *ptr?ACL_SUCCESS:1;
}
extern "C" aclError aclrtFree(void* ptr) { std::free(ptr); return ACL_SUCCESS; }
extern "C" int aclrtMemset(void* ptr,size_t capacity,int value,size_t count) {
    assert(count<=capacity); std::memset(ptr,value,count); return ACL_SUCCESS;
}
extern "C" aclError aclrtMemcpy(void* dst,size_t capacity,const void* src,size_t count,aclrtMemcpyKind) {
    assert(count<=capacity); std::memcpy(dst,src,count); return ACL_SUCCESS;
}
using namespace draft_quant_test;

int main() {
    GuardedBuffer buffer;
    std::ostringstream empty;
    assert(buffer.InspectGuards("workspace",empty));
    assert(empty.str().find("buffer=workspace status=UNALLOCATED")!=std::string::npos);
    // Verify every byte of BOTH sentinels, including the final guard byte.
    for(bool prefix:{true,false}) for(size_t offset=0;offset<kGuard;++offset) {
        buffer.Allocate(8);
        auto* guard=prefix?static_cast<uint8_t*>(buffer.allocation):static_cast<uint8_t*>(buffer.data)+buffer.bytes;
        guard[offset]=0x42;
        std::ostringstream report;
        assert(!buffer.InspectGuards("y",report));
        const auto text=report.str();
        assert(text.find(std::string("side=")+(prefix?"prefix":"suffix"))!=std::string::npos);
        assert(text.find("first_changed_guard_offset="+std::to_string(offset)+" ")!=std::string::npos);
        assert(text.find("last_changed_guard_offset="+std::to_string(offset)+" ")!=std::string::npos);
        assert(text.find("first_actual=0x42")!=std::string::npos);
        const std::string dataOffset=prefix?"-"+std::to_string(kGuard-offset):std::to_string(8+offset);
        assert(text.find("first_changed_data_offset="+dataOffset+" ")!=std::string::npos);
        bool rejected=false;
        try { buffer.CheckGuards("y"); } catch(const std::runtime_error& e) {
            rejected=std::string(e.what()).find("buffer=y")!=std::string::npos;
        }
        assert(rejected);
    }
    // A hypothetical 32B store reproduces the M%4 pattern on the original
    // allocation, while a precise M*8 store must always leave guards intact.
    for(size_t m=1;m<=15;++m) {
        const size_t logical=m*8,aligned=(logical+31)/32*32;
        buffer.Allocate(logical);
        std::memset(buffer.data,0,logical);
        buffer.CheckGuards("y");
        std::memset(buffer.data,0,aligned);
        std::ostringstream report;
        assert(buffer.InspectGuards("y",report)==(m%4==0));
        if(m%4) {
            assert(report.str().find("changed_bytes="+std::to_string(aligned-logical)+" ")!=std::string::npos);
            assert(report.str().find("first_changed_guard_offset=0 ")!=std::string::npos);
            assert(report.str().find("last_changed_data_offset="+std::to_string(aligned-1)+" ")!=std::string::npos);
        }
    }
    GuardedBuffer other;
    buffer.Allocate(8); other.Allocate(16);
    static_cast<uint8_t*>(buffer.allocation)[0]=0;
    static_cast<uint8_t*>(other.data)[16]=0;
    std::ostringstream all;
    bool rejected=false;
    try { CheckNamedGuards({{"x",&buffer},{"w",&other},{"y",&buffer},{"logits",&other},
                            {"workspace",&buffer},{"modelWeights",&other}},all); }
    catch(const std::runtime_error& e) {
        rejected=std::string(e.what())=="device buffer guard overwritten: x,w,y,logits,workspace,modelWeights";
    }
    assert(rejected);
    for(const char* name:{"x","w","y","logits","workspace","modelWeights"})
        assert(all.str().find(std::string("buffer=")+name+" ")!=std::string::npos);
    buffer.Release(); other.Release();
    std::cout<<"Guard diagnostics CPU model PASS; actual NPU buffer/write boundary NOT_RUN\n";
}
