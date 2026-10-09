// Exercise the real shared guard implementation with CPU memory standing in
// for ACL allocations. User confirmed M1's original write through byte 31 on
// NPU; this model tests the fix but cannot establish its NPU acceptance.
#include <cstddef>
extern "C" const char* aclGetRecentErrMsg();
extern "C" int aclrtMemset(void*,std::size_t,int,std::size_t);
#include "../../draft_quant/test/runner_common.h"
#include "output_buffer.h"
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
size_t lastCopyBytes=0;
extern "C" aclError aclrtMemcpy(void* dst,size_t capacity,const void* src,size_t count,aclrtMemcpyKind) {
    lastCopyBytes=count;
    assert(count<=capacity); std::memcpy(dst,src,count); return ACL_SUCCESS;
}
using namespace draft_quant_test;

int main(int argc,char** argv) {
    assert(argc==2);
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
    // Exercise the exact helpers used by the runner, including a model that
    // reports more storage than the logical tensor. Tail padding may change;
    // the logical IDs and physical end guards remain independent gates.
    const size_t expectedPhysical[15]={32,32,32,32,64,64,64,64,96,96,96,96,128,128,128};
    for(size_t m=1;m<=15;++m) {
        const size_t logical=m*8;
        assert(draft_head_test::NativeTop1PayloadBytes(logical,logical)==expectedPhysical[m-1]);
        for(size_t reported:{logical,logical+7,logical+32}) {
            const size_t physical=draft_head_test::NativeTop1PayloadBytes(logical,reported);
            assert(physical>=reported && physical>=logical && physical%32==0 && physical-reported<32);
            buffer.Allocate(physical);
            std::vector<uint8_t> expected(logical);
            for(size_t row=0;row<m;++row) {
                const int64_t id=100+row;
                std::memcpy(expected.data()+row*8,&id,8);
            }
            for(int padding:{0x7f,0xff}) {
                std::memset(buffer.data,padding,physical);
                std::memcpy(buffer.data,expected.data(),logical);
                buffer.CheckGuards("y");
                const auto actual=draft_head_test::ReadLogicalOutput(buffer,logical);
                assert(lastCopyBytes==logical && actual==expected && actual.size()==m*8);
                WriteBytes(std::string(argv[1])+"/m"+std::to_string(m)+".bin",actual);
            }
            // A changed logical ID still fails equality; it is never padding.
            static_cast<uint8_t*>(buffer.data)[logical-1]^=1;
            assert(draft_head_test::ReadLogicalOutput(buffer,logical)!=expected);
            // Even ONE byte beyond the padded allocation must fail the guard.
            static_cast<uint8_t*>(buffer.data)[physical]=0;
            bool rejected=false;
            try { buffer.CheckGuards("y"); } catch(const std::runtime_error&) { rejected=true; }
            assert(rejected);
            rejected=false;
            try { draft_head_test::ReadLogicalOutput(buffer,physical+1); }
            catch(const std::runtime_error&) { rejected=true; }
            assert(rejected);
        }
    }
    bool overflowRejected=false;
    try { draft_head_test::NativeTop1PayloadBytes(8,std::numeric_limits<size_t>::max()); }
    catch(const std::runtime_error&) { overflowRejected=true; }
    assert(overflowRejected);
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
    std::cout<<"Guard diagnostics and output bounds CPU model PASS; fixed runner NPU acceptance NOT_RUN\n";
}
