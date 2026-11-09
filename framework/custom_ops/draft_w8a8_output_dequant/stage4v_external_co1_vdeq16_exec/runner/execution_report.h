#pragma once
#include <cstdint>
#include <ostream>
#include <stdexcept>
// These exact producers are compiled by both the real Runner and CPU contract tests.
namespace echo_execution_report {
inline void Host(std::ostream& out,uint64_t bytes,int device){
 out<<"{\"status\":\"HOST_TILING_ONLY\",\"npu\":\"NOT_RUN\",\"workspace_bytes\":"<<bytes<<",\"device\":"<<device<<"}\n";
 if(!out)throw std::runtime_error("Host execution report write failed");
}
inline void Echo(std::ostream& out,uint64_t bytes,int device){
 out<<"{\"status\":\"ECHO_EXECUTION_MATCH\",\"npu\":\"COMPLETED\",\"device\":"<<device<<",\"workspace_bytes\":"<<bytes<<",\"repetitions\":2,\"guards\":true,\"readonly\":true,\"poison\":true,\"repeat_equal\":true,\"l0c_guard\":\"RANGE_ONLY_NO_CANARIES\",\"vdeq16\":\"NOT_IMPLEMENTED_PENDING_ECHO_PASS\",\"cpu_fallback\":false}\n";
 if(!out)throw std::runtime_error("Echo execution report write failed");
}
}
