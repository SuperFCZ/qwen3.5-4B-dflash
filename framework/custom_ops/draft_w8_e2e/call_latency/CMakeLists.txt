cmake_minimum_required(VERSION 3.16)
project(draft_call_latency LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(runtime "${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime/cpp")
add_executable(draft_call_latency main.cpp capture.cpp
  "${runtime}/src/acl_chunk.cpp" "${runtime}/src/chunk.cpp"
  "${runtime}/src/generation.cpp" "${runtime}/src/sha256.cpp")
target_include_directories(draft_call_latency PRIVATE "${runtime}/include")
target_compile_options(draft_call_latency PRIVATE -Wall -Wextra -Wpedantic)
# Only the measured copy of acl_chunk.cpp routes through these six forwarding
# calls. The stock runners and the CANN implementation are linked as usual.
set_source_files_properties("${runtime}/src/acl_chunk.cpp" PROPERTIES COMPILE_DEFINITIONS
  "aclrtMemcpyAsync=ObserveCopy;aclmdlExecuteAsync=ObserveExecute;aclrtSynchronizeStream=ObserveSync;aclrtMalloc=ObserveMalloc;aclmdlLoadFromFileWithMem=ObserveLoad;aclmdlUnload=ObserveUnload")
if(PROBE_FAKE_ACL)
  target_sources(draft_call_latency PRIVATE "${runtime}/tests/fake_acl/fake_acl.cpp")
  target_include_directories(draft_call_latency PRIVATE "${runtime}/tests/fake_acl")
  target_compile_definitions(draft_call_latency PRIVATE PROBE_FAKE_ACL=1)
else()
  find_path(acl_include acl/acl.h HINTS "${CANN_ROOT}" PATH_SUFFIXES include REQUIRED)
  find_library(acl_lib ascendcl HINTS "${CANN_ROOT}" PATH_SUFFIXES lib64 aarch64-linux/lib64 REQUIRED)
  target_include_directories(draft_call_latency PRIVATE "${acl_include}")
  target_link_libraries(draft_call_latency PRIVATE "${acl_lib}")
endif()
