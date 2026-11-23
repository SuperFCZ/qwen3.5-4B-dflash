cmake_minimum_required(VERSION 3.16)
project(dflash_vector_bench LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SOC_VERSION Ascend310P3)
set(RUN_MODE npu)
set(ASCEND_KERNEL_LAUNCH_ONLY ON)
include("${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
ascendc_library(vector_ops SHARED vector_kernel.cpp)
ascendc_include_directories(vector_ops PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../draft_gate_scale_broadcast/kernel")
add_executable(vector_bench main.cpp)
add_dependencies(vector_bench vector_ops)
set(probe_image "${CMAKE_CURRENT_BINARY_DIR}/vector_ops_merge_obj_dir/device_aic.o")
target_compile_definitions(vector_bench PRIVATE VECTOR_PROBE_IMAGE="${probe_image}")
target_include_directories(vector_bench PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/include")
target_link_directories(vector_bench PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64")
find_package(Threads REQUIRED)
target_link_libraries(vector_bench PRIVATE ascendc_runtime_static ascendcl runtime ascend_dump
    profapi mmpa unified_dlog c_sec Threads::Threads ${CMAKE_DL_LIBS})
add_custom_command(TARGET vector_bench PRE_LINK COMMAND ${CMAKE_COMMAND}
    "-DGENERATED_DIR=${CMAKE_CURRENT_BINARY_DIR}/auto_gen/vector_ops" "-DIMAGE=${probe_image}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/check_entry.cmake" VERBATIM)
