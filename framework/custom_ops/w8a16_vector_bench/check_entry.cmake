# Launch the generated AiCore payload with a persistent overflow buffer instead of
# the default wrapper's malloc/free per launch. Check the ABI we actually compiled.
file(READ "${GENERATED_DIR}/auto_gen_vector_kernel.cpp" entry)
file(READ "${GENERATED_DIR}/host_stub.cpp" stub)
string(REPLACE "__attribute__((cce_global))" "" entry "${entry}")
string(REGEX REPLACE "[ \t\r\n]+" "" entry "${entry}")
string(REGEX REPLACE "[ \t\r\n]+" "" stub "${stub}")
string(FIND "${entry}" "auto_gen_dflash_vector_probe_kernel(uint8_t*codes,uint8_t*scales,uint8_t*output,uint32_tmode,uint32_tn,uint32_tkTiles,uint32_titerations,GM_ADDRoverflow_status)" abi)
string(FIND "${stub}" ".aic_file_len,2,&g_kernel_handle_aic)" ai_core)
string(FIND "${stub}" "launch_and_profiling_dflash_vector_probe(0,numBlocks,stream," entry_zero)
if(abi EQUAL -1 OR ai_core EQUAL -1 OR entry_zero EQUAL -1 OR NOT EXISTS "${IMAGE}")
    message(FATAL_ERROR "Unsupported SDK direct-launch ABI. Expected the CANN 9 Ascend310P3 AiCore entry 0 and overflow argument. See ${GENERATED_DIR}.")
endif()
