#include "vector_probe.h"
extern "C" __global__ __aicore__ void d_flash_cast_probe(GM_ADDR values,GM_ADDR vector_fp32,GM_ADDR vector_fp16,
    GM_ADDR workspace,GM_ADDR tiling) { RunVectorProbe(values,vector_fp32,vector_fp16); }
