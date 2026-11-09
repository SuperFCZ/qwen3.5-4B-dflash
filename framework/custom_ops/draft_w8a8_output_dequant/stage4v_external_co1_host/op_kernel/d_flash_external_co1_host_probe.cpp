// COMPILE_ONLY registration placeholder. No calculation, memory access or executable test route.
#include "kernel_operator.h"
extern "C" __global__ __aicore__ void d_flash_external_co1_host_probe(
    GM_ADDR a,GM_ADDR b,GM_ADDR scale,GM_ADDR o,GM_ADDR workspace,GM_ADDR tiling) {
    (void)a;(void)b;(void)scale;(void)o;(void)workspace;(void)tiling;
}
