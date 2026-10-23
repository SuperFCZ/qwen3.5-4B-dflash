#define W8A8_RAW_OUTPUT 0
#include "kernel_common.h"
extern "C" __global__ __aicore__ void d_flash_w8_a8_fixpipe(GM_ADDR a,GM_ADDR b,GM_ADDR scale,GM_ADDR y,GM_ADDR workspace,GM_ADDR tiling) {
 GET_TILING_DATA(td,tiling);
 W8Process(a,b,scale,y,td);
}
