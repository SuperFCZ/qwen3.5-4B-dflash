#include "head_kernel.h"
extern "C" __global__ __aicore__ void d_flash_draft_lm_head_top1(GM_ADDR hidden, GM_ADDR weight, GM_ADDR token_id,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(data, tiling);
    RunHead<false>(hidden, weight, token_id, nullptr, workspace, data);
}
