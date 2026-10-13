#include "head_kernel.h"
#include "vector_probe.h"
// Separate diagnostic operator, never registered into production Draft.
extern "C" __global__ __aicore__ void d_flash_head_probe(GM_ADDR hidden,GM_ADDR weight,GM_ADDR values,
    GM_ADDR token_id,GM_ADDR logits,GM_ADDR precast,GM_ADDR vector_fp32,GM_ADDR vector_fp16,
    GM_ADDR workspace,GM_ADDR tiling) {
    GET_TILING_DATA(data,tiling);
    // Execute the actual complete Head body; record tiles 0/1 before conversion.
    RunHead<true,true>(hidden,weight,token_id,logits,workspace,data,precast);
    if(AscendC::GetBlockIdx()!=0) return;
    RunVectorProbe(values,vector_fp32,vector_fp16);
}
