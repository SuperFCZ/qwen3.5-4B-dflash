#pragma once
#include "kernel_operator.h"
// Only six register tests, never a per-logit scalar scan. A zero mask is invalid.
__aicore__ inline uint32_t HeadFirstBit(uint64_t mask) {
    if(!mask) return 64;
    uint32_t index=0;
    if(!(mask&0xffffffffULL)) { mask>>=32; index+=32; }
    if(!(mask&0xffffULL)) { mask>>=16; index+=16; }
    if(!(mask&0xffULL)) { mask>>=8; index+=8; }
    if(!(mask&0xfULL)) { mask>>=4; index+=4; }
    if(!(mask&3ULL)) { mask>>=2; index+=2; }
    return index+((mask&1ULL)?0:1);
}
__aicore__ inline void HeadVectorToScalar(AscendC::TPipe& pipe) {
    const auto ready=pipe.FetchEventID(AscendC::HardEvent::V_S);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(ready);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(ready);
}
// Input is an EXACT widening of the already-rounded FP16 logits, not FP32
// Matmul sums. scratch is 64 B in the unused padded row 15 of c32 (M<=15).
__aicore__ inline uint32_t HeadVectorWinner(const AscendC::LocalTensor<float>& row,
    const AscendC::LocalTensor<uint8_t>& scratch, AscendC::TPipe& pipe) {
    using namespace AscendC;
    auto mask=scratch; auto maximum=scratch[32].ReinterpretCast<float>();
    Compare(mask,row,row,CMPMODE::NE,64); // NaNs first, independently of Reduce's NaN behavior
    HeadVectorToScalar(pipe);
    const uint64_t nanMask=mask.ReinterpretCast<uint64_t>().GetValue(0);
    if(nanMask) return HeadFirstBit(nanMask);
    WholeReduceMax(maximum,row,64,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
    HeadVectorToScalar(pipe);
    const float value=maximum.GetValue(0); // ignore the SDK's tie index
    CompareScalar(mask,row,value,CMPMODE::EQ,64);
    HeadVectorToScalar(pipe);
    return HeadFirstBit(mask.ReinterpretCast<uint64_t>().GetValue(0));
}
