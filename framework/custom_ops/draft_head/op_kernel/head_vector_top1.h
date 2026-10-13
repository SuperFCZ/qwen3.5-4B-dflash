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
// 704 B scratch, all subregions 32-byte aligned and disjoint from raw/logit rows:
// mask[0:32], maximum[32:64], abs/delta s16[64:192], constant u16[192:320],
// numeric half marker[320:448], numeric float marker[448:704].
// Classification is solely from original FP16 exponent/mantissa, NOT NaN arithmetic.
__aicore__ inline void HeadRawNanMask(const AscendC::LocalTensor<half>& raw,
    const AscendC::LocalTensor<uint8_t>& scratch) {
    using namespace AscendC;
    auto delta=scratch[64].ReinterpretCast<uint16_t>();
    auto constant=scratch[192].ReinterpretCast<uint16_t>();
    auto h=scratch[320].ReinterpretCast<half>();
    auto f=scratch[448].ReinterpretCast<float>();
    Duplicate(constant,uint16_t(0x7fff),64); PipeBarrier<PIPE_V>();
    And(delta,raw.ReinterpretCast<uint16_t>(),constant,64); PipeBarrier<PIPE_V>();
    // abs_bits in [0,32767]; delta in [-31744,1023], no int16 overflow.
    Adds(delta.ReinterpretCast<int16_t>(),delta.ReinterpretCast<int16_t>(),int16_t(-0x7c00),64);
    PipeBarrier<PIPE_V>();
    // dav_m200 supports s16->half->float, NOT direct s16->float.
    // These are finite numeric markers; rounding negative markers cannot change their sign.
    Cast(h,delta.ReinterpretCast<int16_t>(),RoundMode::CAST_NONE,64); PipeBarrier<PIPE_V>();
    Cast(f,h,RoundMode::CAST_NONE,64); PipeBarrier<PIPE_V>();
    CompareScalar(scratch,f,0.0f,CMPMODE::GT,64);
}
__aicore__ inline void HeadTraceMask64(AscendC::LocalTensor<uint32_t> trace,uint32_t offset,uint64_t value) {
    trace.SetValue(offset,uint32_t(value)); trace.SetValue(offset+1,uint32_t(value>>32));
}
// row is an exact widening of one ALREADY-ROUNDED FP16 row. Trace is probe-only
// [240] uint32 words; ordinary Fused/Audit do not execute the legacy NaN path.
template<bool Diagnose=false>
__aicore__ inline uint32_t HeadVectorWinner(const AscendC::LocalTensor<half>& raw,
    const AscendC::LocalTensor<float>& row, const AscendC::LocalTensor<uint8_t>& scratch,
    AscendC::TPipe& pipe, AscendC::LocalTensor<uint32_t> trace={}, const AscendC::LocalTensor<float>& legacyRow={}) {
    using namespace AscendC;
    auto mask=scratch; auto maximum=scratch[32].ReinterpretCast<float>();
    if constexpr(Diagnose) {
        DataCopy(trace,legacyRow.ReinterpretCast<uint32_t>(),64);
        DataCopy(trace[176],row.ReinterpretCast<uint32_t>(),64);
        Duplicate(mask.ReinterpretCast<uint32_t>(),uint32_t(0xa5a5a5a5),8); PipeBarrier<PIPE_V>();
        Compare(mask,legacyRow,legacyRow,CMPMODE::NE,64);
        HeadVectorToScalar(pipe);
        const auto legacyNan=mask.ReinterpretCast<uint64_t>().GetValue(0);
        DataCopy(trace[64],mask.ReinterpretCast<uint32_t>(),8); PipeBarrier<PIPE_V>();
        // Deliberately measure even for NaN rows; this is diagnostic only.
        Duplicate(maximum.ReinterpretCast<uint32_t>(),uint32_t(0xa5a5a5a5),8); PipeBarrier<PIPE_V>();
        WholeReduceMax(maximum,legacyRow,64,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
        HeadVectorToScalar(pipe);
        const float value=maximum.GetValue(0);
        const auto maxBits=maximum.ReinterpretCast<uint32_t>().GetValue(0);
        const auto maxIndex=maximum.ReinterpretCast<uint32_t>().GetValue(1);
        Duplicate(mask.ReinterpretCast<uint32_t>(),uint32_t(0xa5a5a5a5),8); PipeBarrier<PIPE_V>();
        CompareScalar(mask,legacyRow,value,CMPMODE::EQ,64);
        HeadVectorToScalar(pipe);
        const auto legacyEq=mask.ReinterpretCast<uint64_t>().GetValue(0);
        DataCopy(trace[72],mask.ReinterpretCast<uint32_t>(),8); PipeBarrier<PIPE_V>();
        HeadTraceMask64(trace,96,legacyNan); HeadTraceMask64(trace,98,legacyEq);
        trace.SetValue(104,maxBits); trace.SetValue(105,maxIndex);
        trace.SetValue(106,HeadFirstBit(legacyNan?legacyNan:legacyEq));
        trace.SetValue(109,uint32_t(0x48543102));
        trace.SetValue(110,HeadFirstBit(legacyNan)); trace.SetValue(111,HeadFirstBit(legacyEq));
        Duplicate(mask.ReinterpretCast<uint32_t>(),uint32_t(0xa5a5a5a5),8); PipeBarrier<PIPE_V>();
    }
    HeadRawNanMask(raw,scratch);
    HeadVectorToScalar(pipe);
    const uint64_t nanMask=mask.ReinterpretCast<uint64_t>().GetValue(0);
    if constexpr(Diagnose) {
        DataCopy(trace[80],mask.ReinterpretCast<uint32_t>(),8);
        DataCopy(trace[112],scratch[448].ReinterpretCast<uint32_t>(),64); PipeBarrier<PIPE_V>();
        HeadTraceMask64(trace,100,nanMask);
        trace.SetValue(107,uint32_t(0));
    }
    if(nanMask) {
        const auto winner=HeadFirstBit(nanMask);
        if constexpr(Diagnose) trace.SetValue(108,winner);
        return winner;
    }
    WholeReduceMax(maximum,row,64,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
    HeadVectorToScalar(pipe);
    const float value=maximum.GetValue(0); // ignore the SDK's tie index
    CompareScalar(mask,row,value,CMPMODE::EQ,64);
    HeadVectorToScalar(pipe);
    const auto equal=mask.ReinterpretCast<uint64_t>().GetValue(0);
    const auto winner=HeadFirstBit(equal);
    if constexpr(Diagnose) {
        DataCopy(trace[88],mask.ReinterpretCast<uint32_t>(),8); PipeBarrier<PIPE_V>();
        HeadTraceMask64(trace,102,equal); trace.SetValue(107,uint32_t(1)); trace.SetValue(108,winner);
    }
    return winner;
}
