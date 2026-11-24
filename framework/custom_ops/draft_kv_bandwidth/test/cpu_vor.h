#pragma once
#include "kernel_operator.h"

namespace AscendC {
inline void Or(Tensor<uint16_t> dst, Tensor<uint16_t> a, Tensor<uint16_t> b, uint64_t mask,
               uint8_t repeat, const BinaryRepeatParams& p)
{
    assert(mask==128 && (repeat==32 || repeat==80));
    for (unsigned r=0; r<repeat; ++r) {
        for (unsigned block=0; block<8; ++block) {
            const auto d=(r*p.dstRepStride+block*p.dstBlkStride)*16;
            const auto i=(r*p.src0RepStride+block*p.src0BlkStride)*16;
            const auto j=(r*p.src1RepStride+block*p.src1BlkStride)*16;
            assert(d+16<=dst.count && i+16<=a.count && j+16<=b.count);
            VectorAccess(Bytes(dst.data+d,32),Bytes(a.data+i,32));
            CheckReadable(Bytes(b.data+j,32),false);
            for(unsigned l=0;l<16;++l) dst.data[d+l]=a.data[i+l]|b.data[j+l];
        }
    }
}
}
