// A portable, deliberately explicit transpose for testing the COMPLETE generated kernel on CPU.
// The separate tikicpulib probe validates the real SDK API and addresses.
namespace AscendC {
inline unsigned cpuScaleCalls = 0;
struct TransDataTo5HDParams {
    bool dstHighHalf, srcHighHalf;
    uint8_t repeatTimes;
    uint16_t dstRepStride, srcRepStride;
};
inline void SetMaskNorm() {}
inline void ResetMask() {}
template <typename T>
inline void TransDataTo5HD(uint64_t dst[16], uint64_t src[16], const TransDataTo5HDParams& p) {
    static_assert(sizeof(T) == 2);
    assert(!p.dstHighHalf && !p.srcHighHalf && p.repeatTimes==4 && p.dstRepStride==16 && p.srcRepStride==1);
    ++cpuScaleCalls;
    auto* output=reinterpret_cast<T*>(dst[0]);
    const auto* input=reinterpret_cast<const T*>(src[0]);
    VectorAccess(Bytes(output,2048),Bytes(input,128));
    for(unsigned repeat=0;repeat<p.repeatTimes;++repeat)
        for(unsigned column=0;column<16;++column)
            for(unsigned row=0;row<16;++row) {
                assert(src[row]==src[0] && dst[column]==dst[0]+32*column);
                const auto* from=reinterpret_cast<const T*>(src[row])+repeat*p.srcRepStride*16+column;
                auto* to=reinterpret_cast<T*>(dst[column])+repeat*p.dstRepStride*16+row;
                std::memcpy(to,from,2);
            }
}
} // namespace AscendC
