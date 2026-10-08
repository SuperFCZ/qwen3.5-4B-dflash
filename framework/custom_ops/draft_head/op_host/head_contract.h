#pragma once
#include <cstdint>

#ifdef DFLASH_HEAD_DEVICE
#define DFLASH_HEAD_INLINE __aicore__ inline
#else
#define DFLASH_HEAD_INLINE inline
#endif

namespace head_contract {
constexpr uint32_t kK=2560, kN=248320, kTileN=64, kPaddedM=16, kMaxCores=64;
constexpr uint32_t kTiles=kN/kTileN;
DFLASH_HEAD_INLINE bool Rows(uint32_t m) { return m>=1 && m<=15; }
DFLASH_HEAD_INLINE uint32_t Partitions(uint32_t available, uint32_t cap=0) {
    uint32_t p=available<kMaxCores?available:kMaxCores;
    if(cap && cap<p) p=cap;
    return p;
}
// Logical partial_max[P,M] and partial_id[P,M], physical row stride 16.
DFLASH_HEAD_INLINE uint64_t MaxOffset(uint32_t) { return 0; }
DFLASH_HEAD_INLINE uint64_t IdOffset(uint32_t p) { return uint64_t(p)*32; }
DFLASH_HEAD_INLINE uint64_t SyncOffset(uint32_t p) { return uint64_t(p)*160; }
DFLASH_HEAD_INLINE uint64_t WorkspaceBytes(uint32_t p) { return uint64_t(p)*192; }
DFLASH_HEAD_INLINE uint64_t UserUbBytes(uint32_t p) { return 16*2560*2 + 16*64*2 + 32+128 + p*32; }
DFLASH_HEAD_INLINE bool IsNan(uint16_t b) { return (b&0x7c00)==0x7c00 && (b&0x03ff)!=0; }
DFLASH_HEAD_INLINE uint32_t Rank(uint16_t b) { return b&0x8000 ? 0x8000-(b&0x7fff) : 0x8000+(b&0x7fff); }
// Compare the already-rounded FP16 bit pattern. ±0 tie; first NaN wins.
// Native OM characterization is a hard device gate, including NaNs/infinities.
DFLASH_HEAD_INLINE bool Better(uint16_t value, int64_t id, uint16_t best, int64_t bestId) {
    if(bestId<0) return true;
    const bool a=IsNan(value), b=IsNan(best);
    if(a!=b) return a;
    if(a || Rank(value)==Rank(best)) return id<bestId;
    return Rank(value)>Rank(best);
}
}
