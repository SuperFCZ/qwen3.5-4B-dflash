// GM order [304 N tiles, 10 K tiles, 64 channels, 256 K lanes].
__aicore__ inline void CopyGateTileNd(AscendC::LocalTensor<int8_t> dst, AscendC::GlobalTensor<int8_t> src,
                                     uint32_t nBegin, uint32_t kBegin)
{
    constexpr uint32_t tileK = 256;
    constexpr uint32_t kTiles = 2560 / tileK;
    const uint32_t tile = (nBegin / kTileN) * kTiles + kBegin / tileK;
    AscendC::DataCopy(dst, src[tile * kTileN * tileK], kTileN * tileK);
}

__aicore__ inline void CastGateTileNd(AscendC::LocalTensor<half> dst, AscendC::LocalTensor<int8_t> src)
{
    // Same B16 [64,256] presented to the Group128 Mul and SetTensorB.
    const AscendC::UnaryRepeatParams dense{1, 1, 8, 4};
    AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, uint64_t(128), uint8_t(128), dense);
}
