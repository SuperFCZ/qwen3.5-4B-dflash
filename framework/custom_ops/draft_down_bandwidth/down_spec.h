#pragma once
#include <cstdint>

namespace down_panel {
constexpr uint32_t M = 16, K = 9728, N = 2560, Columns = 64, Group = 128;
constexpr uint32_t PanelK = 4864, ChunkK = 2432, ChunkN = 16;
constexpr uint32_t Tiles = N / Columns, Panels = K / PanelK, Groups = PanelK / Group;
constexpr uint32_t ChunksPerStripe = PanelK / ChunkK, Pieces = (Columns / ChunkN) * ChunksPerStripe;
constexpr uint32_t Elements = ChunkK * ChunkN, W16Banks = DFLASH_DOWN_W16_BANKS;
static_assert(K == 2 * PanelK && PanelK % 256 == 0 && PanelK % ChunkK == 0 &&
              ChunkK % Group == 0 && N % Columns == 0, "unaligned Down panels");
static_assert(W16Banks == 1 || W16Banks == 2, "unsupported W16 buffer count");
constexpr uint32_t RawBanks = 1;
constexpr uint32_t CodeBytes = RawBanks * Elements, ScaleBytes = Groups * Columns * 2;
constexpr uint32_t WeightBytes = W16Banks * Elements * 2;
constexpr uint32_t UserBytes = CodeBytes + ScaleBytes + WeightBytes + 8192;
constexpr uint32_t ResidentL1 = 2 * (M * K + Columns * PanelK);
} // namespace down_panel
