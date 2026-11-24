#pragma once
#include <cstdint>

namespace o_panel {
constexpr uint32_t M = 16, K = 4096, N = 2560, Columns = 64, Group = 128;
constexpr uint32_t PanelK = K, Panels = 1, Tiles = N / Columns, Groups = K / Group;
constexpr uint32_t ChunkK = 2048, Chunks = K / ChunkK, GroupsPerChunk = ChunkK / Group;
constexpr uint32_t Stripe = ChunkK * 16, Pieces = (Columns / 16) * Chunks;
constexpr bool FullL1 = true;
constexpr uint32_t RawBanks = 1, W16Banks = 2;
constexpr uint32_t CodeBytes = Stripe, ScaleBytes = Groups * Columns * 2;
constexpr uint32_t WeightBytes = W16Banks * Stripe * 2;
constexpr uint32_t ResidentL1 = (M + Columns) * K * 2;
constexpr uint32_t UserBytes = CodeBytes + ScaleBytes + WeightBytes + 8192;
static_assert(K % ChunkK == 0 && ChunkK % Group == 0 && N % Columns == 0, "complete Group128 and N64");
static_assert(Pieces * Stripe == Columns * K && UserBytes == 176128, "O panel allocation");
} // namespace o_panel
