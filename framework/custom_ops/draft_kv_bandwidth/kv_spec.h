#pragma once
#include <cstdint>

namespace kv_panel {
constexpr uint32_t K = 2560, N = 2048, Columns = 64, Group = 128;
constexpr uint32_t Tiles = N / Columns, PanelK = K, Panels = 1, Groups = K / Group;
constexpr uint32_t Stripe = K * 16, RawBanks = 1, W16Banks = 2;
constexpr uint32_t CodeBytes = Stripe, ScaleBytes = 2 * Groups * Columns;
constexpr uint32_t WeightBytes = 2 * W16Banks * Stripe;
constexpr uint32_t UserBytes = CodeBytes + ScaleBytes + WeightBytes + 8192;
constexpr uint32_t StagingRows = WeightBytes / (2 * K);
constexpr uint32_t BaseK(uint32_t m) { return m == 32 ? 256 : 160; }
constexpr uint32_t ResidentL1(uint32_t m) { return (m + Columns) * K * 2; }
static_assert(Tiles == 32 && StagingRows == 32 && UserBytes == 215552, "KV buffer geometry");
} // namespace kv_panel
