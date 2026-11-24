#pragma once
#include <cstdint>

namespace q_panel {
constexpr uint32_t M = 16, K = 2560, N = 4096, Columns = 64, Group = 128;
constexpr uint32_t PanelK = K, Panels = 1, Tiles = N / Columns, Groups = K / Group;
constexpr bool FullL1 = true;
constexpr bool Lookahead = DFLASH_Q_RAW_LOOKAHEAD != 0;
static_assert(DFLASH_Q_RAW_LOOKAHEAD == 0 || DFLASH_Q_RAW_LOOKAHEAD == 1, "unknown Q schedule");
constexpr uint32_t RawBanks = Lookahead ? 2 : 1;
constexpr uint32_t W16Banks = 2, Stripe = K * 16;
constexpr uint32_t CodeBytes = RawBanks * Stripe;
constexpr uint32_t ScaleBytes = Groups * Columns * 2;
constexpr uint32_t WeightBytes = W16Banks * Stripe * 2;
constexpr uint32_t ResidentL1 = (M + Columns) * K * 2;
constexpr uint32_t UserBytes = CodeBytes + ScaleBytes + WeightBytes + 8192;
static_assert(Stripe == Groups * Group * 16 && Tiles == 64, "incomplete Q tile");
}  // namespace q_panel
