#pragma once
#include <cstdint>

// The private build prepends DFLASH_PANEL_K to both copies of this header.
namespace gate_panel {
constexpr uint32_t M = 16, K = 2560, N = 19456, Columns = 64, Group = 128;
constexpr uint32_t PanelK = DFLASH_PANEL_K, Panels = K / PanelK, Tiles = N / Columns;
static_assert(PanelK == 256 || PanelK == 512 || PanelK == 1280, "unsupported Gate panel");
static_assert(K % PanelK == 0 && N % Columns == 0, "only complete panels");
constexpr uint32_t Groups = PanelK / Group;
constexpr bool SplitRaw = PanelK == 1280;
constexpr uint32_t RawBanks = SplitRaw ? 1 : 2;
constexpr uint32_t RawElements = (SplitRaw ? Columns / 2 : Columns) * PanelK;
constexpr uint32_t CodeBytes = RawBanks * RawElements;
constexpr uint32_t ScaleBytes = RawBanks * Groups * Columns * 2;
constexpr uint32_t WeightBytes = Columns * PanelK * 2;
constexpr uint32_t ReservedBytes = 8192; // Same unallocated SDK allowance as the B16 control.
constexpr uint32_t UserBytes = CodeBytes + ScaleBytes + WeightBytes + ReservedBytes;
}  // namespace gate_panel
