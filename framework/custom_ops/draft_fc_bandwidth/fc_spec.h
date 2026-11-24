#pragma once
#include <cstdint>

namespace fc_panel {
constexpr uint32_t K = 12800, N = 2560, Columns = 64, Group = 128, Tiles = N / Columns;
constexpr uint32_t PanelK = DFLASH_FC_PANEL_K, Panels = K / PanelK;
constexpr bool StreamA = PanelK == 6400;
constexpr uint32_t ChunkK = StreamA ? 1280 : 2560;
constexpr uint32_t Parts = PanelK / ChunkK, Groups = PanelK / Group, GroupsPerPart = ChunkK / Group;
constexpr uint32_t Elements = ChunkK * 16, Pieces = (Columns / 16) * Parts;
constexpr uint32_t RawBanks = 1;
constexpr uint32_t CodeBytes = Elements, ScaleBytes = 2 * Groups * Columns;
constexpr uint32_t WeightBytes = 2 * 2 * Elements, UserBytes = CodeBytes + ScaleBytes + WeightBytes + 8192;
static_assert((PanelK == 2560 || PanelK == 6400) && K % PanelK == 0 && PanelK % ChunkK == 0 &&
              ChunkK % Group == 0 && Pieces * Elements == PanelK * Columns, "incomplete FC panel");

template<uint32_t M>
struct Activation {
    static_assert(M == 16 || M == 64, "unsupported FC gear");
    static constexpr bool Whole = !StreamA && M == 16;
    static constexpr uint32_t LocalK = StreamA ? 0 : (Whole ? K : PanelK);
};
constexpr uint32_t ResidentL1(uint32_t m) {
    return 2 * (Columns * PanelK + m * (m == 16 ? Activation<16>::LocalK : Activation<64>::LocalK));
}
} // namespace fc_panel
