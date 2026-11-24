#pragma once
#include <cstdint>

namespace kv_panel {
constexpr uint32_t K = 2560, N = 2048, Columns = DFLASH_KV_WIDTH, Group = 128;
constexpr uint32_t PanelK = 1280, Panels = K / PanelK, Tiles = N / Columns;
constexpr uint32_t Groups = PanelK / Group, GroupsPerPart = Groups, Parts = 1;
constexpr uint32_t Pieces = Columns / 16, Elements = 16 * PanelK, RawBanks = 1;
constexpr uint32_t CodeBytes = Elements, ScaleBytes = Groups * Columns * 2;
constexpr uint32_t WeightBytes = 4 * Elements, UserBytes = CodeBytes + ScaleBytes + WeightBytes + 8192;
constexpr bool StreamA = false;
template<uint32_t M> struct Activation {
    // N128 revisits A for a second group of columns on this core.
    // Each N256 block owns one complete column group and consumes A exactly once.
    static constexpr bool Whole = Columns == 128;
    static constexpr uint32_t LocalK = Whole ? K : PanelK;
};
constexpr uint32_t BaseK(uint32_t) { return Columns == 128 ? 128 : 64; }
constexpr uint32_t ResidentL1(uint32_t m) { return 2 * (m * Activation<32>::LocalK + Columns * PanelK); }
static_assert((Columns == 128 || Columns == 256) && N % Columns == 0 && Panels == 2,
              "two full K panels and disjoint output tiles");
} // namespace kv_panel
