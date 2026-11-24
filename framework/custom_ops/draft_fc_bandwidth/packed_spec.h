#pragma once
#include <cstdint>

namespace packed_fc {
constexpr uint32_t K=12800, N=2560, Group=128, Columns=DFLASH_FC_COLUMNS;
constexpr bool Wide=Columns==320, StreamA=false;
constexpr uint32_t BaseK=Wide?32:256, PanelK=Wide?1280:2560, ChunkK=PanelK;
constexpr uint32_t Tiles=N/Columns, Panels=K/PanelK, Groups=PanelK/Group;
constexpr uint32_t Parts=1, GroupsPerPart=Groups, Pieces=Columns/16, Elements=ChunkK*16;
constexpr uint32_t RawBanks=1, CodeBytes=Elements, ScaleBytes=Groups*Columns*2;
constexpr uint32_t WeightBytes=2*2*Elements, UserBytes=CodeBytes+ScaleBytes+WeightBytes+8192;
template<uint32_t M> struct Activation {static constexpr bool Whole=false; static constexpr uint32_t LocalK=PanelK;};
constexpr uint32_t ResidentL1(uint32_t m){return 2*(m+Columns)*PanelK;}
static_assert((Columns==64||Wide) && K%PanelK==0 && N%Columns==0 && PanelK%BaseK==0, "complete FC panels");
} // namespace packed_fc
