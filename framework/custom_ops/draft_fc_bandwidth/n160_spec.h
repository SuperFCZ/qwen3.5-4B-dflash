#pragma once
#include <cstdint>

namespace packed_fc {
constexpr uint32_t MRows=64, K=12800, N=2560, Group=128, Columns=160, BaseK=80;
constexpr bool StreamA=false, Banked=DFLASH_FC_BANKED_L1!=0;
constexpr uint32_t PanelK=Banked?640:1280, ChunkK=PanelK, Panels=K/PanelK, Tiles=N/Columns;
constexpr uint32_t Parts=1, Groups=PanelK/Group, GroupsPerPart=Groups, Pieces=Columns/16, Elements=ChunkK*16;
constexpr uint32_t RawBanks=1, CodeBytes=Banked?Columns*PanelK:Elements, ScaleBytes=2*Groups*Columns;
constexpr uint32_t WeightBytes=4*Elements, UserBytes=CodeBytes+ScaleBytes+WeightBytes+8192;
constexpr uint32_t StepKa=Banked?8:2, StepKb=Banked?8:16;
template<uint32_t M> struct Activation {static constexpr bool Whole=false;static constexpr uint32_t LocalK=PanelK;};
constexpr uint32_t ResidentL1(uint32_t m){return (Banked?2:1)*2*(m+Columns)*PanelK;}
static_assert(N%Columns==0 && K%PanelK==0 && PanelK%Group==0 && PanelK%BaseK==0,"complete N160 panels");
} // namespace packed_fc
