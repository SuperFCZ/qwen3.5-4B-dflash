#pragma once
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TargetGateUpPanelTiling)
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cube);
    TILING_DATA_FIELD_DEF(uint32_t, rows);
    TILING_DATA_FIELD_DEF(uint32_t, tileN);
    TILING_DATA_FIELD_DEF(uint32_t, cores);
    TILING_DATA_FIELD_DEF(uint32_t, packedA);
    TILING_DATA_FIELD_DEF(uint32_t, prefetchB);
    TILING_DATA_FIELD_DEF(uint32_t, cubeDequant);
    TILING_DATA_FIELD_DEF(uint32_t, localBytes);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(TargetGateUpPanel, TargetGateUpPanelTiling)
}
