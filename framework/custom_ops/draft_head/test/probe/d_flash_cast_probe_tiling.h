#pragma once
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(DFlashCastProbeTilingData)
TILING_DATA_FIELD_DEF(uint32_t, count);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(DFlashCastProbe,DFlashCastProbeTilingData)
}
