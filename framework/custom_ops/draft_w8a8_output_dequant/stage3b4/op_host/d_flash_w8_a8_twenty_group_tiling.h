#pragma once
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(DFlashW8A8TwentyGroupTilingData)
TILING_DATA_FIELD_DEF(uint64_t, localMemSize);
TILING_DATA_FIELD_DEF(uint32_t, outputMode);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(DFlashW8A8TwentyGroup, DFlashW8A8TwentyGroupTilingData)
}
