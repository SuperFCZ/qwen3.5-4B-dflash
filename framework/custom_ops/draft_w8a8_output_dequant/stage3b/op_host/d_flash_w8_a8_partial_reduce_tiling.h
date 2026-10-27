#pragma once
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(DFlashW8A8PartialReduceTilingData)
TILING_DATA_FIELD_DEF(uint32_t, groupCount);
TILING_DATA_FIELD_DEF(uint32_t, elements);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(DFlashW8A8PartialReduce, DFlashW8A8PartialReduceTilingData)
}
