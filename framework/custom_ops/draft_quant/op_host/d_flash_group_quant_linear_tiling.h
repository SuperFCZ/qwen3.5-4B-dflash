#ifndef D_FLASH_GROUP_QUANT_LINEAR_TILING_H
#define D_FLASH_GROUP_QUANT_LINEAR_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DFlashGroupQuantLinearTilingData)
TILING_DATA_FIELD_DEF(uint64_t, matmulUbBytes);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DFlashGroupQuantLinear, DFlashGroupQuantLinearTilingData)
}  // namespace optiling

#endif
