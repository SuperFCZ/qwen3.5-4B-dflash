#ifndef SMOKE_MATMUL_TILING_H
#define SMOKE_MATMUL_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SmokeMatmulTilingData)
TILING_DATA_FIELD_DEF(uint64_t, localMemSize);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SmokeMatmul, SmokeMatmulTilingData)
}  // namespace optiling

#endif  // SMOKE_MATMUL_TILING_H
