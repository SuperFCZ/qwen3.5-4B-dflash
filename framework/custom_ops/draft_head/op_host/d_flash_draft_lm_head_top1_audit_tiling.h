#pragma once
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(DFlashDraftLmHeadTop1AuditTilingData)
TILING_DATA_FIELD_DEF(uint32_t, m);
TILING_DATA_FIELD_DEF(uint32_t, partitions);
TILING_DATA_FIELD_DEF(uint64_t, matmulUbBytes);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(DFlashDraftLmHeadTop1Audit, DFlashDraftLmHeadTop1AuditTilingData)
}
