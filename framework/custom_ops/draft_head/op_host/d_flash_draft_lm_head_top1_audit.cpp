#include "d_flash_draft_lm_head_top1_audit_tiling.h"
#include "head_host.h"
namespace ops {
class DFlashDraftLmHeadTop1Audit : public OpDef {
public:
    explicit DFlashDraftLmHeadTop1Audit(const char *name) : OpDef(name) {
        this->Input("hidden").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("token_id").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Output("logits").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->SetInferShape(head_host::InferShape<true>).SetInferDataType(head_host::InferType<true>);
        this->AICore().SetTiling(head_host::Tiling<optiling::DFlashDraftLmHeadTop1AuditTilingData, true>).AddConfig("ascend310p");
    }
};
OP_ADD(DFlashDraftLmHeadTop1Audit);
}
