#include "d_flash_w8_a8_two_group_tiling.h"
#include "host_common.h"
namespace optiling {
static ge::graphStatus Tiling(gert::TilingContext *c) { return S3B3Tiling<DFlashW8A8TwoGroupTilingData>(c); }
}
namespace ge {
static graphStatus Infer(gert::InferShapeContext *c) { return S3B3Shape(c); }
static graphStatus Dtype(gert::InferDataTypeContext *c) { c->SetOutputDataType(0, DT_FLOAT16);return GRAPH_SUCCESS; }
}
namespace ops {
class DFlashW8A8TwoGroup: public OpDef {
public:
 explicit DFlashW8A8TwoGroup(const char *n):OpDef(n) {
  Input("a").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND});
  Input("b").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND});
  Input("scale").ParamType(REQUIRED).DataType({ge::DT_UINT64}).Format({ge::FORMAT_ND});
  Output("o").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
  Attr("output_mode").AttrType(OPTIONAL).Int(0);
  SetInferShape(ge::Infer).SetInferDataType(ge::Dtype);
  AICore().SetTiling(optiling::Tiling).AddConfig("ascend310p");
 }
};
OP_ADD(DFlashW8A8TwoGroup);
}
