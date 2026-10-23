#include "d_flash_w8_a8_accumulator_tiling.h"
#include "host_common.h"
namespace optiling {
static ge::graphStatus Tiling(gert::TilingContext *c) { return W8Tiling<DFlashW8A8AccumulatorTilingData>(c, true); }
}
namespace ge {
static graphStatus Infer(gert::InferShapeContext *c) { return W8Shape(c); }
static graphStatus Dtype(gert::InferDataTypeContext *c) { c->SetOutputDataType(0, DT_INT32);return GRAPH_SUCCESS; }
}
namespace ops {
class DFlashW8A8Accumulator: public OpDef {
public:
 explicit DFlashW8A8Accumulator(const char *n):OpDef(n) {
  Input("a").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND});
  Input("b").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND});
  Output("y").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
  SetInferShape(ge::Infer).SetInferDataType(ge::Dtype);
  AICore().SetTiling(optiling::Tiling).AddConfig("ascend310p");
 }
};
OP_ADD(DFlashW8A8Accumulator);
}
