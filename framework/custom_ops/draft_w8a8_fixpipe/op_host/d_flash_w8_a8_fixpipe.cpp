#include "d_flash_w8_a8_fixpipe_tiling.h"
#include "host_common.h"
namespace optiling {
static ge::graphStatus Tiling(gert::TilingContext *c) { return W8Tiling<DFlashW8A8FixpipeTilingData>(c, false); }
}
namespace ge {
static graphStatus Infer(gert::InferShapeContext *c) { return W8Shape(c); }
static graphStatus Dtype(gert::InferDataTypeContext *c) { c->SetOutputDataType(0, DT_FLOAT16);return GRAPH_SUCCESS; }
}
namespace ops {
class DFlashW8A8Fixpipe: public OpDef {
public:
 explicit DFlashW8A8Fixpipe(const char *n):OpDef(n) {
  Input("a").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND});
  Input("b").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({W8A8_B_NZ ? ge::FORMAT_FRACTAL_NZ : ge::FORMAT_ND});
  Input("scale").ParamType(REQUIRED).DataType({ge::DT_UINT64}).Format({ge::FORMAT_ND});
  Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
  SetInferShape(ge::Infer).SetInferDataType(ge::Dtype);
  AICore().SetTiling(optiling::Tiling).AddConfig("ascend310p");
 }
};
OP_ADD(DFlashW8A8Fixpipe);
}
