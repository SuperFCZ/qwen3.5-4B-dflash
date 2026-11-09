#include "d_flash_full_k_int32_probe_tiling.h"
#include "host_common.h"
namespace optiling {
static ge::graphStatus Tiling(gert::TilingContext *c) { return S4RAWTiling<DFlashFullKInt32ProbeTilingData>(c); }
}
namespace ge {
static graphStatus Infer(gert::InferShapeContext *c) { return S4RAWShape(c); }
static graphStatus Dtype(gert::InferDataTypeContext *c) { c->SetOutputDataType(0, DT_INT32);return GRAPH_SUCCESS; }
}
namespace ops {
class DFlashFullKInt32Probe: public OpDef {
public:
 explicit DFlashFullKInt32Probe(const char *n):OpDef(n) {
  Input("a").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND});
  Input("b").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND});
  Output("o").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
  SetInferShape(ge::Infer).SetInferDataType(ge::Dtype);
  AICore().SetTiling(optiling::Tiling).AddConfig("ascend310p");
 }
};
OP_ADD(DFlashFullKInt32Probe);
}
