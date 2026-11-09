#include "d_flash_external_co1_echo_probe_tiling.h"
#include "host_common.h"
namespace optiling {
static ge::graphStatus Tiling(gert::TilingContext *c) { return S4CO1Tiling<DFlashExternalCo1EchoProbeTilingData>(c); }
}
namespace ge {
static graphStatus Infer(gert::InferShapeContext *c) { return S4CO1Shape(c); }
static graphStatus Dtype(gert::InferDataTypeContext *c) { c->SetOutputDataType(0, DT_INT32);c->SetOutputDataType(1, DT_UINT64);return GRAPH_SUCCESS; }
}
namespace ops {
class DFlashExternalCo1EchoProbe: public OpDef {
public:
 explicit DFlashExternalCo1EchoProbe(const char *n):OpDef(n) {
  Input("a").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND});
  Input("b").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND});
  Input("scale").ParamType(REQUIRED).DataType({ge::DT_UINT64}).Format({ge::FORMAT_ND});
  Output("o").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
  Output("witness").ParamType(REQUIRED).DataType({ge::DT_UINT64}).Format({ge::FORMAT_ND});
  SetInferShape(ge::Infer).SetInferDataType(ge::Dtype);
  AICore().SetTiling(optiling::Tiling).AddConfig("ascend310p");
 }
};
OP_ADD(DFlashExternalCo1EchoProbe);
}
