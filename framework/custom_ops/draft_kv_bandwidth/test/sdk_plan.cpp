#ifdef DFLASH_KV_WIDTH
#include "kv_wide_spec.h"
#else
#include "kv_spec.h"
#endif
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include <iostream>

int main() {
    using namespace kv_panel;
    using namespace matmul_tiling;
    using Mem = platform_ascendc::CoreMemType;
    auto* p = platform_ascendc::PlatformAscendCManager::GetInstance("Ascend310P3");
    if (!p) return 1;
    uint64_t l1 = 0, ub = 0;
    p->GetCoreMemSize(Mem::L1, l1);
    p->GetCoreMemSize(Mem::UB, ub);
    for (unsigned m : {32u, 80u}) {
        MatmulApiTiling mm(*p);
        mm.SetAType(TPosition::A1, CubeFormat::NZ, DataType::DT_FLOAT16);
        mm.SetBType(TPosition::B1, CubeFormat::NZ, DataType::DT_FLOAT16, false);
        mm.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
        mm.SetBiasType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
        mm.SetBias(false);
        mm.SetShape(m, Columns, PanelK);
        mm.SetOrgShape(m, N, PanelK, PanelK);
        mm.SetFixSplit(m, Columns, BaseK(m));
        if (ub <= UserBytes || l1 <= ResidentL1(m)) return 2;
        mm.SetBufferSpace(l1 - ResidentL1(m), -1, ub - UserBytes);
        optiling::TCubeTiling t;
        if (mm.GetTiling(t) != 0) return 3;
        std::cout << "M " << m << " base " << t.get_baseM() << "/" << t.get_baseN() << "/" << t.get_baseK()
                  << " steps " << t.get_stepKa() << "/" << t.get_stepKb() << " depths "
                  << t.get_depthA1() << "/" << t.get_depthB1() << " DB " << t.get_dbL0A() << "/"
                  << t.get_dbL0B() << "/" << t.get_dbL0C() << " trans " << t.get_transLength() << '\n';
        if (t.get_baseK()!=BaseK(m) || t.get_baseM()!=m || t.get_baseN()!=Columns ||
            t.get_singleCoreK()!=PanelK || t.get_transLength()<=0 || t.get_transLength()>ub-UserBytes) return 4;
    }
}
