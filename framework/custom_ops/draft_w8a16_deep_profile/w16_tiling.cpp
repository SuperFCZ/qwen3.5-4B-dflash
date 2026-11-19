// Standalone CPU SDK query. No ACL initialization, stream or device execution.
#include <tiling/platform/platform_ascendc.h>
#include <tiling/tiling_api.h>
#include <cstring>
#include <iostream>
#include <vector>
using namespace matmul_tiling;

struct Plan {
    std::vector<uint8_t> bytes;
    int rc, depth_a, depth_b, db_a, db_b, db_c, m, n, k, trans;
};
Plan query(platform_ascendc::PlatformAscendC& platform, int scratch) {
    MatmulApiTiling mm(platform);
    mm.SetAType(TPosition::GM,CubeFormat::ND,DataType::DT_FLOAT16);
    mm.SetBType(TPosition::VECOUT,CubeFormat::ND,DataType::DT_FLOAT16,true);
    mm.SetCType(TPosition::GM,CubeFormat::ND,DataType::DT_FLOAT16);
    mm.SetBiasType(TPosition::GM,CubeFormat::ND,DataType::DT_FLOAT);
    mm.SetBias(false); mm.SetShape(16,64,256); mm.SetOrgShape(16,19456,2560,256);
    mm.SetFixSplit(16,64,256); mm.SetBufferSpace(-1,-1,scratch);
    optiling::TCubeTiling cube; Plan plan{};plan.rc=mm.GetTiling(cube);
    if(plan.rc==0) {
        plan.bytes.resize(cube.GetDataSize());cube.SaveToBuffer(plan.bytes.data(),plan.bytes.size());
        plan.depth_a=cube.get_depthA1();plan.depth_b=cube.get_depthB1();
        plan.db_a=cube.get_dbL0A();plan.db_b=cube.get_dbL0B();plan.db_c=cube.get_dbL0C();
        plan.m=cube.get_baseM();plan.n=cube.get_baseN();plan.k=cube.get_baseK();plan.trans=cube.get_transLength();
    }
    std::cout<<"{\"scratch\":"<<scratch<<",\"get_tiling_rc\":"<<plan.rc<<",\"base\":["<<plan.m<<","<<plan.n<<","<<plan.k
      <<"],\"depth\":["<<plan.depth_a<<","<<plan.depth_b<<"],\"db\":["<<plan.db_a<<","<<plan.db_b<<","<<plan.db_c
      <<"],\"transLength\":"<<plan.trans<<",\"cube_hex\":\"";
    const char* digits="0123456789abcdef";for(auto byte:plan.bytes)std::cout<<digits[byte>>4]<<digits[byte&15];
    std::cout<<"\"}";return plan;
}
int main() {
    auto* platform=platform_ascendc::PlatformAscendCManager::GetInstance("Ascend310P3");
    if(!platform || platform->GetSocVersion()!=platform_ascendc::SocVersion::ASCEND310P || static_cast<unsigned>(platform->GetCurNpuArch())!=2002)return 2;
    uint64_t ub=0,l0b=0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B,l0b);
    std::cout<<"{\"scope\":\"LOCAL_SDK_NAMED_PLATFORM_NOT_FORMAL_CONTEXT\",\"UB\":"<<ub<<",\"L0B\":"<<l0b<<",\"queries\":[";
    auto raw=query(*platform,185600);std::cout<<",";auto two=query(*platform,152832);
    bool match=raw.rc==0 && two.rc==0 && raw.bytes.size()==200 && raw.bytes==two.bytes &&
       raw.m==16 && raw.n==64 && raw.k==256 && raw.depth_a==1 && raw.depth_b==1 && raw.db_a==2 && raw.db_b==2 && raw.db_c==1 && raw.trans==34816 && ub==262144 && l0b==65536;
    std::cout<<"],\"same_accepted_geometry_and_cube\":"<<(match?"true":"false")<<"}\n";return match?0:1;
}
