// CPU only: exercise the actual SDK parent serializer with independently recognizable field markers.
#include "d_flash_group_quant_linear_tiling.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

template<class Cube> void mark(Cube& cube) {
#include "cube_setters.inc"
}

int main() {
    optiling::DFlashGroupQuantLinearTilingData parent;
    optiling::TCubeTiling cube;
    parent.set_globalM(1); parent.set_globalK(2); parent.set_globalN(3); parent.set_tileK(4);
    parent.set_matmulUbBytes(UINT64_C(0x0102030405060708));
    mark(parent.cubeTilingData); mark(cube);
    if (cube.GetDataSize()!=200 || parent.cubeTilingData.GetDataSize()!=200 || parent.GetDataSize()!=224) return 1;
    std::vector<uint8_t> a(224), b(200);
    parent.SaveToBuffer(a.data(),a.size()); cube.SaveToBuffer(b.data(),b.size());
    const uint32_t prefix[]={1,2,3,4}; const uint64_t ub=UINT64_C(0x0102030405060708);
    if (std::memcmp(a.data(),prefix,16)||std::memcmp(a.data()+16,&ub,8)||std::memcmp(a.data()+24,b.data(),200))return 2;
    for (uint32_t i=0;i<50;++i) {
        uint32_t value=0;std::memcpy(&value,b.data()+4*i,4);
        if (value!=0x13570000u+i)return 3;
    }
    std::cout<<"GATE8_PARENT_224_PREFIX_24_CUBE_200_FIELDS_50_MATCH\n";
    return 0;
}
