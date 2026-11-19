// Native CPU witness for the very same loop included in both private device builds.
#include <cstdint>
#include <cstdlib>
#include <iostream>
#define __aicore__
#include "../kernel/tile_schedule.h"

struct Witness {
    int fail, issued = -1;
    void emit(const char *op, int k) { std::cout << op << ' ' << k << '\n'; }
    void Copy(uint32_t n, uint32_t k) { if(n!=128) std::abort(); emit("copy", k); pending = k; }
    int pending=-1;
    void WaitRaw() { if(pending<0) std::abort(); emit("wait_s",pending); emit("wait_v",pending); pending=-1; }
    void Dequant(uint32_t k) { emit("dequant", k); }
    bool Iterate(uint32_t k) {
        issued=k;
        emit("v_to_s",k);
        emit(int(k)==fail ? "iterate_false" : "iterate",k);
        return int(k)!=fail;
    }
    bool copied=false;
    void Fence() { emit(copied?"output_all":"all", copied?-1:issued); }
    void Output(uint32_t n) { if(n!=128) std::abort(); copied=true; emit("get_c",-1); }
};

int main(int argc, char **argv) {
    if(argc!=3) return 2;
    Witness model{std::atoi(argv[2])};
    bool success = std::atoi(argv[1]) ? RunW16NTile<true>(model,128) : RunW16NTile<false>(model,128);
    return success==(model.fail<0) && model.pending<0 ? 0 : 1;
}
