#pragma once
#include "kernel_operator.h"
#include <limits>
namespace matmul {
enum class CubeFormat { ND };
template<AscendC::TPosition P,CubeFormat F,typename T,bool Trans=false> struct MatmulType {};
// Sparse virtual full-vocabulary weights avoid allocating a 1.27 GB CPU fixture.
// Products are evaluated in FP32, then rounded to half ONLY in GetTensorC.
template<class A,class B,class C,class Bias> class Matmul {
    AscendC::LocalTensor<half> x;
    uint32_t column=0; bool live=false,checked=false;
public:
    void Init(CpuCube* t) { assert(t->singleM==16 && t->singleN==64 && t->singleK==2560); }
    void SetLocalWorkspace(AscendC::LocalTensor<uint8_t>) {}
    void SetTensorA(AscendC::LocalTensor<half> value) {
        assert(value.count==16*2560); x=value;
        if(checked) return;
        checked=true;
        for(uint32_t r=0;r<run->m;++r) for(uint32_t k=0;k<2560;++k)
            assert(x.data[r*2560+k]==run->hidden[r*2560+k]);
        for(uint32_t r=run->m;r<16;++r) for(uint32_t k=0;k<2560;++k) assert(x.data[r*2560+k]==0);
    }
    void SetTensorB(AscendC::GlobalTensor<half> w,bool transpose) {
        assert(transpose && w.base==run->weight && w.offset%2560==0);
        column=w.offset/2560; assert(column%64==0 && column<248320);
    }
    bool Iterate(bool partial) {
        assert(!partial && !live);
        if(int(core)==run->failCore) return false;
        { std::lock_guard<std::mutex> guard(run->mutex);
          assert((column/64)%run->p==core); assert(run->visits[column/64]++==0); }
        live=true; ++run->calls; return true;
    }
    void GetTensorC(AscendC::LocalTensor<half> out,int atomic,bool sequential) {
        assert(live && !atomic && sequential && out.count==16*64); live=false;
        for(uint32_t r=0;r<16;++r) for(uint32_t n=0;n<64;++n) {
            const auto id=column+n; float w0=0,w1=0,wlast=0;
            switch(run->pattern) {
            case 0: break;
            case 1: if(id==64 || id==53760) w0=3; break;
            case 2: if(id==248319) wlast=2; break;
            case 3: if(id==63 || id==64) w0=1; if(id==64) w1=.5f; break;
            case 4: w0=id==248319?-.5f:-1; break;
            case 5: if(id==63 || id==64) w0=65504; break;
            case 6: if(id==63) w0=std::numeric_limits<float>::quiet_NaN(); if(id==64) w0=std::numeric_limits<float>::infinity(); break;
            default: break;
            }
            float sum=0;
            if(run->pattern==7) {
                if(id>=1 && id<=15) sum=float(x.data[r*2560+id-1])*2;
            } else {
                sum+=float(x.data[r*2560])*w0; sum+=float(x.data[r*2560+1])*w1;
                sum+=float(x.data[r*2560+2559])*wlast;
            }
            out.data[r*64+n]=static_cast<half>(sum);
        }
    }
    void End() { assert(!live); }
};
}
#define REGIST_MATMUL_OBJ(pipe,workspace,mm,tiling) mm.Init(tiling)
