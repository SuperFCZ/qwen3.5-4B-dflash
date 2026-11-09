#pragma once
namespace matmul{using namespace AscendC;template<TPosition P,CubeFormat F,class T>struct MatmulType{};
template<class A,class B,class C,class Bias>struct Matmul{GlobalTensor<int8_t>a,b;bool wa=false,aa=false,bb=false,active=false;
void SetLocalWorkspace(LocalTensor<uint8_t>w){Require(w.size==202240);wa=true;}
void SetTensorA(GlobalTensor<int8_t>x){Require(x.size==40960);a=x;aa=true;}
void SetTensorB(GlobalTensor<int8_t>x,bool trans){Require(x.size==655360&&!trans);b=x;bb=true;}
bool Iterate(bool accum,LocalTensor<int32_t>o){Require(!accum&&wa&&aa&&bb&&!active&&o.pos==TPosition::CO1&&o.size==4096&&o.address==0);active=true;++iterations;
for(int m=0;m<16;++m)for(int n=0;n<256;++n){int64_t v=0;for(int k=0;k<2560;++k)v+=int64_t(a.data[m*2560+k])*b.data[k*256+n];o.data[n/16*256+m*16+n%16]=v;}
if(fault>=1&&fault<=4)buffers[fault-1][0]^=1;if(fault==5)workspaceBase[0]^=1;if(fault==6)o.data[0]+=1;return true;}
void End(){Require(active);active=false;++ends;}
};}
#define REGIST_MATMUL_OBJ(pipe,workspace,...) AscendC::RegisterWorkspace(workspace)
