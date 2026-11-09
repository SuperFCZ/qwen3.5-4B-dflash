#pragma once
namespace matmul{using namespace AscendC;
template<TPosition P,CubeFormat F,class T>struct MatmulType{};
template<class A,class B,class C,class Bias>struct Matmul{
 GlobalTensor<int8_t>a,b;LocalTensor<uint8_t>w;bool hasA=false,hasB=false,hasW=false,active=false;
 void SetLocalWorkspace(LocalTensor<uint8_t>x){Require(!active&&x.size==202240);w=x;hasW=true;}
 void SetTensorA(GlobalTensor<int8_t>x){Require(!active&&x.size==40960);a=x;hasA=true;}
 void SetTensorB(GlobalTensor<int8_t>x,bool trans){Require(!active&&!trans&&x.size==655360);b=x;hasB=true;}
 bool Iterate(bool acc){Require(!acc&&!active&&hasA&&hasB&&hasW);active=true;++iterations;return true;}
 void GetTensorC(LocalTensor<int32_t>o,int atomic,bool sync){Require(active&&atomic==0&&sync&&o.size==4096);
 for(int m=0;m<16;++m)for(int n=0;n<256;++n){int64_t v=0;for(int k=0;k<2560;++k)v+=int64_t(a.data[m*2560+k])*b.data[k*256+n];Require(v>=INT32_MIN&&v<=INT32_MAX);o.data[(n/16)*256+m*16+n%16]=int32_t(v);}
 if(fault==1)w.data[-1]^=1;if(fault==2)o.data[-1]^=1;if(fault==3)workspaceBase[0]^=1;}
 void End(){Require(active);active=false;hasA=hasB=hasW=false;++ends;}
};}
#define REGIST_MATMUL_OBJ(pipe,workspace,...) AscendC::RegisterWorkspace(workspace)
