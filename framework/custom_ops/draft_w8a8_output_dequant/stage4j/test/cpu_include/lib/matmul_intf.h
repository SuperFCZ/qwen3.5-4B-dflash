// CPU scheduling/lifetime double: cached independent partials, not a Matmul emulator.
#pragma once
namespace matmul {
using namespace AscendC;
template<TPosition P,CubeFormat F,class T>struct MatmulType {};
template<class A,class B,class C,class Bias>struct Matmul {
 GlobalTensor<int8_t>a,b;GlobalTensor<uint64_t>s;LocalTensor<uint8_t>w;
 bool hasA=false,hasB=false,hasS=false,hasW=false,active=false;
 void SetLocalWorkspace(LocalTensor<uint8_t>v){Require(v.size==226816);w=v;hasW=true;}
 void SetTensorA(GlobalTensor<int8_t>v){a=v;hasA=true;}
 void SetTensorB(GlobalTensor<int8_t>v,bool trans){Require(!trans);b=v;hasB=true;}
 void SetQuantVector(GlobalTensor<uint64_t>v){s=v;hasS=true;}
 bool Iterate(bool acc){Require(!acc&&!active&&hasA&&hasB&&hasS&&hasW);active=true;++iterations;return true;}
 void GetTensorC(LocalTensor<half>out,int atomic,bool seq){
  Require(active&&atomic==0&&seq&&out.size==2048);
  size_t index=(b.data-weightBase)/16384,tile=index/20,g=index%20;
  Require(tile<152&&g==(iterations-1)%20&&a.data==activationBase+g*2048&&s.data==scaleBase+index*128);
  for(size_t m=0;m<16;++m)for(size_t n=0;n<128;++n)out.data[(n/16)*256+m*16+n%16]=partialBase[index*2048+m*128+n];
  if(fault==7)w.data[-1]=0;
  if(fault==8)out.data[0]=static_cast<half>(out.data[0]+1);
  if(fault==6)registeredWorkspace[-1]=0;
 }
 void End(){Require(active);active=false;hasA=hasB=hasS=hasW=false;++ends;}
};
}
#define REGIST_MATMUL_OBJ(pipe,workspace,...) AscendC::RegisterWorkspace(workspace)
