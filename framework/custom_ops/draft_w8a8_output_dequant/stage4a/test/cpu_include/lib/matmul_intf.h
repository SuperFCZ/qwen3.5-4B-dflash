// Scalar reference double for exercising the real Kernel's control flow, not SDK emulation.
#pragma once
namespace matmul {
using namespace AscendC;
template<TPosition P,CubeFormat F,class T>struct MatmulType {};
template<class A,class B,class C,class Bias>struct Matmul {
 GlobalTensor<int8_t>a,b;GlobalTensor<uint64_t>s,firstScale;
 LocalTensor<half> firstOutput;LocalTensor<uint8_t>w;bool hasA=false,hasB=false,hasS=false,hasW=false,active=false;
 void SetLocalWorkspace(LocalTensor<uint8_t>v){Require(v.size==239104);w=v;hasW=true;}
 void SetTensorA(GlobalTensor<int8_t>v){a=v;hasA=true;}
 void SetTensorB(GlobalTensor<int8_t>v,bool trans){Require(!trans);b=v;hasB=true;}
 void SetQuantVector(GlobalTensor<uint64_t>v){s=v;hasS=true;}
 bool Iterate(bool accumulate){Require(!accumulate&&!active&&hasA&&hasB&&hasS&&hasW);active=true;++iterations;if(iterations==1)firstScale=s;return true;}
 void GetTensorC(LocalTensor<half>out,int atomic,bool sequential){
  Require(active&&atomic==0&&sequential&&out.size==1024&&a.size>=2048&&b.size>=8192&&s.size>=64);
  if(iterations==1)firstOutput=out;
  if(fault==3)out=firstOutput; // injected stale Local output pointer
  auto scales=s;
  if(fault==1){scales.data=scaleBase+((iterations-1)%20)*64;scales.size=64;}
  auto inputB=b;
  if(fault==5){size_t slot=(b.data-weightBase)/(20*8192);size_t g=(iterations-1)%20;inputB.data=weightBase+(((slot+1)%8)*20+g)*8192;}
  if(fault==6)registeredWorkspace[-1]=0;

  for(int m=0;m<16;++m)for(int n=0;n<64;++n){
   int32_t z=0;for(int k=0;k<128;++k)z+=int32_t(a.data[m*128+k])*int32_t(inputB.data[k*64+n]);
   uint32_t bits=uint32_t(scales.data[n]);float scale;std::memcpy(&scale,&bits,4);
   out.data[(n/16)*256+m*16+n%16]=half(float(z)*scale);
  }
  if(fault==2&&iterations==size_t(targetGroup+1))w.data[-1]^=1; // mutate in allocated prefix redzone, not outside allocation
 }
 void End(){Require(active);active=false;hasA=hasB=hasS=hasW=false;++ends;}
};
}
#define REGIST_MATMUL_OBJ(pipe,workspace,...) AscendC::RegisterWorkspace(workspace)
