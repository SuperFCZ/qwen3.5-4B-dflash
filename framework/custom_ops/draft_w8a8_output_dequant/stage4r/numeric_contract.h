#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <stdexcept>
// Fixed before device execution. No tolerance CLI or adaptive calibration.
inline double HalfValue(uint16_t u){
 const int e=(u>>10)&31,m=u&1023;double v=e==0?std::ldexp(double(m),-24):e==31?(m?NAN:INFINITY):std::ldexp(double(1024+m),e-25);
 return (u&32768)?-v:v;
}
inline void CheckReference(const std::vector<uint8_t>& out,const std::vector<uint8_t>& ref){
 if(out.size()!=622592||ref.size()!=311296*8)throw std::runtime_error("numeric shape mismatch");
 long double err2=0,ref2=0;
 for(size_t i=0;i<311296;++i){uint16_t bits;double x;std::memcpy(&bits,out.data()+2*i,2);std::memcpy(&x,ref.data()+8*i,8);double y=HalfValue(bits),d=y-x;
  if(!std::isfinite(x)||!std::isfinite(y)||std::abs(d)>0.002+0.002*std::abs(x))throw std::runtime_error("independent FP64 reference tolerance failure");
  err2+=(long double)d*d;ref2+=(long double)x*x;
 }
 if(err2>0.000001L*ref2)throw std::runtime_error("independent relative L2 exceeds 0.001");
}
