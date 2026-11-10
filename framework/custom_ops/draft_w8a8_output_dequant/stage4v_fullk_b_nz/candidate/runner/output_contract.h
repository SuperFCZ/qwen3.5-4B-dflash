#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>
namespace fullk_explore {
constexpr size_t Elements=16*19456,Bytes=Elements*2;
inline uint16_t Bits(const std::vector<uint8_t>&x,size_t i){return uint16_t(x[2*i])|(uint16_t(x[2*i+1])<<8);}
inline int Ordered(uint16_t x){return x&0x8000?32768-int(x&32767):32768+int(x);}
inline double Value(uint16_t x){int e=(x>>10)&31;double v=e?std::ldexp(double(1024+(x&1023)),e-25):std::ldexp(double(x&1023),-24);return x&32768?-v:v;}
struct Difference{size_t row,channel;uint16_t actual,expected;int ulp;double absolute;};
struct Numeric{size_t active=0,mismatches=0;double max_abs=0;int max_ulp=0;std::vector<Difference> rows;};
inline Numeric Inspect(const std::vector<uint8_t>&a,const std::vector<uint8_t>&g,const std::vector<uint8_t>&first,int64_t tile,uint16_t poison){
 if(a.size()!=Bytes||g.size()!=Bytes||(!first.empty()&&first.size()!=Bytes)||(tile!=0&&tile!=-1))throw std::runtime_error("output contract shape/scope");
 Numeric n;
 for(size_t i=0;i<Elements;++i){size_t c=i%19456;uint16_t x=Bits(a,i),y=Bits(g,i);
  if(tile==0&&c>=256){if(x!=poison)throw std::runtime_error("inactive Tile poison changed");continue;}
  ++n.active;
  if((x&0x7c00)==0x7c00||(y&0x7c00)==0x7c00)throw std::runtime_error("nonfinite active output/reference or Poison residue");
  if(!first.empty()&&x!=Bits(first,i))throw std::runtime_error("active output Repeat drift");
  double err=std::abs(Value(x)-Value(y));int u=std::abs(Ordered(x)-Ordered(y));
  n.max_abs=std::max(n.max_abs,err);n.max_ulp=std::max(n.max_ulp,u);
  if(x!=y){++n.mismatches;n.rows.push_back({i/19456,c,x,y,u,err});}
 }
 if(n.active!=(tile==0?4096:Elements))throw std::runtime_error("active coverage incomplete");return n;
}
inline void WriteNumeric(const std::string&path,const Numeric&n){
 std::ofstream f(path);f<<std::setprecision(17)<<"{\"numeric_status\":\""<<(n.mismatches?"NUMERIC_MISMATCH_UNRESOLVED":"BITWISE_REFERENCE_OBSERVATION")<<"\",\"active_elements\":"<<n.active<<",\"mismatch_count\":"<<n.mismatches<<",\"max_ulp\":"<<n.max_ulp<<",\"max_absolute_error\":"<<n.max_abs<<",\"differences\":[";
 for(size_t i=0;i<n.rows.size();++i){const auto&d=n.rows[i];f<<(i?",":"")<<"{\"row\":"<<d.row<<",\"channel\":"<<d.channel<<",\"actual_bits\":"<<d.actual<<",\"expected_bits\":"<<d.expected<<",\"ulp\":"<<d.ulp<<",\"absolute_error\":"<<d.absolute<<"}";}
 f<<"]}\n";if(!f)throw std::runtime_error("numeric report write failed");
}
}
