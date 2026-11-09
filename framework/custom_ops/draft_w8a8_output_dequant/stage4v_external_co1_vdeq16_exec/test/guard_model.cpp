#include "../op_kernel/d_flash_external_co1_echo_probe.cpp"
#include <iostream>
int main(){using namespace AscendC;TPipe p;TBuf<> b[4];uint32_t sizes[4]={202240,16384,2048,8192};int count=0;
for(int k=0;k<4;++k){p.InitBuffer(b[k],sizes[k]+512);auto x=b[k].Get<uint32_t>();Init(x,sizes[k]);Require(Delta(x,sizes[k])==0);
for(int side=0;side<2;++side)for(int word=0;word<64;++word)for(int bit=0;bit<32;++bit){size_t i=side?(256+sizes[k])/4+word:word;auto v=x.GetValue(i);x.SetValue(i,v^(uint32_t(1)<<bit));Require(Delta(x,sizes[k])!=0);x.SetValue(i,v);Require(Delta(x,sizes[k])==0);++count;}}
Require(count==16384);std::cout<<"CPU_ONLY_ALL_512_WORDS_32_BITS_DETECTED\n";}
