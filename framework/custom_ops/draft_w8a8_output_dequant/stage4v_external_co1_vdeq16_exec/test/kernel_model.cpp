#include "../op_kernel/d_flash_external_co1_echo_probe.cpp"
#include <iostream>
int main(int argc,char**argv){using namespace AscendC;fault=argc>1?std::stoi(argv[1]):0;
std::vector<int8_t>a(40960),b(655360);for(size_t i=0;i<a.size();++i)a[i]=int(i%19)-9;for(size_t i=0;i<b.size();++i)b[i]=int((i/256+3*(i%256))%23)-11;auto oa=a,ob=b;
std::vector<uint64_t>scale(256,0x40003f800000ULL),w(32,0x7f7f7f7f7f7f7f7fULL);auto os=scale;
std::vector<int32_t>o(4224,0x7f7f7f7f);std::vector<uint8_t>ws(10240,0x6d);outputBase=o.data()+64;workspaceBase=ws.data();TestTiling t{202240,0,0,1,4096,4096,5120,0};
d_flash_external_co1_echo_probe((uint8_t*)a.data(),(uint8_t*)b.data(),(uint8_t*)scale.data(),(uint8_t*)outputBase,(uint8_t*)w.data(),ws.data(),(uint8_t*)&t);
Require(a==oa&&b==ob&&scale==os&&ubBytes==230912&&l0bytes==16384);
for(int i=0;i<64;++i)Require(o[i]==0x7f7f7f7f&&o[4160+i]==0x7f7f7f7f);for(size_t i=5120;i<ws.size();++i)Require(ws[i]==0x6d);
for(int s:evstate)Require(s==0);
if(fault!=0&&fault!=6){Require(copies==0&&w[2]!=0);std::cout<<"CPU_GUARD_REJECTED";return 0;}
Require(copies==256&&ends==1&&iterations==1&&w[2]==0&&w[5]==230912&&w[7]==0);
int bad=0;for(int m=0;m<16;++m)for(int n=0;n<256;++n){int64_t dot=0;for(int k=0;k<2560;++k)dot+=int64_t(a[m*2560+k])*b[k*256+n];bad+=outputBase[m*256+n]!=dot;Require(writes[m*256+n]==1);}
Require(bad==(fault==6?1:0));for(int i=0;i<8;++i)Require(w[8+i]<8&&w[16+i]>0);std::cout<<"CPU_SOURCE_MODEL_ONLY";}
