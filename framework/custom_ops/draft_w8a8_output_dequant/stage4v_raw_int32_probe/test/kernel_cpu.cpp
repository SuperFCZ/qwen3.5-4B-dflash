#include "../op_kernel/d_flash_full_k_int32_probe.cpp"
#include <algorithm>
#include <iostream>
int main(int argc,char**argv){using namespace AscendC;fault=argc>1?std::stoi(argv[1]):0;
 std::vector<int8_t>a(40960),b(655360);for(size_t i=0;i<a.size();++i)a[i]=int(i%19)-9;for(size_t i=0;i<b.size();++i)b[i]=int((i/256+i%256*3)%23)-11;
 auto oldA=a,oldB=b;std::vector<int32_t>out(4096+128,0x7f7f7f7f);std::vector<uint8_t>ws(10240,0x6d);outputBase=out.data()+64;workspaceBase=ws.data();
 TestTiling t{202240,0,0,1,4096,4096,5120,0};if(fault==4)t.localMemSize++;if(fault==5)blockNum=2;
 d_flash_full_k_int32_probe(reinterpret_cast<uint8_t*>(a.data()),reinterpret_cast<uint8_t*>(b.data()),reinterpret_cast<uint8_t*>(outputBase),ws.data(),reinterpret_cast<uint8_t*>(&t));
 Require(a==oldA&&b==oldB);for(int i=0;i<64;++i)Require(out[i]==0x7f7f7f7f&&out[4160+i]==0x7f7f7f7f);
 for(size_t i=5120;i<ws.size();++i)Require(ws[i]==0x6d);
 if(fault){Require(copies==0);for(int x:writes)Require(x==0);std::cout<<"CPU_REJECTED_BEFORE_GM\n";return 0;}
 Require(ubBytes==220672&&ubBytes+8192==228864&&copies==256&&iterations==1&&ends==1);
 for(int m=0;m<16;++m)for(int n=0;n<256;++n){int64_t expected=0;for(int k=0;k<2560;++k)expected+=int64_t(a[m*2560+k])*b[k*256+n];Require(outputBase[m*256+n]==expected&&writes[m*256+n]==1);}
 std::cout<<"CPU_SOURCE_MODEL_MATCH; HARDWARE_NOT_RUN\n";}
