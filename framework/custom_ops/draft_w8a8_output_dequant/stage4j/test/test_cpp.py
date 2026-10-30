"""Actual new C++ sources with CPU-only ABI/Matmul doubles."""
import shutil
import subprocess
import pytest
from test_stage4j import HERE,c,ref
import test_stage4j as fixtures
packed=fixtures.packed  # pytest fixture export

def test_runner_syntax_with_new_acl_abi(tmp_path):
    # Reuse only the old test's public ABI-double construction (not its assertions).
    old=(HERE.parent/'stage4b/test/test_stage4b.py').read_text()
    start=old.index(' (tmp_path/\'acl\').mkdir()')
    end=old.index(' result=subprocess.run',start)
    construction=old[start:end]
    construction=construction.replace('aclnnDFlashW8A8FullGate','aclnnDFlashW8A8N128Gate').replace('int64_t,int64_t,aclTensor*','int64_t,int64_t,int64_t,aclTensor*')
    exec('if True:\n'+construction,{'tmp_path':tmp_path,'c':c})
    result=subprocess.run([shutil.which('c++'),'-std=c++17','-fsyntax-only','-I'+str(tmp_path),'-I'+str(c.common.QUANT/'test'),str(HERE/'test/runner.cpp')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr

@pytest.fixture(scope='module')
def kernel_model(tmp_path_factory):
    root=tmp_path_factory.mktemp('kernel-model');src=root/'main.cpp'
    src.write_text('#include <fstream>\n#include "'+str(HERE/'op_kernel/d_flash_w8_a8_n128_gate.cpp')+'"\n'+r'''
int main(int argc,char**argv){
 using namespace AscendC;std::vector<int8_t>a(40960),b(49807360);std::vector<uint64_t>s(389120);std::vector<half>partials(152*20*2048),o(311296);
 std::ifstream(argv[1],std::ios::binary).read((char*)a.data(),40960);std::ifstream(argv[2],std::ios::binary).read((char*)b.data(),49807360);
 std::ifstream(argv[3],std::ios::binary).read((char*)s.data(),3112960);std::ifstream(argv[4],std::ios::binary).read((char*)partials.data(),12451840);
 std::memset(o.data(),std::stoi(argv[10]),622592);outputBase=o.data();activationBase=a.data();weightBase=b.data();scaleBase=s.data();partialBase=partials.data();
 blockNum=std::stoi(argv[6]);uint32_t mode=std::stoi(argv[7]);fault=std::stoi(argv[8]);int tile=std::stoi(argv[9]);std::vector<uint8_t>w(blockNum*5120);workspaceBase=w.data();
 TestTiling td{226816,mode,tile,blockNum,4096,4096,5120,0};
 for(blockIdx=0;blockIdx<blockNum;++blockIdx){
  ubBytes=adds=casts=copies=ends=iterations=0;
  d_flash_w8_a8_n128_gate((uint8_t*)a.data(),(uint8_t*)b.data(),(uint8_t*)s.data(),(uint8_t*)o.data(),w.data(),(uint8_t*)&td);
  Require(ubBytes==253952); // 8192B reserve remains unused
  if(fault!=6&&fault!=7){size_t count=tile<0?((151-blockIdx)/blockNum+1):(uint32_t(tile)%blockNum==blockIdx?1:0);
   Require(iterations==count*20&&ends==iterations&&adds==iterations&&copies==count*128);}
  else Require(copies==0);
 }
 if(fault!=6&&fault!=7)for(size_t i=0;i<writes.size();++i)Require(writes[i]==(tile<0||int((i%19456)/128)==tile?1:0));
 std::ofstream(argv[5],std::ios::binary).write((char*)o.data(),622592);
}
''')
    exe=root/'model';p=subprocess.run([shutil.which('c++'),'-std=c++17','-O1','-I'+str(HERE/'test/cpu_include'),str(src),'-o',str(exe)],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    return exe

@pytest.mark.parametrize('blocks,mode,tile',[(7,0,-1),(2,0,-1),(1,0,-1),(7,21,-1),(7,1,0),(7,20,0),(7,0,75),(7,1,76),(7,20,151)])
def test_actual_kernel_addressing_partial_retention_and_lifecycle(packed,kernel_model,tmp_path,blocks,mode,tile):
    _,case,_,_=packed;maps=ref.verify_case(case);out=[]
    for poison in (127,255):
        dest=tmp_path/f'{poison}.bin'
        subprocess.run([str(kernel_model),*[case['files'][n]['path'] for n in ('a_i8.bin','b_i8.bin','quant_u64_doc_candidate.bin','partials_fp16.bin')],str(dest),str(blocks),str(mode),'0',str(tile),str(poison)],check=True)
        out.append(dest.read_bytes())
    assert ref.compare(maps,mode,blocks,tile,out)['status']=='BITWISE_MATCH'

@pytest.mark.parametrize('fault',[4,6,7,8])
def test_actual_kernel_injected_missing_reset_arena_ub_numeric_fault(packed,kernel_model,tmp_path,fault):
    _,case,_,_=packed;maps=ref.verify_case(case);out=[]
    for poison in (127,255):
        dest=tmp_path/f'{poison}.bin'
        subprocess.run([str(kernel_model),*[case['files'][n]['path'] for n in ('a_i8.bin','b_i8.bin','quant_u64_doc_candidate.bin','partials_fp16.bin')],str(dest),'7','0',str(fault),'-1',str(poison)],check=True)
        out.append(dest.read_bytes())
    assert ref.compare(maps,0,7,-1,out)['status']=='FAIL_NUMERICAL'

def test_every_prefix_and_suffix_guard_word_is_checked(tmp_path):
    src=tmp_path/'guard.cpp';src.write_text('#include "'+str(HERE/'op_kernel/d_flash_w8_a8_n128_gate.cpp')+'"\n'+r'''
int main(){using namespace AscendC;
 for(uint32_t bytes:{226816U,4096U,4096U,8192U,8192U}){
  TPipe pipe;TBuf<> buf;pipe.InitBuffer(buf,bytes+512);auto raw=buf.Get<uint32_t>();InitGuard(raw,bytes);Require(CheckGuard(raw,bytes));
  for(uint32_t side=0;side<2;++side)for(uint32_t i=0;i<64;++i){
   auto idx=side?(256+bytes)/4+i:i;raw.data[idx]^=1;Require(!CheckGuard(raw,bytes));raw.data[idx]^=1;Require(CheckGuard(raw,bytes));
  }
 }
}
''')
    exe=tmp_path/'guard';p=subprocess.run([shutil.which('c++'),'-std=c++17','-O1','-I'+str(HERE/'test/cpu_include'),str(src),'-o',str(exe)],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    subprocess.run([str(exe)],check=True)
