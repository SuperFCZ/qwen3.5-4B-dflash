"""Actual C++ candidate, cached Matmul double, never hardware correctness proof."""
import sys,shutil,subprocess
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import guard_transform as g
import n_common as n

def compile_model(tmp_path,kind):
 inc=tmp_path/'inc';shutil.copytree(HERE.parent/'stage4j/test/cpu_include',inc)
 p=inc/'kernel_operator.h';s=p.read_text().replace('V_S,MTE2_S','V_S,MTE2_S,S_MTE3').replace('inline size_t ubBytes=0','inline size_t guardReads=0;\ninline size_t ubBytes=0')
 s=s.replace('T GetValue(size_t i)const{Require(i<size);return data[i];}', 'T GetValue(size_t i)const{Require(i<size);++guardReads;return data[i];}\n void SetValue(size_t i,T value)const{Require(i<size);data[i]=value;}')
 p.write_text(s)
 original=(g.BASE/'op_kernel/d_flash_w8_a8_n128_gate.cpp').read_bytes()
 (tmp_path/'kernel.cpp').write_bytes(g.diagnostic(original) if kind=='fault' else g.transform(original))
 return inc

def build(tmp_path,body,kind='xor'):
 inc=compile_model(tmp_path,kind);src=tmp_path/'main.cpp';src.write_text('#include "kernel.cpp"\n'+body);exe=tmp_path/'model'
 r=subprocess.run([shutil.which('c++'),'-std=c++17','-O2','-I'+str(inc),str(src),'-o',str(exe)],capture_output=True,text=True)
 assert r.returncode==0,r.stderr
 return exe

def test_exhaustive_every_word_bit_group_no_short_circuit(tmp_path):
 exe=build(tmp_path,r'''
int main(){using namespace AscendC;std::vector<uint32_t>v[5];uint32_t bytes[5]={226816,4096,4096,8192,8192};LocalTensor<uint32_t> t[5];
 for(int b=0;b<5;++b){v[b].resize((bytes[b]+512)/4);t[b]={v[b].data(),v[b].size()};InitGuard(t[b],bytes[b]);}
 auto check=[&](){guardReads=0;auto d=GroupDelta(t[0],bytes[0],t[1],t[2],t[3],t[4]);Require(guardReads==640);return d;};Require(check()==0);
 for(int group=0;group<20;++group)for(int b=0;b<5;++b)for(int side=0;side<2;++side)for(int word=0;word<64;++word)for(int bit=0;bit<32;++bit){
  auto idx=side?(256+bytes[b])/4+word:word;auto mask=uint32_t(1)<<bit;t[b].SetValue(idx,guardWord^mask);Require(check()==mask);t[b].SetValue(idx,guardWord);Require(check()==0);
 }
 // Equal errors in different words must not cancel (OR of XOR, not XOR of XOR).
 t[0].SetValue(0,guardWord^0xffffffffU);t[4].SetValue(0,guardWord^0xffffffffU);Require(check()==0xffffffffU);
}
''')
 subprocess.run([str(exe)],check=True,timeout=90)

BODY=r'''
int main(int argc,char**argv){using namespace AscendC;int mode=std::stoi(argv[1]);std::vector<int8_t>a(40960),b(49807360);std::vector<uint64_t>s(389120);std::vector<half>parts(152*20*2048),o(311296);std::vector<uint8_t>w(5120);
std::memset(o.data(),127,622592);outputBase=o.data();activationBase=a.data();weightBase=b.data();scaleBase=s.data();partialBase=parts.data();workspaceBase=w.data();blockIdx=0;blockNum=1;
TestTiling td{226816,uint32_t(mode),0,1,4096,4096,5120,0};
d_flash_w8_a8_n128_gate((uint8_t*)a.data(),(uint8_t*)b.data(),(uint8_t*)s.data(),(uint8_t*)o.data(),w.data(),(uint8_t*)&td);
Require(ubBytes==253952);
if(mode==0){Require(iterations==20&&ends==20&&adds==20&&copies==128);for(int m=0;m<16;++m)for(int n=0;n<128;++n)Require(o[m*19456+n]==half(0));}
else{Require(iterations==size_t(mode)&&ends==size_t(mode)&&adds==size_t(mode)&&copies==0);const uint32_t expected[8]={0x534e4641U,uint32_t(mode-1),0x80000001U,640,4,1,63,uint32_t(mode)};Require(std::memcmp(o.data(),expected,32)==0);for(int i=32;i<622592;++i)Require(((uint8_t*)o.data())[i]==127);}
}
'''

def test_instrumented_actual_kernel_all_sweeps_and_abort_timing(tmp_path):
 exe=build(tmp_path,BODY,'fault')
 for mode in range(21):subprocess.run([str(exe),str(mode)],check=True,timeout=120)

def test_transient_real_kernel_stops_before_next_group(tmp_path):
 inc=compile_model(tmp_path,'xor');p=inc/'lib/matmul_intf.h';s=p.read_text().replace('if(fault==6)registeredWorkspace[-1]=0;', 'if(g==3)w.data[-1]=0; if(g==4)w.data[-1]=0x5a;');p.write_text(s)
 src=tmp_path/'test.cpp';body=BODY[:BODY.index('if(mode==0)')]+ 'Require(iterations==4&&ends==4&&copies==0);}\n'
 src.write_text('#include "kernel.cpp"\n'+body);exe=tmp_path/'t'
 p=subprocess.run([shutil.which('c++'),'-std=c++17','-O2','-I'+str(inc),str(src),'-o',str(exe)],capture_output=True,text=True);assert p.returncode==0,p.stderr
 subprocess.run([str(exe),'0'],check=True)

def test_transform_keeps_tile_arena_math_sync():
 original=(g.BASE/'op_kernel/d_flash_w8_a8_n128_gate.cpp').read_bytes();candidate=g.transform(original)
 for token in (b'mm.End()',b'SetQuantVector',b'SetLocalWorkspace',b'PipeBarrier',b'SetFlag',b'WaitFlag',b'InitBuffer',b'Add(sum',b'CheckArena'):
  assert candidate.count(token)==original.count(token)
 assert candidate.count(b'CheckGuard(')==original.count(b'CheckGuard(')-5
 assert b'GetValue' in g.HELPER.encode() and b'uint64_t' not in g.HELPER.encode()
 with pytest.raises(ValueError):g.transform(candidate)

def test_missing_device_fault_gate():
 from fault_validation import verify_faults
 with pytest.raises(ValueError,match='fault-report'):verify_faults(None,{})

@pytest.mark.parametrize('name',['Full','Xor','Fault'])
def test_runners_compile_and_protocol(name,tmp_path):
 import os,json
 stub=HERE.parent/'stage4k/test/N128';inc=tmp_path/'api';shutil.copytree(stub/'cpu_include',inc)
 code=(stub/'cpu_api.cpp').read_text()
 if name=='Fault':
  code=code.replace('if(tile!=-1)std::abort();','if(tile!=0)std::abort();currentMode=mode;').replace('static int prepares=0','static int64_t currentMode=0;\nstatic int prepares=0')
  code=code.replace('std::memset(output->data,0,622592);','if(currentMode==0){for(int row=0;row<16;++row)std::memset((char*)output->data+row*19456*2,0,256);}else{uint32_t v[8]={0x534e4641U,uint32_t(currentMode-1),0x80000001U,640,4,1,63,uint32_t(currentMode)};std::memcpy(output->data,v,32);}')
 (tmp_path/'api.cpp').write_text(code);exe=tmp_path/'runner'
 p=subprocess.run([shutil.which('c++'),'-std=c++17','-O1','-I'+str(inc),'-I'+str(n.common.QUANT/'test'),str(HERE/name/'runner.cpp'),str(tmp_path/'api.cpp'),'-o',str(exe)],capture_output=True,text=True)
 assert p.returncode==0,p.stderr
 for mode in ([0,1,20] if name=='Fault' else [0]):
  folder=tmp_path/str(mode);folder.mkdir()
  for file,size in [('a_i8.bin',40960),('b_i8.bin',49807360),('quant_u64_doc_candidate.bin',3112960),('golden_fp16.bin',622592)]:
   with (folder/file).open('wb') as f:f.truncate(size)
  env=os.environ.copy();env['DFLASH_STAGE4J_EXPECTED_CORE_POLICY']='8,8,7,7,2002'
  argv=[str(exe),'0',str(folder)]+([str(mode),'1','0'] if name=='Fault' else ['0','--benchmark'])
  with (folder/'runner.log').open('w') as log:p=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,env=env)
  assert p.returncode==0,(folder/'runner.log').read_text()
  ex=json.loads((folder/'execution.json').read_text())
  if name!='Fault':assert len(n.check_execution(ex,0,name,True)['execute_sync']['samples_ms'])==30
  else:assert ex['tile_id']==0 and ex['timing']['status']=='NOT_RUN'

def test_materiality_no_go_and_no_profile_denominator():
 rows=[dict(round=i,position=p,variant=v,status='FROZEN_GOLDEN_MATCH',profiled=False,formal_latency_evidence=True,safety_protocol=n.POLICY[v],workspace_bytes=123,plan={},timing={key:n.k.statistics_ms([1]*30,30) for key in ('execute_sync','prepare')}) for i,o in enumerate(n.ORDER) for p,v in enumerate(o)]
 assert n.comparison(rows)['decision']=='NO_GO_NO_MATERIAL_IMPROVEMENT'
 rows[0]['profiled']=True
 with pytest.raises(ValueError):n.comparison(rows)

def test_audit_inventory_and_fault_artifact_cannot_authorize_timing(tmp_path):
 audit=g.materialize(tmp_path/'xor','xor');g.verify_audit(audit)
 bad=dict(audit,files=audit['files'][:-1])
 with pytest.raises(ValueError,match='inventory'):g.verify_audit(bad)
 with pytest.raises(ValueError,match='swapped'):n.check_build(dict(opps=dict(Xor=dict(audit=dict(kind='fault')))))
