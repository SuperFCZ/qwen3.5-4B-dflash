"""CPU-only resource/index/actual-source models; no NPU correctness claim."""
import sys,shutil,subprocess
from pathlib import Path
import numpy as np
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import p_common as c
import p_reference as ref
import p_transform as transform
import run_stage4p as run

def test_exact_source_audit():
 audit,diff=transform.audit();assert audit['status']=='N256_DIMENSION_ONLY_FROM_ACCEPTED_XOR'
 old=transform.ng.transform((transform.J/'op_kernel/d_flash_w8_a8_n128_gate.cpp').read_bytes())
 new=(HERE/'op_kernel/d_flash_w8_a8_n256_gate.cpp').read_bytes()
 for token in (b'PipeBarrier',b'SetFlag',b'WaitFlag',b'mm.End()',b'GetValue',b'InitBuffer',b'SetLocalWorkspace',b'SetQuantVector'):
  assert old.count(token)==new.count(token)
 assert 'Reserve' not in diff or '8192' not in diff # reserve constant unchanged in Kernel
 assert b'nb<16' in new and b'localMemSize!=202240' in new
 subprocess.run(['bash','-n',str(HERE/'run.sh')],check=True)

@pytest.mark.parametrize('cores,limit',[(7,0),(7,2),(7,1)])
def test_complete_indices_and_schedule(cores,limit):
 blocks,owners=c.schedule(cores,limit)
 assert sorted(t for row in owners for t in row)==list(range(76))
 assert c.layout_i.symbolic()['gate_up_boundary']['tile']==38
 for t in c.BOUNDED_TILES:
  _,o=c.schedule(cores,limit,t);assert [v for row in o for v in row]==[t] and t in o[t%blocks]
 assert 202240+2*8192+2*16384+2560+8192==262144
 with pytest.raises(ValueError):c.schedule(7,8)

def test_group_reference_and_half_rounding():
 a=np.ones((20,16,128),'i1');b=np.ones((20,128,256),'i1');b[-1]=-1
 scale=np.full((20,256),1/128,'<f4');scale[0]=scale[-1]=16
 words=np.empty((20,256),'<u8')
 for g in range(20):
  for lo in range(0,256,64):words[g,lo:lo+64]=c.common.stage2a.encode_doc_candidate(scale[g,lo:lo+64])[0]
 parts,out=ref.compute(a,b,words);assert np.all(out==18)
 for half in (slice(0,128),slice(128,256)):
  p,f=c.jref.compute(a,b[:,:,half].copy(),words[:,half].copy())
  assert parts[:,:,half].tobytes()==p.tobytes() and out[:,half].tobytes()==f.tobytes()
 wrong=np.zeros((16,256),'<f2')
 for p in parts:wrong=(wrong+p).astype('<f2')
 assert not np.array_equal(wrong,out)

@pytest.fixture(scope='module')
def model(tmp_path_factory):
 root=tmp_path_factory.mktemp('n256-model');inc=root/'inc';shutil.copytree(transform.J/'test/cpu_include',inc)
 p=inc/'kernel_operator.h';s=p.read_text().replace('V_S,MTE2_S','V_S,MTE2_S,S_MTE3').replace('T GetValue(size_t i)const{Require(i<size);return data[i];}', 'T GetValue(size_t i)const{Require(i<size);return data[i];}\n void SetValue(size_t i,T v)const{Require(i<size);data[i]=v;}');p.write_text(s)
 p=inc/'lib/matmul_intf.h';s=p.read_text().replace('226816','202240').replace('out.size==2048','out.size==4096').replace('/16384','/32768').replace('tile<152','tile<76').replace('index*128','index*256').replace('n<128','n<256').replace('index*2048+m*128+n','index*4096+m*256+n');p.write_text(s)
 return root,inc

BODY=r'''
#include <fstream>
int main(int argc,char**argv){using namespace AscendC;blockNum=std::stoi(argv[2]);int tile=std::stoi(argv[3]);int mode=std::stoi(argv[4]);fault=std::stoi(argv[5]);
 std::vector<int8_t>a(40960),b(49807360);std::vector<uint64_t>s(389120);std::vector<half>parts(76*20*4096),out(311296);std::vector<uint8_t>w(blockNum*5120);
 for(int t=0;t<76;++t)for(int g=0;g<20;++g)for(int m=0;m<16;++m)for(int n=0;n<256;++n)parts[((t*20+g)*16+m)*256+n]=half((g-10)*(n%3+1)*.03125+t*.00390625+m*.015625);
 partialBase=parts.data();activationBase=a.data();weightBase=b.data();scaleBase=s.data();outputBase=out.data();workspaceBase=w.data();std::memset(out.data(),127,622592);
 TestTiling td{202240,uint32_t(mode),tile,blockNum,4096,4096,5120,0};
 for(blockIdx=0;blockIdx<blockNum;++blockIdx){ubBytes=adds=casts=copies=ends=iterations=0;d_flash_w8_a8_n256_gate((uint8_t*)a.data(),(uint8_t*)b.data(),(uint8_t*)s.data(),(uint8_t*)out.data(),w.data(),(uint8_t*)&td);Require(ubBytes==253952);}
 std::ofstream(argv[1],std::ios::binary).write((char*)out.data(),622592);
}
'''

def compile_source(model,diag):
 root,inc=model;src=root/('fault.cpp' if diag else 'normal.cpp');kernel=root/('fault_kernel.cpp' if diag else 'kernel.cpp')
 kernel.write_bytes(transform.kernel((transform.J/'op_kernel/d_flash_w8_a8_n128_gate.cpp').read_bytes(),fault=diag))
 src.write_text('#include "'+str(kernel)+'"\n'+BODY);exe=root/('fault' if diag else 'normal')
 r=subprocess.run([shutil.which('c++'),'-std=c++17','-O2','-I'+str(inc),str(src),'-o',str(exe)],capture_output=True,text=True)
 assert r.returncode==0,r.stderr;return exe

@pytest.fixture(scope='module')
def normal(model):return compile_source(model,False)

@pytest.mark.parametrize('blocks,tile,mode',[(7,-1,0),(2,-1,0),(1,-1,0),(7,0,1),(7,37,20),(7,38,0),(7,75,20)])
def test_actual_kernel_coverage_and_partial(normal,tmp_path,blocks,tile,mode):
 out=tmp_path/'out';subprocess.run([str(normal),str(out),str(blocks),str(tile),str(mode),'0'],check=True)
 a=np.fromfile(out,'<f2').reshape(16,19456)
 for t in range(76):
  v=a[:,t*256:(t+1)*256]
  if tile>=0 and t!=tile:assert np.all(v.view('<u2')==0x7f7f);continue
  parts=np.array([[(g-10)*(np.arange(256)%3+1)*.03125+t*.00390625+m*.015625 for m in range(16)] for g in range(20)],'<f2')
  total=np.zeros((16,256),'<f4')
  for p in parts:total=np.add(total,p.astype('<f4'),dtype=np.float32)
  expected=total.astype('<f2') if mode==0 else parts[mode-1]
  assert v.tobytes()==expected.tobytes()

@pytest.mark.parametrize('fault',[4,6,7,8])
def test_faults_not_silently_passed(normal,tmp_path,fault):
 out=tmp_path/'out';subprocess.run([str(normal),str(out),'7','-1','0',str(fault)],check=True)
 data=out.read_bytes();good=tmp_path/'good';subprocess.run([str(normal),str(good),'7','-1','0','0'],check=True)
 assert data!=good.read_bytes()

def test_device_diagnostic_source_sweep_and_all_abort_groups(model,tmp_path):
 import struct
 exe=compile_source(model,True)
 for mode in range(21):
  out=tmp_path/str(mode);subprocess.run([str(exe),str(out),'1','0',str(mode),'0'],check=True,timeout=120)
  raw=out.read_bytes()
  if mode>0:
   assert raw[:32]==struct.pack('<8I',0x534e4641,mode-1,0x80000001,640,4,1,63,mode)
   assert raw[32:]==bytes([127])*(622592-32)
  else:
   a=np.frombuffer(raw,'<f2').reshape(16,19456);assert np.isfinite(a[:,:256]).all() and np.all(a[:,256:].view('<u2')==0x7f7f)

def test_phase_gates_and_no_benchmark():
 assert run.sequence('bounded')==[(0,m,t) for t in (0,37,38,75) for m in (1,20,0)]
 with pytest.raises(ValueError):run.validate_proof(None,{},None,'full')
 assert 'benchmark' not in (HERE/'run_stage4p.py').read_text().split("choices=")[1].split(')')[0]

def test_complete_comparator_covers_right_half_and_inactive_columns():
 golden=np.zeros((16,19456),'<f2');parts=np.zeros((76,20,16,256),'<f2');words=np.zeros((76,20,256),'<u8')
 maps={'golden_fp16.bin':golden,'partials_fp16.bin':parts,'quant_u64_doc_candidate.bin':words}
 raw=golden.tobytes();assert ref.compare(maps,0,7,-1,[raw]*2)['checked_elements']==311296
 bad=golden.copy();bad[15,19455]=1
 assert ref.compare(maps,0,7,-1,[raw,bad.tobytes()])['status']=='FAIL_NUMERICAL'
 outs=[]
 for val in (0x7f7f,0xffff):
  a=np.full((16,19456),val,'<u2').view('<f2');a[:,38*256:39*256]=0;outs.append(a.tobytes())
 assert ref.compare(maps,0,7,38,outs)['status']=='BITWISE_MATCH'
 a=np.frombuffer(outs[1],'<u2').copy().reshape(16,19456);a[0,39*256+200]=0
 assert not ref.compare(maps,0,7,38,[outs[0],a.tobytes()])['inactive_poison_unchanged']

@pytest.mark.parametrize('which',['test','Fault'])
def test_runner_abi_syntax(which,tmp_path):
 inc=tmp_path/'inc';shutil.copytree(HERE.parent/'stage4k/test/N128/cpu_include',inc)
 old=inc/'aclnn_d_flash_w8_a8_n128_gate.h';(inc/'aclnn_d_flash_w8_a8_n256_gate.h').write_text(old.read_text().replace('N128','N256'));old.unlink()
 r=subprocess.run([shutil.which('c++'),'-std=c++17','-fsyntax-only','-I'+str(inc),'-I'+str(c.common.QUANT/'test'),str(HERE/which/'runner.cpp')],capture_output=True,text=True)
 assert r.returncode==0,r.stderr

def test_full_prepare_hashes_and_frozen_golden_identity(tmp_path,monkeypatch):
 import hashlib
 a=((np.arange(40960)%11)-5).astype('i1').reshape(20,16,128)
 nk=np.broadcast_to((np.arange(19456)%31-15).astype('i1')[:,None],(19456,2560))
 nz=nk.reshape(1216,16,80,32).transpose(2,0,1,3)
 sw=np.broadcast_to(np.float16(.125),(20,19456));src=dict(a=a,q=nk,nz=nz,sw=sw,sa=.25)
 b=np.broadcast_to(nk[:,0].reshape(152,128)[:,None,None,:],(152,20,128,128))
 word=c.common.stage2a.encode_doc_candidate(np.full(64,.03125,'<f4'))[0][0]
 words=np.full((152,20,128),word,'<u8');parts=np.empty((152,20,16,128),'<f2');gold=np.empty((16,19456),'<f2')
 for t in range(152):
  part,out=c.jref.compute(a,b[t].copy(),words[t]);parts[t]=part;gold[:,t*128:(t+1)*128]=out
 files={}
 for name,array in [('a_i8.bin',a),('golden_fp16.bin',gold)]:
  path=tmp_path/name;array.tofile(path);files[name]=c.identity(path)
 bc=tmp_path/'b-case.json';c.write(bc,dict(files=files,source={}))
 jc=tmp_path/'j-case.json';c.write(jc,dict(files=files,stage4b_case=c.identity(bc)))
 maps=dict(**files);maps={'a_i8.bin':a,'b_i8.bin':b,'quant_u64_doc_candidate.bin':words,'partials_fp16.bin':parts,'golden_fp16.bin':gold}
 hb,hs=hashlib.sha256(),hashlib.sha256()
 for t in range(76):
  packed,s=c.layout_i.verify_tile(src,b,words,t);hb.update(packed.tobytes());hs.update(s.tobytes())
 proof=dict(n128_case=c.identity(jc),stage4b_case=c.identity(bc),layout=dict(packed_B_sha256=hb.hexdigest(),packed_scale_sha256=hs.hexdigest()))
 monkeypatch.setattr(ref.jref,'verify_case',lambda *_:maps);monkeypatch.setattr(ref.reference,'load_source',lambda *_:src)
 case=ref.prepare(tmp_path/'new',proof);checked=ref.verify_case(case,numerical=True)
 assert len(case['tiles'])==76 and case['files']['golden_fp16.bin']==files['golden_fp16.bin']
 assert not (tmp_path/'new/golden_fp16.bin').exists()
 assert checked['a_i8.bin'].tobytes()==a.tobytes()
 case['tiles'][-1]['partial_sha256']='0'*64
 with pytest.raises(ValueError):ref.verify_case(case)

def test_host_contract_rejects_l0b_or_basen_changes(tmp_path):
 f={k:0 for k in c.FIELDS};f.update(usedCoreNum=1,singleCoreM=16,singleCoreN=256,singleCoreK=128,baseM=16,baseN=256,baseK=128,depthA1=1,depthB1=1,dbL0A=2,dbL0B=2,dbL0C=1,transLength=34816,shareL1Size=34816,shareL0CSize=16384,shareUbSize=2048,stepM=1,stepN=1,stepKa=1,stepKb=1)
 t=dict(fields=f,capacity=dict(UB=262144,L1=1048576,L0A=65536,L0B=65536,L0C=262144),system_workspace_bytes=4096,scratch=202240)
 (tmp_path/'contract.h').write_text(c.contract_header(t))
 getters='\n'.join(f'int get_{k}()const{{return {v};}}' for k,v in f.items())
 code='#include <cstdint>\n#include "contract.h"\nstruct T{'+getters+'}; int main(){T t;return S4PAcceptedT1(t,262144,1048576,65536,65536,262144,4096,202240)?0:1;}'
 for i,text in enumerate((code,code.replace('get_baseN()const{return 256;}','get_baseN()const{return 128;}'),code.replace('65536,65536,262144','65536,65504,262144'))):
  (tmp_path/'t.cpp').write_text(text);r=subprocess.run([shutil.which('c++'),'-std=c++17',str(tmp_path/'t.cpp'),'-o',str(tmp_path/'t')],capture_output=True,text=True)
  assert r.returncode==0,r.stderr
  r=subprocess.run([str(tmp_path/'t')],capture_output=True);assert r.returncode==(0 if i==0 else 1)
