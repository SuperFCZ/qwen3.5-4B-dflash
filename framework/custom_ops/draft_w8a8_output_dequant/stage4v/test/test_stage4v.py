"""CPU only. Independent quant/reference and actual-source lifecycle models."""
import copy,json,os,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import v_common as c
import reference_v as ref
import resources_v as res
import protocol_v as proto
import run_stage4v as run


def test_requantization_rne_zero_and_scale_not_fitted_to_x():
 w=np.array([[1,3,254,-1,-3,-254],[0,0,0,0,0,0]],'<f2')
 q,s,d=ref.quantize_weight(w)
 assert s.tolist()==[2.,1.] and q.tolist()==[[0,2,127,0,-2,-127],[0]*6]
 assert d['zero_channels']==[1] and d['clipped']==0
 with pytest.raises(ValueError):ref.quantize_weight(np.array([[np.inf]],'<f2'))

@pytest.mark.parametrize('sign',[-1,0,1])
def test_full_k_exact_integer_oracle_and_overflow_bound(sign):
 a=np.full((16,2560),127,'i1');b=np.full((2560,256),127*sign,'i1')
 words,e=ref.encode(np.full(256,1/2048,'<f4'));dot,y=ref.compute(a,b,words)
 assert np.all(dot==41290240*sign) and np.all(y==np.float16(np.float32(41290240*sign)*e[0]))
 b[0,0]=-128
 with pytest.raises(ValueError):ref.compute(a,b,words)


def test_encoding_finite_and_partial_rounding_not_old_golden():
 words,eff=ref.encode(np.array([1,0,.3],'<f4'))
 assert words[0]==(1<<46)|0x3f800000 and eff[-1]!=np.float32(.3)
 a=np.ones((1,2560),'i1');b=np.zeros((2560,3),'i1');b[:128]=1;b[-128:]=-1;b[128]=1
 dot,y=ref.compute(a,b,words);assert np.all(dot==1) and y[0,0]==1
 for bad in (np.array([np.nan],'<f4'),np.array([-1],'<f4'),np.array([1e-45],'<f4')):
  with pytest.raises(ValueError):ref.encode(bad)
 bad=words.copy();bad[0]|=np.uint64(1)
 with pytest.raises(ValueError):ref.compute(a,b,bad)


def test_full_activation_relayout_and_all_packing_indices():
 old=((np.arange(40960)%255)-127).astype('i1').reshape(20,16,128)
 a=old.transpose(1,0,2).reshape(16,2560)
 assert np.array_equal(a.reshape(16,20,128).transpose(1,0,2),old)
 coverage=np.zeros((16,19456),'u1')
 for tile in range(76):
  nk=((np.arange(256)[:,None]+tile*256+np.arange(2560)[None,:])%255-127).astype('i1')
  packed=nk.T.copy()
  for k in (0,127,128,2559):assert np.array_equal(packed[k],nk[:,k])
  idx=(np.arange(256)[None,:]//16)*256+np.arange(16)[:,None]*16+np.arange(256)[None,:]%16
  assert np.array_equal(np.sort(idx.ravel()),np.arange(4096))
  coverage[:,tile*256:(tile+1)*256]+=1
 assert np.all(coverage==1) and 9728//256==38


def fixture_tiling():
 f={k:0 for k in c.p.FIELDS};f.update(usedCoreNum=1,singleCoreM=16,singleCoreN=256,singleCoreK=2560,baseM=16,baseN=256,baseK=128,depthA1=1,depthB1=1,dbL0A=2,dbL0B=2,dbL0C=1,stepM=1,stepN=1,stepKa=1,stepKb=1,transLength=34816,shareL1Size=34816,shareL0CSize=16384,shareUbSize=2048,isBias=0,Ka=2560,Kb=2560,M=16,N=256)
 platform=dict(platform_source='NO_ARGUMENT_MANAGER_ON_SELECTED_DEVICE',soc='Ascend310P3',arch=2002,aic=8,aiv=8,all=8,system_workspace_bytes=4096,capacity=dict(UB=262144,L1=1048576,L0A=65536,L0B=65536,L0C=262144))
 row=dict(name='FULL_K_N256',requested_n=256,requested_base_n=256,scratch=202240,setup_accepted=True,get_tiling_rc=0,fields=f)
 return platform,row


def test_resource_fixture_is_not_real_tiling_and_fails_closed():
 pl,row=fixture_tiling();r=res.validate(pl,row)
 assert r['status']=='HOST_RESOURCE_CHECKS_PASS' and r['details']['total_ub']==220672 and r['details']['l0b_bound']==65536
 for field,val in [('singleCoreK',128),('baseN',128),('dbL0B',3),('transLength',202240),('depthB1',1000)]:
  bad=copy.deepcopy(row);bad['fields'][field]=val
  assert res.validate(pl,bad)['status']=='BLOCKED'
 larger=copy.deepcopy(pl);larger['capacity']['L0B']=131072
 assert res.validate(larger,row)['status']=='BLOCKED'
 bad=copy.deepcopy(row);bad['get_tiling_rc']=-1;bad['fields']=None
 assert res.validate(pl,bad)['status']=='BLOCKED'


def test_source_new_semantics_and_fault_audit(tmp_path):
 kernel=(HERE/'op_kernel/d_flash_w8_a8_full_k_channel.cpp').read_text()
 assert kernel.count('mm.Iterate(false)')==1 and kernel.count('mm.GetTensorC(')==1
 assert 'Cast(' not in kernel and 'Add(' not in kernel and 'g<20' not in kernel
 assert 'tile*2560*256' in kernel and 'sg[tile*256]' in kernel
 for diag in (False,True):
  audit=c.materialize(tmp_path/str(diag),diag);c.verify_audit(audit)
  bad=copy.deepcopy(audit);bad['fault']=not diag
  with pytest.raises(ValueError):c.verify_audit(bad)
 assert (HERE/'op_host/host_common.h').read_text().count('SetShape(16,256,2560)')==1

@pytest.fixture(scope='module')
def kernel_exes(tmp_path_factory):
 root=tmp_path_factory.mktemp('fullk');inc=root/'inc';shutil.copytree(HERE.parent/'stage4j/test/cpu_include',inc)
 f=inc/'kernel_operator.h';s=f.read_text().replace('V_S,MTE2_S','V_S,MTE2_S,S_MTE3')
 s=s.replace('T GetValue(size_t i)const{Require(i<size);return data[i];}', 'T GetValue(size_t i)const{Require(i<size);return data[i];}\n void SetValue(size_t i,T v){Require(i<size);data[i]=v;}')
 s=s.replace('Require(iterations>0&&iterations%20==0&&ends==iterations&&adds==iterations);','Require(iterations>0&&ends==iterations&&adds==0&&casts==0);')
 f.write_text(s)
 (inc/'lib/matmul_intf.h').write_text(r'''
#pragma once
namespace matmul {using namespace AscendC;
template<TPosition P,CubeFormat F,class T>struct MatmulType{};
template<class A,class B,class C,class Bias>struct Matmul {
 GlobalTensor<int8_t>a,b;GlobalTensor<uint64_t>s;LocalTensor<uint8_t>w;
 bool active=false,ha=false,hb=false,hs=false,hw=false;
 void SetLocalWorkspace(LocalTensor<uint8_t>v){Require(v.size==202240);w=v;hw=true;}
 void SetTensorA(GlobalTensor<int8_t>v){a=v;ha=true;}
 void SetTensorB(GlobalTensor<int8_t>v,bool trans){Require(!trans);b=v;hb=true;}
 void SetQuantVector(GlobalTensor<uint64_t>v){s=v;hs=true;}
 bool Iterate(bool acc){Require(!acc&&!active&&ha&&hb&&hs&&hw);active=true;++iterations;return true;}
 void GetTensorC(LocalTensor<half>out,int atomic,bool seq){
  Require(active&&!atomic&&seq&&out.size==4096&&a.data==activationBase);
  size_t tile=(b.data-weightBase)/(2560*256);Require(tile<76&&s.data==scaleBase+tile*256);
  for(size_t m=0;m<16;++m)for(size_t n=0;n<256;++n){int64_t dot=0;for(size_t k=0;k<2560;++k)dot+=int64_t(a.data[m*2560+k])*int64_t(b.data[k*256+n]);
   uint32_t bits=uint32_t(s.data[n]);float scale;std::memcpy(&scale,&bits,4);out.data[(n/16)*256+m*16+n%16]=half(float(dot)*scale);}
  if(fault==1)w.data[-1]=0;
  if(fault==2)registeredWorkspace[-1]=0;
 }
 void End(){Require(active);active=false;ha=hb=hs=hw=false;++ends;}
};}
#define REGIST_MATMUL_OBJ(pipe,workspace,...) AscendC::RegisterWorkspace(workspace)
''')
 body=r'''
#include <fstream>
#include <iostream>
int main(int argc,char**argv){using namespace AscendC;blockNum=std::stoi(argv[2]);int tile=std::stoi(argv[3]),mode=std::stoi(argv[4]);fault=std::stoi(argv[5]);
 std::vector<int8_t>a(40960),b(49807360);std::vector<uint64_t>s(19456);std::vector<half>out(311296);std::vector<uint8_t>w(blockNum*5120);
 for(int m=0;m<16;++m)for(int k=0;k<2560;++k)a[m*2560+k]=(m%3-1)*(k%5-2);
 for(int t=0;t<76;++t)for(int k=0;k<2560;++k)for(int n=0;n<256;++n)b[(t*2560+k)*256+n]=(t%3-1)*(n%5-2)*(k%3-1);
 for(int n=0;n<19456;++n){float f=float(1.0/(1<<(n%4+3)));uint32_t bits;std::memcpy(&bits,&f,4);s[n]=(uint64_t(1)<<46)|bits;}
 activationBase=a.data();weightBase=b.data();scaleBase=s.data();outputBase=out.data();workspaceBase=w.data();std::memset(out.data(),127,622592);
 TestTiling td{202240,uint32_t(mode),tile,blockNum,4096,4096,5120,0};size_t total=0;
 for(blockIdx=0;blockIdx<blockNum;++blockIdx){ubBytes=adds=casts=copies=ends=iterations=0;d_flash_w8_a8_full_k_channel((uint8_t*)a.data(),(uint8_t*)b.data(),(uint8_t*)s.data(),(uint8_t*)out.data(),w.data(),(uint8_t*)&td);Require(ubBytes==212480);total+=iterations;}
 std::cout<<total<<"\n";std::ofstream(argv[1],std::ios::binary).write((char*)out.data(),622592);
}
'''
 exes={}
 for fault in (0,1):
  src=root/f'main{fault}.cpp';src.write_text('#define STAGE4V_FAULT '+str(fault)+'\n#include "'+str(HERE/'op_kernel/d_flash_w8_a8_full_k_channel.cpp')+'"\n'+body)
  exe=root/str(fault);cp=subprocess.run([shutil.which('c++'),'-std=c++17','-O2','-I'+str(inc),str(src),'-o',str(exe)],capture_output=True,text=True)
  assert cp.returncode==0,cp.stderr;exes[fault]=exe
 return exes

@pytest.mark.parametrize('blocks,tile',[(7,-1),(2,-1),(7,0),(7,37),(7,38),(7,75)])
def test_actual_kernel_fullk_mapping_rounding_and_lifecycle(kernel_exes,tmp_path,blocks,tile):
 out=tmp_path/'out';cp=subprocess.run([str(kernel_exes[0]),str(out),str(blocks),str(tile),'0','0'],check=True,capture_output=True,text=True)
 assert int(cp.stdout)==(76 if tile==-1 else 1)
 actual=np.fromfile(out,'<f2').reshape(16,19456)
 factor=sum((k%5-2)*(k%3-1) for k in range(2560))
 for t in range(76):
  if tile>=0 and tile!=t:assert np.all(actual[:,t*256:(t+1)*256].view('<u2')==0x7f7f);continue
  dot=np.array([[factor*(m%3-1)*(t%3-1)*(n%5-2) for n in range(256)] for m in range(16)],np.int64)
  scale=np.array([1/(1<<(n%4+3)) for n in range(256)],'<f4');expected=(dot.astype('<f4')*scale).astype('<f2')
  assert actual[:,t*256:(t+1)*256].tobytes()==expected.tobytes()


def test_actual_fault_sweeps_all_boundaries_and_abort_witnesses(kernel_exes,tmp_path):
 import struct
 for mode in (2,3,4,5):
  out=tmp_path/str(mode);cp=subprocess.run([str(kernel_exes[1]),str(out),'1','0',str(mode),'0'],check=True,capture_output=True,text=True,timeout=120)
  if mode==2:
   raw=out.read_bytes();assert int(cp.stdout)==1
   assert raw[:32]==struct.pack('<8I',0x53565357,3,384,32,36864,0,1,1) and raw[32:]==bytes([127])*(622592-32)
  else:
   raw=out.read_bytes();assert raw[:32]==struct.pack('<8I',0x53564641,mode-3,0x80000001,384,2,1,63,mode-2)
   assert raw[32:]==bytes([127])*(622592-32)
   assert int(cp.stdout)==(0 if mode==3 else 1)

@pytest.mark.parametrize('fault',[1,2])
def test_ub_and_arena_corruption_stops_normal_output(kernel_exes,tmp_path,fault):
 out=tmp_path/'out';subprocess.run([str(kernel_exes[0]),str(out),'1','0','0',str(fault)],check=True,capture_output=True)
 assert out.read_bytes()==bytes([127])*622592

@pytest.fixture(scope='module')
def runner(tmp_path_factory):
 root=tmp_path_factory.mktemp('runner');old=HERE.parent/'stage4k/test/N128';inc=root/'inc';shutil.copytree(old/'cpu_include',inc)
 f=inc/'aclnn_d_flash_w8_a8_n128_gate.h';(inc/'aclnn_d_flash_w8_a8_full_k_channel.h').write_text(f.read_text().replace('DFlashW8A8N128Gate','DFlashW8A8FullKChannel'));f.unlink()
 s=(old/'cpu_api.cpp').read_text().replace('DFlashW8A8N128Gate','DFlashW8A8FullKChannel').replace('d_flash_w8_a8_n128_gate','d_flash_w8_a8_full_k_channel').replace('STAGE4J','STAGE4V')
 s=s.replace('static aclTensor* output=nullptr;', 'static aclTensor* input=nullptr;static aclTensor* output=nullptr;')
 s=s.replace('return new aclTensor{p};','auto*t=new aclTensor{p};if(!input)input=t;return t;')
 s=s.replace('std::memset(output->data,0,622592);','std::memset(output->data,0,622592);if(std::getenv("TEST_OUTPUT_GUARD"))((unsigned char*)output->data)[622592]^=1;if(std::getenv("TEST_READONLY"))((unsigned char*)input->data)[0]^=1;')
 (root/'api.cpp').write_text(s)
 exe=root/'runner';cp=subprocess.run([shutil.which('c++'),'-std=c++17','-O1','-I'+str(inc),'-I'+str(c.common.QUANT/'test'),str(HERE/'runner/runner.cpp'),str(root/'api.cpp'),'-o',str(exe)],capture_output=True,text=True)
 assert cp.returncode==0,cp.stderr;return exe

@pytest.mark.parametrize('fault',[None,'TEST_BAD_GOLDEN','TEST_TIMED_LOG','TEST_OUTPUT_GUARD','TEST_READONLY'])
def test_runner_in_process_golden_timing_and_protocol(runner,tmp_path,fault):
 for name,size in [('a_i8.bin',40960),('b_i8.bin',49807360),('quant_u64_doc_candidate.bin',155648),('expected-0.bin',622592),('expected-1.bin',622592)]:
  with (tmp_path/name).open('wb') as f:f.truncate(size)
 env=os.environ.copy();env['DFLASH_STAGE4V_EXPECTED_CORE_POLICY']='8,8,7,7,2002'
 if fault:env[fault]='1'
 with (tmp_path/'runner.log').open('w') as log:cp=subprocess.run([str(runner),'0',str(tmp_path),'0','0','-1','--benchmark'],env=env,stdout=log,stderr=subprocess.STDOUT)
 ex=json.loads((tmp_path/'execution.json').read_text())
 if fault:
  assert cp.returncode!=0 and (tmp_path/'actual-0.bin').stat().st_size==622592
  return
 assert cp.returncode==0,(tmp_path/'runner.log').read_text()
 assert len(proto.check_execution(ex,0,0,0,-1,True)['execute_sync']['samples_ms'])==30
 assert ex['group_count']==1 and ex['guard_coverage']==c.COVERAGE

@pytest.mark.parametrize('stage',tuple(run.STATES))
def test_cli_and_prior_evidence_fail_closed(stage,tmp_path):
 args=[stage,'--output-dir',str(tmp_path/'out')]
 if stage!='prepare':
  with pytest.raises(SystemExit):run.parse(args)
  args+=['--prepare-report',str(tmp_path/'p')]
 if stage not in ('prepare','host'):args+=['--host-report',str(tmp_path/'h')]
 if stage not in ('prepare','host','build'):args+=['--build-report',str(tmp_path/'b')]
 if stage in ('full','fault','benchmark','profile'):args+=['--prior-report',str(tmp_path/'prior')]
 assert run.parse(args).stage==stage
 with pytest.raises(ValueError):run.prior(None,'full',{})


def test_performance_gate_never_sets_quality_go():
 rows=[dict(round=i,position=j,scheme=n,profiled=False,workspace_bytes=1,timing={k:c.q.statistics_ms([2. if n=='Group128' else 1.]*30,30) for k in ('execute_sync','prepare')}) for i,o in enumerate(run.ORDER) for j,n in enumerate(o)]
 r=run.comparison(rows);assert r['performance_decision']=='GO_FURTHER_REVIEW' and r['quality_go']=='UNRESOLVED'
 rows[0]['profiled']=True
 with pytest.raises(ValueError):run.comparison(rows)


def test_complete_prepare_and_tamper_rejection(tmp_path,monkeypatch):
 # Synthetic frozen capture, real full-shape quantization/INT64/FP64/Packing functions.
 # Only historical filesystem loading is substituted; no hardware proof is manufactured.
 k=2560;n=19456
 a=np.broadcast_to((np.arange(k)%5-2).astype('i1'),(16,k)).copy();sa=.125;x=(a.astype('<f4')*sa).astype('<f2')
 olda=a.reshape(16,20,128).transpose(1,0,2).copy()
 nk=np.broadcast_to((np.arange(n)%31-15).astype('i1')[:,None],(n,k))
 sw=np.broadcast_to((np.arange(20)+1).astype('<f2')[:,None]/np.float16(16),(20,n)).copy()
 nz=nk.reshape(1216,16,80,32).transpose(2,0,1,3)
 src=dict(a=olda,sa=sa,x=x,q=nk,nz=nz,sw=sw)
 def save(name,value):
  f=tmp_path/name;value.tofile(f);return c.identity(f)
 aid=save('old-a.bin',olda);xid=save('x.bin',x)
 w=np.memmap(tmp_path/'w.bin',mode='w+',dtype='<f2',shape=(k,n));f64=np.empty((16,n),'<f8')
 for t in range(76):
  lo=t*256;hi=lo+256;block=(nk[lo:hi].astype(np.float64)*np.repeat(sw[:,lo:hi].T,128,axis=1).astype(np.float64)).astype('<f2')
  w[:,lo:hi]=block.T;f64[:,lo:hi]=ref.reference_block(x,block.T.copy())
 w.flush();fid=save('original-ref.bin',f64);gold=save('old-golden.bin',f64.astype('<f2'))
 native=tmp_path/'native.json';c.write(native,dict(files={'x.bin':xid,'w_fp16_kn.bin':c.identity(tmp_path/'w.bin'),'reference_f64.bin':fid}))
 bc=tmp_path/'b-case.json';c.write(bc,dict(source=dict(source=dict(frozen={'x.bin':xid},activation=dict(scale=sa)))))
 pc=tmp_path/'p-case.json';c.write(pc,dict(stage4b_case=c.identity(bc),files={'a_i8.bin':aid,'golden_fp16.bin':gold}))
 proof=dict(baseline=dict(case=c.identity(pc)),native_reference=c.identity(native))
 monkeypatch.setattr(c.p.reference,'load_source',lambda _:src)
 case=ref.prepare(tmp_path/'new',proof);maps=ref.verify_case(case,numerical=True)
 assert case['files']['a_i8.bin']['sha256']==__import__('hashlib').sha256(a.tobytes()).hexdigest()
 assert len(case['tiles'])==76 and case['files']['quant_u64_doc_candidate.bin']['bytes']==155648
 assert case['files']['golden_fp16.bin']!=gold and case['activation_requantized'] is False
 quality=c.read(c.verify(case['quality']));assert quality['status']=='QUALITY_GO_UNRESOLVED' and len(quality['channels'])==19456
 for t in (0,37,38,75):
  dot,y=ref.compute(a,maps['b_i8.bin'][t],maps['quant_u64_doc_candidate.bin'][t])
  assert y.tobytes()==maps['golden_fp16.bin'][:,t*256:(t+1)*256].copy().tobytes()
 bad=copy.deepcopy(case);bad['tiles'][0]['B_sha256']='0'*64
 with pytest.raises(ValueError):ref.verify_case(bad)
 bad=copy.deepcopy(case);bad['files']['golden_fp16.bin']=gold
 with pytest.raises(ValueError):ref.verify_case(bad)


def test_formal_plan_binding_not_manager_eight_core():
 platform,row=fixture_tiling();t1=dict(fields={k:row['fields'][k] for k in c.p.FIELDS},capacity=platform['capacity'],system_workspace_bytes=4096,scratch=202240)
 policy=dict(aic=8,aiv=8,all=7,selected=7,arch=2002)
 text='STAGE4V_CORE_POLICY aic=8 aiv=8 all=7 selected=7\nSTAGE4V_ARCH_PROBE arch=2002\nSTAGE4V_SCOPE tile=-1\n'
 text+=''.join(f'STAGE4V_T1 {k}={v}\n' for k,v in t1['fields'].items())
 text+='STAGE4V_T1_CONTRACT match=1 effectiveSharedUb=0\nSTAGE4V_SCHEDULE coreLimit=0 blocks=7 cores=7 system=4096 arena=4096 stride=5120 user=35840 total=39936 mode=0\n'
 pl=proto.plan(text,0,0,-1,t1,policy);assert sorted(t for row in pl['owners'] for t in row)==list(range(76))
 for bad in (text.replace('selected=7','selected=8'),text.replace('stride=5120','stride=4096'),text.replace('singleCoreK=2560','singleCoreK=128')):
  with pytest.raises(ValueError):proto.plan(bad,0,0,-1,t1,policy)


def test_failure_keeps_first_difference_even_when_runner_rejects(tmp_path,monkeypatch):
 from types import SimpleNamespace
 files={}
 for name in ('a_i8.bin','b_i8.bin','quant_u64_doc_candidate.bin'):
  f=tmp_path/name;f.write_bytes(b'x');files[name]=c.identity(f)
 case=tmp_path/'case.json';c.write(case,dict(files=files))
 runner=tmp_path/'runner';runner.write_bytes(b'stub')
 maps={'golden_fp16.bin':np.zeros((16,19456),'<f2'),'dot_i32.bin':np.zeros((16,19456),'<i4'),'effective_f32.bin':np.ones(19456,'<f4'),'quant_u64_doc_candidate.bin':np.full((76,256),1<<46,'<u8')}
 monkeypatch.setattr(run,'verify_case',lambda _:maps)
 def fail(argv,log,*args):
  folder=log.parent;out=maps['golden_fp16.bin'].copy();out[0,5]=1;out.tofile(folder/'actual-0.bin');log.write_text('strict Golden mismatch')
  raise RuntimeError('Runner rejected before timing')
 monkeypatch.setattr(run,'command',fail)
 r=dict(case=c.identity(case),device=0,evidence=dict(formal_core_policy=dict(aic=8,aiv=8,all=7,selected=7,arch=2002)))
 b=dict(variants=dict(normal=dict(build=dict(opp=str(tmp_path/'opp'),runner=c.identity(runner)))))
 args=SimpleNamespace(device=0,cann_root=tmp_path/'sdk')
 with pytest.raises(RuntimeError):run.invoke(args,tmp_path/'execute',r,b,0,0,-1)
 d=c.read(tmp_path/'execute/failure-diagnostics.json')
 assert d['outputs'][0]['bit_mismatches']==1 and d['outputs'][0]['first']['channel']==5
 with pytest.raises(ValueError):run.invoke(args,tmp_path/'forbidden',r,b,0,0,-1,timed=True,fault=True)


def test_materiality_and_each_round_rule():
 rows=[dict(round=i,position=j,scheme=n,profiled=False,workspace_bytes=1,timing={k:c.q.statistics_ms([1.02 if n=='Group128' else 1.]*30,30) for k in ('execute_sync','prepare')}) for i,o in enumerate(run.ORDER) for j,n in enumerate(o)]
 assert run.comparison(rows)['performance_decision']=='NO_GO_NO_MATERIAL_IMPROVEMENT'
 for row in rows:
  if row['scheme']=='Group128':row['timing']['execute_sync']=c.q.statistics_ms([2 if row['round']<2 else .9]*30,30)
 assert run.comparison(rows)['performance_decision']=='NO_GO_NO_MATERIAL_IMPROVEMENT'
