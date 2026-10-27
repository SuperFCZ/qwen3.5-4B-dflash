"""CPU-only layout/ownership/reference tests. No CANN/NPU evidence."""
import copy,json,shutil,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import tile_common as c
import run_stage4a as run
from w8_reference import nz_pack

@pytest.fixture(scope='module')
def frozen(tmp_path_factory):
 root=tmp_path_factory.mktemp('stage4a-frozen');rng=np.random.Generator(np.random.PCG64(41027))
 x=(rng.integers(-32,33,size=(16,2560))/64).astype('<f2');x[0,-1]=3
 x.tofile(root/'x.bin');aq,sa,quant=c.activation(x);aq.tofile(root/'saved-activation.bin')
 q=np.memmap(root/'q_nk.bin',mode='w+',dtype='i1',shape=(19456,2560));nz=np.memmap(root/'w_nz.bin',mode='w+',dtype='i1',shape=(80,1216,16,32))
 sw=np.memmap(root/'s_gn.bin',mode='w+',dtype='<f2',shape=(20,19456));bs=[];sc=[]
 for slot,t in enumerate(c.TILES):
  nk=rng.integers(-17,18,size=(64,2560),dtype=np.int8);q[t*64:t*64+64]=nk
  for g in range(20):nz[g*4:g*4+4,t*4:t*4+4]=nz_pack(nk[:,g*128:(g+1)*128].T.copy())
  scale=np.array([[(slot+1)*.001*(1+g/20+n/128) for n in range(64)] for g in range(20)],'<f2')
  sw[:,t*64:t*64+64]=scale
  bs.append(nk.reshape(64,20,128).transpose(1,2,0).copy());sc.append((np.float64(sa)*scale.astype(np.float64)).astype('<f4'))
 q.flush();nz.flush();sw.flush();del q,nz,sw
 a=aq.reshape(16,20,128).transpose(1,0,2).copy();b=np.stack(bs);s=np.stack(sc);d=c.reference(a,b,s)
 inputs={}
 for n,value in [('a_i8.bin',a),('b_i8.bin',b[0]),('quant_u64_doc_candidate.bin',d['words'][0])]:
  p=root/('tile0-'+n);value.tofile(p);inputs[n]=c.identity(p)
 golden=root/'opaque-tile-zero.ref';d['out'][:,:64].tofile(golden)
 return dict(frozen={n:c.identity(root/n) for n in ('x.bin','q_nk.bin','w_nz.bin','s_gn.bin')},
  prepared={'x_full_fp16.bin':c.identity(root/'x.bin'),'activation_full_i8.bin':c.identity(root/'saved-activation.bin')},activation=quant,
  tile0=dict(inputs=inputs,golden=c.identity(golden),outputs=[c.identity(golden)]*2))

@pytest.fixture
def case(tmp_path,frozen):return c.prepare(tmp_path/'data',frozen)


def test_real_index_nz_scale_and_saved_activation(case,frozen):
 d=c.verify_case(case)
 assert [x['original_tile'] for x in case['slots']]==[0,1,15,16,151,152,302,303]
 assert case['slots'][4]['n_range']==[9664,9728] and case['slots'][5]['n_range']==[9728,9792]
 assert case['slots'][-1]['nz_n16_range']==[1212,1216]
 assert d['out'][:,:64].tobytes()==c.verify(frozen['tile0']['golden']).read_bytes()
 a=np.fromfile(case['files']['a_i8.bin']['path'],'i1').reshape(20,16,128)
 saved=np.fromfile(frozen['prepared']['activation_full_i8.bin']['path'],'i1').reshape(16,2560)
 assert a.tobytes()==saved.reshape(16,20,128).transpose(1,0,2).copy().tobytes()


@pytest.mark.parametrize('cores,limit,expected',[(1,1,1),(2,0,2),(3,0,3),(7,0,7),(16,0,8),(16,2,2)])
def test_core_count_and_disjoint_ownership(cores,limit,expected):
 blocks,owners=c.schedule(cores,limit);assert blocks==expected
 assert sorted(x for row in owners for x in row)==list(range(8))
 with pytest.raises(ValueError):c.schedule(1,2)
 with pytest.raises(ValueError):c.schedule(0,0)


def plantext(cores,limit,mode):
 blocks,_=c.schedule(cores,limit);system=4096;arena=4096;stride=5120;user=blocks*stride
 aic=8 if cores==7 else cores
 return f'STAGE4A_CORE_PROBE aic={aic} aiv={aic} all={cores}\nSTAGE4A_ARCH_PROBE arch=2002\nSTAGE4A_CORE_POLICY aic={aic} aiv={aic} all={cores} selected={cores}\nSTAGE4A_SCHEDULE coreLimit={limit} blocks={blocks} cores={cores} system={system} arena={arena} stride={stride} user={user} total={system+user} mode={mode}\n'+'scratch=239104 userUB=12288 ubGuards=2560 reserve=8192 single=16/64/128 base=16/64/128'


def test_workspace_plan_rejects_shared_or_undersized_arenas():
 p=c.plan_from_log(plantext(8,0,0),0,0)
 for i in range(7):assert p['arena_intervals'][i][1]<p['arena_intervals'][i+1][0]
 for old,new in [('stride=5120','stride=4096'),('arena=4096','arena=512'),('blocks=8','blocks=9'),('total=45056','total=40960')]:
  with pytest.raises(ValueError):c.plan_from_log(plantext(8,0,0).replace(old,new),0,0)


def test_bad_offsets_scale_broadcast_missing_slot_and_carry_are_detected(case):
 d=c.verify_case(case);ref=d['out'];a=np.fromfile(case['files']['a_i8.bin']['path'],'i1').reshape(20,16,128)
 b=np.fromfile(case['files']['b_i8.bin']['path'],'i1').reshape(8,20,128,64);s=np.fromfile(case['files']['scale_f32.bin']['path'],'<f4').reshape(8,20,64)
 wrong=[]
 wrong.append(c.reference(a,b,np.broadcast_to(s[0],s.shape).copy())['out'])
 wrong.append(c.reference(a,np.roll(b,1,axis=0),s)['out'])
 sg=s.copy();sg[:,19]=sg[:,18];wrong.append(c.reference(a,b,sg)['out'])
 gate=ref.copy();gate[:,256:320],gate[:,320:384]=ref[:,320:384],ref[:,256:320];wrong.append(gate)
 missing=ref.copy();missing[:,64:128].view('<u2')[:]=0x7f7f;wrong.append(missing)
 flat=np.full(8192,0x7f7f,dtype='<u2')
 for slot in range(8):
  for row in range(16):flat[row*64+slot*64:row*64+slot*64+64]=ref[row,slot*64:(slot+1)*64].view('<u2')
 wrong.append(flat.view('<f2').reshape(16,512))
 wrong.append(c.compact(np.cumsum(d['total'],axis=0,dtype=np.float32).astype('<f2')))
 for bad in wrong:assert c.compare(case,d,0,1,[bad.tobytes()]*2)['status']=='FAIL_NUMERICAL'
 assert c.compare(case,d,0,1,[ref.tobytes()]*2)['status']=='BITWISE_MATCH'


def test_tamper_and_tile_identity_rebinding_fail(case):
 bad=copy.deepcopy(case);bad['slots'][5]['original_tile']=151
 with pytest.raises(ValueError,match='identity'):c.verify_case(bad)
 p=Path(case['files']['b_i8.bin']['path']);raw=bytearray(p.read_bytes());raw[-1]^=1;p.write_bytes(raw)
 with pytest.raises(ValueError,match='identity'):c.verify_case(case)
 case['files']['b_i8.bin']=c.identity(p)
 with pytest.raises(ValueError,match='mismatch'):c.verify_case(case)


def execution(mode,limit):
 return dict(status='EXECUTION_COMPLETED_NUMERICS_UNCHECKED',op=c.NAME,soc='Ascend310P3',shape=[16,2560,512],tile_count=8,group_count=20,
  output_mode=mode,core_limit=limit,resource_admission='BEFORE_EXECUTION',ub_guard_protocol='UB_AND_CORE_ARENA_POISON_ON_FAILURE',guards=True,readonly=True,
  repeat_equal=True,repetitions=2,device=0,cpu_fallback=False,layout='nd',input_count=3,poison_verified=True,poison_patterns=[127,255],
  timing=dict(status='NOT_RUN',warmup=0,repetitions=0),workspace_bytes=60000)


@pytest.mark.parametrize('cores,fail',[ (8,False),(7,False),(1,False),(8,True)])
def test_serial_parallel_driver_gate(case,tmp_path,cores,fail):
 d=c.verify_case(case);runner=tmp_path/'runner';runner.write_text('fixture');bp=tmp_path/'build.json';root=tmp_path/'smoke';root.mkdir()
 r=dict(settings={},sources=[],evidence={},device=0)
 c.write(bp,dict(status='STAGE4A_BUILT_NPU_NOT_RUN',**r,runner=c.identity(runner),opp=str(tmp_path/'opp'),installed={},cann_compile='fixture',installation='fixture'))
 calls=[]
 def command(argv,log,*rest):
  mode=int(argv[-2]);limit=int(argv[-1]);calls.append((limit,mode));blocks,_=c.schedule(cores,limit)
  c.write(log.parent/'execution.json',execution(mode,limit));log.write_text(plantext(cores,limit,mode))
  plan=c.plan_from_log(log.read_text(),mode,limit)
  c.write(log.parent/'resource-admission.json',run.admission_expected(execution(mode,limit),plan,False))
  data=c.target(d,mode,blocks).tobytes()
  if fail and limit==2:data=bytes(16384)
  for i in range(2):(log.parent/f'actual-{i}.bin').write_bytes(data)
 hp=tmp_path/'host-proof.json';c.write(hp,{})
 policy=c.plan_from_log(plantext(cores,1,21),21,1)['core_policy']
 args=SimpleNamespace(multi_build_report=bp,host_probe_report=hp,device=0,cann_root=tmp_path/'sdk',diagnostic_groups=[])
 with patch.object(run,'installed',return_value={}),patch.object(run,'command',side_effect=command),patch.object(run,'validate_host_probe',return_value=policy):
  if cores==1:
   with pytest.raises(ValueError,match='BLOCKED_NO_MULTICORE'):run.smoke(args,root,r,case)
   assert calls==[(1,21),(1,0)]
  elif fail:
   with pytest.raises(ValueError,match='stop without fallback'):run.smoke(args,root,r,case)
   assert calls==[(1,21),(1,0),(2,21)]
  else:
   run.smoke(args,root,r,case);assert r['status']=='STAGE4A_MULTI_TILE_MATCH'
   assert calls==[(1,21),(1,0),(2,21),(2,0),(0,21),(0,0)]


def test_kernel_cpu_multicore_slots_and_workspace(case,tmp_path):
 compiler=shutil.which('c++')
 if not compiler:pytest.skip('no C++ compiler')
 src=tmp_path/'main.cpp';src.write_text('#include <fstream>\n#include "'+str(HERE/'op_kernel/d_flash_w8_a8_multi_tile.cpp')+'"\n'+r'''
int main(int argc,char**argv){
 using namespace AscendC;std::vector<int8_t>a(40960),b(1310720);std::vector<uint64_t>s(10240);std::vector<half>o(8192);
 std::ifstream(argv[1],std::ios::binary).read((char*)a.data(),40960);
 std::ifstream(argv[2],std::ios::binary).read((char*)b.data(),1310720);
 std::ifstream(argv[3],std::ios::binary).read((char*)s.data(),81920);
 std::memset(o.data(),0x7f,16384);outputBase=o.data();weightBase=b.data();scaleBase=s.data();
 blockNum=std::stoi(argv[5]);uint32_t mode=std::stoi(argv[6]);fault=std::stoi(argv[7]);std::vector<uint8_t>w(blockNum*5120);workspaceBase=w.data();
 TestTiling td{239104,mode,blockNum,4096,4096,5120,0};
 for(blockIdx=0;blockIdx<blockNum;++blockIdx){
  ubBytes=adds=casts=copies=ends=iterations=0;
  d_flash_w8_a8_multi_tile((uint8_t*)a.data(),(uint8_t*)b.data(),(uint8_t*)s.data(),(uint8_t*)o.data(),w.data(),(uint8_t*)&td);
  Require(ubBytes==253952);
  if(fault!=6){size_t count=(7-blockIdx)/blockNum+1;Require(iterations==count*20&&ends==iterations&&adds==iterations&&copies==count*64);}
  else Require(copies==0);
 }
 if(fault!=6)for(auto v:writes)Require(v==1);
 std::ofstream(argv[4],std::ios::binary).write((char*)o.data(),16384);
}
''')
 exe=tmp_path/'model';r=subprocess.run([compiler,'-std=c++17','-O1','-I'+str(HERE/'test/cpu_include'),str(src),'-o',str(exe)],capture_output=True,text=True)
 assert r.returncode==0,r.stderr
 d=c.verify_case(case)
 def invoke(blocks,mode,fault=0):
  out=tmp_path/'out.bin';argv=[str(exe),*[case['files'][n]['path'] for n in ('a_i8.bin','b_i8.bin','quant_u64_doc_candidate.bin')],str(out),str(blocks),str(mode),str(fault)]
  subprocess.run(argv,check=True);return out.read_bytes()
 for blocks in (1,2,3,7,8):
  for mode in (0,21,1,20):assert invoke(blocks,mode)==c.target(d,mode,blocks).tobytes()
 for fault in (1,4,5,6):
  data=invoke(2,0,fault);assert c.compare(case,d,0,2,[data]*2)['status']=='FAIL_NUMERICAL'

def test_runner_cpp_syntax_only(tmp_path):
 compiler=shutil.which('c++')
 if not compiler:pytest.skip('no local compiler')
 (tmp_path/'acl').mkdir();(tmp_path/'acl/acl.h').write_text('''#pragma once
#include <cstddef>
#include <cstdint>
using aclrtStream=void*;using aclDataType=int;using aclFormat=int;
constexpr int ACL_SUCCESS=0,ACL_MEM_MALLOC_HUGE_FIRST=0,ACL_MEMCPY_HOST_TO_DEVICE=1,ACL_MEMCPY_DEVICE_TO_HOST=2,ACL_FLOAT16=1,ACL_INT8=2,ACL_UINT64=4,ACL_FORMAT_ND=2;
const char* aclGetRecentErrMsg();const char* aclrtGetSocName();
int aclrtMalloc(void**,size_t,int);int aclrtFree(void*);int aclrtMemset(void*,size_t,int,size_t);int aclrtMemcpy(void*,size_t,const void*,size_t,int);
int aclInit(const char*);int aclFinalize();int aclrtSetDevice(int);int aclrtResetDevice(int);int aclrtCreateStream(aclrtStream*);int aclrtDestroyStream(aclrtStream);int aclrtSynchronizeStream(aclrtStream);
''')
 (tmp_path/f'aclnn_{c.SNAKE}.h').write_text('''#pragma once
struct aclTensor;struct aclOpExecutor;
aclTensor* aclCreateTensor(const int64_t*,uint64_t,int,const int64_t*,int64_t,int,const int64_t*,uint64_t,void*);
int aclDestroyTensor(const aclTensor*);
int aclnnDFlashW8A8MultiTileGetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,int64_t,int64_t,aclTensor*,uint64_t*,aclOpExecutor**);
int aclnnDFlashW8A8MultiTile(void*,uint64_t,aclOpExecutor*,aclrtStream);
''')
 result=subprocess.run([compiler,'-std=c++17','-fsyntax-only','-I'+str(tmp_path),'-I'+str(c.common.QUANT/'test'),str(HERE/'test/runner.cpp')],capture_output=True,text=True)
 assert result.returncode==0,result.stderr



def test_report_pass_cannot_hide_failed_numeric_comparison(tmp_path):
 root=tmp_path/'old';(root/'smoke').mkdir(parents=True)
 case=root/'case.json';c.write(case,dict(name='real_20_groups'))
 ex=root/'execution.json';c.write(ex,{})
 out=root/'out.bin';out.write_bytes(bytes(2048))
 modes=[dict(mode=m,status='BITWISE_MATCH',execution=c.identity(ex),outputs=[c.identity(out)]*2,comparison=dict(status='FAIL_NUMERICAL')) for m in c.prev.MODES]
 rows=[dict(name=n,status='PASS',case=c.identity(case),modes=modes) for n in c.prev.CASE_NAMES]
 bp=root/'build.json';base=dict(settings=c.prev.SETTINGS,sources=[],evidence={},device=0)
 c.write(bp,dict(status='STAGE3B4_BUILT_NPU_NOT_RUN',**base))
 c.write(root/'smoke/report.json',dict(status='STAGE3B4_20_GROUP_FUSED_MATCH',phase='complete',performance='NOT_RUN',**base,cases=rows,fused_build_report=c.identity(bp)))
 with patch.object(c.prev,'evidence',return_value={}),patch.object(c.prev,'sources',return_value=[]),patch.object(c.prev,'check_execution'),patch.object(c.prev,'compare',return_value=dict(status='FAIL_NUMERICAL')):
  with pytest.raises(ValueError,match='numerical proof'):c.evidence(SimpleNamespace(stage3b4_root=root,device=0))


def test_actual_cpp_resource_admission_precedes_device_call(tmp_path):
 compiler=shutil.which('c++')
 if not compiler:pytest.skip('no compiler')
 source=(HERE/'test/runner.cpp').read_text();helper=source[source.index('static void AdmitWorkspace'):source.index('struct Tensor')]
 cpp=tmp_path/'admit.cpp';cpp.write_text('#include <cstdlib>\n#include <algorithm>\n#include <cstdint>\n#include <cstdio>\n#include <fstream>\n#include <iterator>\n#include <regex>\n#include <stdexcept>\n#include <string>\n#include <vector>\n'+helper+'\nint main(int argc,char**argv){try{AdmitWorkspace(argv[1],std::stoull(argv[2]),0,1,argc==3);return 0;}catch(...){return 2;}}\n')
 exe=tmp_path/'admit';r=subprocess.run([compiler,'-std=c++17',str(cpp),'-o',str(exe)],capture_output=True,text=True);assert r.returncode==0,r.stderr
 for name,log,size,ok in [('valid',plantext(8,1,0),9216,True),('undersized',plantext(8,1,0),9215,False),('missing','',9216,False),('overlap',plantext(8,1,0).replace('stride=5120','stride=4096'),9216,False),('drift',plantext(8,1,0)+'\n'+plantext(4,1,0),9216,False)]:
  folder=tmp_path/name;folder.mkdir();(folder/'runner.log').write_text(log)
  p=subprocess.run([str(exe),str(folder),str(size)])
  assert (p.returncode==0)==ok and (folder/'resource-admission.json').exists()==ok


def test_cpp_admission_binds_real_8_8_7_policy_before_execution(tmp_path):
 import os
 compiler=shutil.which('c++')
 if not compiler:pytest.skip('no compiler')
 source=(HERE/'test/runner.cpp').read_text();helper=source[source.index('static void AdmitWorkspace'):source.index('struct Tensor')]
 cpp=tmp_path/'admit.cpp';cpp.write_text('#include <cstdlib>\n#include <algorithm>\n#include <cstdint>\n#include <cstdio>\n#include <fstream>\n#include <iterator>\n#include <regex>\n#include <stdexcept>\n#include <string>\n#include <vector>\n'+helper+'\nint main(int argc,char**argv){try{AdmitWorkspace(argv[1],10000,0,1,false);return 0;}catch(...){return 2;}}\n')
 exe=tmp_path/'admit';r=subprocess.run([compiler,'-std=c++17',str(cpp),'-o',str(exe)],capture_output=True,text=True);assert r.returncode==0,r.stderr
 for expected,ok in [(None,False),('8,8,7,7,2002',True),('8,8,8,8,2002',False)]:
  folder=tmp_path/str(expected);folder.mkdir();(folder/'runner.log').write_text(plantext(7,1,0))
  env=os.environ.copy();env.pop('DFLASH_STAGE4A_EXPECTED_CORE_POLICY',None)
  if expected:env['DFLASH_STAGE4A_EXPECTED_CORE_POLICY']=expected
  p=subprocess.run([str(exe),str(folder)],env=env)
  assert (p.returncode==0)==ok
  if ok:assert json.loads((folder/'resource-admission.json').read_text())['core_policy']==dict(aic=8,aiv=8,all=7,selected=7,arch=2002)
