"""CPU-only tests. Cached Matmul partials test full-kernel control/coverage, not hardware."""
import copy,json,shutil,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import full_common as c
import full_reference as ref
import run_stage4b as run
from w8_reference import activation,nz_pack

@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
 root=tmp_path_factory.mktemp('full-gate-cpu');frozen=root/'frozen';frozen.mkdir()
 x=((np.arange(16*2560)%65-32)/64).astype('<f2').reshape(16,2560);x[0,-1]=3
 x.tofile(frozen/'x.bin');aq,sa,quant=activation(x);aq.tofile(frozen/'activation.bin')
 q=np.memmap(frozen/'q_nk.bin',mode='w+',dtype='i1',shape=(19456,2560));nz=np.memmap(frozen/'w_nz.bin',mode='w+',dtype='i1',shape=(80,1216,16,32))
 sw=np.memmap(frozen/'s_gn.bin',mode='w+',dtype='<f2',shape=(20,19456));a=aq.reshape(16,20,128).transpose(1,0,2).copy()
 oldb=[];olds=[]
 for t in range(304):
  nk=((np.arange(64)[:,None]*11+np.arange(2560)[None,:]*7+t*13)%31-15).astype('i1');q[t*64:(t+1)*64]=nk
  for g in range(20):nz[g*4:g*4+4,t*4:t*4+4]=nz_pack(nk[:,g*128:(g+1)*128].T.copy())
  scale=np.array([[(.001+t*.000001)*(1+g/20+n/128) for n in range(64)] for g in range(20)],'<f2');sw[:,t*64:(t+1)*64]=scale
  if t in c.HISTORY:
   oldb.append(nk.reshape(64,20,128).transpose(1,2,0).copy());olds.append((np.float64(sa)*scale.astype(np.float64)).astype('<f4'))
 q.flush();nz.flush();sw.flush();del q,nz,sw
 source=dict(frozen={n:c.identity(frozen/n) for n in ('x.bin','q_nk.bin','w_nz.bin','s_gn.bin')},
  prepared={'x_full_fp16.bin':c.identity(frozen/'x.bin'),'activation_full_i8.bin':c.identity(frozen/'activation.bin')},activation=quant)
 old=c.old.reference(a,np.stack(oldb),np.stack(olds));history=root/'old-eight';history.mkdir();files={}
 for n,v in [('a_i8.bin',a),('b_i8.bin',np.stack(oldb)),('quant_u64_doc_candidate.bin',old['words']),('partials_fp16.bin',old['parts']),('golden_fp16.bin',old['out'])]:
  v.tofile(history/n);files[n]=c.identity(history/n)
 proof=dict(source=source,history=dict(files=files,tiles=c.HISTORY,npu_outputs=[files['golden_fp16.bin']]*2),native_comparison=dict(status='SKIPPED',reason='CPU fixture only'))
 case=ref.prepare(root/'data',proof)
 return root,proof,case


def test_full_preparation_and_eight_historical_bindings(prepared):
 root,proof,case=prepared;maps=ref.verify_case(case)
 assert len(case['tiles'])==304 and sum(r['nz_groups'] for r in case['tiles'])==6080
 assert case['files']['b_i8.bin']['bytes']==49807360 and case['files']['quant_u64_doc_candidate.bin']['bytes']==3112960
 assert case['files']['golden_fp16.bin']['bytes']==622592
 assert maps['a_i8.bin'].tobytes()==c.verify(proof['history']['files']['a_i8.bin']).read_bytes()
 assert [r['tile'] for r in case['tiles'] if r['historical_stage4a']=='BITWISE_MATCH']==c.HISTORY
 assert all(v['status']=='COMPLETED' for v in case['w8a16_reference']['accuracy'].values())
 assert case['native_comparison']['status']=='SKIPPED'
 assert json.loads((root/'data/progress.json').read_text())==dict(status='COMPLETE',completed_tiles=304,nz_groups=6080)


def test_group_reference_failure_coordinates(prepared):
 _,proof,_=prepared;src=ref.load_source(proof)
 bad=dict(src);bad['sw']=np.array(src['sw']);bad['sw'][10,152*64+3]=np.inf
 with pytest.raises(ref.ReferenceFailure) as ex:ref.compute_tile(bad,152)
 assert ex.value.detail==dict(kind='WEIGHT_SCALE_NONFINITE',tile=152,group=10,row=None,channel=9731,k=None)
 bad=dict(src);bad['nz']=np.array(src['nz']);bad['nz'][79,1215,15,31]^=1
 with pytest.raises(ref.ReferenceFailure) as ex:ref.compute_tile(bad,303)
 assert ex.value.detail['kind']=='NZ_INVERSE_MISMATCH' and ex.value.detail['channel']==19455 and ex.value.detail['k']==2559


def test_case_incomplete_tamper_and_rebinding(prepared,tmp_path):
 _,_,case=prepared;bad=copy.deepcopy(case);bad['tiles'].pop()
 with pytest.raises(ValueError,match='304'):ref.verify_case(bad)
 bad=copy.deepcopy(case);p=tmp_path/'changed.bin';raw=bytearray(c.verify(case['files']['golden_fp16.bin']).read_bytes());raw[-2]^=1;p.write_bytes(raw)
 bad['files']['golden_fp16.bin']=c.identity(p)
 with pytest.raises(ValueError,match='digest'):ref.verify_case(bad,numerical=False)


@pytest.mark.parametrize('cores,limit,blocks',[(7,0,7),(8,0,8),(304,0,304),(7,1,1),(7,2,2)])
def test_all_304_tiles_have_one_owner(cores,limit,blocks):
 b,owners=c.schedule(cores,limit);assert b==blocks
 assert sorted(x for row in owners for x in row)==list(range(304))
 if blocks==7:assert [len(x) for x in owners]==[44,44,44,43,43,43,43]
 with pytest.raises(ValueError):c.schedule(7,8)


def test_strict_full_and_historical_output_checks(prepared):
 _,_,case=prepared;maps=ref.verify_case(case,numerical=False);out=maps['golden_fp16.bin'].tobytes()
 r=ref.compare(case,maps,0,7,[out]*2)
 assert r['status']=='BITWISE_MATCH' and r['total_bit_mismatches']==[0,0] and r['total_elements']==311296
 assert [x['tile'] for x in r['historical_stage4a']]==c.HISTORY
 bad=bytearray(out);bad[-1]^=1
 assert ref.compare(case,maps,0,7,[out,bytes(bad)])['status']=='FAIL_NUMERICAL'
 missing=np.frombuffer(out,'<f2').copy().reshape(16,19456);missing[:,151*64:152*64].view('<u2')[:]=0x7f7f
 assert ref.compare(case,maps,0,7,[missing.tobytes()]*2)['status']=='FAIL_NUMERICAL'


def test_prepare_failure_prevents_build_without_references(tmp_path):
 with pytest.raises(ValueError,match='required'):run.prepared(None,{})
 p=tmp_path/'prepare.json';c.write(p,dict(status='FAILED',phase='cpu_304_tile_preparation'))
 with pytest.raises(ValueError,match='mismatch'):run.prepared(p,dict(settings={},sources=[],evidence={},device=0))


def test_kernel_same_numeric_and_workspace_sequence():
 old=(HERE.parent/'stage4a/op_kernel/d_flash_w8_a8_multi_tile.cpp').read_text().splitlines()[1:]
 new=(HERE/'op_kernel/d_flash_w8_a8_full_gate.cpp').read_text().splitlines()[1:]
 text='\n'.join(new).replace('d_flash_w8_a8_full_gate','d_flash_w8_a8_multi_tile').replace('Stage4B','Stage4A').replace('STAGE4B','STAGE4A').replace('S4B','S4A')
 text=text.replace('>304','>8').replace('304*20','8*20').replace('slot<304','slot<8').replace('16*19456','16*512').replace('m*19456+','m*512+')
 assert text=='\n'.join(old)
 subprocess.run(['bash','-n',str(HERE/'run.sh')],check=True)

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
int aclnnDFlashW8A8FullGateGetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,int64_t,int64_t,aclTensor*,uint64_t*,aclOpExecutor**);
int aclnnDFlashW8A8FullGate(void*,uint64_t,aclOpExecutor*,aclrtStream);
''')
 result=subprocess.run([compiler,'-std=c++17','-fsyntax-only','-I'+str(tmp_path),'-I'+str(c.common.QUANT/'test'),str(HERE/'test/runner.cpp')],capture_output=True,text=True)
 assert result.returncode==0,result.stderr




def test_actual_kernel_full_coverage_and_cross_tile_reset(prepared,tmp_path):
 _,_,case=prepared;compiler=shutil.which('c++')
 if not compiler:pytest.skip('no compiler')
 src=tmp_path/'main.cpp';src.write_text('#include <fstream>\n#include "'+str(HERE/'op_kernel/d_flash_w8_a8_full_gate.cpp')+'"\n'+r'''
int main(int argc,char**argv){
 using namespace AscendC;std::vector<int8_t>a(40960),b(49807360);std::vector<uint64_t>s(389120);std::vector<half>partials(304*20*1024),o(311296);
 std::ifstream(argv[1],std::ios::binary).read((char*)a.data(),40960);std::ifstream(argv[2],std::ios::binary).read((char*)b.data(),49807360);
 std::ifstream(argv[3],std::ios::binary).read((char*)s.data(),3112960);std::ifstream(argv[4],std::ios::binary).read((char*)partials.data(),12451840);
 std::memset(o.data(),0x7f,622592);outputBase=o.data();activationBase=a.data();weightBase=b.data();scaleBase=s.data();partialBase=partials.data();
 blockNum=std::stoi(argv[6]);uint32_t mode=std::stoi(argv[7]);fault=std::stoi(argv[8]);std::vector<uint8_t>w(blockNum*5120);workspaceBase=w.data();
 TestTiling td{239104,mode,blockNum,4096,4096,5120,0};
 for(blockIdx=0;blockIdx<blockNum;++blockIdx){
  ubBytes=adds=casts=copies=ends=iterations=0;
  d_flash_w8_a8_full_gate((uint8_t*)a.data(),(uint8_t*)b.data(),(uint8_t*)s.data(),(uint8_t*)o.data(),w.data(),(uint8_t*)&td);
  Require(ubBytes==253952);
  if(fault!=6){size_t count=(303-blockIdx)/blockNum+1;Require(iterations==count*20&&ends==iterations&&adds==iterations&&copies==count*64);}
  else Require(copies==0);
 }
 if(fault!=6)for(auto v:writes)Require(v==1);
 std::ofstream(argv[5],std::ios::binary).write((char*)o.data(),622592);
}
''')
 exe=tmp_path/'kernel-model';p=subprocess.run([compiler,'-std=c++17','-O1','-I'+str(HERE/'test/cpu_include'),str(src),'-o',str(exe)],capture_output=True,text=True)
 assert p.returncode==0,p.stderr
 maps=ref.verify_case(case,numerical=False)
 def invoke(blocks,mode,fault):
  out=tmp_path/'out.bin';argv=[str(exe),*[case['files'][n]['path'] for n in ('a_i8.bin','b_i8.bin','quant_u64_doc_candidate.bin','partials_fp16.bin')],str(out),str(blocks),str(mode),str(fault)]
  subprocess.run(argv,check=True);return out.read_bytes()
 for blocks,mode in [(1,0),(2,0),(7,0),(7,21),(7,1),(7,20)]:
  data=invoke(blocks,mode,0)
  assert ref.compare(case,maps,mode,blocks,[data]*2)['status']=='BITWISE_MATCH'
 for fault in (4,6):
  data=invoke(7,0,fault);assert ref.compare(case,maps,0,7,[data]*2)['status']=='FAIL_NUMERICAL'


def plantext(limit,mode):
 blocks,_=c.schedule(7,limit);user=blocks*5120
 return f'STAGE4B_CORE_PROBE aic=8 aiv=8 all=7\nSTAGE4B_ARCH_PROBE arch=2002\nSTAGE4B_CORE_POLICY aic=8 aiv=8 all=7 selected=7\nSTAGE4B_SCHEDULE coreLimit={limit} blocks={blocks} cores=7 system=4096 arena=4096 stride=5120 user={user} total={4096+user} mode={mode}\nscratch=239104 userUB=12288 ubGuards=2560 reserve=8192 single=16/64/128 base=16/64/128'


@pytest.mark.parametrize('fail',[False,True])
def test_smoke_auto_then_control_requires_full_outputs(prepared,tmp_path,fail):
 _,_,case=prepared;maps=ref.verify_case(case,numerical=False);root=tmp_path/'smoke';root.mkdir()
 runner=tmp_path/'runner';runner.write_text('fixture only');hp=tmp_path/'host.json';c.write(hp,{})
 prep=tmp_path/'prep.json';c.write(prep,{});ci=Path(case['files']['a_i8.bin']['path']).parent/'case.json'
 report=dict(settings={},sources=[],evidence={},device=0,prepare_report=c.identity(prep),case=c.identity(ci))
 bp=tmp_path/'build.json';c.write(bp,dict(status='STAGE4B_BUILT_NPU_NOT_RUN',**report,runner=c.identity(runner),opp=str(tmp_path/'opp'),installed={},cann_compile='fixture',installation='fixture'))
 calls=[];policy=dict(aic=8,aiv=8,all=7,selected=7,arch=2002)
 def command(argv,log,*rest):
  mode,limit=int(argv[-2]),int(argv[-1]);calls.append((limit,mode));plan=c.plan_from_log(plantext(limit,mode),mode,limit)
  ex=dict(status='EXECUTION_COMPLETED_NUMERICS_UNCHECKED',op=c.NAME,soc='Ascend310P3',shape=[16,2560,19456],tile_count=304,group_count=20,output_mode=mode,core_limit=limit,
   resource_admission='BEFORE_EXECUTION',ub_guard_protocol='UB_AND_CORE_ARENA_POISON_ON_FAILURE',guards=True,readonly=True,repeat_equal=True,repetitions=2,device=0,cpu_fallback=False,layout='nd',input_count=3,poison_verified=True,poison_patterns=[127,255],timing=dict(status='NOT_RUN',warmup=0,repetitions=0),workspace_bytes=60000)
  c.write(log.parent/'execution.json',ex);log.write_text(plantext(limit,mode));c.write(log.parent/'resource-admission.json',run.admission_expected(ex,plan,False))
  data=np.concatenate([ref.target_tile(maps,t,mode,plan['blocks']) for t in range(304)],axis=1)
  if fail and mode==0:data[:,303*64:].view('<u2')[:]=0x7f7f
  for i in range(2):data.tofile(log.parent/f'actual-{i}.bin')
 args=SimpleNamespace(multi_build_report=bp,host_probe_report=hp,device=0,cann_root=tmp_path/'sdk',diagnostic_groups=[])
 with patch.object(run,'installed',return_value={}),patch.object(run,'validate_host_probe',return_value=policy),patch.object(run,'command',side_effect=command),patch.object(run,'verify_case',return_value=maps):
  if fail:
   with pytest.raises(ValueError,match='stop without fallback'):run.smoke(args,root,report,case)
   assert calls==[(0,21),(0,0)] and report['runs'][-1]['comparison']['tiles'][-1]['status']=='FAIL_NUMERICAL'
  else:
   run.smoke(args,root,report,case);assert report['status']=='STAGE4B_FULL_GATE_MATCH'
   assert calls==[(0,21),(0,0),(2,21),(2,0)] and report['summary']['tiles']==304


def test_diagnostics_never_require_w8a16_bitwise(prepared):
 _,_,case=prepared;maps=ref.verify_case(case,numerical=False)
 report=ref.regions(maps['w8a16_cpu_fp16.bin'],maps['golden_fp16.bin'])
 assert set(report)=={'full','gate','up'} and all(x['status']=='COMPLETED' for x in report.values())
 assert any(x['relative_l2_error']>0 for x in report.values())
 expected=np.array(maps['w8a16_cpu_fp16.bin']);expected[0,0]=np.inf
 assert ref.regions(expected,maps['golden_fp16.bin'])['full']['status'].startswith('SKIPPED')
 assert ref.compare(case,maps,0,7,[maps['golden_fp16.bin'].tobytes()]*2)['status']=='BITWISE_MATCH'


def test_optional_w8a16_nonfinite_diagnostic_does_not_repair_candidate(prepared):
 _,proof,_=prepared;src=ref.load_source(proof)
 src=dict(src);src['a']=np.zeros_like(src['a']);src['sw']=np.array(src['sw']);src['sw'][0,3]=65504
 d=ref.compute_tile(src,0,True)
 assert np.isfinite(d['out']).all() and not d['out'].any()
 assert d['w8a16_error']['kind']=='W8A16_EFFECTIVE_WEIGHT_NONFINITE'
 assert d['w8a16_error']['tile']==0 and d['w8a16_error']['group']==0 and d['w8a16_error']['channel']==3
 assert not np.isfinite(d['w8a16'][:,3]).all()
 assert np.isfinite(d['w8a16'][:,4:]).all() # actual IEEE propagation, not blanket NaN replacement
