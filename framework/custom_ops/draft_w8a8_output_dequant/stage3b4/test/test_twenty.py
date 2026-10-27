"""CPU fixtures only: no test here constitutes CANN or NPU evidence."""
import copy,json,shutil,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import twenty_common as c
import run_stage3b4 as run

@pytest.fixture
def proof(tmp_path):
 a,b,s=c.synthetic('synthetic_distinct');d=c.reference(a,b,s);groups=[]
 for g in range(20):
  folder=tmp_path/f'old-group-{g}';folder.mkdir()
  arrays={'a_i8.bin':a[g],'b_i8.bin':b[g],'scale_f32.bin':s[g],'z_i32.bin':d['z'][g].astype('<i4'),
   'quant_u64_doc_candidate.bin':d['words'][g],'scale_effective_f32.bin':d['effective'][g],
   'ideal_fp16.bin':c.common.stage2a.convert(d['z'][g],s[g]),'doc_candidate_fp16.bin':d['parts'][g]}
  for n,v in arrays.items():v.tofile(folder/n)
  c.write(folder/'case.json',dict(name=f'group-{g}',files={n:c.identity(folder/n) for n in arrays}))
  outs=[]
  for i in range(2):p=folder/f'actual-{i}.bin';d['parts'][g].tofile(p);outs.append(c.identity(p))
  groups.append(dict(group_index=g,case=c.identity(folder/'case.json'),outputs=outs))
 # Deliberately opaque filenames: all consumers must follow report identities.
 half=tmp_path/'opaque-frozen-final.dat';d['out'].tofile(half)
 fp=tmp_path/'opaque-frozen-sum.dat';d['total'].tofile(fp)
 final=dict(fp16=c.identity(half),fp32=c.identity(fp),npu_outputs=[c.identity(half)]*2)
 return dict(groups=groups,final=final)

@pytest.fixture
def cases(tmp_path,proof):
 return [json.loads(c.verify(i).read_text()) for i in c.prepare(tmp_path/'data',proof)]


def test_original_bytes_and_golden_identity_are_reused(cases,proof):
 case=cases[0];c.verify_case(case)
 assert case['files']['golden_fp16.bin']['sha256']==proof['final']['fp16']['sha256']
 for n in ('a_i8.bin','b_i8.bin','quant_u64_doc_candidate.bin'):
  old=b''.join(c.verify(json.loads(c.verify(g['case']).read_text())['files'][n]).read_bytes() for g in proof['groups'])
  assert c.verify(case['files'][n]).read_bytes()==old
 assert c.SETTINGS['scratch']+12288+2560+8192==262144


def test_mutation_witnesses_cover_middle_and_last_groups(cases):
 for case in cases[1:3]:
  rows=case['controls']['groups'];assert [r['group'] for r in rows]==list(range(1,20))
  assert all(v>0 for r in rows for v in r['mismatches'].values())
  for g in (1,9,10,18,19):assert rows[g-1]['mismatches']['scale_previous']>0


def test_fp32_order_rounding_witness(cases):
 case=cases[3];d=c.verify_case(case)
 assert np.all(d['out'][:,:32]==1) and np.all(d['out'][:,32:]==np.float16(2**-12))
 assert all(case['controls'][k]==512 for k in ('fp16_accumulator','fp32_tree','reversed_order'))


def test_every_partial_mode_and_final_are_strict(cases):
 for case in cases:
  d=c.verify_case(case)
  for mode in c.MODES:
   x=d['out'] if mode==0 else d['parts'][mode-1]
   assert c.compare(case,mode,[x.tobytes()]*2)['status']=='BITWISE_MATCH'
   bad=bytearray(x.tobytes());bad[0]^=1
   assert c.compare(case,mode,[x.tobytes(),bytes(bad)])['status']=='FAIL_NUMERICAL'
 with pytest.raises(ValueError):c.compare(cases[0],21,[bytes(2048)]*2)


@pytest.mark.parametrize('name',['a_i8.bin','b_i8.bin','quant_u64_doc_candidate.bin','partials_fp16.bin','golden_fp16.bin'])
def test_tampering_or_rebinding_is_rejected(cases,name):
 case=cases[0];p=Path(case['files'][name]['path']);data=bytearray(p.read_bytes());data[-2]^=1;p.write_bytes(data)
 with pytest.raises(ValueError):c.verify_case(case)
 case['files'][name]=c.identity(p)
 with pytest.raises(ValueError):c.verify_case(case)


def test_missing_or_reordered_groups_rejected(proof,tmp_path):
 bad=copy.deepcopy(proof);bad['groups'][9],bad['groups'][10]=bad['groups'][10],bad['groups'][9]
 with pytest.raises(ValueError,match='order'):c.prepare(tmp_path/'reordered',bad)
 bad=copy.deepcopy(proof);bad['groups'].pop()
 with pytest.raises(ValueError,match='order'):c.prepare(tmp_path/'missing',bad)


def execution(mode=0):
 return dict(status='EXECUTION_COMPLETED_NUMERICS_UNCHECKED',op=c.NAME,soc='Ascend310P3',shape=[16,2560,64],group_count=20,output_mode=mode,
  ub_guard_protocol='POISON_ON_GUARD_FAILURE',guards=True,readonly=True,repeat_equal=True,repetitions=2,device=0,cpu_fallback=False,
  layout='nd',input_count=3,poison_verified=True,poison_patterns=[127,255],timing=dict(status='NOT_RUN',warmup=0,repetitions=0))


def test_safety_and_timing_gates():
 for mode in c.MODES:c.check_execution(execution(mode),0,mode)
 for k,v in [('guards',False),('readonly',False),('repeat_equal',False),('poison_verified',False),('output_mode',21),('group_count',2),('ub_guard_protocol','NONE'),('timing',dict(status='MEASURED',warmup=5,repetitions=30))]:
  x=execution();x[k]=v
  with pytest.raises(ValueError):c.check_execution(x,0,0)


@pytest.mark.parametrize('fail_mode',[None,10])
def test_driver_requires_all_84_modes_or_stops_without_fallback(cases,tmp_path,fail_mode):
 root=tmp_path/'smoke';root.mkdir();runner=tmp_path/'runner';runner.write_text('fixture only');bp=tmp_path/'build.json'
 report=dict(settings={},sources=[],evidence={},device=0)
 built=dict(status='STAGE3B4_BUILT_NPU_NOT_RUN',**report,installed={},opp=str(tmp_path/'opp'),runner=c.identity(runner),cann_compile='fixture',installation='fixture')
 c.write(bp,built);calls=[];lookup={x['name']:c.verify_case(x) for x in cases}
 def command(argv,log,*unused):
  mode=int(argv[-1]);name=log.parent.parent.name;calls.append((name,mode));c.write(log.parent/'execution.json',execution(mode))
  log.write_text(f'STAGE3B4_OUTPUT_MODE={mode} groups=20 M=16 K=2560 groupK=128 N=64 scratch=239104 userUB=12288 ubGuards=2560 reserve=8192 single=16/64/128 base=16/64/128')
  d=lookup[name];data=(d['out'] if mode==0 else d['parts'][mode-1]).tobytes()
  if mode==fail_mode:data=bytes(2048)
  for i in range(2):(log.parent/f'actual-{i}.bin').write_bytes(data)
 args=SimpleNamespace(fused_build_report=bp,device=0,cann_root=tmp_path/'sdk')
 ids=[c.identity(Path(x['files']['a_i8.bin']['path']).parent/'case.json') for x in cases]
 with patch.object(run,'installed',return_value={}),patch.object(run,'command',side_effect=command):
  if fail_mode is None:
   run.smoke(args,root,report,ids);assert report['status']=='STAGE3B4_20_GROUP_FUSED_MATCH'
   assert len(calls)==84 and all(x['checked_partials']==20 for x in report['cases'])
  else:
   with pytest.raises(ValueError,match='no next'):run.smoke(args,root,report,ids)
   assert len(calls)==10 and report['cases'][0]['modes'][9]['status']=='FAIL_NUMERICAL'
   assert all(x['status']=='NOT_RUN' for x in report['cases'][0]['modes'][10:])
   assert all(x['status']=='NOT_RUN' for x in report['cases'][1:])


def test_actual_kernel_all_modes_and_faults(cases,tmp_path):
 compiler=shutil.which('c++')
 if not compiler:pytest.skip('no C++ compiler')
 src=tmp_path/'main.cpp';src.write_text('#include <fstream>\n#include "'+str(HERE/'op_kernel/d_flash_w8_a8_twenty_group.cpp')+'"\n'+r'''
int main(int argc,char**argv){
 using namespace AscendC;
 std::vector<int8_t>a(40960),b(163840);std::vector<uint64_t>s(1280);std::vector<half>o(1024);
 std::ifstream(argv[1],std::ios::binary).read((char*)a.data(),40960);
 std::ifstream(argv[2],std::ios::binary).read((char*)b.data(),163840);
 std::ifstream(argv[3],std::ios::binary).read((char*)s.data(),10240);
 std::memset(o.data(),0x7f,2048);TestTiling td{239104,uint32_t(std::stoi(argv[5])),0};fault=std::stoi(argv[6]);targetGroup=std::stoi(argv[7]);
 d_flash_w8_a8_twenty_group((uint8_t*)a.data(),(uint8_t*)b.data(),(uint8_t*)s.data(),(uint8_t*)o.data(),nullptr,(uint8_t*)&td);
 Require(ubBytes==253952);
 if(fault!=2)Require(iterations==20&&ends==20&&adds==20&&casts==(td.outputMode==0?21:20)&&copies==64&&firstDmaAfterTwo);
 else Require(copies==0);
 std::ofstream(argv[4],std::ios::binary).write((char*)o.data(),2048);
}
''')
 exe=tmp_path/'kernel-model';done=subprocess.run([compiler,'-std=c++17','-O0','-I'+str(HERE/'test/cpu_include'),str(src),'-o',str(exe)],capture_output=True,text=True)
 assert done.returncode==0,done.stderr
 def invoke(case,mode,fault=0,target=19):
  out=tmp_path/'model-output.bin';argv=[str(exe),*[case['files'][n]['path'] for n in ('a_i8.bin','b_i8.bin','quant_u64_doc_candidate.bin')],str(out),str(mode),str(fault),str(target)]
  subprocess.run(argv,check=True);return out.read_bytes()
 for case in (cases[1],cases[3]):
  d=c.verify_case(case)
  for mode in c.MODES:
   data=invoke(case,mode);assert data==(d['out'] if mode==0 else d['parts'][mode-1]).tobytes()
 for g in (1,9,10,18,19):
  for fault in (1,2):
   data=invoke(cases[1],g+1,fault,g)
   assert c.compare(cases[1],g+1,[data]*2)['status']=='FAIL_NUMERICAL'
 for mode in (1,2,10,11,19,20):
  data=invoke(cases[1],mode,3)
  assert c.compare(cases[1],mode,[data]*2)['status']=='FAIL_NUMERICAL'

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
int aclnnDFlashW8A8TwentyGroupGetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,int64_t,aclTensor*,uint64_t*,aclOpExecutor**);
int aclnnDFlashW8A8TwentyGroup(void*,uint64_t,aclOpExecutor*,aclrtStream);
''')
 result=subprocess.run([compiler,'-std=c++17','-fsyntax-only','-I'+str(tmp_path),'-I'+str(c.common.QUANT/'test'),str(HERE/'test/runner.cpp')],capture_output=True,text=True)
 assert result.returncode==0,result.stderr


def test_prior_two_group_contract_missing_mode_cannot_pass(tmp_path,proof):
 ids=c.prior.prepare(tmp_path/'prior-data',dict(groups=proof['groups'][:2]));entries=[]
 for ci in ids:
  case=json.loads(c.verify(ci).read_text());d=c.prior.verify_case(case);modes=[]
  for mode in (1,2,0):
   folder=tmp_path/case['name']/str(mode);folder.mkdir(parents=True)
   ex=execution(mode);ex.update(op=c.prior.NAME,shape=[16,256,64],group_count=2);c.write(folder/'execution.json',ex)
   data=d['out'] if mode==0 else d['parts'][mode-1];outs=[]
   for i in range(2):p=folder/f'actual-{i}.bin';data.tofile(p);outs.append(c.identity(p))
   modes.append(dict(mode=mode,status='BITWISE_MATCH',execution=c.identity(folder/'execution.json'),outputs=outs,comparison=c.prior.compare(case,mode,[data.tobytes()]*2)))
  consistency=[c.metrics(d['out'],d['out'])]*2
  entries.append(dict(name=case['name'],case=ci,status='PASS',modes=modes,partial_final_consistency=consistency))
 x=[c.verify(e['modes'][-1]['outputs'][0]) for e in entries[1:]]
 r=dict(cases=entries,scale_swap_observation=c.metrics(np.fromfile(x[0],'<f2'),np.fromfile(x[1],'<f2')))
 c.validate_two_group(r,0)
 bad=copy.deepcopy(r);bad['cases'][0]['modes'].pop()
 with pytest.raises(ValueError,match='incomplete'):c.validate_two_group(bad,0)
 bad=copy.deepcopy(r);bad['cases'][1]['modes'][1]['comparison']['status']='FAKE'
 with pytest.raises(ValueError,match='changed'):c.validate_two_group(bad,0)
