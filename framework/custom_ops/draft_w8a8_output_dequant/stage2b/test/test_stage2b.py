"""CPU and Runner syntax tests only; all device reports here are synthetic fixtures."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import s2b_common as c
import s2b_reference as ref
import run_stage2b as run


@pytest.fixture
def evidence(tmp_path):
 folder=tmp_path/'stage2a';c.stage2a.make_cases(folder)
 path=folder/'manifest.json'
 return dict(stage2a_manifest=c.identity(path),stage2a_case_files=c.checked_cases(path))


def execution(device=0):
 return dict(status='EXECUTION_COMPLETED_NUMERICS_UNCHECKED',op=c.NAME,soc='Ascend310P3',shape=[16,128,64],
    guards=True,readonly=True,repeat_equal=True,repetitions=2,device=device,cpu_fallback=False,layout='nd',input_count=3,
    poison_verified=True,poison_patterns=[127,255],timing=dict(status='NOT_RUN',warmup=0,repetitions=0))


def test_reuses_stage2a_cpu_goldens_without_changing_them(evidence,tmp_path):
 before=copy.deepcopy(evidence)
 cases=ref.prepare(tmp_path/'basic',evidence,'basic');assert len(cases)==4
 for case in cases:
  _,_,_,ideal,candidate=ref.verify_case(case)
  result=ref.compare(case,[candidate.tobytes()]*2)
  assert result['status']=='CANDIDATE_BITWISE_MATCH' and result['codec_status']=='EXPERIMENTAL_CODEC_CANDIDATE'
  assert ideal.tobytes()==candidate.tobytes()
  if case['name'] in evidence['stage2a_case_files']:
   for name,old in evidence['stage2a_case_files'][case['name']].items():assert case['files'][name]['sha256']==old['sha256']
 c.verify_tree(before)


def test_nonpower_ideal_and_candidate_are_distinct(evidence,tmp_path):
 case=ref.prepare(tmp_path/'nonpower',evidence,'nonpower')[0]
 _,_,eff,ideal,candidate=ref.verify_case(case)
 assert np.all(eff==1) and ideal.tobytes()!=candidate.tobytes()
 result=ref.compare(case,[candidate.tobytes()]*2)
 assert result['status']=='CANDIDATE_BITWISE_MATCH' and result['ideal_fp32_reference'][0]['bit_mismatches']>0
 assert result['codec_status']==c.CODEC
 assert ref.compare(case,[ideal.tobytes()]*2)['status']=='FAIL_NUMERICAL'


def test_zero_preserves_reference_signed_zero_and_strict_bits(evidence,tmp_path):
 cases=ref.prepare(tmp_path/'zero',evidence,'zero');case=next(x for x in cases if x['name']=='scale_zero')
 acc,_,_,_,gold=ref.verify_case(case)
 assert (gold.view('<u2')[acc<0]==0x8000).all()
 assert ref.compare(case,[gold.tobytes()]*2)['status']=='CANDIDATE_BITWISE_MATCH'
 assert ref.compare(case,[bytes(2048)]*2)['status']=='FAIL_NUMERICAL'


def test_per_channel_groups_detect_ignored_and_block_aliased_scales(evidence,tmp_path):
 cases=ref.prepare(tmp_path/'basic',evidence,'basic')
 for name in ('per_channel_powers','per_channel_blocks'):
  case=next(x for x in cases if x['name']==name);acc,_,eff,_,gold=ref.verify_case(case)
  uniform=c.stage2a.convert(acc,np.ones(64,'<f4'))
  assert ref.compare(case,[uniform.tobytes()]*2)['status']=='FAIL_NUMERICAL'
  if name=='per_channel_blocks':
   wrong=c.stage2a.convert(acc,np.roll(eff,16))
   assert ref.compare(case,[wrong.tobytes()]*2)['status']=='FAIL_NUMERICAL'


@pytest.mark.parametrize('filename',['a_i8.bin','quant_u64_doc_candidate.bin','doc_candidate_fp16.bin'])
def test_changed_data_or_rewritten_golden_is_rejected(evidence,tmp_path,filename):
 case=ref.prepare(tmp_path/'basic',evidence,'basic')[0];path=Path(case['files'][filename]['path'])
 data=bytearray(path.read_bytes());data[0]^=1;path.write_bytes(data)
 with pytest.raises(ValueError,match='identity'):ref.verify_case(case)
 case['files'][filename]=c.identity(path)
 with pytest.raises(ValueError,match='model/Golden differs'):ref.verify_case(case)


def test_repeat_and_guard_poison_gates(evidence,tmp_path):
 case=ref.prepare(tmp_path/'basic',evidence,'basic')[0];gold=Path(case['files']['doc_candidate_fp16.bin']['path']).read_bytes()
 wrong=bytearray(gold);wrong[0]^=1
 result=ref.compare(case,[gold,bytes(wrong)])
 assert result['status']=='FAIL_NUMERICAL' and not result['repeat_equal']
 c.check_execution(execution(),0)
 for key,value in [('guards',False),('readonly',False),('repeat_equal',False),('poison_verified',False),('poison_patterns',[0,0]),
                   ('input_count',2),('shape',[16,256,64]),('timing',dict(status='MEASURED',warmup=5,repetitions=30))]:
  ex=execution();ex[key]=value
  with pytest.raises(ValueError):c.check_execution(ex,0)


def basic_proof(tmp_path,evidence):
 cases=ref.prepare(tmp_path/'basic-data',evidence,'basic');entries=[]
 for case in cases:
  folder=Path(case['files']['a_i8.bin']['path']).parent
  ex=folder/'execution.json';c.write(ex,execution())
  gold=Path(case['files']['doc_candidate_fp16.bin']['path']).read_bytes()
  outputs=[]
  for i in range(2):p=folder/f'actual-{i}.bin';p.write_bytes(gold);outputs.append(c.identity(p))
  entries.append(dict(name=case['name'],case=c.identity(folder/'case.json'),execution=c.identity(ex),outputs=outputs,
                      comparison=ref.compare(case,[gold]*2)))
 report=dict(status='STAGE2B_BASIC_MATCH',group='basic',phase='complete',build_report={},upstream=evidence,device=0,codec_status=c.CODEC,cases=entries)
 path=tmp_path/'basic.json';c.write(path,report);return path,report


def test_advanced_requires_complete_bound_basic_proof(evidence,tmp_path):
 with pytest.raises(ValueError,match='basic-report'):run.check_basic(None,{},evidence,0)
 p,r=basic_proof(tmp_path,evidence)
 run.check_basic(p,{},evidence,0)
 with pytest.raises(ValueError,match='basic proof'):run.check_basic(p,{'other':True},evidence,0)
 r['cases']=r['cases'][:3];c.write(p,r)
 with pytest.raises(ValueError,match='complete basic'):run.check_basic(p,{},evidence,0)


def test_numerical_failure_stops_next_case_and_no_fallback(evidence,tmp_path):
 built=dict(status='STAGE2B_BUILT_NPU_NOT_RUN',settings={},upstream=evidence,sources=[],installed={},opp=str(tmp_path/'opp'),
    cann_compile='fixture',installation='fixture')
 runner=tmp_path/'runner';runner.write_text('fixture only');built['runner']=c.identity(runner)
 bp=tmp_path/'build.json';c.write(bp,built);root=tmp_path/'smoke';root.mkdir()
 args=SimpleNamespace(build_report=bp,group='basic',device=0,cann_root=tmp_path/'sdk')
 report=dict(settings={},upstream=evidence,sources=[]);calls=[]
 def command(argv,log,environment,r,label,timeout=0):
  calls.append(label);c.write(log.parent/'execution.json',execution())
  log.write_text('STAGE2B_PLAN arch=2002 M=16 K=128 N=64 blockDim=1 HostC=DT_FLOAT16 dequant=TENSOR single=16/64/128 base=16/64/128')
  for i in range(2):(log.parent/f'actual-{i}.bin').write_bytes(bytes(2048))
 with patch.object(run,'installed',return_value={}),patch.object(run,'command',side_effect=command):
  with pytest.raises(ValueError,match='numerical mismatch'):run.smoke(args,root,report)
 assert calls==['case_scale_one'] and report['cases'][0]['comparison']['status']=='FAIL_NUMERICAL'


def test_stage1_evidence_is_checked_offline_without_any_runner(tmp_path):
 from w8_raw import prepare,compare
 root=tmp_path/'stage1';root.mkdir();cases=prepare(root/'data');items=[]
 for case in cases:
  folder=Path(case['files']['a.bin']['path']).parent;payload=np.fromfile(folder/'acc_int64.bin','<i8').astype('<i4').tobytes()
  outputs=[]
  for i in range(2):p=folder/f'actual-{i}.bin';p.write_bytes(payload);outputs.append(c.identity(p))
  ex=execution();ex.update(op='DFlashW8A8Accumulator',raw_accumulator=True,input_count=2)
  ex.pop('poison_verified');ex.pop('poison_patterns');c.write(folder/'execution.json',ex)
  items.append(dict(name=case['name'],case_file=c.identity(folder/'case.json'),raw=dict(outputs=outputs,execution_file=c.identity(folder/'execution.json')),comparison=compare(case,[payload]*2)))
 cann=tmp_path/'sdk';pf=root/'preflight.json';c.write(pf,dict(cann_root=str(cann),soc='Ascend310P3'))
 bp=root/'build.json';c.write(bp,dict(status='BUILT_NOT_NPU_VERIFIED',raw_only=True,layout='nd'))
 report=dict(status='RAW_INT32_ND_PASS',phase='complete',raw_only=True,layout='nd',device=0,cases=items,preflight=c.identity(pf),build_report=c.identity(bp))
 rp=root/'report.json';c.write(rp,report)
 with patch('subprocess.run',side_effect=AssertionError('Raw must never execute')):
  assert c.stage1_evidence(rp,0,cann)['rerun'] is False
 report['cases'][0]['comparison']['comparisons'][0]['mismatches']=1;c.write(rp,report)
 with pytest.raises(ValueError,match='numeric evidence'):c.stage1_evidence(rp,0,cann)


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
int aclnnDFlashW8A8OutputDequantGetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**);
int aclnnDFlashW8A8OutputDequant(void*,uint64_t,aclOpExecutor*,aclrtStream);
''')
 result=subprocess.run([compiler,'-std=c++17','-fsyntax-only','-I'+str(tmp_path),'-I'+str(c.QUANT/'test'),str(HERE/'test/runner.cpp')],capture_output=True,text=True)
 assert result.returncode==0,result.stderr


def test_fixed_abi_source_and_no_explicit_weight_dequant():
 spec=json.loads((HERE/'operator.json').read_text());assert len(spec)==1 and spec[0]['op']==c.NAME
 assert [x['type'] for x in spec[0]['input_desc']]==[['int8'],['int8'],['uint64']]
 host=(HERE/'op_host/host_common.h').read_text();kernel=(HERE/'op_kernel'/f'{c.SNAKE}.cpp').read_text()
 assert 'mm.SetCType(TPosition::GM,CubeFormat::ND,DataType::DT_FLOAT16)' in host and 'SetDequantType(DequantType::TENSOR)' in host
 assert 'SetQuantVector(scales)' in kernel and 'CubeFormat::ND,half' in kernel and 'CubeFormat::ND,int8_t' in kernel
 for token in ('Cast(', 'Brcb(', 'Muls(', 'Mul(', 'SetAntiQuant'):
  assert token not in kernel
 subprocess.run(['bash','-n',str(HERE/'run.sh')],check=True)


def test_cli_no_performance_or_expanded_shapes(tmp_path):
 for flags in (['--repetitions','30'],['--k','2560'],['--layout','nz']):
  r=subprocess.run([sys.executable,'-B',str(HERE/'run_stage2b.py'),'smoke','--output-dir',str(tmp_path/'new'),*flags],capture_output=True,text=True)
  assert r.returncode==2


def test_compile_failure_does_not_install_or_build_raw(tmp_path):
 root=tmp_path/'build';root.mkdir();tools={}
 for name in ('msopgen','cmake','c++'):
  path=tmp_path/name;path.write_text('fixture executable');tools[name]=dict(file=c.identity(path))
 pf=tmp_path/'preflight.json';c.write(pf,dict(tools=tools))
 args=SimpleNamespace(cann_root=tmp_path/'sdk');report=dict(upstream={'stage1':{'preflight':c.identity(pf)}});calls=[]
 def command(argv,log,environment,r,label,timeout=1800):
  calls.append((list(map(str,argv)),label))
  if label=='msopgen':
   (root/'CustomOp/op_host').mkdir(parents=True);(root/'CustomOp/op_kernel').mkdir()
  elif label=='build':raise RuntimeError('CPU simulated CANN compiler diagnostic')
 with patch.object(run.sys,'platform','linux'),patch.object(run.platform,'machine',return_value='aarch64'),\
      patch.object(run.shutil,'which',side_effect=lambda name:tools[name]['file']['path']),patch.object(run,'command',side_effect=command):
  with pytest.raises(RuntimeError,match='compiler diagnostic'):run.build(args,root,report)
 assert [label for _,label in calls]==['msopgen','build']
 assert calls[0][0][-1]==c.NAME and all('DFlashW8A8Accumulator' not in ' '.join(argv) for argv,_ in calls)
 assert not (root/'opp').exists()


def test_command_keeps_return_code_and_error_log(tmp_path):
 r={};log=tmp_path/'compile.log'
 with pytest.raises(RuntimeError,match='returned 7'):
  run.command([sys.executable,'-c','print("CPU simulated SDK compiler error");raise SystemExit(7)'],log,{},r,'build')
 assert r['commands']['build']['returncode']==7
 assert r['commands']['build']['log_identity']==c.identity(log)
 assert 'SDK compiler error' in log.read_text()


def test_sdk_index_tampering_is_rejected_before_build(evidence,tmp_path):
 sdk=tmp_path/'sdk';asc=sdk/'aarch64-linux/asc';asc.mkdir(parents=True)
 index=tmp_path/'index.json';m=tmp_path/'stage2a/manifest.json'
 required=['impl/adv_api/detail/matmul/matmul_impl_base.h',
    'impl/adv_api/detail/matmul/stage/copy_cube_out/copy_cube_out_datacopy.h',
    'impl/adv_api/detail/matmul/stage/copy_cube_out/quant/quant_processor_datacopy.h',
    'impl/adv_api/detail/matmul/stage/copy_cube_out/quant/quant_processor_utils.h']
 entries=[]
 for rel in required:
  p=asc/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('// CPU fixture')
  item=c.identity(p);entries.append(dict(relative=rel,state='INDEXED_NOT_REVIEWED',bytes=item['bytes'],sha256=item['sha256']))
 c.write(index,dict(stage='2A',status='UNRESOLVED_SOURCE_REVIEW_REQUIRED',source_root=str(asc),files=entries))
 args=SimpleNamespace(stage1=tmp_path/'raw.json',device=0,cann_root=sdk,stage2a_index=index,stage2a_cases=m)
 with patch.object(c,'stage1_evidence',return_value={'fixture_only':True}):
  result=c.upstream(args)
  assert result['codec']==c.CODEC and result['stage2a_review_status']=='UNRESOLVED_SOURCE_REVIEW_REQUIRED'
  (asc/required[0]).write_text('changed')
  with pytest.raises(ValueError,match='SDK changed'):c.upstream(args)
