"""CPU contracts and scalar model of the actual new Kernel. Not NPU evidence."""
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
import s3b_common as c
import run_stage3b as run
from vdeq_inspect import inspect


def partials():
 p=((np.arange(20*1024)*37)%255-127).astype('<f2').reshape(20,16,64)/np.float16(64)
 return p.astype('<f2')


@pytest.fixture
def prior(tmp_path):
 root=tmp_path/'prior';root.mkdir();groups=[];cases=[]
 a=((np.arange(16*128)*17)%33-16).astype('i1').reshape(16,128)
 b=((np.arange(128*64)*13)%31-15).astype('i1').reshape(128,64)
 for g in range(20):
  folder=root/f'g{g}';folder.mkdir();acc=c.common.stage2a.dot(a,b)
  scale=np.full(64,(g+1)/256,dtype='<f4');words,eff=c.common.stage2a.encode_doc_candidate(scale)
  gold=c.common.stage2a.convert(acc,eff)
  data={'a_i8.bin':a,'b_i8.bin':b,'z_i32.bin':acc.astype('<i4'),'scale_f32.bin':scale,
    'quant_u64_doc_candidate.bin':words,'scale_effective_f32.bin':eff,'ideal_fp16.bin':c.common.stage2a.convert(acc,scale),'doc_candidate_fp16.bin':gold}
  for name,v in data.items():v.tofile(folder/name)
  case=dict(name=f'group-{g:02d}',group_index=g,k_range=[128*g,128*(g+1)],n_range=[0,64],files={n:c.identity(folder/n) for n in data})
  c.write(folder/'case.json',case);ci=c.identity(folder/'case.json');cases.append(ci)
  ex=dict(status='EXECUTION_COMPLETED_NUMERICS_UNCHECKED',op='DFlashW8A8OutputDequant',soc='Ascend310P3',shape=[16,128,64],
    guards=True,readonly=True,repeat_equal=True,repetitions=2,device=0,cpu_fallback=False,layout='nd',input_count=3,poison_verified=True,poison_patterns=[127,255],timing=dict(status='NOT_RUN',warmup=0,repetitions=0))
  c.write(folder/'execution.json',ex);outputs=[]
  for i in range(2):p=folder/f'actual-{i}.bin';gold.tofile(p);outputs.append(c.identity(p))
  groups.append(dict(group_index=g,status='PASS',case=ci,execution=c.identity(folder/'execution.json'),outputs=outputs,
    comparison=c.e2c.reference.compare(case,[gold.tobytes()]*2)))
 stacks=np.stack([np.fromfile(g['outputs'][0]['path'],'<f2').reshape(16,64) for g in groups]);total,half=c.reduce_cpu(stacks)
 total.tofile(root/'partial_f32.bin');half.tofile(root/'partial_f16.bin')
 refs={'partial_f32':c.identity(root/'partial_f32.bin'),'partial_f16':c.identity(root/'partial_f16.bin')}
 manifest=root/'refs.json';c.write(manifest,dict(group_count=20,group_order=list(range(20)),cases=cases,references=refs))
 cann=tmp_path/'sdk';pf=root/'pf.json';c.write(pf,dict(cann_root=str(cann),soc='Ascend310P3',files=[],tools={}))
 r=dict(stage='3A',status='STAGE3A_20_GROUP_MATCH',phase='complete',shape=[16,2560,64],group_count=20,device=0,npu='ALL_20_GROUPS_EXECUTED',performance='NOT_RUN',
    summary=dict(passed=20,failed=0,not_run=0),groups=groups,sources=[],cpu_manifest=c.identity(manifest),
    aggregation=dict(status='MATCH',repeat_equal=True,order=list(range(20)),outputs=[dict(fp32=refs['partial_f32'],fp16=refs['partial_f16'])]*2),
    baseline={'stage2b':{'sources':[],'upstream':{'stage1':{'preflight':c.identity(pf)}}}},
    native_semantics_status='UNRESOLVED',native={'error':'Reshape_1 output descriptor unavailable'})
 path=root/'report.json';c.write(path,r);return path,r,cann,stacks


def test_stage3a_is_reused_readonly_and_native_unresolved_does_not_block(prior,tmp_path):
 path,r,cann,original=prior;before=path.read_bytes()
 with patch('subprocess.run',side_effect=AssertionError('must not execute old kernels')):
  stack,total,half,proof,pf=c.evidence(path,0,cann)
 assert path.read_bytes()==before and stack.tobytes()==original.tobytes()
 assert proof['upstream_rerun'] is False and pf['cann_root']==str(cann)
 case=c.prepare(tmp_path/'data',stack,total,half,proof);assert c.verify_case(case).tobytes()==half.tobytes()
 result=c.compare(case,[half.tobytes()]*2);assert result['status']=='REDUCTION_BITWISE_MATCH'
 wrong=bytearray(half.tobytes());wrong[0]^=1
 assert c.compare(case,[bytes(wrong)]*2)['status']=='FAIL_NUMERICAL'


@pytest.mark.parametrize('change',['missing','reorder','duplicate','bad_guard','bad_repeat','golden_changed'])
def test_corrupt_or_incomplete_stage3a_evidence_fails(prior,change):
 path,r,cann,_=prior
 if change=='missing':r['groups'].pop()
 elif change=='reorder':r['groups'].reverse()
 elif change=='duplicate':r['groups'][19]=r['groups'][0]
 elif change=='bad_guard':
  p=Path(r['groups'][0]['execution']['path']);ex=json.loads(p.read_text());ex['guards']=False;c.write(p,ex);r['groups'][0]['execution']=c.identity(p)
 elif change=='bad_repeat':
  p=Path(r['groups'][0]['outputs'][1]['path']);data=bytearray(p.read_bytes());data[0]^=1;p.write_bytes(data);r['groups'][0]['outputs'][1]=c.identity(p)
 else:
  m=json.loads(Path(r['cpu_manifest']['path']).read_text());Path(m['references']['partial_f16']['path']).write_bytes(bytes(2048))
 c.write(path,r)
 with pytest.raises(ValueError):c.evidence(path,0,cann)


def test_order_fp32_and_single_final_rounding():
 p=np.zeros((20,16,64),'<f2');p[0]=2048;p[1]=1;p[2]=-2048
 total,half=c.reduce_cpu(p);assert (total==1).all() and (half==1).all()
 p[0]=65504;p[1]=np.float16(2**-24);p[2]=-65504
 assert (c.reduce_cpu(p)[1]==0).all()
 reordered=p.copy();reordered[[1,2]]=p[[2,1]]
 assert (c.reduce_cpu(reordered)[1].view('<u2')==1).all()
 with pytest.raises(ValueError):c.reduce_cpu(p[:19])
 p[0,0,0]=np.nan
 with pytest.raises(ValueError):c.reduce_cpu(p)


def test_actual_kernel_scalar_model_matches_ordered_cpu_reference(tmp_path):
 compiler=shutil.which('c++')
 if not compiler:pytest.skip('local C++ compiler unavailable')
 harness=tmp_path/'main.cpp'
 harness.write_text('#include <fstream>\n#include "'+str(HERE/'op_kernel/d_flash_w8_a8_partial_reduce.cpp')+'"\n'+r'''
int main(int argc,char**argv) {
 std::vector<AscendC::half> input(20*1024),output(1024);
 std::ifstream f(argv[1],std::ios::binary);f.read(reinterpret_cast<char*>(input.data()),40960);
 TestTiling td{20,1024};
 d_flash_w8_a8_partial_reduce(reinterpret_cast<uint8_t*>(input.data()),reinterpret_cast<uint8_t*>(output.data()),nullptr,reinterpret_cast<uint8_t*>(&td));
 AscendC::Require(AscendC::ubBytes==12288&&AscendC::addCalls==20&&AscendC::casts==21&&AscendC::copiesIn==20&&AscendC::copiesOut==1);
 std::ofstream out(argv[2],std::ios::binary);out.write(reinterpret_cast<char*>(output.data()),2048);
}
''')
 exe=tmp_path/'model';r=subprocess.run([compiler,'-std=c++17','-O0','-I'+str(HERE/'test/cpu_include'),str(harness),'-o',str(exe)],capture_output=True,text=True)
 assert r.returncode==0,r.stderr
 cases=[partials(),np.zeros((20,16,64),'<f2')]
 for v in (1,np.float16(2**-24)):
  p=np.zeros((20,16,64),'<f2');p[0]=2048 if v==1 else 65504;p[1]=v;p[2]=-p[0];cases.append(p)
 for i,p in enumerate(cases):
  inp=tmp_path/f'in{i}.bin';out=tmp_path/f'out{i}.bin';p.tofile(inp)
  subprocess.run([str(exe),str(inp),str(out)],check=True)
  assert out.read_bytes()==c.reduce_cpu(p)[1].tobytes()


def test_execution_gates_and_report_resolution(tmp_path):
 ex=dict(status='EXECUTION_COMPLETED_NUMERICS_UNCHECKED',op=c.NAME,soc='Ascend310P3',input_shape=[20,16,64],output_shape=[16,64],group_count=20,
    guards=True,readonly=True,repeat_equal=True,repetitions=2,device=0,cpu_fallback=False,layout='nd',input_count=1,poison_verified=True,poison_patterns=[127,255],timing=dict(status='NOT_RUN',warmup=0,repetitions=0))
 c.check_execution(ex,0)
 for key in ('guards','readonly','repeat_equal','poison_verified'):
  bad=copy.deepcopy(ex);bad[key]=False
  with pytest.raises(ValueError):c.check_execution(bad,0)
 (tmp_path/'report.json').write_text('{}');assert c.resolve_report(tmp_path,None)==tmp_path/'report.json'
 (tmp_path/'npu').mkdir();(tmp_path/'npu/report.json').write_text('{}')
 with pytest.raises(ValueError,match='ambiguous'):c.resolve_report(tmp_path,None)


def test_fixed_file_vdeq_inspection_does_not_infer_support(tmp_path):
 cann=tmp_path/'sdk';asc=cann/'aarch64-linux/asc';asc.mkdir(parents=True)
 p=asc/'impl/basic_api/dav_m200/kernel_operator_data_copy_impl.h';p.parent.mkdir(parents=True);p.write_text('VDEQ16\ncopy_matrix_cc_to_ubuf\n')
 result=inspect(cann,tmp_path/'inspect',{'files':[]})
 assert result['low_level_intrinsic_review'].startswith('UNRESOLVED')
 assert len(result['files'])==4 and result['npu']=='NOT_RUN'
 subprocess.run(['bash','-n',str(HERE/'run.sh')],check=True)
 assert not list(HERE.glob('*.so'))


def test_runner_cpp_syntax_only(tmp_path):
 compiler=shutil.which('c++')
 if not compiler:pytest.skip('local compiler unavailable')
 (tmp_path/'acl').mkdir();(tmp_path/'acl/acl.h').write_text('''#pragma once
#include <cstddef>
#include <cstdint>
using aclrtStream=void*;using aclDataType=int;using aclFormat=int;
constexpr int ACL_SUCCESS=0,ACL_MEM_MALLOC_HUGE_FIRST=0,ACL_MEMCPY_HOST_TO_DEVICE=1,ACL_MEMCPY_DEVICE_TO_HOST=2,ACL_FLOAT16=1,ACL_FORMAT_ND=2;
const char* aclGetRecentErrMsg();const char* aclrtGetSocName();
int aclrtMalloc(void**,size_t,int);int aclrtFree(void*);int aclrtMemset(void*,size_t,int,size_t);int aclrtMemcpy(void*,size_t,const void*,size_t,int);
int aclInit(const char*);int aclFinalize();int aclrtSetDevice(int);int aclrtResetDevice(int);int aclrtCreateStream(aclrtStream*);int aclrtDestroyStream(aclrtStream);int aclrtSynchronizeStream(aclrtStream);
''')
 (tmp_path/f'aclnn_{c.SNAKE}.h').write_text('''#pragma once
struct aclTensor;struct aclOpExecutor;
aclTensor* aclCreateTensor(const int64_t*,uint64_t,int,const int64_t*,int64_t,int,const int64_t*,uint64_t,void*);
int aclDestroyTensor(const aclTensor*);
int aclnnDFlashW8A8PartialReduceGetWorkspaceSize(const aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**);
int aclnnDFlashW8A8PartialReduce(void*,uint64_t,aclOpExecutor*,aclrtStream);
''')
 r=subprocess.run([compiler,'-std=c++17','-fsyntax-only','-I'+str(tmp_path),'-I'+str(c.common.QUANT/'test'),str(HERE/'test/runner.cpp')],capture_output=True,text=True)
 assert r.returncode==0,r.stderr


def test_poison_readonly_repeat_and_no_timing_gate():
 ex=dict(status='EXECUTION_COMPLETED_NUMERICS_UNCHECKED',op=c.NAME,soc='Ascend310P3',input_shape=[20,16,64],output_shape=[16,64],group_count=20,
    guards=True,readonly=True,repeat_equal=True,repetitions=2,device=0,cpu_fallback=False,layout='nd',input_count=1,poison_verified=True,poison_patterns=[127,255],timing=dict(status='NOT_RUN',warmup=0,repetitions=0))
 for k,v in [('poison_patterns',[0,0]),('input_shape',[19,16,64]),('input_count',3),('timing',dict(status='MEASURED',warmup=5,repetitions=30))]:
  bad=copy.deepcopy(ex);bad[k]=v
  with pytest.raises(ValueError):c.check_execution(bad,0)


def test_tampered_packed_input_cannot_be_rebound(prior,tmp_path):
 path,r,cann,_=prior;stack,total,half,proof,pf=c.evidence(path,0,cann)
 case=c.prepare(tmp_path/'data',stack,total,half,proof)
 p=Path(case['files']['partials_fp16.bin']['path']);data=bytearray(p.read_bytes());data[0]^=1;p.write_bytes(data)
 with pytest.raises(ValueError,match='identity'):c.verify_case(case)
 case['files']['partials_fp16.bin']=c.identity(p)
 with pytest.raises(ValueError,match='actual Stage3A'):c.verify_case(case)
