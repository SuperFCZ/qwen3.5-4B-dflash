"""CPU-only two-group checks; synthetic device records are fixtures, not NPU proof."""
import copy,json,shutil,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import fused_common as c
import run_stage3b3 as run

@pytest.fixture
def proof(tmp_path):
 a,b,s=c.synthetic();d=c.reference(a,b,s);groups=[]
 for g in range(2):
  folder=tmp_path/f'prior-g{g}';folder.mkdir();acc=d['z'][g]
  data={'a_i8.bin':a[g],'b_i8.bin':b[g],'scale_f32.bin':s[g],'z_i32.bin':acc.astype('<i4'),
   'quant_u64_doc_candidate.bin':d['words'][g],'scale_effective_f32.bin':d['effective'][g],
   'ideal_fp16.bin':c.common.stage2a.convert(acc,s[g]),'doc_candidate_fp16.bin':d['parts'][g]}
  for n,v in data.items():v.tofile(folder/n)
  case=dict(name=f'group-{g}',files={n:c.identity(folder/n) for n in data});c.write(folder/'case.json',case)
  outs=[]
  for i in range(2):p=folder/f'actual-{i}.bin';d['parts'][g].tofile(p);outs.append(c.identity(p))
  groups.append(dict(group_index=g,case=c.identity(folder/'case.json'),outputs=outs))
 return dict(groups=groups)

@pytest.fixture
def cases(tmp_path,proof):
 return [json.loads(c.verify(i).read_text()) for i in c.prepare(tmp_path/'data',proof)]


def test_real_chunks_and_old_goldens_are_reused_without_requantizing(cases,proof):
 case=cases[0];d=c.verify_case(case)
 assert c.verify(case['files']['partials_fp16.bin']).read_bytes()==b''.join(c.verify(g['outputs'][0]).read_bytes() for g in proof['groups'])
 for mode in (0,1,2):
  data=d['out'] if mode==0 else d['parts'][mode-1]
  assert c.compare(case,mode,[data.tobytes()]*2)['status']=='BITWISE_MATCH'
 assert case['settings']['scratch']==239104 and case['settings']['scratch']+12288+2560+8192==256*1024


def test_scale_and_pointer_update_witnesses(cases):
 for case in cases[1:]:
  assert all(n>0 for n in case['controls'].values())
  a=np.fromfile(case['files']['a_i8.bin']['path'],'i1').reshape(2,16,128)
  b=np.fromfile(case['files']['b_i8.bin']['path'],'i1').reshape(2,128,64)
  s=np.fromfile(case['files']['scale_f32.bin']['path'],'<f4').reshape(2,64)
  d=c.reference(a,b,s);wrong=c.common.stage2a.convert(d['z'][1],d['effective'][0])
  assert c.compare(case,2,[wrong.tobytes()]*2)['status']=='FAIL_NUMERICAL'
 assert cases[1]['files']['golden_fp16.bin']['sha256']!=cases[2]['files']['golden_fp16.bin']['sha256']


def test_audit_detects_errors_hidden_by_final_match(cases):
 case=cases[1];d=c.verify_case(case)
 wrong=d['parts'][0].copy();wrong[0,0]=np.float16(float(wrong[0,0])+1)
 assert c.compare(case,0,[d['out'].tobytes()]*2)['status']=='BITWISE_MATCH'
 assert c.compare(case,1,[wrong.tobytes()]*2)['status']=='FAIL_NUMERICAL'


@pytest.mark.parametrize('name',['a_i8.bin','quant_u64_doc_candidate.bin','golden_fp16.bin'])
def test_hash_and_reference_rebinding_rejected(cases,name):
 case=cases[0];p=Path(case['files'][name]['path']);data=bytearray(p.read_bytes());data[0]^=1;p.write_bytes(data)
 with pytest.raises(ValueError):c.verify_case(case)
 case['files'][name]=c.identity(p)
 with pytest.raises(ValueError):c.verify_case(case)


def test_group_order_and_saved_partial_repeat_rejected(cases):
 case=cases[0];bad=copy.deepcopy(case);bad['source_groups'].reverse()
 with pytest.raises(ValueError,match='group order'):c.verify_case(bad)
 p=Path(case['source_groups'][1]['outputs'][1]['path']);p.write_bytes(bytes(2048))
 with pytest.raises(ValueError):c.verify_case(case)


def execution(mode=0):
 return dict(status='EXECUTION_COMPLETED_NUMERICS_UNCHECKED',op=c.NAME,soc='Ascend310P3',shape=[16,256,64],group_count=2,output_mode=mode,
  ub_guard_protocol='POISON_ON_GUARD_FAILURE',guards=True,readonly=True,repeat_equal=True,repetitions=2,device=0,cpu_fallback=False,
  layout='nd',input_count=3,poison_verified=True,poison_patterns=[127,255],timing=dict(status='NOT_RUN',warmup=0,repetitions=0))


def test_execution_gates():
 for mode in (0,1,2):c.check_execution(execution(mode),0,mode)
 for k,v in [('guards',False),('readonly',False),('poison_verified',False),('repeat_equal',False),('output_mode',2),('ub_guard_protocol','NONE')]:
  ex=execution();ex[k]=v
  with pytest.raises(ValueError):c.check_execution(ex,0,0)


def test_mismatch_stops_before_next_diagnostic(cases,tmp_path):
 root=tmp_path/'smoke';root.mkdir();runner=tmp_path/'runner';runner.write_text('fixture');bp=tmp_path/'build.json'
 r=dict(settings={},sources=[],evidence={},device=0)
 b=dict(status='STAGE3B3_BUILT_NPU_NOT_RUN',**r,installed={},opp=str(tmp_path/'opp'),runner=c.identity(runner),cann_compile='fixture',installation='fixture')
 c.write(bp,b);casefile=Path(cases[0]['files']['a_i8.bin']['path']).parent/'case.json';calls=[]
 def command(argv,log,*rest):
  calls.append(argv);c.write(log.parent/'execution.json',execution(1))
  log.write_text('STAGE3B3_OUTPUT_MODE=1 groups=2 M=16 K=256 groupK=128 N=64 scratch=239104 userUB=12288 ubGuards=2560 reserve=8192 single=16/64/128 base=16/64/128')
  for i in range(2):(log.parent/f'actual-{i}.bin').write_bytes(bytes(2048))
 args=SimpleNamespace(fused_build_report=bp,device=0,cann_root=tmp_path/'sdk')
 with patch.object(run,'installed',return_value={}),patch.object(run,'command',side_effect=command):
  with pytest.raises(ValueError,match='no next'):run.smoke(args,root,r,[c.identity(casefile)])
 assert len(calls)==1 and r['cases'][0]['modes'][0]['mode']==1


def test_actual_kernel_cpu_protocol_and_arithmetic(cases,tmp_path):
 compiler=shutil.which('c++')
 if not compiler:pytest.skip('no C++ compiler')
 src=tmp_path/'main.cpp';src.write_text('#include <fstream>\n#include "'+str(HERE/'op_kernel/d_flash_w8_a8_two_group.cpp')+'"\n'+r'''
int main(int argc,char**argv){
 using namespace AscendC;
 std::vector<int8_t>a(4096),b(16384);std::vector<uint64_t>s(128);std::vector<half>o(1024);
 std::ifstream(argv[1],std::ios::binary).read((char*)a.data(),4096);
 std::ifstream(argv[2],std::ios::binary).read((char*)b.data(),16384);
 std::ifstream(argv[3],std::ios::binary).read((char*)s.data(),1024);
 std::memset(o.data(),0x7f,2048);TestTiling td{239104,uint32_t(std::stoi(argv[5])),0};fault=std::stoi(argv[6]);
 d_flash_w8_a8_two_group((uint8_t*)a.data(),(uint8_t*)b.data(),(uint8_t*)s.data(),(uint8_t*)o.data(),nullptr,(uint8_t*)&td);
 Require(ubBytes==253952);
 if(fault!=2)Require(iterations==2&&ends==2&&adds==2&&casts==(td.outputMode==0?3:2)&&copies==64&&firstDmaAfterTwo);
 else Require(copies==0); // UB guard failure leaves every GM byte poisoned
 std::ofstream(argv[4],std::ios::binary).write((char*)o.data(),2048);
}
''')
 exe=tmp_path/'kernel-model';done=subprocess.run([compiler,'-std=c++17','-O0','-I'+str(HERE/'test/cpu_include'),str(src),'-o',str(exe)],capture_output=True,text=True)
 assert done.returncode==0,done.stderr
 case=cases[1]
 for mode in (0,1,2):
  for fault in (0,1,2):
   out=tmp_path/f'out-{mode}-{fault}.bin'
   argv=[str(exe),*[case['files'][n]['path'] for n in ('a_i8.bin','b_i8.bin','quant_u64_doc_candidate.bin')],str(out),str(mode),str(fault)]
   subprocess.run(argv,check=True)
   result=c.compare(case,mode,[out.read_bytes()]*2)
   assert (result['status']=='BITWISE_MATCH')==(fault==0 or (fault==1 and mode==1))

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
int aclnnDFlashW8A8TwoGroupGetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,int64_t,aclTensor*,uint64_t*,aclOpExecutor**);
int aclnnDFlashW8A8TwoGroup(void*,uint64_t,aclOpExecutor*,aclrtStream);
''')
 result=subprocess.run([compiler,'-std=c++17','-fsyntax-only','-I'+str(tmp_path),'-I'+str(c.common.QUANT/'test'),str(HERE/'test/runner.cpp')],capture_output=True,text=True)
 assert result.returncode==0,result.stderr
