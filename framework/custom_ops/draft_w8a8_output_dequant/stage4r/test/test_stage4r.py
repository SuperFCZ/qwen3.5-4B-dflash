import sys,shutil,subprocess,json,os
from pathlib import Path
import numpy as np
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import reference_r as r

def test_fixed_precision_is_not_a_w8_golden():
 x=np.ones((16,19456));assert r.check(x,x.astype('<f2'))['status']=='INDEPENDENT_CPU_NUMERICS_PASS'
 for bad in (x*0,x+.01,x*np.nan):
  with pytest.raises(ValueError):r.check(x,bad)
 assert r.POLICY['adaptive_tolerance'] is False

def test_cpp_numeric_contract_rejects_bad_before_timing(tmp_path):
 src=tmp_path/'test.cpp';src.write_text('#include "'+str(HERE/'numeric_contract.h')+'"\nint main(){std::vector<uint8_t> o(622592),r(311296*8);CheckReference(o,r);double v=1;std::memcpy(r.data(),&v,8);try{CheckReference(o,r);}catch(const std::exception&){return 0;}return 1;}')
 p=subprocess.run([shutil.which('c++'),'-std=c++17',str(src),'-o',str(tmp_path/'t')],capture_output=True,text=True);assert p.returncode==0,p.stderr
 subprocess.run([str(tmp_path/'t')],check=True)

@pytest.fixture(scope='module')
def native(tmp_path_factory):
 root=tmp_path_factory.mktemp('native-cpu');old=HERE.parent/'stage4k/test/N128';inc=root/'inc';shutil.copytree(old/'cpu_include',inc)
 header=inc/'native.h';header.write_text((inc/'aclnn_d_flash_w8_a8_n128_gate.h').read_text().replace('aclnnDFlashW8A8N128Gate','aclnnMm').replace('const aclTensor*,const aclTensor*,const aclTensor*,int64_t,int64_t,int64_t,aclTensor*','const aclTensor*,const aclTensor*,const aclTensor*,int8_t'))
 s=(old/'cpu_api.cpp').read_text().replace('#include "aclnn_d_flash_w8_a8_n128_gate.h"','#include "native.h"').replace('aclnnDFlashW8A8N128Gate','aclnnMm')
 a=s.index('int aclnnMmGetWorkspaceSize(');z=s.index(' volatile int clockResolutionWork',a)
 s=s[:a]+'int aclnnMmGetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*out,int8_t,uint64_t*size,aclOpExecutor**e){\n int64_t mode=0,limit=0;\n'+s[z:]
 s=s.replace('output=out','output=const_cast<aclTensor*>(out)')
 (root/'api.cpp').write_text(s);exe=root/'runner'
 p=subprocess.run([shutil.which('c++'),'-std=c++17','-O1','-DDFLASH_NATIVE_MM_HEADER="'+str(header)+'"','-I'+str(inc),'-I'+str(r.q.common.QUANT/'test'),str(HERE/'native_mm.cpp'),str(root/'api.cpp'),'-o',str(exe)],capture_output=True,text=True)
 assert p.returncode==0,p.stderr;return exe

@pytest.mark.parametrize('fault',[None,'TEST_BAD_GOLDEN','TEST_TIMED_LOG'])
def test_native_process_cpu_gate_and_log_stability(native,tmp_path,fault):
 for file,size in [('x.bin',81920),('w_fp16_kn.bin',99614720),('reference_f64.bin',311296*8),('golden_fp16.bin',622592)]:
  with (tmp_path/file).open('wb') as f:f.truncate(size)
 env=os.environ.copy()
 if fault:env[fault]='1'
 # TEST_BAD_GOLDEN changes one output bit to a tiny subnormal; force a large error instead.
 if fault=='TEST_BAD_GOLDEN':np.ones(311296,dtype='<f8').tofile(tmp_path/'reference_f64.bin')
 with (tmp_path/'runner.log').open('w') as log:p=subprocess.run([str(native),'0',str(tmp_path),'5','30','--continuous'],stdout=log,stderr=subprocess.STDOUT,env=env)
 if fault:assert p.returncode!=0;return
 assert p.returncode==0,(tmp_path/'runner.log').read_text()
 ex=json.loads((tmp_path/'execution.json').read_text());assert ex['independent_cpu_reference'] and ex['prepare_calls']==38 and ex['verify_calls']==4
 assert len(ex['timing']['execute_sync']['samples_ms'])==30

def test_effective_weights_and_independent_dot(tmp_path):
 q=(np.arange(16*128)%255-128).astype('i1').reshape(16,128);s=np.full((1,16),.0078125,'<f2')
 nz=q.reshape(1,16,4,32).transpose(2,0,1,3).copy();files={}
 for name,data in [('q_nk.bin',q),('w_nz.bin',nz),('s_gn.bin',s)]:
  data.tofile(tmp_path/name);files[name]=r.q.identity(tmp_path/name)
 weights=r.fc.generate_weights(files,tmp_path/'weights',16,128)
 w=np.fromfile(r.q.verify(weights['files']['w_fp16_kn.bin']),'<f2').reshape(128,16)
 assert w.T.tobytes()==(q.astype(np.float64)*s[0,:,None]).astype('<f2').tobytes()
 x=(np.arange(16*128)%9-4).astype('<f2').reshape(16,128)
 expected=np.array([[sum(float(x[m,k])*float(w[k,n]) for k in range(128)) for n in range(16)] for m in range(16)])
 assert np.array_equal(r.reference_block(x,w),expected)


def test_native_unadmitted_golden_cannot_authorize_timing(tmp_path):
 import run_stage4r as run
 with pytest.raises(ValueError):run.prior(None,'smoke',{})
 path=tmp_path/'r.json';r.q.write(path,dict(status='BLOCKED',phase='complete'))
 with pytest.raises(ValueError):run.prior(path,'smoke',dict(sources=[],evidence={},device=0,candidate='N128',policy=r.POLICY))
