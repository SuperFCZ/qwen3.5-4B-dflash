import copy,json,os,shutil,subprocess,sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import q_common as q
import run_stage4q as run

@pytest.fixture(scope='module',params=['N128','N256'])
def exe(request,tmp_path_factory):
 name=request.param;root=tmp_path_factory.mktemp(name);old=HERE.parent/'stage4k/test/N128';inc=root/'inc';shutil.copytree(old/'cpu_include',inc)
 s=(old/'cpu_api.cpp').read_text()
 if name=='N256':
  s=s.replace('N128','N256').replace('n128','n256').replace('STAGE4J','STAGE4P')
  p=inc/'aclnn_d_flash_w8_a8_n128_gate.h';(inc/'aclnn_d_flash_w8_a8_n256_gate.h').write_text(p.read_text().replace('N128','N256'));p.unlink()
 (root/'api.cpp').write_text(s);out=root/'runner'
 p=subprocess.run([shutil.which('c++'),'-std=c++17','-O1','-I'+str(inc),'-I'+str(q.common.QUANT/'test'),str(HERE/name/'runner.cpp'),str(root/'api.cpp'),'-o',str(out)],capture_output=True,text=True)
 assert p.returncode==0,p.stderr;return name,out

@pytest.mark.parametrize('fault',[None,'TEST_BAD_GOLDEN','TEST_TIMED_LOG'])
def test_actual_runner_protocol(exe,tmp_path,fault):
 name,runner=exe
 for file,size in [('a_i8.bin',40960),('b_i8.bin',49807360),('quant_u64_doc_candidate.bin',3112960),('golden_fp16.bin',622592)]:
  with (tmp_path/file).open('wb') as f:f.truncate(size)
 env=os.environ.copy();env['DFLASH_'+('STAGE4P' if name=='N256' else 'STAGE4J')+'_EXPECTED_CORE_POLICY']='8,8,7,7,2002'
 if fault:env[fault]='1'
 with (tmp_path/'runner.log').open('w') as log:p=subprocess.run([str(runner),'0',str(tmp_path),'0','--benchmark'],env=env,stdout=log,stderr=subprocess.STDOUT)
 ex=json.loads((tmp_path/'execution.json').read_text())
 if fault:assert p.returncode!=0 and ex['status']=='FAIL';return
 assert p.returncode==0;assert len(q.check_execution(ex,0,name,True)['execute_sync']['samples_ms'])==30
 assert (tmp_path/'postcheck.bin').read_bytes()==bytes(622592)

def test_materiality_pairing_and_profile_rejection():
 rows=[dict(round=i,position=j,variant=n,profiled=False,formal_latency_evidence=True,status='FROZEN_GOLDEN_MATCH',timing={key:q.statistics_ms([1]*30,30) for key in ('execute_sync','prepare')}) for i,o in enumerate(q.ORDER) for j,n in enumerate(o)]
 assert q.comparison(rows)['decision']=='NO_GO_NO_MATERIAL_IMPROVEMENT'
 bad=copy.deepcopy(rows);bad[0]['profiled']=True
 with pytest.raises(ValueError):q.comparison(bad)
 with pytest.raises(ValueError):q.comparison(rows[:-1])
 with pytest.raises(ValueError):run.prior(None,'smoke',{})
