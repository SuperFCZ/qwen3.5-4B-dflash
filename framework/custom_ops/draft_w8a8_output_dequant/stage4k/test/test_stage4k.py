"""CPU-only ABI/timing doubles. No fixture timing is real NPU evidence."""
import copy,json,os,shutil,subprocess,sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import k_common as k
import run_stage4k as run
from profile_metrics_k import collect

@pytest.fixture(scope='module',params=['N64','N128'])
def runner(request,tmp_path_factory):
 name=request.param;root=tmp_path_factory.mktemp(name);exe=root/'runner';d=HERE/'test'/name
 r=subprocess.run([shutil.which('c++'),'-std=c++17','-O1','-I'+str(d/'cpu_include'),'-I'+str(k.common.QUANT/'test'),str(HERE/name/'runner.cpp'),str(d/'cpu_api.cpp'),'-o',str(exe)],capture_output=True,text=True)
 assert r.returncode==0,r.stderr
 return name,exe

def execute(runner,root,timed,fail=None):
 name,exe=runner;folder=root/'case';folder.mkdir()
 for file,size in [('a_i8.bin',40960),('b_i8.bin',49807360),('quant_u64_doc_candidate.bin',3112960),('golden_fp16.bin',622592)]:
  with (folder/file).open('wb') as f:f.truncate(size)
 env=os.environ.copy();prefix='STAGE4B' if name=='N64' else 'STAGE4J';env['DFLASH_'+prefix+'_EXPECTED_CORE_POLICY']='8,8,7,7,2002'
 if fail:env[fail]='1'
 with (folder/'runner.log').open('w') as f:r=subprocess.run([str(exe),'0',str(folder),'0']+(['--benchmark'] if timed else []),stdout=f,stderr=subprocess.STDOUT,env=env)
 return folder,r,json.loads((folder/'execution.json').read_text())

@pytest.mark.parametrize('timed',[False,True])
def test_protocol_and_abi(runner,tmp_path,timed):
 folder,result,ex=execute(runner,tmp_path,timed)
 assert result.returncode==0,(folder/'runner.log').read_text()
 stats=k.check_execution(ex,0,runner[0],timed)
 assert ex['prepare_calls']==(38 if timed else 2) and ex['verify_calls']==(4 if timed else 2)
 if timed:
  assert len(stats['execute_sync']['samples_ms'])==30
  ex['timing']['per_timed_call_readback']=True
  with pytest.raises(ValueError):k.check_execution(ex,0,runner[0],True)
 proof=dict(files={'golden_fp16.bin':k.identity(folder/'golden_fp16.bin')})
 assert len(k.bc.verify_outputs(proof,folder,timed))==(4 if timed else 2)
 (folder/'actual-0.bin').write_bytes(b'bad')
 with pytest.raises(ValueError):k.bc.verify_outputs(proof,folder,timed)

@pytest.mark.parametrize('failure',['TEST_BAD_GOLDEN','TEST_TIMED_LOG'])
def test_fail_before_accepting_samples(runner,tmp_path,failure):
 folder,result,ex=execute(runner,tmp_path,True,failure)
 assert result.returncode!=0 and ex['status']=='FAIL'
 assert not (folder/'postcheck.bin').exists()
 if failure=='TEST_BAD_GOLDEN':assert ex['timing']['execute_sync']['samples_ms']==[]
 else:assert 'reject formal timing' in (folder/'runner.log').read_text()

def rows():
 return [dict(round=i,position=j,variant=n,profiled=False,formal_latency_evidence=True,status='FROZEN_GOLDEN_MATCH',workspace_bytes=123,
 timing={key:k.statistics_ms([2 if n=='N64' else 1]*30,30) for key in ('execute_sync','prepare')}) for i,order in enumerate(k.ORDER) for j,n in enumerate(order)]

def test_paired_statistics_no_historical_or_profile_samples():
 r=rows();v=k.comparison(r)
 assert v['n64_over_n128']==2 and all(x['n64_over_n128']==2 for x in v['rounds'])
 assert len(v['variants']['N64']['execute_sync']['samples_ms'])==90
 for bad in (r[:-1],list(reversed(r))):
  with pytest.raises(ValueError):k.comparison(bad)
 for key,val in [('profiled',True),('formal_latency_evidence',False),('status','FAILED')]:
  bad=copy.deepcopy(r);bad[0][key]=val
  with pytest.raises(ValueError):k.comparison(bad)
 bad=copy.deepcopy(r);bad[0]['timing']['execute_sync']['samples_ms'][0]=float('nan')
 with pytest.raises(ValueError):k.comparison(bad)

def test_profile_operator_isolation(tmp_path):
 (tmp_path/'summary.csv').write_text('Op Name,Task Duration(us),aic_scalar_time(us),aic_mte2_time(us)\nDFlashW8A8FullGate,20,15,0\nDFlashW8A8N128Gate,10,8,N/A\n')
 for name in ('DFlashW8A8FullGate','DFlashW8A8N128Gate'):
  result=collect(tmp_path,name)
  assert len(result['counter_rows'])==1 and result['counter_rows'][0]['names']==[name]
  assert not result['formal_latency_evidence'] and result['availability']['cube']=='UNAVAILABLE'

def test_missing_failed_and_wrong_identity_prior_blocked(tmp_path):
 with pytest.raises(ValueError):run.prior(None,'build',{})
 p=tmp_path/'r.json';base=dict(sources=[],evidence={},device=0)
 k.write(p,dict(**base,status=run.STATES['build'],phase='complete'))
 with pytest.raises(ValueError,match='two isolated'):run.prior(p,'build',base)
 for change in (dict(device=1),dict(status='FAILED'),dict(sources=['modified'])):
  doc=dict(**base,status=run.STATES['build'],phase='complete');doc.update(change);k.write(p,doc)
  with pytest.raises(ValueError):run.prior(p,'build',base)

def test_runner_delta_only_abi_and_fixed_auto():
 base=(HERE.parent/'stage4c/test/runner.cpp').read_text()
 for name in ('N64','N128'):
  expected=base
  if name=='N128':expected=expected.replace('d_flash_w8_a8_full_gate','d_flash_w8_a8_n128_gate').replace('DFlashW8A8FullGate','DFlashW8A8N128Gate').replace('STAGE4B','STAGE4J').replace('Stage4B','Stage4J').replace('304','152').replace('{152,20,128,64}','{152,20,128,128}').replace('{152,20,64}','{152,20,128}').replace('mode,limit,r.y.desc','mode,limit,int64_t(-1),r.y.desc')
  expected=expected.replace('(limit!=0&&limit!=1&&limit!=2)','limit!=0').replace('only core limits 1/2/0 are benchmarked','Stage4K requires auto Core limit')
  assert (HERE/name/'runner.cpp').read_text()==expected
 assert not list(HERE.glob('op_kernel/*')) and not list(HERE.glob('op_host/*'))
 subprocess.run(['bash','-n',str(HERE/'run.sh')],check=True)

def test_all_tile_packing_relation_and_wrong_channel_rejected():
 import numpy as np
 oldn=(np.arange(304)[:,None]%2*64+np.arange(64)[None,:]).astype('i1')
 old={'b_i8.bin':np.broadcast_to(oldn[:,None,None,:],(304,20,128,64)),
      'quant_u64_doc_candidate.bin':np.broadcast_to(oldn.astype('<u8')[:,None,:],(304,20,64))}
 new={'b_i8.bin':np.broadcast_to(np.arange(128,dtype='i1'),(152,20,128,128)),
      'quant_u64_doc_candidate.bin':np.broadcast_to(np.arange(128,dtype='<u8'),(152,20,128))}
 k.verify_packing(old,new)
 bad=dict(new);bad['quant_u64_doc_candidate.bin']=new['quant_u64_doc_candidate.bin'].copy();bad['quant_u64_doc_candidate.bin'][151,19,127]^=np.uint64(1)
 with pytest.raises(ValueError,match='tile=151'):k.verify_packing(old,bad)
