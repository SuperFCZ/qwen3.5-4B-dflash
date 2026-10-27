"""CPU timing protocol tests; fixture runtimes/timings are never NPU evidence."""
import copy,json,os,shutil,subprocess,sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import bench_common as c
from profile_metrics import collect


def test_statistics_raw_samples_and_exact_protocol():
 x=c.statistics_ms(list(range(1,31)),30)
 assert x['median_ms']==15.5 and x['p95_ms']==29 and x['min_ms']==1 and x['max_ms']==30
 for data in ([1]*29,[1]*29+[0],[1]*29+[float('nan')]):
  with pytest.raises(ValueError):c.statistics_ms(data,30)
 assert c.ORDER==((1,2,0),(2,0,1),(0,1,2))


def test_preprocessing_scope_and_comparison_not_run(tmp_path):
 f=tmp_path/'data';f.write_bytes(b'fixture');proof=dict(files={n:c.identity(f) for n in ('a_i8.bin','b_i8.bin','quant_u64_doc_candidate.bin')})
 r=c.preprocessing(proof)
 assert r['status']=='ONLINE_PREPROCESSING_NOT_RUN'
 assert all(not x['included'] for x in r['components'] if x['kind']=='DYNAMIC_PER_INPUT')
 assert sum(x.get('bytes',0) for x in r['components'] if x['name']=='input_upload')==52961280
 result=c.a31_readiness(tmp_path/'missing.json',proof)
 assert result['status']=='NOT_RUN' and result['historical_numbers_used_as_samples'] is False


def test_raw_profile_fields_no_sum_or_invented_hotspots(tmp_path):
 (tmp_path/'op_summary.csv').write_text('Op Name,Task Duration(us),aic_mte2_time(us),aic_vec_ratio(%)\nDFlashW8A8FullGate,100,80,70\nOther,200,40,30\n')
 (tmp_path/'msprof_1.json').write_text(json.dumps([dict(name='DFlashW8A8FullGate',ph='X',pid=10,tid=0,ts='1000000000000000.123',dur=100,args={'Task Type':'AI_CORE','Task Id':1}),dict(name='Iteration 1',ph='X',ts=0,dur=200)]))
 r=collect(tmp_path)
 assert len(r['counter_rows'])==1 and len(r['device_tasks'])==1
 assert r['availability']['mte2']=='AVAILABLE_RAW_FIELDS' and r['availability']['cube']=='UNAVAILABLE'
 assert r['formal_latency_evidence'] is False and r['tail_analysis'].startswith('UNRESOLVED')
 assert r['device_tasks'][0]['event']['ts']=='1000000000000000.123'
 assert all(v.startswith('UNAVAILABLE') for v in r['instruction_hotspots'].values())


@pytest.fixture(scope='module')
def executable(tmp_path_factory):
 compiler=shutil.which('c++')
 if not compiler:pytest.skip('no local compiler')
 root=tmp_path_factory.mktemp('bench-cpu');exe=root/'runner'
 r=subprocess.run([compiler,'-std=c++17','-O1','-I'+str(HERE/'test/cpu_include'),'-I'+str(c.common.QUANT/'test'),str(HERE/'test/runner.cpp'),str(HERE/'test/cpu_api.cpp'),'-o',str(exe)],capture_output=True,text=True)
 assert r.returncode==0,r.stderr
 return exe


def inputs(folder):
 folder.mkdir()
 for name,size in [('a_i8.bin',40960),('b_i8.bin',49807360),('quant_u64_doc_candidate.bin',3112960),('golden_fp16.bin',622592)]:
  with (folder/name).open('wb') as f:f.truncate(size)
 return dict(files={'golden_fp16.bin':c.identity(folder/'golden_fp16.bin')})


@pytest.mark.parametrize('timed',[False,True])
def test_real_runner_continuous_schedule_has_no_timed_copies(executable,tmp_path,timed):
 folder=tmp_path/'case';proof=inputs(folder);env=os.environ.copy();env['DFLASH_STAGE4B_EXPECTED_CORE_POLICY']='8,8,7,7,2002'
 argv=[str(executable),'0',str(folder),'1']+(['--benchmark'] if timed else [])
 with (folder/'runner.log').open('w') as log:r=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode==0,(folder/'runner.log').read_text()
 ex=json.loads((folder/'execution.json').read_text());stats=c.check_execution(ex,0,1,timed)
 assert len(c.verify_outputs(proof,folder,timed))==(4 if timed else 2)
 assert ex['prepare_calls']==(38 if timed else 2) and ex['verify_calls']==(4 if timed else 2)
 if timed:
  assert len(stats['execute_sync']['samples_ms'])==30 and len(stats['prepare']['samples_ms'])==30
  bad=copy.deepcopy(ex);bad['timing']['per_timed_call_readback']=True
  with pytest.raises(ValueError):c.check_execution(bad,0,1,True)
  bad=copy.deepcopy(ex);bad['timing']['timed_tail_checked']=False
  with pytest.raises(ValueError):c.check_execution(bad,0,1,True)
  (folder/'postcheck.bin').write_bytes(bytes([1])*622592)
  with pytest.raises(ValueError,match='Golden'):c.verify_outputs(proof,folder,True)


def test_bad_precheck_cannot_enter_timing(executable,tmp_path):
 folder=tmp_path/'bad';inputs(folder);env=os.environ.copy();env.update(DFLASH_STAGE4B_EXPECTED_CORE_POLICY='8,8,7,7,2002',TEST_BAD_GOLDEN='1')
 with (folder/'runner.log').open('w') as log:r=subprocess.run([str(executable),'0',str(folder),'1','--benchmark'],env=env,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode!=0
 ex=json.loads((folder/'execution.json').read_text());assert ex['status']=='FAIL' and ex['timing']['execute_sync']['samples_ms']==[]
 assert not (folder/'benchmark-last.bin').exists()


def test_timed_sdk_logging_is_reported_not_silently_muted(executable,tmp_path):
 folder=tmp_path/'logged';inputs(folder);env=os.environ.copy();env.update(DFLASH_STAGE4B_EXPECTED_CORE_POLICY='8,8,7,7,2002',TEST_TIMED_LOG='1')
 with (folder/'runner.log').open('w') as log:r=subprocess.run([str(executable),'0',str(folder),'1','--benchmark'],env=env,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode!=0 and 'reject formal timing' in (folder/'runner.log').read_text()
 ex=json.loads((folder/'execution.json').read_text());assert ex['status']=='FAIL' and len(ex['timing']['execute_sync']['samples_ms'])==30
 assert (folder/'benchmark-last.bin').exists() and not (folder/'postcheck.bin').exists()


def test_profiled_or_incomplete_reports_cannot_authorize_next_stage(tmp_path):
 import run_stage4c as run
 base=dict(sources=[],evidence={},device=0,runner_build_report={})
 p=tmp_path/'smoke.json'
 c.write(p,dict(status='STAGE4C_SMOKE_MATCH',phase='complete',**base,runs=[dict(core_limit=x,status='FROZEN_GOLDEN_MATCH',profiled=True) for x in (1,2,0)]))
 with pytest.raises(ValueError,match='profile scope'):run.prior_report(p,'STAGE4C_SMOKE_MATCH',base)
 c.write(p,dict(status='STAGE4C_SMOKE_MATCH',phase='complete',**base,runs=[]))
 with pytest.raises(ValueError,match='coverage'):run.prior_report(p,'STAGE4C_SMOKE_MATCH',base)


def test_profile_na_is_unavailable_but_zero_is_preserved(tmp_path):
 (tmp_path/'summary.csv').write_text('Op Name,Task Duration(us),aic_mte1_time(us),aic_mte2_time(us),Machine Id\nDFlashW8A8FullGate,N/A,N/A,0,42\n')
 r=collect(tmp_path)
 assert r['availability']['duration']=='UNAVAILABLE' and r['availability']['mte1']=='UNAVAILABLE'
 assert r['availability']['mte2']=='AVAILABLE_RAW_FIELDS' and r['availability']['mac']=='UNAVAILABLE'
 assert r['counter_rows'][0]['fields']['aic_mte1_time(us)']=='N/A'
