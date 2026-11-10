from pathlib import Path
import sys
import numpy as np
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import perf_contract as c
import run_perf as run


def inputs():
 raw=np.arange(4096,dtype='<i4').reshape(16,256)-2048
 f=np.full((16,256),1,dtype='<f2');g=f.copy();g[0,84]=np.nextafter(np.float16(1),np.float16(2))
 return [raw.copy(),raw.copy()],[f.copy(),f.copy()],raw,f,g,{k:True for k in c.SAFETY}


def test_one_ulp_does_not_become_numeric_pass_or_forbid_exploration():
 args=inputs();r=c.smoke_observation(*args)
 assert r['exploration_only'] and not r['numeric_cpu_bitwise'] and r['quality']=='UNRESOLVED'
 assert all(x['cpu']['bit_mismatches']==1 and x['cpu']['max_ulp']==1 for x in r['comparisons'])
 assert r['status']=='OFFLINE_OBSERVATION_ANALYZED_NOT_DEVICE_ADMISSION'


@pytest.mark.parametrize('key',c.SAFETY)
def test_safety_still_mandatory(key):
 args=inputs();args[-1][key]=False
 with pytest.raises(ValueError):c.smoke_observation(*args)


def test_raw_repeat_poison_finiteness_and_completeness():
 args=inputs();args[0][1][15,255]+=1
 with pytest.raises(ValueError):c.smoke_observation(*args)
 for bad in (np.nan,np.inf,-np.inf):
  args=inputs();args[1][0][0,0]=bad;args[1][1][0,0]=bad
  with pytest.raises(ValueError):c.smoke_observation(*args)
 args=inputs();args[1][1][1,1]=2
 with pytest.raises(ValueError):c.smoke_observation(*args)
 args=inputs();args[1][0]=args[1][0][:8];args[1][1]=args[1][1][:8]
 with pytest.raises(ValueError):c.smoke_observation(*args)
 args=inputs();args[0].pop()
 with pytest.raises(ValueError):c.smoke_observation(*args)


def test_bitwise_negative_zero_and_negative_ulp():
 a=np.full((16,256),-1,dtype='<f2');b=a.copy();b[0,0]=np.nextafter(np.float16(-1),np.float16(0))
 assert c.compare_half(a,b)['max_ulp']==1
 a[0,0]=np.float16(-0.);b[0,0]=np.float16(0.)
 r=c.compare_half(a,b);assert r['bit_mismatches']==1 and r['max_ulp']==0


def test_historical_stride_is_not_flat_tile():
 a=np.zeros((16,19456),dtype='<f2');a[:,256:]=17;a[1,0]=23
 t=c.historical_tile0(a.tobytes());assert t[1,0]==23 and t[0,-1]==0
 with pytest.raises(ValueError):c.historical_tile0(t.tobytes())


def rounds():
 return [dict(round=i,warmup=5,repetitions=30,execute_sync_ms=[.02+j*.001 for j in range(30)],prepare_ms=[1.0],
 instrumented=False,host_logs_during_timing=0,workspace_drift=False,two_poison_prechecks=True,timed_tail=True,
 poison_postcheck=True,safety=True,identities_bound=True) for i in range(3)]


def test_statistics_no_samples_from_other_runs_or_device_inference():
 r=c.benchmark_summary(rounds());assert r['execute_sync']['count']==90
 assert len(r['execute_sync']['samples_ms'])==90 and r['execute_sync']['median_ms']==pytest.approx(.0345)
 assert r['execute_sync']['p95_ms']==pytest.approx(.048)
 assert r['device_execution_time'].startswith('UNAVAILABLE') and r['layer_projection']=='PROHIBITED'
 assert all(x['prepare']['median_ms']==1 for x in r['rounds'])


@pytest.mark.parametrize('key,value',[('instrumented',True),('host_logs_during_timing',1),('workspace_drift',True),('warmup',4),('repetitions',29),('round',7),('two_poison_prechecks',False),('timed_tail',False),('poison_postcheck',False),('safety',False),('identities_bound',False)])
def test_invalid_timing_round(key,value):
 r=rounds();r[0][key]=value
 with pytest.raises(ValueError):c.benchmark_summary(r)


@pytest.mark.parametrize('bad',[0,-1,float('nan'),float('inf'),True])
def test_invalid_timing_samples(bad):
 r=rounds();r[0]['execute_sync_ms'][3]=bad
 with pytest.raises(ValueError):c.benchmark_summary(r)


def test_required_three_rounds_no_fake_success():
 with pytest.raises(ValueError):c.benchmark_summary(rounds()[:2])
 assert not c.admission()['device_execution_allowed']
 assert c.admission()['smoke']==c.admission()['benchmark']=='NOT_RUN'
 assert c.admission()['ub_accounted']==240384 and c.admission()['reserve']==8192


@pytest.mark.parametrize('phase',['host','build','smoke','benchmark'])
def test_execution_gates_cannot_invoke_sdk_or_be_opened_by_cli(tmp_path,monkeypatch,phase):
 def forbidden(*a,**k):raise AssertionError('device-facing work forbidden')
 monkeypatch.setattr(run,'binding',forbidden)
 monkeypatch.setattr(run.subprocess,'run',forbidden)
 a=run.parse([phase,'--output-dir',str(tmp_path/phase)])
 assert run.main(a)==2
 r=run.e.read(tmp_path/phase/'report.json')
 assert r['status']=='BLOCKED_NORMAL_PATH_EVENT_BUDGET' and not r['device_execution_allowed']
 assert r['build']==r['smoke']==r['benchmark']=='NOT_RUN'


def test_no_device_assets_or_sdk_source_snapshots():
 assert not any((HERE/p).exists() for p in ('op_kernel','op_host','runner','operator.json'))
 text=(HERE/'compile_check/interfaces.cpp').read_text()
 assert '__global__ void' not in text and 'int main(' not in text
