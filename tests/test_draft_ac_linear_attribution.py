"""Offline-only CANN9 API pairing/shape classification tests."""
import importlib
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_draft_ac_c_profile import modules,timeline


def prefixed(path,fp16=True):
    timeline(path);data=json.loads(path.read_text());origin=Decimal('1792389259308275.990')
    for e in data['traceEvents']:
        if 'ts' in e:e['ts']=str(origin+Decimal(str(e['ts'])))
        if e.get('name') in ('aclmdlExecuteAsync','aclrtSynchronizeStream'):
            api=e['name'];e['name']='AscendCL@'+api
            e['args']=dict(level='acl',id=api,connection_id=len(str(e)),**{'Thread Id':1})
    if fp16:data['traceEvents'].append(dict(ph='X',pid=2,tid=1,name='AscendCL@aclrtSynchronizeStream',ts=str(origin-100),dur=5,args={'level':'acl','id':'aclrtSynchronizeStream'}))
    data['traceEvents'].append(dict(ph='X',pid=2,tid=1,name='AscendCL@aclrtSynchronizeStream',ts=str(origin+379500),dur=5,args={'level':'acl','id':'aclrtSynchronizeStream'}))
    data['traceEvents'].append(dict(ph='M',name='process_name',pid=1,args={'name':'Ascend Hardware'}))
    path.write_text(json.dumps(data));return data


@pytest.mark.parametrize('fp16',[True,False])
def test_observed_ascendcl_names_38_exec_40_or_39_sync_not_zipped(tmp_path,monkeypatch,fp16):
    _,trace,_=modules(monkeypatch);p=tmp_path/'msprof.json';prefixed(p,fp16)
    r=trace.parse_timeline(p,{},set())
    assert r['status']=='WINDOWS_IDENTIFIED' and r['execute_count']==38 and r['sync_count']==(40 if fp16 else 39)
    assert len(r['all_iterations'])==38 and [c['index'] for c in r['iterations']]==list(range(7,37))
    assert len(r['unpaired_syncs'])==(2 if fp16 else 1)
    c=r['iterations'][0]
    assert Decimal(c['sync_api']['timestamp_us'])-Decimal(c['execute_api']['timestamp_us'])==10
    assert c['phase']=='timed' and r['all_iterations'][37]['phase']=='postcheck'


def test_epoch_fraction_preserved_and_conflicting_id_rejected(tmp_path,monkeypatch):
    _,trace,_=modules(monkeypatch);p=tmp_path/'msprof.json';d=prefixed(p)
    execute=next(e for e in d['traceEvents'] if e['name']=='AscendCL@aclmdlExecuteAsync')
    d['traceEvents'].append(dict(ph='M',name='thread_name',pid=2,tid=1,ts=0,args={'name':'host'}))
    execute['dur']=9.98
    sync=next(e for e in d['traceEvents'] if e['name']=='AscendCL@aclrtSynchronizeStream')
    sync['ts']=str(Decimal(execute['ts'])+Decimal('10.01'))
    p.write_text(json.dumps(d));r=trace.parse_timeline(p,{},set())
    assert r['all_iterations'][0]['host_window_us']==pytest.approx(9010.01)
    execute['args']['id']='aclrtSynchronizeStream';p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='conflicting'):trace.parse_timeline(p,{},set())


def test_wrong_thread_and_duplicate_execute_never_claim_windows(tmp_path,monkeypatch):
    _,trace,_=modules(monkeypatch);p=tmp_path/'msprof.json';d=prefixed(p)
    for e in d['traceEvents']:
        if e['name']=='AscendCL@aclrtSynchronizeStream':e['tid']=99
    p.write_text(json.dumps(d));assert trace.parse_timeline(p,{},set())['status']=='UNRESOLVED_WINDOWS'
    d=prefixed(p);d['traceEvents'].append(next(e for e in d['traceEvents'] if e['name']=='AscendCL@aclmdlExecuteAsync'))
    p.write_text(json.dumps(d));assert trace.parse_timeline(p,{},set())['execute_count']==39


def test_shape_roles_respect_distinct_fp16_checkpoint_and_unknown(monkeypatch):
    modules(monkeypatch);m=importlib.import_module('linear_attribution')
    def task(k,n):return dict(op_type='MatMulV2',shape_evidence={'status':'CSV','fields':{'Input Shapes':f'1,16,{k};{k},{n}','Output Shapes':f'1,16,{n}'}})
    assert m.role(task(2560,19456),'w8_a')[0]=='GATE_UP'
    assert m.role(task(9728,2560),'w8_a')[0]=='DOWN'
    assert m.role(task(2560,18432),'fp16')[0]=='GATE_UP'
    assert m.role(task(9216,2560),'fp16')[0]=='DOWN'
    assert m.role(task(2560,19456),'fp16')[0]=='UNKNOWN'
    assert m.role(dict(op_type='MatMulV2',duration_us=2060),'w8_a')[0]=='UNKNOWN'
    assert m.role(task(4096,2560),'w8_a')[0]=='OTHER_LINEAR'
    assert m.intervals_overlap({'gate':[(0,10)],'down':[(5,15)]})==5


def test_offline_entry_no_subprocess_no_evidence_writes(tmp_path,monkeypatch):
    modules(monkeypatch);m=importlib.import_module('linear_attribution');c=importlib.import_module('profile_recovery_checks')
    sources={};rows=[]
    for model in ('fp16','w8_a'):
        p=tmp_path/model/'profile'/'profiler';p.mkdir(parents=True);prefixed(p/'msprof.json',model=='fp16')
        (p/'op_summary.csv').write_text('Op Name,OP Type,Task Duration(us),Input Shapes,Output Shapes\nhead_mm,MatMulV2,2000,"15,2560;2560,248320","15,248320"\n')
        sources[model]=p
        rows.append(dict(model=model,profile_origin=str(p.parent),unprofiled_latency={'median_ms':17 if model=='fp16' else 31}))
    rows[1]['profiler_inventory']={'files':c.snapshot(sources['w8_a'],profiler=True)}
    report=tmp_path/'old'/'report.json';report.parent.mkdir();report.write_text(json.dumps(dict(status='DIAGNOSTIC_ONLY',historical_fp16_status='FAIL_REPLAY_CHECKS',cases=rows,
        recovery={'fp16_profiler_integrity':{'files':c.snapshot(sources['fp16'],profiler=True)}})))
    monkeypatch.setattr(m,'graph_shapes',lambda *a:({},dict(status='UNPARSED')))
    import subprocess
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:pytest.fail('offline analysis must not execute a program'))
    before={k:c.snapshot(v,profiler=True) for k,v in sources.items()}
    m.run(SimpleNamespace(report=report,output_dir=tmp_path/'new'))
    r=json.loads((tmp_path/'new/linear-timeline.json').read_text())
    assert r['analysis_status']=='WINDOWS_VERIFIED' and r['device_execution']=='NOT_RUN_OFFLINE_ONLY'
    for model in sources:
        assert len(r['models'][model]['analysis']['iterations'])==38
        assert r['models'][model]['analysis']['region_statistics_us']['HEAD']['median']==2000
        assert c.snapshot(sources[model],profiler=True)==before[model]


def test_unavailable_csv_shapes_allow_only_exact_frozen_graph_fallback(monkeypatch):
    _,trace,_=modules(monkeypatch);m=importlib.import_module('linear_attribution')
    event={'name':'gate','args':{}}
    meta={('name','gate'):[{'fields':{'Input Shapes':'N/A','Output Shapes':'N/A'}}]}
    evidence=trace.task_shapes(event,meta);assert evidence['status']=='MISSING'
    task=dict(name='gate',op_type='DFlashGroupQuantLinear',shape_evidence=evidence)
    profile={'timelines':[{'all_iterations':[{'tasks':[task]}]}]}
    mapping={'gate':{'type':'DFlashGroupQuantLinear','fields':{'Input Shapes':[[16,2560]],'Output Shapes':[[16,19456]]}}}
    m.add_graph_shapes(profile,mapping)
    assert m.role(task,'w8_a')[0]=='GATE_UP'
    assert m.shapes('[(16,2560),(19456,2560)]')==[[16,2560],[19456,2560]]


def test_frozen_graph_shape_fallback_uses_exported_name_and_hash(tmp_path,monkeypatch):
    modules(monkeypatch);m=importlib.import_module('linear_attribution')
    evidence=tmp_path/'evidence';folder=evidence/'draft-a-c16';artifacts=folder/'artifacts';artifacts.mkdir(parents=True)
    pb=artifacts/'dynamo.pbtxt'
    x='name: "x"\ndtype: DT_FLOAT16\nshape { dim:16 dim:2560 }\n'
    y='name: "y"\ndtype: DT_FLOAT16\nshape { dim:16 dim:19456 }\n'
    pb.write_text('node { name:"gate" op:"DFlashGroupQuantLinear" attr{key:"input_desc_0" value{s:'+repr(x)+'}} attr{key:"output_desc_0" value{s:'+repr(y)+'}}}')
    def rec(p,root):return dict(path=str(p.relative_to(root)),sha256=m.sha256(p),bytes=p.stat().st_size)
    air=artifacts/'air.json';air.write_text(json.dumps({'graphs':[{'name':'draft_ac_a_c16','payload_files':[rec(pb,artifacts)]}]}))
    report={'cases':[{'model':'w8_a','model_identity':{'evidence':str(evidence),'artifacts':{'air':rec(air,folder)}}}],
            'source_watch':{str(air):m.sha256(air),str(pb):m.sha256(pb)}}
    mapping,audit=m.graph_shapes(report,'w8_a')
    assert audit['status']=='PARSED' and mapping['gate']['fields']['Output Shapes']==[[16,19456]]
    pb.write_text('tampered')
    with pytest.raises(ValueError,match='changed'):m.graph_shapes(report,'w8_a')
