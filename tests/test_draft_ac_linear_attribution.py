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
    assert r['analysis_status']=='UNRESOLVED_DEVICE_ASSIGNMENT' and r['host_windows']=='WINDOWS_VERIFIED' and r['device_execution']=='NOT_RUN_OFFLINE_ONLY'
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


@pytest.mark.parametrize('value,want',[
    ('"16,2560;80,1216,16,32;20,19456"',[[16,2560],[80,1216,16,32],[20,19456]]),
    ('"16,19456"',[[16,19456]]),
    (json.dumps(json.dumps('16,19456')),[[16,19456]]),
    ('[[16,2560],[2560,19456]]',[[16,2560],[2560,19456]]),
    ('[(16,2560),(2560,19456)]',[[16,2560],[2560,19456]]),
    ('junk [16,2560]',[]),('16,,2560',[]),('16 2560',[]),('16,-1',[]),
    ('16,2560;garbage',[]),('[true,16]',[]),('[16,2560.0]',[]),('"16,2560',[]),
])
def test_strict_quoted_shape_decode(monkeypatch,value,want):
    modules(monkeypatch);m=importlib.import_module('linear_attribution')
    assert m.shapes(value)==want


def test_real_csv_double_encoding_and_physical_stream_identity(tmp_path,monkeypatch):
    import csv
    _,trace,_=modules(monkeypatch);m=importlib.import_module('linear_attribution')
    p=tmp_path/'op_summary_1.csv'
    with p.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['Op Name','OP Type','Task Duration(us)','Stream Id','Task Id','Input Shapes','Output Shapes','Input Formats','Output Formats'])
        w.writerow(['gate','DFlashGroupQuantLinear',2060,13,581,'"16,2560;80,1216,16,32;20,19456"','"16,19456"','ND;FRACTAL_NZ;ND','ND'])
    _,meta=trace.csv_tables(tmp_path)
    event={'name':'compiled_task','args':{'Physic Stream Id':13,'Task Id':581}}
    label,name,kind=trace.event_label(event,meta,set())
    task=dict(name=name,op_type=kind,label=label,shape_evidence=trace.task_shapes(event,meta))
    assert m.role(task,'w8_a')[0]=='GATE_UP' and name=='gate'


def test_nz_requires_exact_frozen_node_dtype_format_and_shape_proof(monkeypatch):
    modules(monkeypatch);m=importlib.import_module('linear_attribution')
    def one(name='gate',dtype='FLOAT16',fmt='FRACTAL_NZ',output='"1152,1,16,16"'):
        return dict(name=name,op_type='MatMul',shape_evidence={'status':'CSV','fields':{'Input Shapes':'"160,1,16,16;160,1152,16,16"','Output Shapes':output,'Input Formats':fmt+';'+fmt,'Output Formats':fmt,'Input Data Types':dtype+';'+dtype,'Output Data Types':dtype}})
    mapping={'gate':{'type':'MatMul','fields':{'Input Shapes':[[16,2560]],'Output Shapes':[[16,18432]]}}}
    for task,expected in [(one(),'GATE_UP'),(one(name='gate_suffix'),'UNKNOWN'),(one(dtype='INT8'),'UNKNOWN'),(one(fmt='FRACTAL_Z'),'UNKNOWN'),(one(output='"1153,1,16,16"'),'UNKNOWN')]:
        assert m.role(task,'fp16')[0]=='UNKNOWN'
        m.add_graph_shapes({'timelines':[{'all_iterations':[{'tasks':[task]}]}]},mapping)
        assert m.role(task,'fp16')[0]==expected
    assert m.descriptor_matches([15520,1,16,16],[15,248320],'FRACTAL_NZ')


def step_timeline(path,model='fp16'):
    """CPU fixture: actual CANN event fields, deterministic shifted device clock.

    Head/ArgMax cross the next Host start. Device Step starts still anchor
    uniquely inside the issuing Host call. Connection IDs deliberately repeat.
    """
    events=[dict(ph='M',name='process_name',pid=1,args={'name':'Ascend Hardware'}),
        dict(ph='M',name='thread_name',pid=1,tid=99,args={'name':'Step Trace(Model ID:1)'}),
        dict(ph='M',name='thread_name',pid=1,tid=0,args={'name':'Stream 0'})]
    origin=Decimal('1792389259308275.990')
    layers=6 if model=='fp16' else 5
    meta={}
    for i in range(38):
        base=i*20000
        events += [dict(ph='X',name='AscendCL@aclmdlExecuteAsync',pid=2,tid=1,ts=str(origin+base),dur=20,args={'level':'acl','id':'aclmdlExecuteAsync','connection_id':1000+i*12}),
            dict(ph='X',name='AscendCL@aclrtSynchronizeStream',pid=2,tid=1,ts=str(origin+base+21),dur=19969,args={'level':'acl','id':'aclrtSynchronizeStream','connection_id':1003+i*12}),
            dict(ph='X',name=f'Iteration {i+1}',pid=1,tid=99,cat='Iteration Time',ts=str(origin+base+300),dur=19900.0001,args={'Iteration ID':i+1,'FP Start':'N/A','BP End':'N/A','Iteration End':str(origin+base+20200),'Iteration Time(us)':19900.0001})]
        dims={'GATE_UP':(2560,18432 if model=='fp16' else 19456),'DOWN':(9216 if model=='fp16' else 9728,2560),'OTHER_LINEAR':(4096,2560)}
        task_id=0
        for region,count in [('GATE_UP',layers),('DOWN',layers),('OTHER_LINEAR',1+3*layers)]:
            for j in range(count):
                k,n=dims[region];name=f'{region}_{j}';kind='MatMul' if model=='fp16' else 'DFlashGroupQuantLinear'
                fields={'Input Shapes':json.dumps(f'16,{k};{k},{n}'),'Output Shapes':json.dumps(f'16,{n}')}
                meta['name',name]=[dict(op_name=name,op_type=kind,fields=fields)]
                events.append(dict(ph='X',name=name,pid=1,tid=0,ts=str(origin+base+400+task_id*200),dur=100,args={'Model Id':1,'Task Type':'AI_CORE','Physic Stream Id':13,'Task Id':task_id,'Batch Id':0,'Subtask Id':4294967295,'connection_id':task_id+200}))
                task_id+=1
        for name,kind,offset,duration in [('head','MatMul',13000,7100),('argmax','ArgMaxV2',20101,80)]:
            meta['name',name]=[dict(op_name=name,op_type=kind,fields={'Input Shapes':'"15,2560;248320,2560"' if name=='head' else '"15,248320"','Output Shapes':'"15,248320"' if name=='head' else '"15"'})]
            events.append(dict(ph='X',name=name,pid=1,tid=0,ts=str(origin+base+offset),dur=duration,args={'Model Id':1,'Task Type':'AI_CORE','Physic Stream Id':13,'Task Id':581 if name=='head' else 582,'Batch Id':0,'Subtask Id':4294967295,'connection_id':560 if name=='head' else 561}))
    path.write_text(json.dumps(events));return events,meta


@pytest.mark.parametrize('model',['fp16','w8_a'])
def test_step_assignment_keeps_full_crossing_head_once_and_inventory(tmp_path,monkeypatch,model):
    _,trace,_=modules(monkeypatch);m=importlib.import_module('linear_attribution')
    p=tmp_path/'msprof.json';_,meta=step_timeline(p,model)
    r=trace.parse_timeline(p,meta,set())
    assert r['device_assignment']=='DEVICE_STEPS_VERIFIED'
    assert len(r['assignment_evidence']['excluded_iteration_markers'])==38
    analysis=m.analyze({'selected_timeline':r},model)
    assert analysis['status']=='ATTRIBUTION_VERIFIED' and analysis['head_inventory']=='COMPLETE'
    assert analysis['ffn_inventory']=='MATCHES_EXPECTED_LAYERS'
    assert analysis['region_statistics_us']['HEAD']['median']==7180
    assert len({t['event_index'] for c in r['all_iterations'] for t in c['tasks']})==sum(len(c['tasks']) for c in r['all_iterations'])
    for call in analysis['iterations'][7:37]:
        assert call['head_components']['matmul']['duration_us']==7100
        assert call['head_components']['argmax']['count']==1
        assert call['counts']['OTHER_LINEAR']==(19 if model=='fp16' else 16)
        assert all(t['cat']!='Iteration Time' for t in call['tasks'])
        assert call['device_occupancy_us']<call['device_window']['end_us']-call['device_window']['start_us']
        assert any(t['owner']==call['index']-1 for t in call['boundary_crossing_tasks'])


@pytest.mark.parametrize('damage',['missing_step','wrong_model','duplicate_task','bad_track','step_start','step_end','duplicate_step','task_outside_step'])
def test_unprovable_device_assignment_never_promotes_host_windows(tmp_path,monkeypatch,damage):
    _,trace,_=modules(monkeypatch);m=importlib.import_module('linear_attribution')
    p=tmp_path/'msprof.json';events,meta=step_timeline(p)
    marker=next(e for e in events if e.get('name')=='Iteration 8')
    task=next(e for e in events if e.get('name')=='head')
    if damage=='missing_step':events.remove(marker)
    elif damage=='wrong_model':task['args']['Model Id']=2
    elif damage=='duplicate_task':events.append(dict(task))
    elif damage=='bad_track':next(e for e in events if e.get('name')=='thread_name' and e['tid']==99)['args']['name']='unknown track'
    elif damage=='step_start':marker['ts']=str(Decimal(marker['ts'])-50000)
    elif damage=='step_end':marker['args']['Iteration End']=str(Decimal(marker['args']['Iteration End'])-100)
    elif damage=='duplicate_step':events.append(dict(marker))
    elif damage=='task_outside_step':task['dur']+=200
    p.write_text(json.dumps(events));r=trace.parse_timeline(p,meta,set())
    assert r['status']=='WINDOWS_IDENTIFIED' and len(r['iterations'])==30
    assert r['device_assignment']=='UNRESOLVED_DEVICE_ASSIGNMENT'
    a=m.analyze({'selected_timeline':r},'fp16')
    assert a['status']=='UNRESOLVED_DEVICE_ASSIGNMENT' and a['host_windows']=='WINDOWS_VERIFIED'


def test_actual_iteration_raw_timestamp_precision(monkeypatch):
    modules(monkeypatch);d=importlib.import_module('profile_device_assignment')
    marker=dict(event_index=1,name='Iteration 8',pid=391709088,tid=452528,ts=Decimal('370.833'),raw_timestamp_us='1792389259308646.823',dur=17210.65014527864,cat='Iteration Time',ph='X',args={'Iteration ID':8,'Iteration End':'1792389259325857.473','FP Start':'N/A','BP End':'N/A'})
    threads={(391709088,452528):{'Step Trace(Model ID:1)'}}
    assert d.step_marker(marker,threads)=={'model_id':'1','iteration_id':8}
    # Repeated graph connection IDs never establish call ownership.
    fake_task=dict(event_index=2,name='MatMul_31_ascend_mbatch_batch_0',pid=391709088,tid=0,ts=10617.812,dur=6782.187,args={'Model Id':1,'Task Type':'AI_CORE','Physic Stream Id':13,'Task Id':581,'connection_id':560})
    events=[dict(ph='M',name='thread_name',pid=391709088,tid=452528,args={'name':'Step Trace(Model ID:1)'})]
    owners,_,audit,_=d.assign(events,[marker,fake_task],[{'start':0,'launch_end':22.32,'end':17342.98}],'1792389259308275.990')
    assert not owners and audit['status']=='UNRESOLVED_DEVICE_ASSIGNMENT' # one sample is not 38 verified iterations
    assert not any('inconsistent Iteration End' in e['reason'] for e in audit['errors'])


def test_host_enclosed_tasks_without_steps_remain_unresolved(tmp_path,monkeypatch):
    _,trace,_=modules(monkeypatch);m=importlib.import_module('linear_attribution')
    p=tmp_path/'msprof.json';timeline(p)
    r=trace.parse_timeline(p,{}, {'head_mm','argmax'})
    assert r['status']=='WINDOWS_IDENTIFIED'
    assert r['device_assignment']=='UNRESOLVED_DEVICE_ASSIGNMENT'
    assert m.analyze({'selected_timeline':r},'fp16')['status']=='UNRESOLVED_DEVICE_ASSIGNMENT'


def test_full_device_assignment_still_requires_all_linear_and_head_nodes(tmp_path,monkeypatch):
    _,trace,_=modules(monkeypatch);m=importlib.import_module('linear_attribution')
    p=tmp_path/'msprof.json';events,meta=step_timeline(p,'w8_a')
    meta['name','OTHER_LINEAR_0'][0]['fields']={'Input Shapes':'"16,invalid"'}
    r=trace.parse_timeline(p,meta,set())
    a=m.analyze({'selected_timeline':r},'w8_a')
    assert a['device_assignment']=='DEVICE_STEPS_VERIFIED'
    assert a['status']=='INCOMPLETE_KERNEL_CLASSIFICATION'
    assert a['ffn_inventory']=='INCOMPLETE_OR_DIFFERENT_SHAPES'
    assert a['iterations'][7]['counts']['UNKNOWN']==1
    events=[e for e in events if e.get('name')!='argmax']
    p.write_text(json.dumps(events));r=trace.parse_timeline(p,meta,set())
    assert m.analyze({'selected_timeline':r},'w8_a')['head_inventory']=='INCOMPLETE'


def test_untrusted_iteration_name_is_quarantined_not_counted_as_kernel(tmp_path,monkeypatch):
    _,trace,_=modules(monkeypatch)
    p=tmp_path/'msprof.json';timeline(p)
    events=json.loads(p.read_text())['traceEvents']
    events.append(dict(ph='X',name='Iteration 8',pid=1,tid=44,ts=70100,dur=8800,args={}))
    p.write_text(json.dumps(events));r=trace.parse_timeline(p,{}, {'head_mm','argmax'})
    assert len(r['assignment_evidence']['unconfirmed_markers'])==1
    assert r['device_assignment']=='UNRESOLVED_DEVICE_ASSIGNMENT'
    assert all(not t['name'].startswith('Iteration ') for c in r['iterations'] for t in c['tasks'])
    assert r['metrics']['device_occupancy_us']['median']==2500
