"""CPU profiler/parser fixtures; never device performance evidence."""
import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import importlib.util
import sys


def modules(monkeypatch):
    tools=Path(__file__).resolve().parents[1]/'framework/custom_ops/draft_ac/test'
    monkeypatch.syspath_prepend(str(tools))
    spec=importlib.util.spec_from_file_location('common',tools/'common.py')
    common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    monkeypatch.setitem(sys.modules,'common',common)
    return tuple(importlib.import_module(n) for n in ('c_profile_graph','c_profile_trace','c_profile'))


def test_pbtxt_inspection_skips_large_constant_and_keeps_head_descriptors(tmp_path,monkeypatch):
    graph,_,_=modules(monkeypatch)
    path=tmp_path/'dynamo.pbtxt'
    path.write_text('''graph {
 op {
 name: "weight"
 type: "Const"
 output_desc { dtype: DT_FLOAT16 shape { dim: 248320 dim: 2560 } layout: "ND" }
 attr { key: "value" value { t {
 data: "'''+('x'*200000)+'''"
 } } }
 }
 op { name: "head_mm" type: "MatMulV2" input: "hidden:0" input: "weight:0" }
 op { name: "head_cast" type: "Cast" input: "head_mm:0" }
 op { name: "top" type: "ArgMaxV2" input: "head_cast:0" }
 op { name: "attn_mm" type: "MatMulV2" input: "attention:0" }
}''')
    summary=graph.summarize(graph.read_pbtxt(path),'pre_atc')
    assert summary['node_count']==5
    assert summary['head_anchors']==['head_mm']
    assert {n['name'] for n in summary['head_related_nodes']}=={'head_mm','head_cast','top','weight'}
    assert 'xxxxxx' not in json.dumps(summary)
    assert summary['category_counts']['matmul']==2


def test_compiled_json_preserves_layout_and_redacts_payload(tmp_path,monkeypatch):
    graph,_,_=modules(monkeypatch);p=tmp_path/'om.json'
    p.write_text(json.dumps({'graph':[{'op':[{'name':'head','type':'MatMulV2','input':[],
        'input_desc':[{'shape':{'dim':[248320,2560]},'layout':'FRACTAL_NZ','dtype':'DT_FLOAT16'}],
        'attr':{'value':{'t':{'data':'private weight bytes'}}}}]}]}))
    summary=graph.summarize(graph.read_om_json(p),'compiled_om')
    assert summary['head_anchors']==['head']
    assert 'FRACTAL_NZ' in json.dumps(summary) and 'private weight' not in json.dumps(summary)


def timeline(path,custom=False,missing=False):
    events=[dict(ph='M',name='process_name',pid=1,args={'name':'Ascend Hardware'})]
    for i in range(38):
        base=10000*i
        events.extend([dict(ph='X',name='aclmdlExecuteAsync',pid=2,tid=1,ts=base,dur=10),
                       dict(ph='X',name='aclrtSynchronizeStream',pid=2,tid=1,ts=base+10,dur=9000)])
        if custom:
            events.append(dict(ph='X',name='DFlashDraftLmHeadTop1',pid=1,tid=7,ts=base+100,dur=4000,args={'Op Type':'DFlashDraftLmHeadTop1'}))
        else:
            events.append(dict(ph='X',name='head_mm',pid=1,tid=7,ts=base+100,dur=2000,args={'Op Type':'MatMulV2'}))
            if not (missing and i==8): events.append(dict(ph='X',name='argmax',pid=1,tid=7,ts=base+2100,dur=500,args={'Op Type':'ArgMaxV2'}))
        # Overlap head work: naive sums must NOT count it as extra wall time.
        events.append(dict(ph='X',name='other',pid=1,tid=8,ts=base+200,dur=1000,args={}))
        # Readback/validation outside Execute/Sync must not be attributed to Head.
        events.append(dict(ph='X',name='Memcpy',pid=1,tid=8,ts=base+9200,dur=500,args={}))
    path.write_text(json.dumps({'traceEvents':events}))


def test_timeline_uses_30_windows_and_union_not_duration_sum(tmp_path,monkeypatch):
    _,trace,_=modules(monkeypatch);p=tmp_path/'msprof_1.json';timeline(p)
    result=trace.parse_timeline(p,{}, {'head_mm','argmax'})
    assert result['status']=='WINDOWS_IDENTIFIED' and result['head_mapping']=='COMPLETE_BY_LABEL'
    assert [c['index'] for c in result['iterations']]==list(range(7,37))
    metrics=result['metrics']
    assert metrics['head_occupancy_us']['median']==2500
    assert metrics['device_occupancy_us']['median']==2500 # not 3500 (overlap) or 4000 (readback)
    assert metrics['other_exclusive_occupancy_us']['median']==0
    assert metrics['host_sync_api_us']['median']==9000
    assert metrics['device_envelope_us']['median']==2500


def test_missing_head_iteration_and_multiple_exports_are_not_silently_accepted(tmp_path,monkeypatch):
    _,trace,_=modules(monkeypatch)
    p=tmp_path/'msprof_1.json';timeline(p,missing=True)
    assert trace.parse_timeline(p,{}, {'head_mm','argmax'})['head_mapping']=='INCOMPLETE'
    timeline(p);timeline(tmp_path/'msprof_2.json')
    result=trace.collect(tmp_path,{'head_related_nodes':[{'name':'head_mm'},{'name':'argmax'}]})
    assert result['selected_timeline'] is None and len(result['timelines'])==2


def test_csv_units_shapes_and_raw_pipe_counters(tmp_path,monkeypatch):
    _,trace,_=modules(monkeypatch)
    (tmp_path/'op_summary_1.csv').write_text('Op Name,OP Type,Task Duration(us),Input Shapes,Input Formats,aic_mte2_time(us)\n'
        'head,MatMulV2,8000,"15x2560;248320x2560","ND;FRACTAL_NZ",6000\n'
        'attention,MatMulV2,50,16x128,ND,10\n')
    result=trace.collect(tmp_path,{})
    rows=result['csv_tables'][0]['rows']
    assert rows[0]['label']=='head_matmul' and rows[0]['duration_us']==8000
    assert rows[1]['label']=='unmapped_matmul'
    assert rows[0]['fields']['aic_mte2_time(us)']=='6000'
    assert result['selected_timeline'] is None # no inferred per-call total from CSV alone


def accepted_fixture(tmp_path,tool,gear=16,mode="c"):
    source=tmp_path/'source';source.mkdir(exist_ok=True)
    def save(p,value): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p
    def rec(p,root): return dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=tool.sha256(p))
    run_report=dict(status='PASS',guards=True,readonly=True,repeat=True,cpu_fallback=False,row_permutation_checked=False,
        timing=dict(protocol='continuous-v1',warmup=5,repetitions=30,timed_tail_checked=True,
                    execute_sync=dict(median_ms=40.,p95_ms=41.,samples_ms=[40.]*30)))
    runner=tmp_path/'runner';runner.write_text('CPU fixture')
    folder=source/f'draft-{mode}-c{gear}';folder.mkdir()
    inputs=[dict(name='input'+str(i),dtype='float16',shape=[1]) for i in range(15)]
    outputs=[dict(name=n,dtype='int64' if n=='draft_top1' else 'float16',shape=[1]) for n in tool.OUTPUTS]
    counts={'DFlashGroupQuantLinear':26 if mode in ('a','ac') else 0,'DFlashDraftLmHeadTop1':int(mode in ('c','ac')),'DFlashDraftLmHeadTop1Audit':0}
    graph=dict(metadata={'draft_ac':{'mode':mode,'gear':gear},'tensor_abi':dict(inputs=inputs,outputs=outputs)},
        runtime_input_abi={'draft_ac':{'status':'PASS','counts':counts}},
        custom_op_audit=[{'ge_op_type':k,'ge_node_occurrences':v} for k,v in {**counts,'WeightQuantBatchMatmulV2':26-counts['DFlashGroupQuantLinear']}.items()],payload_files=[])
    artifacts=folder/'artifacts';artifacts.mkdir();om=artifacts/'graph.om';om.write_text('CPU fake OM')
    air=save(artifacts/'air.json',dict(status='PASS',graphs=[graph]))
    compiled={**copy.deepcopy(graph),'om':rec(om,artifacts)}
    dep=save(artifacts/'dep.json',dict(status='PASS',graphs=[compiled],air_manifest=dict(path=air.name,sha256=tool.sha256(air))))
    (folder/'input').mkdir()
    hashes={}
    for item in inputs:
        p=folder/'input'/(item['name']+'.bin');p.write_bytes(b'\0\0');hashes[item['name']]=tool.sha256(p)
    comparison={'status':'PASS','outputs':{}}
    for side in ('candidate','native'):
        directory=folder/side;directory.mkdir();save(directory/'execution.json',run_report)
        for item in outputs:
            value=b'\0'* (8 if item['dtype']=='int64' else 2)
            for suffix in ('actual-0.bin','actual-1.bin','benchmark-last.bin','postcheck.bin'):
                (directory/(item['name']+'-'+suffix)).write_bytes(value)
            digest=tool.sha256(directory/(item['name']+'-actual-0.bin'))
            comparison['outputs'][item['name']]=dict(equal=True,bytes=len(value),expected_sha256=digest,actual_sha256=digest)
        for name,rows in (('inputs.tsv',inputs),('outputs.tsv',outputs)):
            (directory/name).write_text(''.join(t['name']+' '+t['dtype']+' '+' '.join(map(str,t['shape']))+'\n' for t in rows))
    save(folder/'comparison.json',comparison)
    save(folder/'candidate/command.json',[str(runner),'0',str(om),str(folder/'candidate'),str(folder/'candidate/inputs')])
    row=dict(gear=gear,mode=mode,status='PASS',inputs=hashes,custom=run_report,native=run_report,
             artifacts={k:rec(p,folder) for k,p in [('air',air),('deployment',dep),('om',om)]})
    return source,row,runner


@pytest.mark.parametrize('damage',[None,'om','input','golden','audit','timing','tsv'])
def test_frozen_evidence_identity_and_correctness_are_required(tmp_path,monkeypatch,damage):
    _,_,tool=modules(monkeypatch);source,row,runner=accepted_fixture(tmp_path,tool)
    folder=source/'draft-c-c16'
    if damage=='om': (folder/'artifacts/graph.om').write_text('changed')
    elif damage=='input': (folder/'input/input0.bin').write_bytes(b'changed')
    elif damage=='golden': (folder/'native/draft_top1-postcheck.bin').write_bytes(b'bad')
    elif damage=='audit': row['custom']=copy.deepcopy(row['custom']);row['custom']['guards']=False
    elif damage=='timing': row['custom']=copy.deepcopy(row['custom']);row['custom']['timing']['protocol']='not-continuous'
    elif damage=='tsv': (folder/'candidate/outputs.tsv').write_text('changed')
    if damage:
        with pytest.raises(ValueError):tool.prepare_case(source,row,runner)
    else:
        case=tool.prepare_case(source,row,runner)
        assert case['historical']['median_ms']==40
        assert tool.check_outputs(case,folder/'candidate')['status']=='PASS'


def test_report_does_not_claim_root_cause_or_latency_gain_without_profile(tmp_path,monkeypatch):
    _,_,tool=modules(monkeypatch)
    report=dict(profiling_status='NOT_RUN',gears=[16],confirmed=[],unconfirmed=['needs timeline'],cases=[
        dict(gear=16,mode=m,historical={'median_ms':v,'p95_ms':v+1},profiling_status='NOT_RUN')
        for m,v in [('off',45.7116),('a',31.2055),('c',55.0590),('ac',40.5132)]])
    tool.finish(report,tmp_path)
    assert report['pairs'][0]['historical_delta_ms']==pytest.approx(9.3474)
    assert report['pairs'][1]['historical_delta_ms']==pytest.approx(9.3077)
    assert all(p['root_cause']=='UNPROVEN' for p in report['pairs'])
    assert 'NOT_RUN' in (tmp_path/'c-profile-analysis.md').read_text()


@pytest.mark.parametrize('prepare_only',[True,False])
def test_profile_entry_reuses_runner_and_oms_never_builds_or_exports(tmp_path,monkeypatch,prepare_only):
    import shlex,shutil,subprocess
    _,_,tool=modules(monkeypatch)
    from a4_contract import CASE_NAMES
    rows=[]
    for gear in (16,64):
        for mode in tool.MODES:
            source,row,runner=accepted_fixture(tmp_path,tool,gear,mode);rows.append(row)
    (source/'suite.json').write_text(json.dumps(dict(status='PASS',single_op_graph='PASS',draft_only='PASS')))
    (source/'draft-suite.json').write_text(json.dumps(dict(status='PASS',cases=rows)))
    (source/'a-probes.json').write_text(json.dumps(dict(status='PASS',cases=[dict(name=f'c{g}-{n}',status='PASS') for g in (16,64) for n in CASE_NAMES])))
    (source/'c-probes.json').write_text(json.dumps(dict(status='PASS',cases=[dict(gear=g,audit=a,status='PASS') for g in (16,64) for a in (False,True)])))
    reuse=dict(runner=str(runner),environment={})
    monkeypatch.setattr(tool,'load_reuse',lambda p:(SimpleNamespace(native_base=tmp_path/'native'),reuse))
    monkeypatch.setattr(tool,'verify_reuse',lambda r:None)
    profiler=tmp_path/'msprof';profiler.write_text('CPU mock')
    monkeypatch.setattr(tool.shutil,'which',lambda name:str(profiler) if name=='msprof' else None)
    calls=[]
    def run(command,**kwargs):
        assert command[0]==str(profiler) and any(a.startswith('--application=') for a in command)
        calls.append(command)
        app=shlex.split(next(a.split('=',1)[1] for a in command if a.startswith('--application=')))
        output=Path(app[3]);fixture=next(c for c in source.glob('draft-*') if c.name.split('-')[1]==output.parent.name.split('-')[1] and c.name.endswith('c16'))/'candidate'
        for p in fixture.iterdir():
            if p.suffix in ('.bin','.json') and p.name!='command.json':shutil.copyfile(p,output/p.name)
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(tool.subprocess,'run',run)
    before={str(p):tool.sha256(p) for p in source.rglob('*') if p.is_file()}
    output=tmp_path/'profile'
    tool.run(SimpleNamespace(evidence=source,output_dir=output,gears=[16],prepare_only=prepare_only,msprof='msprof',atc='atc',skip_om_dump=False,metrics='PipeUtilization'))
    report=json.loads((output/'c-profile-report.json').read_text())
    assert len(report['cases'])==4 and all(c['gear']==16 for c in report['cases'])
    assert len(calls)==(0 if prepare_only else 4)
    assert report['profiling_status']==('NOT_RUN' if prepare_only else 'COLLECTED')
    assert all('profile_delta_us' not in p for p in report['pairs'])
    assert before=={str(p):tool.sha256(p) for p in source.rglob('*') if p.is_file()}
    assert all(c['historical']['median_ms']==40 for c in report['cases'])


def test_begin_end_trace_and_incomplete_calls_are_explicit(tmp_path,monkeypatch):
    _,trace,_=modules(monkeypatch);p=tmp_path/'msprof.json';timeline(p,custom=True)
    payload=json.loads(p.read_text()); converted=[]
    for e in payload['traceEvents']:
        if e.get('ph')=='X' and e['name']=='aclmdlExecuteAsync':
            converted.extend([{**e,'ph':'B'},{**e,'ph':'E','ts':e['ts']+e['dur']}])
        else: converted.append(e)
    p.write_text(json.dumps({'traceEvents':converted}))
    result=trace.parse_timeline(p,{},set())
    assert result['status']=='WINDOWS_IDENTIFIED' and result['head_mapping']=='COMPLETE_BY_LABEL'
    converted=[e for e in converted if not (e.get('name')=='aclmdlExecuteAsync' and e.get('ts')==0)]
    p.write_text(json.dumps({'traceEvents':converted}))
    assert trace.parse_timeline(p,{},set())['status']=='UNRESOLVED_WINDOWS'
