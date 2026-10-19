"""CPU-only structural parser and recovery tests; no device claims."""
import copy
import importlib
import json
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace
import pytest
from test_draft_ac_c_profile import modules,accepted_fixture,timeline
from test_draft_ac_c16_diagnostic import failed,fake_gear


def test_textformat_message_lists_adjacent_strings_any_and_comments(tmp_path,monkeypatch):
    graph,_,_=modules(monkeypatch);p=tmp_path/'dynamo.pbtxt'
    p.write_text('''graph { op {
 name: "head_" 'mm' type: "MatMulV2"
 input: ["x:0", "w:0"]
 input_desc { dtype: DT_FLOAT16 shape { dim: [2560, 248320] } }
 attr { key: "descriptors" value { list {
   td: [{name: "x" shape < dim: [15, 2560] >},
        {name: "w" shape { dim: [248320, 2560] }}]
 } } }
 attr { key: "any" value { [type.googleapis.com/ge.Tensor] { f: 1.25e-05 i: - 2 } } }
 # The old lexer also consumed comments as field/value tokens: { } nonsense
} }''')
    nodes=graph.read_pbtxt(p)
    assert nodes[0]['name']=='head_mm' and nodes[0]['inputs']==['x:0','w:0']
    values=list(graph.values(nodes[0]));assert 248320 in values and 1.25e-5 in values and -2 in values
    assert graph.summarize(nodes,'pre_atc')['head_anchors']==['head_mm']
    assert None not in nodes[0]['attributes'][0]['value'][0]['list'][0]
    assert len(nodes[0]['attributes'][0]['value'][0]['list'][0]['td'])==2


def test_bounded_inline_payload_keeps_same_line_closing_braces(tmp_path,monkeypatch):
    graph,_,_=modules(monkeypatch);p=tmp_path/'dynamo.pbtxt'
    p.write_text('graph { op { name:"w" type:"Const" attr { key:"value" value {t{data:"'+('\\377'*50000)+'"}}} } op {name:"m" type:"MatMulV2"} }')
    nodes=graph.read_pbtxt(p)
    assert [n['name'] for n in nodes]==['w','m'] and 'data' not in json.dumps(nodes)


@pytest.mark.parametrize('text',['graph {op {name:"x" attr { {key:"bad"} }}}','graph {op {name:"x" type:"MatMulV2"','graph {op { name: "x" shape { dim: [1 2] } }}'])
def test_malformed_textformat_is_unparsed_not_null_key(tmp_path,monkeypatch,text):
    graph,_,_=modules(monkeypatch);p=tmp_path/'bad.pbtxt';p.write_text(text)
    with pytest.raises(ValueError,match='pbtxt line'):graph.read_pbtxt(p)
    for function in (graph.redact,lambda value:list(graph.values(value))):
        with pytest.raises(ValueError,match='field key'):function({None:{'shape':[248320]}})


def test_metadata_failure_keeps_unknown_mapping(tmp_path,monkeypatch):
    modules(monkeypatch);m=importlib.import_module('profile_metadata');p=tmp_path/'dynamo.pbtxt';p.write_text('graph { op { { } } }')
    out={};graph,heads,linear=m.inspect({'payloads':[p]},out,tmp_path)
    assert graph=={} and not heads and not linear and out['pre_atc_graph']['status']=='UNPARSED'
    assert out['mapping_source']=='UNKNOWN'


def fixture(tmp_path,monkeypatch):
    ev,d,dep,replay=failed(tmp_path,monkeypatch);_,_,cp=modules(monkeypatch)
    recovery=importlib.import_module('recover_c16_profile');checks=importlib.import_module('profile_recovery_checks')
    meta=importlib.import_module('profile_metadata');import numpy as np
    source,row,wrunner=accepted_fixture(tmp_path,cp,16,'a')
    for side in ('candidate','native'):
        row['custom']['timing']['execute_sync']['p95_ms']=40.
        (source/f'draft-a-c16/{side}/execution.json').write_text(json.dumps(row['custom']))
    (source/'suite.json').write_text(json.dumps(dict(status='PASS')))
    (source/'draft-suite.json').write_text(json.dumps(dict(status='PASS',cases=[row])))
    reuse=dict(source=str(source),runner=str(wrunner),environment={'ASCEND_CUSTOM_OPP_PATH':'accepted','LD_LIBRARY_PATH':'accepted/lib'})
    oldargs=SimpleNamespace(native_base=tmp_path/'native')
    for module in (d,recovery):
        monkeypatch.setattr(module,'load_reuse',lambda s:(oldargs,reuse));monkeypatch.setattr(module,'verify_reuse',lambda r:None)
    true_controls=d.controls
    control=lambda c:true_controls(c) if c['model']=='fp16' else dict(start_position=0,valid_rows=16,proposal_count=15)
    monkeypatch.setattr(d,'controls',control);monkeypatch.setattr(recovery,'controls',control)
    original=tmp_path/'interrupted';fp=d.fp16_case(replay,tmp_path,dep)
    actual_build=d.build_runner
    def build(folder,env,reuse=None):
        folder.mkdir();binary=folder/'c16_diagnostic_runner';binary.write_bytes(b'CPU fixture')
        paths=['framework/custom_ops/draft_ac/test/fp16_dynamic_runner.cpp','framework/custom_ops/draft_ac/test/c16_diagnostic_metrics.h',
               'framework/custom_ops/draft_ac/test/c16_diagnostic/CMakeLists.txt','framework/custom_ops/draft_quant/test/runner_common.h']
        r=dict(status='DIAGNOSTIC_ONLY',build_kind='c16_diagnostic_only',sources={p:ev.sha256(d.ROOT/p) for p in paths},
               runner=dict(path=str(binary),sha256=ev.sha256(binary)),cann_root='fixture')
        (folder/'build.json').write_text(json.dumps(r));return binary,r
    monkeypatch.setattr(d,'build_runner',build)
    def materialize(folder,phase):
        names=[f'numerics-{i}' for i in range(30)] if phase=='numerics' else list(recovery.SUFFIXES)
        observed=[]
        for i,name in enumerate(names):
            outputs={}
            for spec in fp['abi']['outputs']:
                key=spec['name'];ref=fp['golden'][key]
                if key=='draft_top1':
                    data=ref.read_bytes();outputs[key]=dict(bitwise_equal=True,mismatches=0,actual_ids=[0]*15)
                else:
                    b=np.memmap(ref,dtype='<u2',mode='r',shape=tuple(spec['shape']));a=np.array(b);a.flat[0]=0x3c00 if i%2==0 else 0x4000
                    data=a.tobytes();x=np.memmap(fp['inputs'][key],dtype='<u2',mode='r',shape=tuple(spec['shape']))
                    regions={'all':(a,b),'valid_prefix':(a[:,:,:16,:],b[:,:,:16,:]),'written_valid':(a[:,:,:16,:],b[:,:,:16,:]),
                        'written_padding':(a[:,:,16:16,:],b[:,:,16:16,:]),'untouched_prefix_vs_input':(a[:,:,:0,:],x[:,:,:0,:]),
                        'untouched_tail_vs_input':(a[:,:,16:,:],x[:,:,16:,:])}
                    outputs[key]={'against':'historical_native_reference_not_accepted_golden',**{k:{**checks.tensor_metrics(*v),'error_domain':'finite pairs only; nonfinite counts never discarded'} for k,v in regions.items()}}
                if phase!='numerics':(folder/(key+'-'+name)).write_bytes(data)
            observed.append(dict(status='DIAGNOSTIC_ONLY',observation=name,outputs=outputs))
        (folder/'observations.jsonl').write_text(''.join(json.dumps(o)+'\n' for o in observed))
        r=dict(status='DIAGNOSTIC_ONLY',guards=True,readonly=True,cpu_fallback=False,repeat_equal=False,observed_calls=len(names))
        if phase=='numerics':r['phase']='numerics'
        else:r['timing']=copy.deepcopy(row['custom']['timing'])
        (folder/'execution.json').write_text(json.dumps(r));fake_gear(fp,folder)
    def profiler(folder):
        folder.mkdir();timeline(folder/'msprof_1.json')
        (folder/'op_summary_1.csv').write_text('Op Name,OP Type,Task Duration(us)\nhead_mm,MatMulV2,2000\nargmax,ArgMaxV2,500\n')
    calls=[]
    def execute(command,**kw):
        if command[0]=='npu-smi' or command[-1]=='--version':return subprocess.CompletedProcess(command,0)
        profile=command[0]=='msprof'
        app=shlex.split(next(x.split('=',1)[1] for x in command if x.startswith('--application='))) if profile else command
        folder=Path(app[3]);model=folder.parent.name;phase=folder.name;calls.append((model,phase,profile))
        if model=='fp16':materialize(folder,phase)
        else:
            import shutil
            for f in (source/'draft-a-c16/candidate').iterdir():
                if f.name=='execution.json' or f.suffix=='.bin':shutil.copyfile(f,folder/f.name)
        if profile:profiler(folder/'profiler')
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(d.subprocess,'run',execute)
    real_inspect=meta.inspect
    monkeypatch.setattr(meta,'inspect',lambda *a,**k:(_ for _ in ()).throw(AttributeError("'NoneType' object has no attribute 'lower'")))
    args=SimpleNamespace(diagnostic_only=True,w8_evidence=source,output_dir=original,fp16_replay=replay,
        search_root=tmp_path,fp16_deployment=dep,prepare_only=False,device_id=0,msprof='msprof',atc='atc',diagnostic_build=None)
    with pytest.raises(AttributeError):d.run(args)
    monkeypatch.setattr(meta,'inspect',real_inspect);monkeypatch.setattr(d,'build_runner',actual_build)
    assert calls==[('fp16','numerics',False),('fp16','timing',False),('w8_a','timing',False),('fp16','profile',True)]
    calls.clear()
    return recovery,checks,original,calls


@pytest.mark.parametrize('validate_only',[False,True])
def test_recovery_runs_only_missing_w8_msprof_and_preserves_every_original_file(tmp_path,monkeypatch,validate_only):
    r,c,source,calls=fixture(tmp_path,monkeypatch);before=c.snapshot(source)
    args=SimpleNamespace(source_evidence=source,output_dir=tmp_path/'recovered',expected_fp16_profiler_files=2,msprof=None,validate_only=validate_only)
    r.run(args);out=json.loads((args.output_dir/'c16-diagnostic-report.json').read_text());old=json.loads((source/'c16-diagnostic-report.json').read_text())
    assert calls==([] if validate_only else [('w8_a','profile',True)])
    assert c.snapshot(source)==before and out['status']=='DIAGNOSTIC_ONLY' and out['historical_fp16_status']=='FAIL_REPLAY_CHECKS'
    assert out['end_to_end']=='NOT_RUN' and out['recovery']['fp16_execution']=='NOT_RERUN'
    for row,original in zip(out['cases'],old['cases']):assert row['unprofiled_latency']==original['unprofiled_latency']
    assert out['cases'][0]['profile_status']=='REUSED_OFFLINE'
    assert out['cases'][0]['attribution']['unknown_ratio']['union_fraction']['mean']==1


@pytest.mark.parametrize('damage',['output','input','observations','runner','timing','profile_missing','profile_json','profile_csv','report'])
def test_corruption_blocks_before_any_profile_execution(tmp_path,monkeypatch,damage):
    r,c,source,calls=fixture(tmp_path,monkeypatch)
    paths={'output':'fp16/timing/d0_key-actual-0.bin','input':'fp16/timing/inputs/features.bin',
           'observations':'fp16/numerics/observations.jsonl','runner':'runner/c16_diagnostic_runner',
           'profile_json':'fp16/profile/profiler/msprof_1.json','profile_csv':'fp16/profile/profiler/op_summary_1.csv'}
    if damage in paths:(source/paths[damage]).write_bytes(b'corrupt')
    elif damage=='profile_missing':(source/'fp16/profile/profiler/msprof_1.json').unlink()
    elif damage=='timing':
        p=source/'w8_a/timing/execution.json';v=json.loads(p.read_text());v['timing']['execute_sync']['median_ms']=1;p.write_text(json.dumps(v))
    else:
        p=source/'c16-diagnostic-report.json';v=json.loads(p.read_text());v['cases'][0]['unprofiled_latency']['median_ms']=2;p.write_text(json.dumps(v))
    args=SimpleNamespace(source_evidence=source,output_dir=tmp_path/'recovered',expected_fp16_profiler_files=2,msprof=None,validate_only=False)
    with pytest.raises((ValueError,KeyError)):r.run(args)
    assert not calls
    out=json.loads((args.output_dir/'c16-diagnostic-report.json').read_text())
    assert out['status']=='DIAGNOSTIC_ONLY' and out['execution']=='RECOVERY_FAILED'


def test_server_single_quoted_nested_ge_descriptors_are_atomic_strings(tmp_path,monkeypatch):
    graph,_,_=modules(monkeypatch)
    descriptor='name: "y"\ndtype: DT_INT64\nlayout: "ND"\nattr {\n  key: "_meta"\n  value {\n    s: "<class \'torch.SymInt\'>(s65)"\n  }\n}\nattr {\n  key: "_fx_tensor_name"\n  value {\n    s: "sym_size-aten.sym_size.OUTPUT.0"\n  }\n}\n'
    empty='s: ""\n'
    tensor='t {\n desc { dtype: DT_INT64 shape {} layout: "ND" }\n data: "\\000<\\000\\000\\000\\000\\000\\000"\n}\n'
    p=tmp_path/'dynamo.pbtxt'
    p.write_text('node { name:"sym_size" op:"Shape" '+''.join('attr { key:"field'+str(i)+'" value { s: '+repr(s)+' } }' for i,s in enumerate((empty,descriptor,tensor)))+' }')
    nodes=graph.read_pbtxt(p)
    assert len(nodes)==1 and nodes[0]['name']=='sym_size' and nodes[0]['type']=='Shape'
    assert [a['value'][0]['s'][0] for a in nodes[0]['attributes']]==[empty,descriptor,tensor]
    assert list(graph.values(nodes)) # no None dictionary keys, no inner attr nodes
    assert graph.summarize(nodes,'pre_atc')['head_anchors']==[]


def test_complete_serialized_descriptor_can_identify_head_but_plain_text_cannot(tmp_path,monkeypatch):
    g,_,_=modules(monkeypatch);p=tmp_path/'dynamo.pbtxt'
    descriptor='name: "x2"\ndtype: DT_FLOAT16\nshape {dim: 248320 dim: 2560}\nlayout: "ND"\n'
    p.write_text('node {name:"head" op:"MatMulV2" attr {key:"input_desc_1" value {s:'+repr(descriptor)+'}}}\n'
                 'node {name:"other" op:"MatMulV2" attr {key:"note" value {s:"not a descriptor: 248320"}}}')
    nodes=g.read_pbtxt(p);assert len(nodes[0]['serialized_descriptors'])==1
    assert g.summarize(nodes,'pre_atc')['head_anchors']==['head']


def test_vocabulary_number_in_unrelated_attributes_is_not_head_evidence(tmp_path,monkeypatch):
    g,_,_=modules(monkeypatch);p=tmp_path/'dynamo.pbtxt'
    p.write_text('op {name:"ordinary_mm" type:"MatMulV2" attr {key:"offset" value {i:248320}}}')
    assert g.summarize(g.read_pbtxt(p),'pre_atc')['head_anchors']==[]


def test_adjacent_byte_literals_decode_utf8_after_concatenation(tmp_path,monkeypatch):
    g,_,_=modules(monkeypatch);p=tmp_path/'dynamo.pbtxt'
    p.write_text(r'''node {name: '\xe4' '\xb8\xad' op: 'Identity'}''')
    assert g.read_pbtxt(p)[0]['name']=='中'
