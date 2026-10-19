"""No NPU: independent model identities, frozen replay, timing/profile separation."""
import copy
import importlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace
import pytest
from test_draft_ac_c_profile import modules, accepted_fixture


def tools(monkeypatch):
    modules(monkeypatch)
    return importlib.import_module('fp16_evidence'), importlib.import_module('fp16_w8_profile')


def fp16_fixture(root, ev):
    root.mkdir(); bundle=root/'bundle';bundle.mkdir()
    inputs=[dict(name='features',dtype='float16',shape=[1,64,2])]
    inputs += [dict(name=n,dtype='int64',shape=[1]) for n in ('start_position','valid_rows','anchor','proposal_count')]
    states=[dict(name=f'd{i}_{k}',dtype='float16',shape=[1,1,4,1]) for i in range(6) for k in ('key','value')]
    inputs+=states
    outputs=[dict(name='draft_top1',dtype='int64',shape=[1,15])]+copy.deepcopy(states)
    abi=dict(inputs=inputs,outputs=outputs)
    meta=dict(tensor_abi=abi,draft_quantization='fp16',dynamic_input_axes={'features':[1]},
              incremental_contract=dict(draft_quantization='fp16',draft_states=states,feature_width=2))
    binding=dict(bindings=[dict(data_node_name='arg'+str(i)) for i in range(17)])
    payload=bundle/'draft.air';payload.write_bytes(b'FP16 immutable AIR fixture')
    graph=dict(name='draft',metadata=meta,runtime_input_abi=binding,custom_op_audit=[],
               payload_files=[dict(path=payload.name,bytes=payload.stat().st_size,sha256=ev.sha256(payload))])
    oms=[]
    for gear in (16,64):
        om=bundle/f'c{gear}.om';om.write_bytes(f'FP16-six-{gear}'.encode())
        shapes=[]
        for t,b in zip(inputs,binding['bindings']):
            shape=list(t['shape'])
            if t['name']=='features':shape[1]=gear
            shapes.append(b['data_node_name']+':'+','.join(map(str,shape)))
        oms.append(dict(rows=gear,om=dict(path=om.name,bytes=om.stat().st_size,sha256=ev.sha256(om)),
                        atc_command=['atc','--input_format=ND','--input_shape='+';'.join(shapes)]))
    graph.update(om=oms[0]['om'],static_gear_oms=oms)
    air=bundle/'air-fp16.json';air.write_text(json.dumps(dict(status='PASS',graphs=[graph],factory_config={'checkpoint':'fp16-real'})))
    dep=bundle/'deployment-manifest.json'
    dep.write_text(json.dumps(dict(status='PASS',target={'soc_version':'Ascend310P3'},graphs=[graph],
                    air_manifest=dict(path=air.name,bytes=air.stat().st_size,sha256=ev.sha256(air)))))
    replays={}
    widths={'float16':2,'int64':8}
    for gear in (16,64):
        d=root/f'replay{gear}';d.mkdir();ip=d/'inputs';ip.mkdir();gold=d/'outputs/reference';gold.mkdir(parents=True)
        hashes={};lines=['qwen35-draft-replay-inputs-v1',oms[0]['om']['sha256'],'4 0 15','static64 '+oms[1]['om']['sha256'],'1 2 3 ']
        for t in inputs:
            n=t['name'];size=widths[t['dtype']]
            for dim in t['shape']:size*=dim
            value=struct.pack('<q',gear) if n=='valid_rows' else bytes(size)
            (ip/(n+'.bin')).write_bytes(value);hashes[n]=ev.sha256(ip/(n+'.bin'))
            lines.append(n+' '+t['dtype']+' '+str(size)+' '+' '.join(map(str,t['shape'])))
        (ip/'contract.txt').write_text('\n'.join(lines)+'\n')
        (ip/'sha256.txt').write_text(''.join(n+' '+h+'\n' for n,h in hashes.items()))
        out={}
        for t in outputs:
            size=widths[t['dtype']]
            for dim in t['shape']:size*=dim
            (gold/(t['name']+'.bin')).write_bytes(bytes(size));out[t['name']]=ev.sha256(gold/(t['name']+'.bin'))
        phases=[dict(phase=n,iterations=20,**{k:0 for k in ('token_mismatch_iterations','input_restore_mismatch_iterations','input_mutation_iterations','full_output_hash_mismatch_iterations','valid_kv_mismatch_iterations')}) for n in ('isolated','interleaved_prefill')]
        report=d/'private.json';report.write_text(json.dumps(dict(status='PASS_REPLAY_CHECKS',fake_acl=False,
            kv_output_audit={'context_rows':gear},draft_om_sha256=oms[0]['om']['sha256'],draft_static64_om_sha256=oms[1]['om']['sha256'],
            input_directory=str(ip),snapshot_sha256=hashes,reference_output_sha256=out,phases=phases)))
        replays[gear]=report
    return dep,replays


def test_discover_separate_fp16_and_readonly_gear_prefix(tmp_path,monkeypatch):
    ev,p=tools(monkeypatch);dep,replays=fp16_fixture(tmp_path/'fp16',ev)
    cases,diagnostic=ev.resolve(tmp_path)
    assert [c['gear'] for c in cases]==[16,64]
    assert all(c['layers']==6 and len(c['abi']['inputs'])==17 and len(c['abi']['outputs'])==13 for c in cases)
    c=cases[0];before=c['inputs']['features'].read_bytes()
    command=p.replay_command(c,tmp_path/'replay',Path('/old/runner'),0)
    assert command[0]=='/old/runner'
    assert (tmp_path/'replay/inputs/features.bin').read_bytes()==before[:64]
    assert c['inputs']['features'].read_bytes()==before
    assert c['identity']['factory_config']['checkpoint']=='fp16-real'


@pytest.mark.parametrize('mutation',['input','golden','air','om','replay_hash','gear','repeat'])
def test_fp16_binding_failures(tmp_path,monkeypatch,mutation):
    ev,p=tools(monkeypatch);dep,replays=fp16_fixture(tmp_path/'fp16',ev)
    base=ev.deployment(dep)
    if mutation=='input': (tmp_path/'fp16/replay16/inputs/features.bin').write_bytes(b'bad')
    elif mutation=='golden': (tmp_path/'fp16/replay16/outputs/reference/draft_top1.bin').write_bytes(b'bad')
    elif mutation in ('air','om'):
        (dep.parent/('air-fp16.json' if mutation=='air' else 'c16.om')).write_bytes(b'bad')
        with pytest.raises((ValueError,KeyError)):ev.deployment(dep)
        return
    else:
        r=ev.load(replays[16])
        if mutation=='replay_hash':r['draft_om_sha256']='0'*64
        elif mutation=='gear':r['kv_output_audit']['context_rows']=64
        else:r['phases'][0]['full_output_hash_mismatch_iterations']=1
        replays[16].write_text(json.dumps(r))
    with pytest.raises(ValueError):ev.replay(base,replays[16],16)


def test_missing_ambiguous_and_dynamic_report_blockers(tmp_path,monkeypatch):
    ev,p=tools(monkeypatch)
    with pytest.raises(ValueError,match='FP16_ARTIFACTS_BLOCKED'):ev.resolve(tmp_path)
    dep,replays=fp16_fixture(tmp_path/'fp16',ev)
    (replays[16].parent/'duplicate.json').write_bytes(replays[16].read_bytes())
    with pytest.raises(ValueError,match='found 2'):ev.resolve(tmp_path)
    assert len(ev.resolve(tmp_path,dep,replays)[0])==2
    value=ev.load(dep);del value['graphs'][0]['static_gear_oms'];dep.write_text(json.dumps(value))
    with pytest.raises((ValueError,KeyError)):ev.deployment(dep)


def test_regions_do_not_call_attention_matmul_linear(monkeypatch):
    ev,p=tools(monkeypatch)
    assert p.region('anonymous','MatMulV2',set(),set()).startswith('Unmapped')
    assert p.region('attn','SoftmaxV2',set(),set())=='Attention'
    assert p.region('x','ScatterNdUpdate',set(),set())=='KV Cache update'
    assert p.region('x','RmsNorm',set(),set())=='Norm'
    assert p.region('head','MatMulV2',{'head'},set())=='LM Head'
    assert p.region('x','DFlashGroupQuantLinear',set(),set())=='Linear'
    assert p.region('x','TransData',set(),set())=='TransData'


def test_profile_region_union_not_sum(monkeypatch):
    ev,p=tools(monkeypatch)
    tasks=[dict(name='x',op_type='RmsNorm',start_us=0,duration_us=10),dict(name='x',op_type='RmsNorm',start_us=5,duration_us=10)]
    a=p.attribution({'selected_timeline':{'iterations':[{'tasks':tasks}]*30}},set(),set())
    assert a['region_task_sum_us']['Norm']['mean']==20
    assert a['region_occupancy_us']['Norm']['mean']==15
    assert p.attribution({},set(),set())['status']=='UNRESOLVED'


def test_report_separates_fresh_formal_and_profiled_times(tmp_path,monkeypatch):
    ev,p=tools(monkeypatch)
    r=dict(status='COLLECTED',device_status='FORMAL_REPLAY_PASS',cases=[])
    for model,t in (('fp16',40),('w8_a',30)):
        r['cases'].append(dict(model=model,gear=16,formal_latency={'median_ms':t,'p95_ms':t+1},
                              profiled_execution={'timing':{'execute_sync':{'median_ms':999}}}))
    p.finish(r,tmp_path)
    saved=ev.load(tmp_path/'fp16-w8-report.json')
    assert saved['comparisons'][0]['fp16_over_w8_a']==40/30
    assert 'NOT quantization-only' in saved['comparisons'][0]['interpretation']
    assert '999' not in (tmp_path/'fp16-w8-analysis.md').read_text()


def test_blocked_run_writes_not_run_without_launch(tmp_path,monkeypatch):
    ev,p=tools(monkeypatch);source=tmp_path/'old';source.mkdir()
    args=SimpleNamespace(output_dir=tmp_path/'new',w8_evidence=source,search_root=tmp_path,device_id=0,prepare_only=False)
    def blocked(*a):raise ValueError('missing accepted artifacts')
    monkeypatch.setattr(p,'load_reuse',blocked)
    monkeypatch.setattr(p.subprocess,'run',lambda *a,**k:pytest.fail('must not launch'))
    with pytest.raises(ValueError,match='missing'):p.run(args)
    r=ev.load(args.output_dir/'fp16-w8-report.json')
    assert r['status']=='BLOCKED' and r['device_status']=='NOT_RUN' and not r['cases']


@pytest.mark.parametrize('prepare_only',[True,False])
@pytest.mark.parametrize('dynamic',[False,True])
def test_four_case_entry_uses_own_goldens_and_separates_launches(tmp_path,monkeypatch,prepare_only,dynamic):
    import shlex, subprocess
    ev,p=tools(monkeypatch);_,_,old=modules(monkeypatch)
    if dynamic:
        from test_draft_ac_fp16_legacy import dynamic_fixture
        dep,replays=dynamic_fixture(tmp_path,ev)
    else: dep,replays=fp16_fixture(tmp_path/'fp16',ev)
    # FP16 token IDs deliberately differ from W8; only within-model equality is legal.
    for rp in replays.values():
        r=ev.load(rp);gold=Path(r['input_directory']).parent/'outputs/reference/draft_top1.bin'
        gold.write_bytes(struct.pack('<15q',*([7]*15)));r['reference_output_sha256']['draft_top1']=ev.sha256(gold)
        rp.write_text(json.dumps(r))
    rows=[]
    for gear in (16,64):
        for mode in old.MODES:
            source,row,runner=accepted_fixture(tmp_path,old,gear,mode);rows.append(row)
    (source/'suite.json').write_text(json.dumps(dict(status='PASS',single_op_graph='PASS',draft_only='PASS')))
    (source/'draft-suite.json').write_text(json.dumps(dict(status='PASS',cases=rows)))
    reuse=dict(runner=str(runner),environment={'ASCEND_CUSTOM_OPP_PATH':'accepted-joint-opp'})
    monkeypatch.setattr(p,'load_reuse',lambda s:(SimpleNamespace(native_base=tmp_path/'native'),reuse))
    monkeypatch.setattr(p,'verify_reuse',lambda r:None)
    monkeypatch.setattr(p.shutil,'which',lambda n:'/existing/msprof' if n=='msprof' else None)
    dynamic_binary=tmp_path/'dynamic-runner';dynamic_binary.write_bytes(b'CPU dynamic adapter fixture')
    dynamic_module=importlib.import_module('fp16_dynamic')
    monkeypatch.setattr(dynamic_module,'runner',lambda *a:(dynamic_binary,dict(status='CPU_TEST_DOUBLE')))
    launches=[]
    def execute(command,**kwargs):
        if command[0]=='npu-smi' or command[-1]=='--version':return subprocess.CompletedProcess(command,0)
        profile=command[0]=='/existing/msprof'
        if profile:
            command=shlex.split(next(x.split('=',1)[1] for x in command if x.startswith('--application=')))
        output=Path(command[3]);model,gear=output.parent.name.split('-c');gear=int(gear)
        assert command[0]==str(dynamic_binary if dynamic and model=='fp16' else runner)
        assert kwargs['env']['ASCEND_CUSTOM_OPP_PATH']=='accepted-joint-opp'
        if dynamic and model=='fp16':
            abi=ev.deployment(dep)['graph']['metadata']['tensor_abi']
            gears=[[dim if t['name']!='features' or j!=1 else rows for t in abi['inputs'] for j,dim in enumerate(t['shape'])] for rows in (16,64)]
            (output/'dynamic-gear.json').write_text(json.dumps(dict(status='PASS',selected_rows=gear,actual_gears=gears)))
        launches.append((model,gear,profile))
        if model=='fp16':
            r=ev.load(replays[gear]);gold=Path(r['input_directory']).parent/'outputs/reference'
            by_name={x.stem:x for x in gold.glob('*.bin')}
        else:
            gold=source/f'draft-a-c{gear}'/'native'
            by_name={x.name.removesuffix('-actual-0.bin'):x for x in gold.glob('*-actual-0.bin')}
        for name,path in by_name.items():
            for suffix in ('actual-0.bin','actual-1.bin','benchmark-last.bin','postcheck.bin'):
                (output/(name+'-'+suffix)).write_bytes(path.read_bytes())
        report=copy.deepcopy(rows[0]['custom']);value=999 if profile else 40 if model=='fp16' else 30
        report['timing']['execute_sync']=dict(median_ms=value,p95_ms=value,samples_ms=[value]*30)
        (output/'execution.json').write_text(json.dumps(report))
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(p.subprocess,'run',execute)
    before={str(x):ev.sha256(x) for base in (source,tmp_path/'fp16') for x in base.rglob('*') if x.is_file()}
    args=SimpleNamespace(output_dir=tmp_path/'new',w8_evidence=source,search_root=tmp_path,device_id=0,
        fp16_deployment=dep,fp16_c16_replay=replays[16],fp16_c64_replay=replays[64],prepare_only=prepare_only,msprof='msprof',atc='atc')
    p.run(args)
    report=ev.load(args.output_dir/'fp16-w8-report.json')
    assert len(report['cases'])==(0 if prepare_only else 4)
    if prepare_only:
        assert not launches and report['device_status']=='NOT_RUN' and report['status']=='PREFLIGHT_ONLY'
    else:
        assert len(launches)==8 and all(not x[2] for x in launches[:4]) and all(x[2] for x in launches[4:])
        assert {x[:2] for x in launches}=={(m,g) for m in ('fp16','w8_a') for g in (16,64)}
        assert report['status']=='COLLECTED'
        assert all(x['fp16_over_w8_a']==40/30 for x in report['comparisons'])
        assert all(x['own_golden_bitwise']=='PASS' for x in report['cases'])
    assert before=={path:ev.sha256(path) for path in before}
