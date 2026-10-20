"""CPU fixtures for selected-layer Gate acceptance; no NPU execution."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from a2_common import record,sha256,write_json
from a3_launch import configure_build,load_build_config
from run_gate_prefetch_pair import parse_args
from test_gate_prefetch import plan
import run_gate_copy as gate
import run_gate_copy_pair as pair

NAMES=gate.selected_cases([0,1,2,3,4])


def execution(kind, timed=True):
    r=dict(status='PASS',runtime='AscendCL ACLNN' if kind=='custom' else 'AscendCL native OM',op='DFlashGroupQuantLinear',
           cpu_fallback=False,input_readonly=True,guards_intact=True,repetitions=2,device_id=0,m=16,k=2560,n=19456,
           tile_n=64,tile_k=256,tile_m=16,weight_reuse_rows=16,workspace_bytes=2097152,tracked_device_allocation_bytes=2100000,io_validated=True)
    r['timing']=dict(status='MEASURED' if timed else 'NOT_RUN',warmup=5 if timed else 0,repetitions=30 if timed else 0,
        protocol='continuous-v1' if timed else 'checked-v1',per_timed_call_readback=False,per_timed_call_poison=False,
        correctness_before_calls=2,correctness_after_calls=1,timed_tail_checked=True,
        execute_sync={'samples_ms':[2.]*30 if timed else []},prepare={'samples_ms':[.01]*30 if timed else []})
    return r


def fixture(root):
    paths={n:root/n for n in ('bundle','native_om_manifest','golden_suite','build_config','runner','om_runner')}
    for p in paths.values():p.write_text('CPU fixture only')
    cases=[];compiled=[];old=[];expected={}
    for i,name in enumerate(NAMES):
        folder=root/name;folder.mkdir();files={}
        # Tiny identity fixtures: load_bundle/layout are mocked; checksums aren't.
        for f in ('x.bin','q_nk.bin','w_nz.bin','s_gn.bin','eager-0.bin','eager-1.bin'):
            p=folder/f;p.write_bytes(bytes([i,1]));files[f]=record(p,root)
        cases.append(dict(name=name,files=files))
        native=dict(name=name)
        for f in ('om','deployment','air_manifest'):
            p=folder/f;p.write_text(name+' CPU fixture '+f);native[f]=record(p,root)
        compiled.append(native)
        expected[name]=(0x3c00+i*256).to_bytes(2,'little')*(16*19456)
        p=folder/'golden.bin';p.write_bytes(expected[name]);g=record(p,root)
        old.append(dict(name=name,status='PASS',native_om=native['om'],executions={'native_om':execution('native_om')},outputs={'native_om':[g,g]},bracket_outputs={'native_om':[g,g]}))
    write_json(paths['golden_suite'],dict(status='PASS',context_rows=16,bundle_sha256=sha256(paths['bundle']),native_om_manifest_sha256=sha256(paths['native_om_manifest']),cases=old))
    cfg=dict(k_tile=256,scale_mode='broadcast',dequant_mode='batched',gate_copy_mode='loop',gate_prefetch_mode='raw')
    args=SimpleNamespace(**paths,experiment='prefetch',output_dir=root/'run',layers=list(range(5)),device_id=0,warmup=5,repetitions=30,timeout=20)
    return args,cfg,dict(context_rows=16,cases=cases),dict(cases=compiled),old,expected


def test_cli_default_explicit_and_invalid_selection():
    common=['--a51-evidence','old.json','--output-dir','new']
    assert parse_args(common).layers==[0]
    assert parse_args(common+['--layers','0','1','2','3','4']).layers==list(range(5))
    assert parse_args(common+['--layers','4','0']).layers==[0,4]
    for values in (['0','0'],['5'],[]):
        with pytest.raises(SystemExit):parse_args(common+['--layers',*values])
    for values in ([],[True],[0,0],[-1],[5],['0']):
        with pytest.raises(ValueError):gate.selected_layers(values)


@pytest.mark.parametrize('failure_layer',[None,0,4])
def test_all_selected_layers_must_pass_before_any_timing(tmp_path,failure_layer):
    args,cfg,bundle,native,old,expected=fixture(tmp_path);calls=[];timed=[]
    def launch(command,**kwargs):
        folder=Path(command[2]);name=folder.name;calls.append(name)
        assert len(command)==6 # No timing arguments; exactly two correctness calls.
        write_json(folder/'execution.json',execution('custom',False))
        (folder/'runner.log').write_text(gate.COPY_PREFIX+'{"version":1,"mode":"loop","selected":"loop"}\n')
        for j in (0,1):
            raw=expected[name]
            if name==f'layer-{failure_layer}-gate_up' and j==1:raw=bytes([raw[0]^1])+raw[1:]
            (folder/f'actual-{j}.bin').write_bytes(raw)
    def timing(child):
        assert calls==list(NAMES);assert child.case_filter==NAMES
        timed.append(True);child.output_dir.mkdir();rows=[]
        for name in NAMES:
            row=dict(name=name,outputs={},bracket_outputs={},executions={})
            for kind in ('custom','native_om'):
                folder=child.output_dir/name/kind;folder.mkdir(parents=True)
                (folder/'runner.log').write_text(gate.COPY_PREFIX+'{"version":1,"mode":"loop","selected":"loop"}\n')
                entries=[]
                for label in ('actual-0','actual-1','benchmark-last','postcheck'):
                    p=folder/(label+'.bin');p.write_bytes(expected[name]);entries.append(record(p,child.output_dir))
                row['outputs'][kind]=entries[:2];row['bracket_outputs'][kind]=entries[2:];row['executions'][kind]=execution(kind)
            rows.append(row)
        write_json(child.output_dir/'suite.json',dict(status='PASS',cases=rows));return 0
    with patch.object(gate,'load_build_config',return_value=cfg),patch.object(gate,'load_bundle',return_value=bundle),patch.object(gate.run_a2,'validate_om_manifest',return_value=native),patch.object(gate,'validate_layout'),patch.object(gate,'launch_plan',return_value=plan('raw')),patch.object(gate.subprocess,'run',side_effect=launch),patch.object(gate.run_a2,'run',side_effect=timing):
        code=gate.run(args)
    report=json.loads((args.output_dir/'suite.json').read_text())
    assert report['full_draft_token_kv']=='NOT_RUN'
    if failure_layer is not None:
        assert code==1 and not timed and calls==list(NAMES[:failure_layer+1])
        assert report['performance']=='NOT_RUN' and report['phase']=='untimed_correctness'
    else:
        assert code==0 and timed==[True] and len(report['checks'])==5
        assert tuple(report['input_identities'])==NAMES
        for name in NAMES:
            identity=report['input_identities'][name]
            assert set(identity['capture_files'])==set(bundle['cases'][0]['files'])
            assert identity['native']['om']['sha256']==sha256(tmp_path/name/'om')
            assert all(g['sha256']==sha256(tmp_path/name/'golden.bin') for g in identity['golden'])


@pytest.mark.parametrize('artifact',['w_nz.bin','s_gn.bin','x.bin','om','golden.bin'])
def test_per_layer_hash_tamper_blocks_execution(tmp_path,artifact):
    args,cfg,bundle,native,old,expected=fixture(tmp_path)
    (tmp_path/NAMES[0]/artifact).write_bytes(b'changed')
    with patch.object(gate,'load_build_config',return_value=cfg),patch.object(gate,'load_bundle',return_value=bundle),patch.object(gate.run_a2,'validate_om_manifest',return_value=native),patch.object(gate,'validate_layout'),patch.object(gate.subprocess,'run') as launch,patch.object(gate.run_a2,'run') as timing:
        assert gate.run(args)==1;launch.assert_not_called();timing.assert_not_called()
    assert json.loads((args.output_dir/'suite.json').read_text())['performance']=='NOT_RUN'


@pytest.mark.parametrize('truncated',[False,True])
def test_five_layer_pair_selection_is_forwarded_and_cannot_be_silently_truncated(tmp_path,truncated):
    files={}
    for name in ('bundle','native','golden','accepted'):
        p=tmp_path/name;p.write_text('CPU fixture');files[name]=str(p)
    source=dict(bundle=files['bundle'],native=files['native'],golden=files['golden'],config=dict(core_limit=0,pipeline_mode='prefetch',kv_m80_mode='a-ub'))
    def launch(command,**kwargs):
        env=kwargs['env'];mode=env['DFLASH_GATE_PREFETCH_MODE']
        assert env['GATE_LAYERS']=='0,1,2,3,4' and env['DFLASH_GATE_COPY_MODE']=='loop'
        run=tmp_path/mode;data=run/'data';data.mkdir(parents=True)
        write_json(data/'suite.json',dict(status='PASS',selected_cases=[NAMES[0]] if truncated else list(NAMES)))
        for name in ('sdk-copy.json','opp-preflight.json'):write_json(run/name,dict(status='PASS',scope='CPU fixture'))
        (run/'source-commit.txt').write_text('CPU fixture')
        Path(env['DFLASH_SUMMARY_POINTER']).write_text(str(data/'suite.json'))
    args=SimpleNamespace(experiment='prefetch',output_dir=tmp_path/'pair',a51_evidence=Path(files['accepted']),layers=list(range(5)),device_id=0,order='baseline-first')
    def finish(a,b,output,**kwargs):write_json(output,dict(status='PASS',scope='CPU fixture'));return dict(status='PASS')
    with patch.object(pair,'source_evidence',return_value=source),patch.object(pair.subprocess,'run',side_effect=launch) as calls,patch.object(pair,'compare_runs',side_effect=finish) as compare:
        assert pair.run(args)==int(truncated);assert calls.call_count==(1 if truncated else 2)
        if truncated:compare.assert_not_called()
        else:assert compare.call_count==1
    report=json.loads((args.output_dir/'suite.json').read_text())
    assert report['selected_cases']==list(NAMES) and report['full_draft_token_kv']=='NOT_RUN'
    if truncated:assert 'exactly the requested' in report['error']


def paired_reports(root):
    args,_,bundle,native,old,_=fixture(root)
    identities={c['name']:gate.layer_identity(c,n,o,root,root,root) for c,n,o in zip(bundle['cases'],native['cases'],old)}
    paths=[]
    for mode in ('off','raw'):
        folder=root/mode;folder.mkdir()
        configure_build(folder/'host.h',folder/'build.json',0,'batched',folder/'kernel.h','prefetch','a-ub','broadcast',256,'loop',mode)
        cfg=load_build_config(folder/'build.json')
        timing=dict(status='PASS',selected_cases=list(NAMES),context_rows=16,device_id=0,bundle_sha256=sha256(args.bundle),native_om_manifest_sha256=sha256(args.native_om_manifest),timing_protocol='continuous-v1',cases=[])
        for name in NAMES:timing['cases'].append(dict(name=name,executions={k:execution(k) for k in ('custom','native_om')}))
        write_json(folder/'timing.json',timing)
        stage=dict(status='PASS',experiment='prefetch',selected_cases=list(NAMES),checks={name:dict(status='PASS') for name in NAMES},build=cfg,build_config_path=str(folder/'build.json'),build_config_sha256=sha256(folder/'build.json'),input_identities=copy.deepcopy(identities),timing=record(folder/'timing.json',folder))
        p=folder/'suite.json';write_json(p,stage);paths.append(p)
    return paths


def fake_projection(old,new,*args,**kwargs):
    assert kwargs['gate_prefetch'] is True
    index=NAMES.index(old['name']);assert old['name']==new['name']
    a=2.+index*.1;b=3. if index==4 else 1.
    def stats(value):return dict(median_ms=value,p95_ms=value+.1,samples_ms=[value]*30)
    return dict(name=old['name'],status='PASS',baseline=stats(a),candidate=stats(b),native_before=stats(4.),native_after=stats(4.1),
                baseline_launch=plan('off'),candidate_launch=plan('raw'),median_speedup=a/b,candidate_slower_median=b>a,candidate_slower_p95=b>a)


def test_comparison_five_layers_samples_totals_and_regressions(tmp_path):
    paths=paired_reports(tmp_path)
    with patch.object(pair,'projection',side_effect=fake_projection) as proj,patch.object(pair,'copy_evidence',return_value=dict(selected='loop')):
        r=pair.compare_runs(*paths,tmp_path/'comparison.json','prefetch')
    assert proj.call_count==5 and [c['name'] for c in r['cases']]==list(NAMES)
    assert r['summary']['projection_count']==5 and r['summary']['baseline']['sum_projection_median_ms']==pytest.approx(11.)
    assert r['summary']['candidate']['sum_projection_median_ms']==7.
    assert r['regressions']==[dict(name=NAMES[4],median=True,p95=True)]
    assert all(len(c[k]['samples_ms'])==30 for c in r['cases'] for k in ('baseline','candidate','native_before','native_after'))
    assert r['full_draft_token_kv']=='NOT_RUN' and 'NOT full Draft' in r['summary']['interpretation']
    assert 'sum' in (tmp_path/'comparison.md').read_text().lower()


@pytest.mark.parametrize('damage',['missing_timing_layer','missing_identity','swapped_scale','golden_changed'])
def test_comparison_rejects_incomplete_or_mismatched_layer_identity(tmp_path,damage):
    left,right=paired_reports(tmp_path);stage=json.loads(right.read_text())
    if damage=='missing_timing_layer':
        timing=json.loads((right.parent/'timing.json').read_text());timing['cases'].pop()
        write_json(right.parent/'timing.json',timing);stage['timing']=record(right.parent/'timing.json',right.parent)
    elif damage=='missing_identity':stage['input_identities'].pop(NAMES[4])
    elif damage=='swapped_scale':stage['input_identities'][NAMES[4]]['capture_files']['s_gn.bin']=stage['input_identities'][NAMES[0]]['capture_files']['s_gn.bin']
    else:(tmp_path/NAMES[4]/'golden.bin').write_bytes(b'changed')
    write_json(right,stage)
    with patch.object(pair,'projection') as proj,pytest.raises(ValueError):pair.compare_runs(left,right,tmp_path/'comparison.json','prefetch')
    proj.assert_not_called()
