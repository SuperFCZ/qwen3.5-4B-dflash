"""Diagnostic observations never grant formal correctness (CPU-only fixtures)."""
import copy
import importlib
import json
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
from types import SimpleNamespace
import pytest
from test_draft_ac_fp16_legacy import dynamic_fixture
from test_draft_ac_fp16_w8_profile import tools
from test_draft_ac_c_profile import modules,accepted_fixture


def failed(tmp_path,monkeypatch):
    ev,_=tools(monkeypatch);dep,replays=dynamic_fixture(tmp_path,ev);p=replays[16]
    d=ev.load(dep);g=d['graphs'][0]
    for group in (g['metadata']['tensor_abi']['inputs'],g['metadata']['tensor_abi']['outputs'],g['metadata']['incremental_contract']['draft_states']):
        for spec in group:
            if spec['name'].startswith('d') and spec['name']!='draft_top1':spec['shape'][2]=32
    g['atc_command']=[x.replace(':1,1,4,1',':1,1,32,1') for x in g['atc_command']]
    air=dep.parent/d['air_manifest']['path'];a=ev.load(air);a['graphs']=[copy.deepcopy(g)];air.write_text(json.dumps(a))
    d['air_manifest']['sha256']=ev.sha256(air);dep.write_text(json.dumps(d))
    r=ev.load(p);ip=Path(r['input_directory'])
    for n in r['snapshot_sha256']:
        if n.startswith('d'):
            (ip/(n+'.bin')).write_bytes(bytes(64));r['snapshot_sha256'][n]=ev.sha256(ip/(n+'.bin'))
            ref=ip.parent/'outputs/reference'/(n+'.bin');ref.write_bytes(bytes(64));r['reference_output_sha256'][n]=ev.sha256(ref)
    contract=ip/'contract.txt';contract.write_text('\n'.join((line.split()[0]+' float16 64 1 1 32 1') if line.startswith(tuple('d'+str(i)+'_' for i in range(6))) else line for line in contract.read_text().splitlines())+'\n')
    (ip/'proposal_count.bin').write_bytes(struct.pack('<q',15))
    r['snapshot_sha256']['proposal_count']=ev.sha256(ip/'proposal_count.bin')
    (ip/'sha256.txt').write_text(''.join(n+' '+h+'\n' for n,h in r['snapshot_sha256'].items()))
    r['status']='FAIL_REPLAY_CHECKS';r['phases'][0]['valid_kv_mismatch_iterations']=2
    r['phases'][0]['full_output_hash_mismatch_iterations']=2;p.write_text(json.dumps(r))
    tool=importlib.import_module('c16_diagnostic')
    return ev,tool,dep,p


def test_failed_replay_only_diagnostic_formal_gate_unchanged(tmp_path,monkeypatch):
    ev,t,dep,p=failed(tmp_path,monkeypatch);before=ev.sha256(p)
    case=t.fp16_case(p,tmp_path,dep)
    assert case['source_status']=='FAIL_REPLAY_CHECKS'
    assert 'NOT_accepted_golden' in case['reference_role']
    with pytest.raises(ValueError,match='status'):ev.replay(ev.deployment(dep),p,16)
    assert ev.sha256(p)==before
    c=case['inputs']['features'];c.write_bytes(b'changed')
    with pytest.raises(ValueError,match='shape/hash'):t.fp16_case(p,tmp_path,dep)


def fake_gear(case,folder):
    gears=[[dim if x['name']!='features' or axis!=1 else rows for x in case['abi']['inputs'] for axis,dim in enumerate(x['shape'])] for rows in (16,64)]
    (folder/'dynamic-gear.json').write_text(json.dumps(dict(status='PASS',selected_rows=16,actual_gears=gears)))


@pytest.mark.parametrize('damage',['gear','pass_status'])
def test_wrong_actual_gear_or_formal_status_rejected(tmp_path,monkeypatch,damage):
    ev,t,dep,p=failed(tmp_path,monkeypatch);case=t.fp16_case(p,tmp_path,dep);folder=tmp_path/'output';folder.mkdir();fake_gear(case,folder)
    if damage=='gear':
        g=ev.load(folder/'dynamic-gear.json');g['actual_gears'][0][1]=32;(folder/'dynamic-gear.json').write_text(json.dumps(g))
    (folder/'execution.json').write_text(json.dumps(dict(status='PASS',cpu_fallback=False,guards=True,readonly=True,repeat_equal=True)))
    with pytest.raises(ValueError,match='gear|status'):t.diagnostic_execution(case,folder,'timing')


def test_diagnostic_report_never_promotes_results(tmp_path,monkeypatch):
    ev,t,_,_=failed(tmp_path,monkeypatch)
    r=dict(status='PASS',execution='COMPLETED',cases=[],historical_fp16_status='FAIL_REPLAY_CHECKS')
    t.save(r,tmp_path)
    out=ev.load(tmp_path/'c16-diagnostic-report.json')
    assert out['status']=='DIAGNOSTIC_ONLY' and out['formal_acceptance']=='NOT_RUN_NOT_ELIGIBLE' and out['end_to_end']=='NOT_RUN'
    assert out['historical_fp16_status']=='FAIL_REPLAY_CHECKS'
    assert t.region('unmapped','MatMulV2',set(),set())=='UNKNOWN'
    assert t.region('a','DFlashGroupQuantLinear',set(),set())=='W8_LINEAR'
    assert t.region('linear','MatMulV2',set(),{'linear'})=='FP16_LINEAR'


def test_cpp_numerical_metrics_and_bhsd_untouched_regions(tmp_path):
    compiler=shutil.which('c++')
    if not compiler:pytest.skip('C++ frontend unavailable')
    root=Path(__file__).resolve().parents[1];source=tmp_path/'metrics.cpp';exe=tmp_path/'metrics'
    source.write_text('''#include <iostream>
#include "c16_diagnostic_metrics.h"
std::vector<uint8_t> bytes(std::initializer_list<uint16_t> v) { std::vector<uint8_t> r;for(auto x:v){r.push_back(x&255);r.push_back(x>>8);}return r; }
int main(){
 auto input=bytes({0x3c00,0x3c00,0x3c00,0x3c00,0x3c00,0x3c00,0x3c00,0x3c00});
 auto a=bytes({0x4000,0x4200,0x3c00,0x3c00,0x3c00,0x7c00,0x3c00,0x8000});
 c16_diagnostic::KV(std::cout,a,input,input,{1,2,4,1},1,1,2);
}''')
    subprocess.run([compiler,'-std=c++17','-I'+str(root/'framework/custom_ops/draft_ac/test'),str(source),'-o',str(exe)],check=True)
    x=json.loads(subprocess.check_output([str(exe)],text=True))
    assert x['all']['bit_mismatches']==4 and x['all']['mismatch_ratio']==.5
    assert x['all']['actual_nonfinite']==1 and x['all']['finite_pairs']==7
    assert x['all']['max_abs']==2 and x['all']['p99_abs']==2
    assert x['all']['rmse']==pytest.approx((6/7)**.5,rel=1e-5)
    assert x['untouched_prefix_vs_input']['bit_mismatches']==1
    assert x['untouched_tail_vs_input']['bit_mismatches']==1
    assert x['written_valid']['actual_nonfinite']==1


def test_diagnostic_and_formal_cpp_syntax_are_separate(tmp_path):
    root=Path(__file__).resolve().parents[1];compiler=shutil.which('c++')
    if not compiler:pytest.skip('C++ frontend unavailable')
    shim=tmp_path/'shim.h';shim.write_text('#include "acl/acl.h"\naclError aclrtMemset(void*,size_t,int32_t,size_t);\nconst char* aclGetRecentErrMsg();\n')
    base=[compiler,'-std=c++17','-fsyntax-only','-I'+str(root/'framework/runtime/cpp/tests/fake_acl'),'-include',str(shim)]
    for define in ([],['-DDFLASH_C16_DIAGNOSTIC_ONLY=1']):
        subprocess.run(base+define+[str(root/'framework/custom_ops/draft_ac/test/fp16_dynamic_runner.cpp')],check=True)


def observations(case,phase):
    names=[f'numerics-{i}' for i in range(30)] if phase=='numerics' else ['actual-0.bin','actual-1.bin','benchmark-last.bin','postcheck.bin']
    result=[]
    for name in names:
        outputs={}
        for spec in case['abi']['outputs']:
            if spec['name']=='draft_top1': outputs[spec['name']]=dict(bitwise_equal=True,mismatches=0,actual_ids=[0]*15)
            else:
                count=1
                for dim in spec['shape']:count*=dim
                m=dict(elements=count,finite_pairs=count,nonfinite_pairs=0,bit_mismatches=1,mismatch_ratio=1/count,
                       actual_nonfinite=0,reference_nonfinite=0,max_abs=.5,rmse=.1,p99_abs=.5)
                outputs[spec['name']]={k:copy.deepcopy(m) for k in ('all','valid_prefix','written_valid','written_padding','untouched_prefix_vs_input','untouched_tail_vs_input')}
        result.append(dict(status='DIAGNOSTIC_ONLY',observation=name,outputs=outputs))
    return result


@pytest.mark.parametrize('prepare_only',[False,True])
def test_only_c16_no_prefill_no_exports_profile_separate_and_no_source_writes(tmp_path,monkeypatch,prepare_only):
    ev,t,dep,p=failed(tmp_path,monkeypatch);_,_,old=modules(monkeypatch)
    source,row,runner=accepted_fixture(tmp_path,old,16,'a')
    (source/'suite.json').write_text(json.dumps(dict(status='PASS')))
    (source/'draft-suite.json').write_text(json.dumps(dict(status='PASS',cases=[row])))
    reuse=dict(runner=str(runner),environment={'ASCEND_CUSTOM_OPP_PATH':'frozen-opp'})
    monkeypatch.setattr(t,'load_reuse',lambda s:(SimpleNamespace(native_base=tmp_path/'native'),reuse))
    monkeypatch.setattr(t,'verify_reuse',lambda r:None)
    diagnostic=tmp_path/'diagnostic';diagnostic.write_bytes(b'CPU fixture')
    monkeypatch.setattr(t,'build_runner',lambda *a:(diagnostic,{'status':'DIAGNOSTIC_ONLY'}))
    monkeypatch.setattr(t.shutil,'which',lambda n:None)
    fp=t.fp16_case(p,tmp_path,dep)
    # The old W8 fixture has placeholder control names; only test its existing strict IO gate here.
    real_controls=t.controls
    monkeypatch.setattr(t,'controls',lambda c:real_controls(c) if c['model']=='fp16' else dict(start_position=0,valid_rows=16,proposal_count=15))
    calls=[]
    def execute(command,**kw):
        if command[0]=='npu-smi' or command[-1]=='--version': return subprocess.CompletedProcess(command,0)
        profile=command[0]=='msprof'
        app=shlex.split(next(s.split('=',1)[1] for s in command if s.startswith('--application='))) if profile else command
        assert app[0] in (str(runner),str(diagnostic)) and len(app)==5
        folder=Path(app[3]);model=folder.parent.name;phase=folder.name;calls.append((model,phase,profile))
        if model=='fp16':
            fake_gear(fp,folder)
            r=dict(status='DIAGNOSTIC_ONLY',guards=True,readonly=True,cpu_fallback=False,repeat_equal=False,observed_calls=30 if phase=='numerics' else 4)
            if phase=='numerics':r['phase']='numerics'
            else:r['timing']=copy.deepcopy(row['custom']['timing'])
            (folder/'execution.json').write_text(json.dumps(r))
            (folder/'observations.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in observations(fp,phase)))
        else:
            original=source/'draft-a-c16/candidate'
            for file in original.iterdir():
                if file.name=='execution.json' or file.suffix=='.bin':shutil.copyfile(file,folder/file.name)
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(t.subprocess,'run',execute)
    before={str(f):ev.sha256(f) for d in (tmp_path/'fp16',source) for f in d.rglob('*') if f.is_file()}
    args=SimpleNamespace(diagnostic_only=True,w8_evidence=source,output_dir=tmp_path/'new',fp16_replay=p,
        search_root=tmp_path,fp16_deployment=dep,prepare_only=prepare_only,device_id=0,msprof='msprof',atc='atc',diagnostic_build=None)
    t.run(args);r=ev.load(args.output_dir/'c16-diagnostic-report.json')
    assert r['status']=='DIAGNOSTIC_ONLY' and r['formal_acceptance']=='NOT_RUN_NOT_ELIGIBLE'
    assert all(c['gear']==16 for c in r['cases']) and r['historical_fp16_status']=='FAIL_REPLAY_CHECKS'
    assert calls==([] if prepare_only else [('fp16','numerics',False),('fp16','timing',False),('w8_a','timing',False),('fp16','profile',True),('w8_a','profile',True)])
    if not prepare_only:assert r['cases'][0]['numerics']['execution']['repeat_equal'] is False and len(r['cases'][0]['numerics']['observations'])==30
    assert all(ev.sha256(f)==h for f,h in before.items())


def test_region_occupancies_do_not_sum_overlapping_tasks(monkeypatch):
    tools(monkeypatch);t=importlib.import_module('c16_diagnostic')
    tasks=[dict(name='norm1',op_type='RmsNorm',start_us=0,duration_us=10),
           dict(name='norm2',op_type='RmsNorm',start_us=5,duration_us=10),
           dict(name='quant',op_type='DFlashGroupQuantLinear',start_us=0,duration_us=10)]
    profile={'selected_timeline':dict(iterations=[dict(tasks=tasks)],metrics={'device_envelope_us':{'mean':15}})}
    result=t.attribute(profile,set(),set())
    assert result['region_union_us']['NORM']['mean']==15
    assert result['region_union_us']['W8_LINEAR']['mean']==10
    assert result['host_and_device_metrics']['device_envelope_us']['mean']==15
    assert 'never sum' in result['note']
