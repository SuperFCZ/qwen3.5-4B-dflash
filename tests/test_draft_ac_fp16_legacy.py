"""Historical SHA-only, dynamic-gear and original-runner capture contracts (CPU)."""
import copy
import importlib
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import pytest
from test_draft_ac_fp16_w8_profile import tools, fp16_fixture


def dynamic_fixture(tmp_path, ev):
    dep,replays=fp16_fixture(tmp_path/'fp16',ev)
    d=ev.load(dep);g=d['graphs'][0];g['static_gear_oms']=[]
    g['metadata']['incremental_contract'].update(draft_context_gears=[16,64],draft_prefill_policy='single_draft16_64_gears',
                                               capacity=256,vocab_size=248320,abi='qwen35-dflash-chunk-v4')
    shapes=[]
    for t,b in zip(g['metadata']['tensor_abi']['inputs'],g['runtime_input_abi']['bindings']):
        shape=list(t['shape'])
        if t['name']=='features':shape[1]=-1
        shapes.append(b['data_node_name']+':'+','.join(map(str,shape)))
    g['atc_command']=['atc','--input_format=ND','--dynamic_dims=16;64','--input_shape='+';'.join(shapes)]
    def strip(value):
        if isinstance(value,dict):
            if 'sha256' in value:value.pop('bytes',None)
            for v in value.values():strip(v)
        elif isinstance(value,list):
            for v in value:strip(v)
    strip(d)
    air=dep.parent/d['air_manifest']['path'];a=ev.load(air);a['graphs']=[copy.deepcopy(g)];strip(a)
    air.write_text(json.dumps(a));d['air_manifest']['sha256']=ev.sha256(air);dep.write_text(json.dumps(d))
    for rp in replays.values():
        r=ev.load(rp);r['draft_static64_om_sha256']=None;rp.write_text(json.dumps(r))
        p=Path(r['input_directory'])/'contract.txt';p.write_text('\n'.join(x for x in p.read_text().splitlines() if not x.startswith('static64 '))+'\n')
    return dep,replays


@pytest.mark.parametrize('bytes_field',[None,3])
def test_legacy_record_measures_real_size_without_rewriting(tmp_path,monkeypatch,bytes_field):
    ev,_=tools(monkeypatch);p=tmp_path/'payload';p.write_bytes(b'abc')
    item=dict(path='payload',sha256=ev.sha256(p))
    if bytes_field:item['bytes']=bytes_field
    original=copy.deepcopy(item);path,audit=ev.legacy_file(tmp_path,item)
    assert item==original and path==p and audit['actual_bytes']==3
    assert audit['declared_bytes']==bytes_field


@pytest.mark.parametrize('damage',['hash','size','absolute','escape','symlink','missing','empty'])
def test_legacy_record_does_not_relax_integrity(tmp_path,monkeypatch,damage):
    ev,_=tools(monkeypatch);root=tmp_path/'bundle';root.mkdir();p=root/'payload';p.write_bytes(b'abc')
    item=dict(path='payload',sha256=ev.sha256(p))
    if damage=='hash':item['sha256']='0'*64
    elif damage=='size':item['bytes']=4
    elif damage=='absolute':item['path']=str(p)
    elif damage=='escape':item['path']='../bundle/payload'
    elif damage=='symlink':
        other=tmp_path/'other';other.write_bytes(b'abc');p.unlink();p.symlink_to(other)
    elif damage=='missing':p.unlink()
    else:p.write_bytes(b'')
    with pytest.raises(ValueError):ev.legacy_file(root,item)


def test_dynamic_sha_only_keeps_original_om_and_full_snapshot(tmp_path,monkeypatch):
    ev,p=tools(monkeypatch);dep,replays=dynamic_fixture(tmp_path,ev)
    before={str(x):ev.sha256(x) for x in dep.parent.iterdir() if x.is_file()}
    base=ev.deployment(dep)
    assert base['dynamic'] and base['oms'][16]==base['oms'][64]
    assert all(a['declared_bytes'] is None and a['actual_bytes']>0 for a in base['file_audit'])
    case=ev.replay(base,replays[16],16)
    assert case['abi']['inputs'][0]['shape'][1]==64
    p.replay_command(case,tmp_path/'new',Path('/existing/dynamic-runner'),0)
    assert (tmp_path/'new/gear.txt').read_text()=='16\n'
    assert (tmp_path/'new/inputs/features.bin').read_bytes()==case['inputs']['features'].read_bytes()
    assert before=={path:ev.sha256(path) for path in before}
    d=ev.load(dep);d['graphs'][0]['atc_command'][2]='--dynamic_dims=32;64';dep.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='gear contract'):ev.deployment(dep)


def test_lightweight_missing_fp16_replays_reports_compatibility_not_false_pass(tmp_path,monkeypatch):
    ev,p=tools(monkeypatch);dep,replays=dynamic_fixture(tmp_path,ev)
    for rp in replays.values():rp.unlink()
    monkeypatch.setattr(p,'load_reuse',lambda *a:pytest.fail('lightweight preflight must not audit/build W8 package'))
    monkeypatch.setattr(p.subprocess,'run',lambda *a,**k:pytest.fail('no device, capture, build or model execution'))
    args=SimpleNamespace(output_dir=tmp_path/'new',w8_evidence=tmp_path/'w8',search_root=tmp_path,
                         device_id=0,prepare_only=True,fp16_deployment=None)
    p.run(args);r=ev.load(args.output_dir/'fp16-w8-report.json')
    c=r['fp16_preflight']['candidates'][0]
    assert r['status']=='BLOCKED' and r['device_status']=='NOT_RUN'
    assert c['manifest_status']=='PASS' and c['shape_mode']=='dynamic'
    assert c['device_gear_query']=='NOT_RUN' and len(c['file_audit'])>=3
    assert all(x['status']=='BLOCKED' for x in c['replays'].values())
    assert all(x['status']=='MISSING' for x in c['capture_requests'].values())


def request_fixture(tmp_path,ev,dep,gear):
    # Original runtime plans enumerate real Deployment OMs, not newly exported models.
    d=ev.load(dep);g=d['graphs'][0]
    if len(d['graphs'])==1:
        d['graphs']=[{**copy.deepcopy(g),'name':n} for n in ('target_prefill','target_verify','draft')]
        air=dep.parent/d['air_manifest']['path'];a=ev.load(air);a['graphs']=copy.deepcopy(d['graphs']);air.write_text(json.dumps(a))
        d['air_manifest']['sha256']=ev.sha256(air);dep.write_text(json.dumps(d))
    folder=tmp_path/f'old-request-{gear}';folder.mkdir()
    runner=tmp_path/'original-runner';runner.write_bytes(b'CPU source provenance fixture')
    c=g['metadata']['incremental_contract']
    lines=[c['abi'],f"capacity {c['capacity']} {c['vocab_size']}",f"draft_prefill_policy {c['draft_prefill_policy']}"]
    for graph in d['graphs']:
        om=(dep.parent/graph['om']['path']).resolve()
        lines.append(f'graph {graph["name"]} {json.dumps(str(om))} {graph["om"]["sha256"]}')
        for side,marker in (('inputs','I'),('outputs','O')):
            for t in graph['metadata']['tensor_abi'][side]:
                lines.append(' '.join(map(str,[marker,t['name'],t['dtype'],len(t['shape']),*t['shape']])))
        lines.append('end')
    lines.append('done');(folder/'plan.txt').write_text('\n'.join(lines)+'\n')
    request=folder/'profile-request.json'
    request.write_text(json.dumps(dict(deployment_manifest=dict(path=str(dep),sha256=ev.sha256(dep)),
        runner=dict(path=str(runner),sha256=ev.sha256(runner)),prompt_token_ids=[1]*gear,eos_token_ids=[2])))
    return request


def test_original_capture_requests_are_bound_to_deployment_plan_and_gear(tmp_path,monkeypatch):
    ev,_=tools(monkeypatch);dep,replays=dynamic_fixture(tmp_path,ev)
    cap=importlib.import_module('fp16_capture');req=request_fixture(tmp_path,ev,dep,16);base=ev.deployment(dep)
    assert cap.capture_request(base,req,16)['ids']==[1]*16
    with pytest.raises(ValueError,match='wrong context gear'):cap.capture_request(base,req,64)
    plan=req.parent/'plan.txt';plan.write_text(plan.read_text().replace('I features float16','I features float32'))
    with pytest.raises(ValueError,match='plan differs'):cap.capture_request(base,req,16)


def test_dynamic_cpp_adapter_syntax_with_existing_fake_headers(tmp_path):
    # Syntax only, NOT CANN compilation or NPU validation.
    root=Path(__file__).resolve().parents[1];compiler=shutil.which('c++')
    if not compiler:pytest.skip('C++ frontend unavailable')
    shim=tmp_path/'shim.h'
    shim.write_text('#include "acl/acl.h"\naclError aclrtMemset(void*,size_t,int32_t,size_t);\nconst char* aclGetRecentErrMsg();\n')
    subprocess.run([compiler,'-std=c++17','-fsyntax-only','-I'+str(root/'framework/runtime/cpp/tests/fake_acl'),
                    '-include',str(shim),str(root/'framework/custom_ops/draft_ac/test/fp16_dynamic_runner.cpp')],check=True)


@pytest.mark.parametrize('damage',[None,'fake','device'])
def test_opt_in_capture_invokes_original_runner_and_validates_generated_evidence(tmp_path,monkeypatch,damage):
    ev,_=tools(monkeypatch);dep,old_replays=dynamic_fixture(tmp_path,ev)
    cap=importlib.import_module('fp16_capture')
    requests={g:request_fixture(tmp_path,ev,dep,g) for g in (16,64)}
    base=ev.deployment(dep);seen=[]
    def execute(command,**kwargs):
        args={command[i]:command[i+1] for i in range(1,len(command),2)}
        assert args['--debug-draft-replay']=='20' and args['--debug-draft-workspace']=='private'
        assert '--debug-draft-inputs' not in args and 'atc' not in command[0]
        gear=len(args['--prompt-token-ids'].split(','));seen.append(gear)
        out=Path(args['--output']);r=ev.load(old_replays[gear]);ip=Path(r['input_directory'])
        replay_dir=Path(str(out)+'.replay');shutil.copytree(ip.parent,replay_dir)
        r['input_directory']=str(replay_dir/'inputs');r['device_id']=1 if damage=='device' else 0
        if damage=='fake':r['fake_acl']=True
        contract=replay_dir/'inputs/contract.txt';lines=contract.read_text().splitlines();lines[2]='256 0 15';lines[3]=' '.join(['1']*gear)
        contract.write_text('\n'.join(lines)+'\n');out.write_text(json.dumps(r))
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(cap.subprocess,'run',execute)
    watch={str(p):ev.sha256(p) for p in tmp_path.rglob('*') if p.is_file()}
    if damage:
        with pytest.raises(ValueError):cap.capture(base,requests,tmp_path/'new-capture',0,{})
    else:
        result=cap.capture(base,requests,tmp_path/'new-capture',0,{})
        assert seen==[16,64] and set(result)=={16,64}
        assert ev.load(tmp_path/'new-capture/capture.json')['status']=='PASS'
    assert all(ev.sha256(p)==h for p,h in watch.items())


def test_reused_dynamic_adapter_requires_matching_source_and_binary(tmp_path,monkeypatch):
    ev,_=tools(monkeypatch);mod=importlib.import_module('fp16_dynamic')
    binary=tmp_path/'runner';binary.write_bytes(b'existing binary fixture')
    build=tmp_path/'build.json';build.write_text(json.dumps(dict(source_sha256=mod.sources(),runner=dict(path=str(binary),sha256=ev.sha256(binary)))))
    monkeypatch.setattr(mod.subprocess,'run',lambda *a,**k:pytest.fail('must reuse, not build'))
    assert mod.runner(tmp_path/'unused',{},build)[0]==binary
    binary.write_bytes(b'changed')
    with pytest.raises(ValueError,match='identity'):mod.runner(tmp_path/'unused',{},build)
