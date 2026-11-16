"""CPU subprocess fixtures, never ATC/NPU evidence."""
import copy
import json
from pathlib import Path
import sys

import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from split_atc import c,ec,SplitAtcRunner,collect_stdout,one_capture,verify_resources,legacy
from atc_evidence import clone_air,resumed_command,compare_om,option,find_export
from worker_derivation import verify_derivation
import run_atc
# Reuse the original synthetic Host record fixture; no historical file changes.
sys.path.insert(0,str(c.HERE/'test'))
from test_raw import host_fixture


def stdout_text(gear=16,mode='off'):
    shapes=[(gear,12800,2560),(gear+16,2560,2048),(16,2560,4096),(16,4096,2560),(16,2560,19456),(16,9728,2560)]
    return ''.join(host_fixture(shape,mode) for shape in shapes)


def cfg(mode):
    return dict(core_limit=0,dequant_mode='batched',pipeline_mode='prefetch',launch_version=5,kv_m80_mode='a-ub',scale_mode='broadcast',k_tile=256,k_tile_policy='m80-synthetic-k128-v1',gate_prefetch_mode=mode)


@pytest.mark.parametrize('warnings',[1,5])
def test_separate_streams_keep_fragmented_cube_and_all_gmm_warnings(tmp_path,warnings):
    text=stdout_text();script=tmp_path/'cpu_atc_fixture.py'
    script.write_text('import os\ntext='+repr(text)+'\nfor line in text.splitlines(True):\n if "_CUBE " in line:\n  n=len(line)//2\n  os.write(1,line[:n].encode())\n  for _ in range('+str(warnings)+'):os.write(2,b"GMM Warning: CPU fixture\\n")\n  os.write(1,line[n:].encode())\n else:os.write(1,line.encode())\n')
    runner=SplitAtcRunner(tmp_path/'logs',c.identity(sys.executable),tmp_path)
    command=[sys.executable,str(script),'--output='+str(tmp_path/'new-model')]
    result=runner(command,tmp_path);capture=one_capture(runner.records,command)
    assert result.returncode==0 and result.stdout==text
    expected=b'GMM Warning: CPU fixture\n'*(6*warnings)
    assert c.verify(capture['stdout']).read_bytes()==text.encode()
    assert c.verify(capture['stderr']).read_bytes()==expected
    assert result.stderr.encode()==expected
    resource=collect_stdout(c.verify(capture['stdout']),tmp_path/'resources',16,cfg('off'))
    assert resource['stdout_schema']['cube_records']==6
    verify_resources(resource,16,cfg('off'))
    c.verify_tree(capture)
    c.verify(capture['stderr']).write_bytes(expected+b'CHANGED')
    with pytest.raises(ValueError):one_capture(runner.records,command)


@pytest.mark.parametrize('kind',['truncated','unknown','missing-field','duplicate-key','missing-shape','interleaved'])
def test_invalid_stdout_is_never_cleaned_or_repaired(tmp_path,kind):
    lines=stdout_text().splitlines();index=5
    if kind=='truncated':lines[index]=lines[index][:-1]
    elif kind=='unknown':lines[index]=lines[index].replace('_CUBE ','_UNKNOWN ')
    elif kind=='missing-field':
        tag,data=lines[index].split(' ',1);value=json.loads(data);del value['depth_a1'];lines[index]=tag+' '+json.dumps(value)
    elif kind=='duplicate-key':lines[index]=lines[index].replace('{','{"version":1,',1)
    elif kind=='missing-shape':lines=lines[:-6]
    else:lines[0],lines[1]=lines[1],lines[0]
    log=tmp_path/'stdout.log';log.write_text('\n'.join(lines)+'\n');before=log.read_bytes()
    with pytest.raises(ValueError):collect_stdout(log,tmp_path/'blocks',16,cfg('off'))
    assert log.read_bytes()==before


@pytest.mark.parametrize('gear',[16,64])
def test_b_r_comparison_keeps_gate_and_non_gate_contracts(tmp_path,gear):
    records=[]
    for mode in ('off','raw'):
        path=tmp_path/(mode+'.log');path.write_text(stdout_text(gear,mode))
        records.append(collect_stdout(path,tmp_path/mode,gear,cfg(mode)))
    legacy.compare(*records)
    bad=copy.deepcopy(records[1]);bad['plans']['16,2560,4096']['cube']['step_ka']+=1
    with pytest.raises(ValueError,match='non-Gate'):legacy.compare(records[0],bad)
    bad=copy.deepcopy(records[1]);bad['plans']['16,2560,19456']['raw_banks']=1
    with pytest.raises(ValueError):legacy.compare(records[0],bad)


def test_failed_subprocess_keeps_both_logs(tmp_path):
    script=tmp_path/'fail.py';script.write_text('import os\nos.write(1,b"incomplete")\nos.write(2,b"GMM fatal diagnostic")\nraise SystemExit(7)\n')
    runner=SplitAtcRunner(tmp_path/'logs',c.identity(sys.executable),tmp_path)
    argv=[sys.executable,str(script),'--output='+str(tmp_path/'new')]
    assert runner(argv,tmp_path).returncode==7
    assert c.verify(runner.records[0]['stderr']).read_bytes()==b'GMM fatal diagnostic'
    with pytest.raises(ValueError):one_capture(runner.records,argv)


def test_air_copy_preserves_originals_and_command_only_relocates_paths(tmp_path):
    old=tmp_path/'old';(old/'graph').mkdir(parents=True)
    data=old/'graph/model.air';data.write_bytes(b'CPU_AIR_FIXTURE')
    entry=dict(c.identity(data),path='graph/model.air')
    manifest=old/'air-manifest.json';c.write(manifest,dict(graphs=[dict(air=entry,payload_files=[entry])]))
    before={p:p.read_bytes() for p in (data,manifest)}
    command=['/verified/atc','--mode=0','--framework=1','--model='+str(data),'--output='+str(old/'om/a'),'--precision_mode=must_keep_origin_dtype','--deterministic=0']
    frozen=dict(air=c.identity(manifest),payloads={'graph/model.air':c.identity(data)},command=command)
    dest=tmp_path/'new';copied=clone_air(frozen,dest)
    new=resumed_command(frozen,copied,dest)
    assert [i for i,(a,b) in enumerate(zip(command,new)) if a!=b]==[3,4]
    assert option(new,'--model')==str(dest/'graph/model.air')
    assert all(p.read_bytes()==v for p,v in before.items())
    assert (dest/'graph/model.air').read_bytes()==data.read_bytes()
    assert (dest/'graph/model.air').stat().st_ino!=data.stat().st_ino
    data.write_bytes(b'CHANGED')
    with pytest.raises(ValueError):clone_air(frozen,tmp_path/'bad')


def test_bad_atc_identity_and_changed_math_flags_block_before_launch(tmp_path):
    argv=[sys.executable,'--output='+str(tmp_path/'om')]
    bad=dict(c.identity(sys.executable),sha256='0'*64)
    runner=SplitAtcRunner(tmp_path/'bad-id',bad,tmp_path)
    with pytest.raises(ValueError,match='identity'):runner(argv,tmp_path)
    runner=SplitAtcRunner(tmp_path/'bad-flags',c.identity(sys.executable),tmp_path,expected_command=argv)
    with pytest.raises(ValueError,match='flags'):runner(argv+['--precision_mode=allow_fp32_to_fp16'],tmp_path)
    assert not runner.records


def test_new_om_hash_difference_is_not_numerical_or_performance_pass(tmp_path):
    a=tmp_path/'old.om';b=tmp_path/'new.om';a.write_bytes(b'OLD');b.write_bytes(b'NEW')
    result=compare_om(c.identity(a),c.identity(b))
    assert result['status']=='DIFFERENT_BINARY_UNATTRIBUTED' and not result['exact_sha_equal']


def test_exact_worker_derivation_and_predecessor_inventory_unchanged():
    verify_derivation()
    assert all(HERE not in Path(x['path']).parents for x in c.sources()+ec.sources())
    assert 'allow_test_double=False' in (HERE/'split_export.py').read_text()


def test_cli_and_bounded_report_discovery(tmp_path):
    with pytest.raises(SystemExit):run_atc.parse(['resume','--output-dir','/tmp/unused'])
    with pytest.raises(SystemExit):run_atc.parse(['complete','--output-dir','/tmp/unused','--admission-report','/tmp/a'])
    with pytest.raises(ValueError):find_export(tmp_path)
    p=tmp_path/'export/report.json';p.parent.mkdir();c.write(p,dict(phase='export',evidence={},precheck_report={}))
    assert find_export(tmp_path)==p


def test_outer_failure_still_records_both_raw_streams(tmp_path,monkeypatch):
    dest=tmp_path/'B/c16';logs=dest/'atc-streams/call-0';logs.mkdir(parents=True)
    (logs/'stdout.log').write_bytes(b'partial stdout')
    (logs/'stderr.log').write_bytes(b'GMM diagnostic must survive')
    request=tmp_path/'request.json';report={}
    def fail(*args,**kwargs):raise RuntimeError('CPU subprocess failure')
    monkeypatch.setattr(c,'command',fail)
    with pytest.raises(RuntimeError):run_atc.launch_worker(report,dest,request,{},'B',16,True)
    records=report['worker_evidence']['B-c16'];c.verify_tree(records)
    assert c.verify(records['stderr']).read_bytes()==b'GMM diagnostic must survive'


@pytest.fixture
def admitted_assets(tmp_path,monkeypatch):
    """Small metadata fixture; not a valid executable AIR or OM."""
    from atc_evidence import COUNTS
    import hashlib
    root=tmp_path/'failed-export';child=root/'B/c16';bundle=child/'artifacts';graphdir=bundle/'graph';graphdir.mkdir(parents=True)
    executable=tmp_path/'atc';executable.write_bytes(b'CPU_BINARY_FIXTURE_NEVER_EXECUTED')
    airfile=graphdir/'model.air';airfile.write_bytes(b'CPU_AIR_FIXTURE')
    ir=graphdir/'dynamo.pbtxt';ir.write_bytes(b'CPU_IR_FIXTURE')
    om=bundle/'om/model.om';om.parent.mkdir();om.write_bytes(b'CPU_OM_FIXTURE')
    abi={'inputs':['CPU15-placeholder'],'outputs':['CPU11-placeholder']}
    g=dict(metadata={'draft_ac':{'mode':'a','gear':16},'tensor_abi':abi},
        runtime_input_abi={'status':'PASS','draft_ac':{'status':'PASS','counts':COUNTS}},
        custom_op_audit=[{'ge_op_type':k,'ge_node_occurrences':v} for k,v in COUNTS.items()],
        air=dict(c.identity(airfile),path='graph/model.air'),payload_files=[dict(c.identity(p),path='graph/'+p.name) for p in (airfile,ir)])
    ap=bundle/'air-manifest.json';c.write(ap,dict(status='PASS',artifact_kind='qwen35-dflash-torchair-bundle',graphs=[g]))
    command=[str(executable),'--mode=0','--framework=1','--soc_version=Ascend310P3','--model='+str(airfile),'--output='+str(om.with_suffix('')),'--precision_mode=must_keep_origin_dtype','--deterministic=0']
    compiled=dict(g,atc_command=command,om=dict(c.identity(om),path='om/model.om'))
    dp=bundle/'deployment-manifest.json';c.write(dp,dict(status='PASS',air_manifest=dict(c.identity(ap),path=ap.name),graphs=[compiled],compiler=dict(path=str(executable),identity='CPU_TEST')))
    log=child/'mixed.log';log.write_bytes(b'original mixed stdout/stderr retained')
    env={'ASCEND_CUSTOM_OPP_PATH':'/private/B/vendor','LD_LIBRARY_PATH':'/private/B/api'}
    monkeypatch.setattr(c,'selected_env',lambda *a:env)
    monkeypatch.setattr(c,'sources',lambda:[])
    e={'H':{'gears':{'16':{'abi':abi}}},'tools':{'atc':c.identity(executable)}}
    prior=dict(current={'device':0,'evidence':e},references={},reports={'build':{'variants':{'B':{'opp':{'install':'/private/B'}}}}})
    model=dict(status='A_ONLY_OM_COMPILED_NOT_EXECUTED',import_boundary={'status':'PASS'},interfaces={'status':'PASS'},
        output_root=str(child),environment=env,counts=COUNTS,abi=abi,graph=c.identity(ir),logical_ir_sha256=hashlib.sha256(ir.read_bytes()).hexdigest(),
        artifacts={'air':c.identity(ap),'deployment':c.identity(dp),'om':c.identity(om)},om=c.identity(om),atc_command=command,atc_log=c.identity(log))
    c.write(child/'report.json',model)
    c.write(root/'B-c16-request.json',dict(variant='B',gear=16,device=0,sources=[],evidence=e,output=str(child)))
    failed=root/'report.json';c.write(failed,dict(status='BLOCKED',phase='export',sources=[],evidence=e))
    return failed,prior,model,airfile,log


def test_admission_seals_worker_and_preserves_old_assets(admitted_assets):
    from atc_evidence import old_worker
    failed,prior,model,air,log=admitted_assets
    before={p:p.read_bytes() for p in (failed,air,log,Path(model['output_root'])/'report.json')}
    r=old_worker(failed,prior)
    assert r['command']==model['atc_command'] and r['air']==model['artifacts']['air']
    assert 'First admission seal' in r['worker_hash_scope']
    assert all(p.read_bytes()==data for p,data in before.items())


@pytest.mark.parametrize('which',['air','log','worker-abi','atc-command'])
def test_admission_rejects_asset_or_contract_changes(admitted_assets,which):
    from atc_evidence import old_worker
    failed,prior,model,air,log=admitted_assets
    if which=='air':air.write_bytes(b'CHANGED')
    elif which=='log':log.write_bytes(b'CHANGED')
    else:
        if which=='worker-abi':model['counts']=dict(model['counts'],DFlashDraftLmHeadTop1=1)
        else:model['atc_command']=model['atc_command'][:-1]+['--deterministic=1']
        c.write(Path(model['output_root'])/'report.json',model)
    with pytest.raises(ValueError):old_worker(failed,prior)
