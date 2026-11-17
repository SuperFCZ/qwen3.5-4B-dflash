"""New eight-core contracts only; CPU fixtures are not NPU evidence."""
import copy,json,sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from c8_common import c,ec
import resource8 as rc
import run_core8 as run
from derive_export import verify
sys.path.insert(0,str(c.HERE/'test'))
from test_raw import host_fixture


def cfg(v):return dict(core_limit=0,gate_copy_mode='loop',gate_prefetch_mode='off' if v=='B' else 'raw',pipeline_mode='prefetch',dequant_mode='batched',kv_m80_mode='a-ub',scale_mode='broadcast',k_tile=256,k_tile_policy='m80-synthetic-k128-v1',launch_version=5)


def sidecar(path,gear,v,cores=8):
    rows=[]
    for key in sorted(rc.shape_keys(gear)):
        shape=tuple(map(int,key.split(',')))
        for line in host_fixture(shape,cfg(v)['gate_prefetch_mode']).splitlines():
            tag,body=line.split(' ',1);kind=tag[len('DFLASH_GROUP_QUANT_'):];p=json.loads(body)
            if kind=='LAUNCH':p.update(available_cores=cores,block_dim=cores)
            rows.append(dict(record=kind,payload=p))
    path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    return c.identity(path)


@pytest.mark.parametrize('gear',[16,64])
def test_all_six_shapes_and_pair_contract_independent_of_history(tmp_path,gear):
    plans={}
    for v in ('B','R'):
        item=sidecar(tmp_path/(v+'.jsonl'),gear,v)
        result=rc.resources(item,tmp_path/v,gear,cfg(v),v);rc.verify_resource(result,gear,cfg(v),v)
        plans[v]=result['plans']
        for p in plans[v].values():assert rc.schedule(p)['coverage']=='EXACT_ONCE'
        gate=plans[v]['16,2560,19456'];assert all(r['tile_count']==38 for r in gate['assignments'])
    rc.compare(plans['B'],plans['R'])
    assert plans['B']['16,9728,2560']['raw_banks']==2  # Existing Down pipeline preserved.
    bad=copy.deepcopy(plans['R']);bad['16,2560,4096']['system_workspace_bytes']+=32
    with pytest.raises(ValueError):rc.compare(plans['B'],bad)
    bad=copy.deepcopy(plans['R']);bad['16,2560,19456']['matmul_ub_bytes']+=32
    with pytest.raises(ValueError):rc.compare(plans['B'],bad)


@pytest.mark.parametrize('cores',[7,9])
def test_no_platform_core_fallback(tmp_path,cores):
    item=sidecar(tmp_path/'plan.jsonl',16,'B',cores)
    with pytest.raises(ValueError,match='8/8'):rc.resources(item,tmp_path/'view',16,cfg('B'),'B')


@pytest.mark.parametrize('field,value',[('block_dim',7),('core_limit',8),('available_cores',True),('n_tiles',303)])
def test_schedule_rejects_misreported_configuration(tmp_path,field,value):
    r=rc.resources(sidecar(tmp_path/'p',16,'B'),tmp_path/'view',16,cfg('B'),'B')
    plan=copy.deepcopy(r['plans']['16,2560,19456']);plan[field]=value
    with pytest.raises(ValueError):rc.schedule(plan)


@pytest.mark.parametrize('field,value',[('block',1),('tile_count',37),('first_column',64),('last_tile_column',0),('tile_stride_columns',448)])
def test_overlap_omission_and_wrong_stride_block(tmp_path,field,value):
    r=rc.resources(sidecar(tmp_path/'p',16,'B'),tmp_path/'view',16,cfg('B'),'B')
    p=copy.deepcopy(r['plans']['16,2560,19456']);p['assignments'][0][field]=value
    with pytest.raises(ValueError):rc.schedule(p)


def test_probe_sidecar_requires_identical_complete_callbacks(tmp_path):
    sidecar(tmp_path/'all',16,'R');lines=(tmp_path/'all').read_text().splitlines()[:6]
    p=tmp_path/'probe';p.write_text('\n'.join(lines+lines)+'\n')
    assert rc.probe_view(p,tmp_path/'log')==2
    changed=copy.deepcopy(lines);obj=json.loads(changed[5]);obj['payload']['db_l0b']=1;changed[5]=json.dumps(obj)
    p.write_text('\n'.join(lines+changed)+'\n')
    with pytest.raises(ValueError,match='drift'):rc.probe_view(p,tmp_path/'bad')
    p.write_text('\n'.join(lines[:-1])+'\n')
    with pytest.raises(ValueError):rc.probe_view(p,tmp_path/'bad')


@pytest.mark.parametrize('stage',run.STATES)
def test_explicit_stage_prerequisites(stage):
    args=[stage,'--output-dir','/tmp/not-executed'];earlier=list(run.STATES)[:list(run.STATES).index(stage)]
    if earlier:
        with pytest.raises(SystemExit):run.parse(args)
    for name in earlier:args+=['--'+name+'-report','/tmp/'+name]
    assert run.parse(args).phase==stage


def test_export_derivation_and_no_historical_equality_gate():
    verify()
    text=(HERE/'resource8.py').read_text()
    assert 'historical_plans' not in text and 'accepted_gate' not in text
    assert all(HERE not in Path(x['path']).parents for x in c.sources()+ec.sources())


def test_source_or_failed_predecessor_cannot_pass(tmp_path):
    p=tmp_path/'r';c.write(p,dict(status='BLOCKED',error='failed'))
    with pytest.raises(ValueError):run.load(p,'admit')
    c.write(p,dict(status=run.STATES['admit'],sources8=[]))
    with pytest.raises(ValueError,match='source drift'):run.load(p,'admit')
