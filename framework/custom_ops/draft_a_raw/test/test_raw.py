import copy,os,shutil,subprocess,sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import raw_common as c
import raw_evidence as ev
import raw_protocol as protocol
import run_experiment as run


def test_precheck_import_has_no_device_or_model_modules():
    code=f"import sys;sys.path.insert(0,{str(HERE)!r});import raw_evidence,c_profile,native_base;import raw_common as c;c.clean();print('NO_DEVICE_IMPORT')"
    p=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True)
    assert p.returncode==0,p.stderr


@pytest.mark.parametrize('stage',run.STATES)
def test_each_cli_requires_exact_predecessors(stage):
    args=[stage,'--output-dir','/tmp/not-executed']
    earlier=list(run.STATES)[:list(run.STATES).index(stage)]
    if earlier:
        with pytest.raises(SystemExit):run.parse(args)
    for p in earlier:args+=['--'+p+'-report','/tmp/'+p]
    assert run.parse(args).phase==stage


def test_precheck_missing_evidence_stays_no_device(tmp_path):
    out=tmp_path/'out'
    p=subprocess.run([sys.executable,'-B',str(HERE/'run_experiment.py'),'precheck','--history',str(tmp_path/'missing'),'--output-dir',str(out)],capture_output=True,text=True)
    r=c.read(out/'report.json')
    assert p.returncode==1 and r['status']=='BLOCKED' and r['npu']==r['compile']=='NOT_RUN'


def test_pins_and_identity_tampering(tmp_path):
    p=tmp_path/'r.json';p.write_text('{}');f=c.identity(p)
    assert ev.pin(tmp_path,{'r.json':f['sha256']})['r.json']==f
    p.write_text('{"changed":true}')
    with pytest.raises(ValueError):ev.pin(tmp_path,{'r.json':f['sha256']})
    with pytest.raises(ValueError):c.verify(f)


def test_generated_headers_change_only_raw_macro(tmp_path):
    b=tmp_path/'b.h';r=tmp_path/'r.h'
    b.write_text('#define DFLASH_GROUP_QUANT_GATE_PREFETCH_MODE 0U\n#define OTHER 3U\n')
    r.write_text(b.read_text().replace('MODE 0U','MODE 1U'));ev.header_pair(b,r)
    r.write_text(r.read_text().replace('OTHER 3U','OTHER 4U'))
    with pytest.raises(ValueError,match='beyond Raw'):ev.header_pair(b,r)


def test_only_raw_config_can_differ():
    b=dict(abi='dflash-group-quant-linear-build-v5',policy='n-tile-cyclic-v1',k_tile_policy='m80-synthetic-k128-v1',core_limit=0,dequant_mode='batched',pipeline_mode='prefetch',kv_m80_mode='a-ub',scale_mode='broadcast',k_tile=256,tiling_abi='full-m-v2',launch_version=5,gate_copy_mode='loop',source_sha256={'a':'hash'},gate_prefetch_mode='off')
    r=dict(b,gate_prefetch_mode='raw');ev.configs(b,r)
    for k in (*ev.FIELDS,'source_sha256'):
        bad=copy.deepcopy(r);bad[k]='changed'
        with pytest.raises(ValueError):ev.configs(b,bad)


def test_independent_environment_never_keeps_h_or_stacks_b_r(tmp_path):
    h=tmp_path/'H/opp';receiver=tmp_path/'receiver'
    e=dict(H={'opp_root':str(h)},historical_environment={'ASCEND_CUSTOM_OPP_PATH':str(h/'vendors/customize')+':'+str(receiver),
        'LD_LIBRARY_PATH':str(h/'vendors/customize/op_api/lib')+':/system/lib'})
    b=c.selected_env(e,tmp_path/'B/opp');r=c.selected_env(e,tmp_path/'R/opp')
    assert str(h) not in b['ASCEND_CUSTOM_OPP_PATH'] and str(h) not in b['LD_LIBRARY_PATH']
    assert '/B/' not in r['ASCEND_CUSTOM_OPP_PATH'] and '/R/' not in b['ASCEND_CUSTOM_OPP_PATH']


def test_new_output_does_not_overwrite_sources_or_history(tmp_path):
    with pytest.raises(ValueError):c.fresh(HERE/'forbidden')
    with pytest.raises(ValueError):c.fresh(tmp_path/'history/new',[tmp_path/'history'])
    p=c.fresh(tmp_path/'new')
    with pytest.raises(FileExistsError):c.fresh(p)


def test_pair_schedule_and_no_historical_samples():
    rows=[]
    for i in range(3):
        for g in (16,64):
            for j,v in enumerate(protocol.order(i,g)):
                rows.append(dict(round=i,gear=g,position=j,variant=v,profiled=False,validation=dict(status='ALL_TOKEN_KV_BITS_AND_SAFETY_MATCH',workspace_bytes=1024,model_weight_bytes=2048,timing=protocol.stats([30. if v=='B' else 29.]*30,30))))
    result=protocol.summarize(rows)
    assert result['gears']['16']['median_speedup']==30/29
    assert not result['historical_samples_used'] and result['adoption']=='NOT_DECIDED'
    with pytest.raises(ValueError):protocol.summarize(rows[:-1])
    rows[0]['profiled']=True
    with pytest.raises(ValueError):protocol.summarize(rows)


@pytest.fixture(scope='module')
def cpp(tmp_path_factory):
    compiler=shutil.which('c++')
    if not compiler:pytest.skip('C++ compiler missing')
    root=tmp_path_factory.mktemp('raw-runner');shim=root/'shim.h'
    shim.write_text('#include <cstddef>\nextern "C" const char* aclGetRecentErrMsg();\nextern "C" const char* aclrtGetSocName();\nextern "C" int aclrtMemset(void*,std::size_t,int,std::size_t);\n')
    exe=root/'cpu-runner'
    subprocess.run([compiler,'-std=c++17','-I'+str(c.REPO/'framework/runtime/cpp/tests/fake_acl'),'-I'+str(c.QUANT/'test'),'-include',str(shim),str(HERE/'runner/runner.cpp'),str(HERE/'test/fake_model.cpp'),'-o',str(exe)],capture_output=True,check=True)
    return exe


def fixture_case(root):
    inp=root/'inputs';gold=root/'golden';inp.mkdir();gold.mkdir()
    inputs=[('features','float16',[1,16,2]),('start_position','int64',[1]),('valid_rows','int16',[1]),('anchor','int64',[1]),('proposal_count','int16',[1])]
    outputs=[('draft_top1','int64',[1,15])]
    for n in protocol.OUTPUTS[1:]:inputs.append((n,'float16',[1,1,2,2]));outputs.append((n,'float16',[1,1,2,2]))
    import math
    for side,rows,folder in [('inputs',inputs,inp),('outputs',outputs,gold)]:
        (root/(side+'.tsv')).write_text(''.join(n+' '+t+' '+' '.join(map(str,s))+'\n' for n,t,s in rows))
        for n,t,s in rows:(folder/(n+'.bin')).write_bytes(bytes(math.prod(s)*(8 if t=='int64' else 2)))
    return inp,gold


@pytest.mark.parametrize('phase,calls',[('smoke',2),('benchmark',38)])
def test_actual_runner_brackets_full_golden(cpp,tmp_path,phase,calls):
    inp,gold=fixture_case(tmp_path)
    with (tmp_path/'runner.log').open('w') as f:p=subprocess.run([cpp,'0','unused_fixture.om',tmp_path,inp,gold,phase],env={**os.environ,'CPU_CASE':str(tmp_path)},stdout=f,stderr=subprocess.STDOUT)
    assert p.returncode==0,(tmp_path/'runner.log').read_text()
    r=c.read(tmp_path/'execution.json')
    assert r['golden_match'] and r['readonly'] and r['guards'] and r['repeat'] and r['poison_verified']
    assert int((tmp_path/'calls.txt').read_text())==calls
    assert len(r['timing']['execute_sync']['samples_ms'])==(30 if phase=='benchmark' else 0)
    # Exercise the actual Python/C++ schema, not a second handwritten report.
    suffixes=['actual-0.bin','actual-1.bin']+(['benchmark-last.bin','postcheck.bin'] if phase=='benchmark' else [])
    row=dict(execution=c.identity(tmp_path/'execution.json'),outputs={n:[c.identity(tmp_path/(n+'-'+s)) for s in suffixes] for n in protocol.OUTPUTS})
    gear=dict(golden={n:c.identity(gold/(n+'.bin')) for n in protocol.OUTPUTS})
    assert protocol.verify_run(row,gear,0,phase=='benchmark')['status']=='ALL_TOKEN_KV_BITS_AND_SAFETY_MATCH'


@pytest.mark.parametrize('fault,when',[('id',1),('kv',1),('readonly',2),('guard',1),('kv',37),('id',38),('log',37),('execute',1)])
def test_bad_token_untouched_kv_guard_or_tail_cannot_pass(cpp,tmp_path,fault,when):
    inp,gold=fixture_case(tmp_path)
    with (tmp_path/'runner.log').open('w') as f:p=subprocess.run([cpp,'0','unused_fixture.om',tmp_path,inp,gold,'benchmark'],env={**os.environ,'CPU_CASE':str(tmp_path),'CPU_FAULT':fault,'CPU_WHEN':str(when)},stdout=f,stderr=subprocess.STDOUT)
    assert p.returncode!=0
    assert int((tmp_path/'calls.txt').read_text())==when
    assert not (tmp_path/'execution.json').exists()  # never emit successful performance


def host_fixture(shape,mode):
    """Synthetic Host records only; values are not receiver resource evidence."""
    import json
    import a3_launch as a
    m,k,n=shape;tile=a.selected_k_tile(shape,256);scale=a.scale_plan(shape,'broadcast')
    plan=a.pipeline_plan(k,n,'prefetch',tile,mode,m)
    staged=shape==(80,2560,2048)
    plan['user_ub_bytes']+=scale['broadcast_bytes']+scale['sdk_reserved_bytes']+(20480 if staged else 0)
    plan.update(version=5,policy=a.POLICY,m=m,k=k,n=n,tile_m=m,tile_n=64,tile_k=tile,
        weight_reuse_rows=m,n_tiles=n//64,core_limit=0,available_cores=7,block_dim=7,
        dequant_mode='batched',matmul_ub_bytes=262144-plan['user_ub_bytes'],system_workspace_bytes=2097152)
    cube=dict(version=1,kv_m80_mode='a-ub',a_source='VECOUT' if staged else 'GM',a_stage_bytes=20480 if staged else 0,
        a_row_stride=128 if staged else k,base_m=m,base_n=64,base_k=tile,weight_dequantizations_per_n_tile=k//tile,
        depth_a1=1,depth_b1=1,step_m=1,step_n=1,step_ka=1,step_kb=1,db_l0a=2,db_l0b=2,db_l0c=1,trans_length=34816)
    kt=dict(version=1,requested_k_tile=256,selected_k_tile=tile,selection_reason=a.k_tile_reason(shape,256),
        l1_bytes=40960,l1_capacity=1048576,l0a_bytes=16384,l0a_capacity=65536,l0b_bytes=65536,l0b_capacity=65536,l0c_bytes=m*64*4,l0c_capacity=262144)
    rows=[plan,dict(version=1,mode=mode,enabled=mode=='raw' and shape==(16,2560,19456)),dict(version=1,mode='loop',selected='loop'),scale,kt,cube]
    import raw_resources as rr
    return ''.join('DFLASH_GROUP_QUANT_'+tag+' '+json.dumps(value)+'\n' for tag,value in zip(rr.PREFIXES,rows))


@pytest.mark.parametrize('gear',[16,64])
def test_actual_host_parser_and_resource_contract(tmp_path,gear):
    import raw_resources as rr
    cfg=dict(core_limit=0,dequant_mode='batched',pipeline_mode='prefetch',launch_version=5,kv_m80_mode='a-ub',scale_mode='broadcast',k_tile=256,k_tile_policy='m80-synthetic-k128-v1')
    shapes=[(gear,12800,2560),(gear+16,2560,2048),(16,2560,4096),(16,4096,2560),(16,2560,19456),(16,9728,2560)]
    plans=[]
    for mode in ('off','raw'):
        log=tmp_path/(mode+'.log');log.write_text(''.join(host_fixture(s,mode) for s in shapes))
        plans.append(rr.collect(log,tmp_path/mode,gear,dict(cfg,gate_prefetch_mode=mode)))
    rr.compare(*plans)
    bad=copy.deepcopy(plans[1]);bad['plans']['16,2560,4096']['cube']['db_l0b']=1
    with pytest.raises(ValueError,match='non-Gate'):rr.compare(plans[0],bad)
    frozen=dict(pairs={'forward':{'comparison':{'cases':[{'baseline_launch':plans[0]['plans']['16,2560,19456']}]}}})
    rr.accepted_gate(plans[0]['plans']['16,2560,19456'],frozen,'B')
    altered=copy.deepcopy(plans[0]['plans']['16,2560,19456']);altered['available_cores']=8
    with pytest.raises(ValueError,match='accepted'):rr.accepted_gate(altered,frozen,'B')
    # Missing and interleaved callbacks must never be paired by guesswork.
    log.write_text(host_fixture(shapes[0],'raw'))
    with pytest.raises(ValueError,match='coverage'):rr.collect(log,tmp_path/'missing',gear,dict(cfg,gate_prefetch_mode='raw'))
    lines=host_fixture(shapes[0],'raw').splitlines();log.write_text(lines[0]+'\n'+lines[0]+'\n')
    with pytest.raises(ValueError,match='interleaved'):rr.collect(log,tmp_path/'mixed',gear,dict(cfg,gate_prefetch_mode='raw'))


def test_private_opp_inventory_hash_and_root_binding(tmp_path):
    vendor=tmp_path/'opp/vendors/customize';kernel=vendor/'op_impl/ai_core/tbe/kernel/ascend310p/example/a.o'
    kernel.parent.mkdir(parents=True);kernel.write_bytes(b'CPU_ELF_FIXTURE');meta=kernel.with_suffix('.json');meta.write_text('{}')
    lib=vendor/'op_api/lib/libcust_opapi.so';lib.parent.mkdir(parents=True);lib.write_bytes(b'CPU_DSO_FIXTURE')
    opp=dict(install=str(tmp_path/'opp'),preflight=dict(vendor=str(vendor),files=[dict(path=str(p),sha256=c.sha(p)) for p in (meta,lib)],
        kernels=[dict(binary=str(kernel),sha256=c.sha(kernel),json=str(meta),json_sha256=c.sha(meta))]))
    c.verify_opp(opp)
    kernel.write_bytes(b'CHANGED')
    with pytest.raises(ValueError,match='Kernel identity'):c.verify_opp(opp)
    kernel.write_bytes(b'CPU_ELF_FIXTURE');extra=vendor/'unknown.so';extra.write_bytes(b'x')
    with pytest.raises(ValueError,match='inventory'):c.verify_opp(opp)


def test_failed_predecessor_and_source_drift_block(tmp_path):
    report=tmp_path/'report.json'
    current=dict(sources=[],device=0,cann_root='/selected')
    c.write(report,dict(current,status='BLOCKED',error='original failure'))
    with pytest.raises(ValueError,match='predecessor'):run.prior(report,'precheck',current)
    c.write(report,dict(current,status=run.STATES['precheck']))
    assert run.prior(report,'precheck',current)['status']==run.STATES['precheck']
    with pytest.raises(ValueError,match='identity drift'):run.prior(report,'precheck',dict(current,device=1))


def test_stage_script_never_has_an_automatic_all(tmp_path):
    p=subprocess.run(['bash',str(HERE/'server_stage.sh'),'all',str(tmp_path/'not-created')],capture_output=True,text=True)
    assert p.returncode==2 and not (tmp_path/'not-created').exists()
