"""CPU fixtures: actual C++ writer, no CANN/ATC or NPU execution."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from sidecar_common import c,ec,audit_host
from sidecar_atc import ENV,SidecarAtcRunner,decode_sidecar,collect_host_evidence,verify_host_evidence
from host_build import tree,compare_install
from derive_worker import verify
from raw_resources import compare
sys.path.insert(0,str(c.HERE/'test'))
from test_raw import host_fixture


def config(mode):
    return dict(core_limit=0,dequant_mode='batched',pipeline_mode='prefetch',launch_version=5,kv_m80_mode='a-ub',scale_mode='broadcast',k_tile=256,k_tile_policy='m80-synthetic-k128-v1',gate_prefetch_mode=mode)


def inputs(root,mode='off',gear=16):
    shapes=[(gear,12800,2560),(gear+16,2560,2048),(16,2560,4096),(16,4096,2560),(16,2560,19456),(16,9728,2560)]
    result=[]
    for i,shape in enumerate(shapes):
        p=root/f'{mode}-{gear}-{i}.txt';p.write_text(host_fixture(shape,mode));result.append(p)
    return result


@pytest.fixture(scope='module')
def writer(tmp_path_factory):
    compiler=shutil.which('c++')
    if not compiler:pytest.skip('C++ compiler unavailable')
    root=tmp_path_factory.mktemp('writer');binary=root/'writer'
    subprocess.run([compiler,'-std=c++11','-pthread','-I'+str(HERE/'op_host'),str(HERE/'test/writer_fixture.cpp'),'-o',str(binary)],check=True,capture_output=True)
    return binary


def test_stdout_189_warnings_and_concurrent_callback_sidecar(writer,tmp_path):
    runner=SidecarAtcRunner(tmp_path/'logs',c.identity(writer),tmp_path)
    result=runner([writer,*inputs(tmp_path),'--output='+str(tmp_path/'model')],tmp_path)
    assert result.returncode==0 and result.stdout.count('GMM Warning')==189
    assert 'DFLASH_GROUP_QUANT_' not in result.stdout and result.stderr==''
    row=runner.records[0];c.verify_tree(row)
    assert c.verify(row['stdout']).read_bytes()==b'GMM Warning CPU fixture\n'*189
    assert len(decode_sidecar(c.verify(row['host_evidence'])))==36
    plan=collect_host_evidence(row['host_evidence'],tmp_path/'plan',16,config('off'))
    verify_host_evidence(plan,16,config('off'))


def test_separate_process_callbacks_are_not_interleaved(writer,tmp_path):
    sidecar=tmp_path/'host-evidence.jsonl';sidecar.touch();env=dict(os.environ,NO_WARN='1',**{ENV:str(sidecar)})
    children=[subprocess.Popen([writer,p],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE) for p in inputs(tmp_path)]
    for child in children:
        stdout,stderr=child.communicate();assert child.returncode==0,(stdout,stderr)
    rows=decode_sidecar(sidecar)
    assert len(rows)==36
    collect_host_evidence(c.identity(sidecar),tmp_path/'resources',16,config('off'))


def valid_jsonl(root,mode='off',gear=16):
    lines=[]
    for p in inputs(root,mode,gear):
        for line in p.read_text().splitlines():
            tag,payload=line.split(' ',1);lines.append(dict(record=tag[len('DFLASH_GROUP_QUANT_'):],payload=json.loads(payload)))
    return lines


@pytest.mark.parametrize('fault',['missing','truncated','order','shape','type','field','unknown'])
def test_bad_sidecar_blocks(tmp_path,fault):
    p=tmp_path/'sidecar.jsonl';rows=valid_jsonl(tmp_path)
    if fault=='missing':
        with pytest.raises(FileNotFoundError):decode_sidecar(p)
        return
    if fault=='order':rows[0],rows[1]=rows[1],rows[0]
    elif fault=='shape':rows[6:12]=copy.deepcopy(rows[:6])
    elif fault=='type':rows[5]['payload']['depth_a1']=True
    elif fault=='field':del rows[5]['payload']['depth_a1']
    elif fault=='unknown':rows[0]['record']='UNKNOWN'
    raw=''.join(json.dumps(x)+'\n' for x in rows)
    if fault=='truncated':
        lines=raw.splitlines();lines[5]=lines[5][:-2];raw='\n'.join(lines)+'\n'
    p.write_text(raw);original=p.read_bytes()
    with pytest.raises(ValueError):collect_host_evidence(c.identity(p),tmp_path/'resources',16,config('off'))
    assert p.read_bytes()==original


@pytest.mark.parametrize('gear',[16,64])
def test_original_b_r_resource_contract(tmp_path,gear):
    records=[]
    for mode in ('off','raw'):
        p=tmp_path/(mode+'.jsonl');p.write_text(''.join(json.dumps(x)+'\n' for x in valid_jsonl(tmp_path,mode,gear)))
        records.append(collect_host_evidence(c.identity(p),tmp_path/mode,gear,config(mode)))
    compare(*records)
    bad=copy.deepcopy(records[1]);bad['plans']['16,2560,4096']['cube']['db_l0b']=1
    with pytest.raises(ValueError):compare(records[0],bad)
    bad=copy.deepcopy(records[1]);bad['plans']['16,2560,19456']['raw_banks']=1
    with pytest.raises(ValueError):compare(records[0],bad)


def test_unset_environment_preserves_original_stdout(writer,tmp_path):
    paths=inputs(tmp_path);env=dict(os.environ,NO_WARN='1');env.pop(ENV,None)
    # One callback avoids imposing order on concurrent stdout legacy logging.
    r=subprocess.run([writer,paths[0]],env=env,capture_output=True)
    assert r.returncode==0 and r.stdout==paths[0].read_bytes() and not r.stderr


@pytest.mark.parametrize('path_kind',['relative','empty','missing','directory','symlink'])
def test_evidence_write_failure_never_falls_back_to_stdout(writer,tmp_path,path_kind):
    data=inputs(tmp_path)[0]
    p=tmp_path/'target';value=str(p)
    if path_kind=='relative':value='relative.jsonl'
    elif path_kind=='empty':value=''
    elif path_kind=='directory':p.mkdir()
    elif path_kind=='symlink':
        dest=tmp_path/'actual';dest.touch();p.symlink_to(dest)
    r=subprocess.run([writer,data],env=dict(os.environ,NO_WARN='1',**{ENV:value}),capture_output=True)
    assert r.returncode!=0 and b'DFLASH_GROUP_QUANT_EVIDENCE_ERROR' in r.stderr
    assert b'DFLASH_GROUP_QUANT_LAUNCH' not in r.stdout


def test_only_host_libraries_can_differ_and_kernel_bytes_stay_fixed(tmp_path):
    a=tmp_path/'old';b=tmp_path/'new'
    for root in (a,b):
        kernel=root/'vendors/customize/op_impl/ai_core/tbe/kernel/ascend310p/a/k.o';kernel.parent.mkdir(parents=True);kernel.write_bytes(b'FIXED_KERNEL')
        lib=root/'host.so';lib.write_bytes(b'HOST_OLD' if root==a else b'HOST_NEW')
        (root/'liboptiling.so').symlink_to('host.so')
    r=compare_install(tree(a),tree(b),{'host.so'});assert r['all_kernel_bytes_equal']
    (b/'vendors/customize/op_impl/ai_core/tbe/kernel/ascend310p/a/k.o').write_bytes(b'CHANGED')
    with pytest.raises(ValueError,match='unexpected OPP'):compare_install(tree(a),tree(b),{'host.so'})


def test_source_audit_and_old_locks_are_unchanged():
    audit_host();verify()
    assert all(HERE not in Path(x['path']).parents for x in c.sources()+ec.sources())
