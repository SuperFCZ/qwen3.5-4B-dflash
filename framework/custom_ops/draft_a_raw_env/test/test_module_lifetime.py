"""CPU lifecycle fixtures: an actual child exit deletes its own TemporaryDirectory."""
import atexit
import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType

import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import raw_env_contract as ec
import raw_env_lifetime as life
import renew_admission as renewal
c=ec.c
DATA=b'# CPU lifecycle fixture, never a real PyTorch module\nvalue = 123\n'


def install_fixture(root):
    package=root/'torch';package.mkdir(exist_ok=True)
    paths={life.GENERATOR:package/'distributed/nn/jit/instantiator.py',life.TEMPLATE:package/'distributed/nn/jit/templates/remote_module_template.py'}
    init=package/'__init__.py';init.write_text('# CPU torch fixture\n')
    torch=ModuleType('torch');torch.__file__=str(init);torch.__version__='2.8.0+cpu';sys.modules['torch']=torch
    for name,path in paths.items():
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# CPU producer fixture\n')
        mod=ModuleType(name);mod.__file__=str(path);sys.modules[name]=mod
    owner=sys.modules[life.GENERATOR]
    owner._TEMP_DIR=tempfile.TemporaryDirectory(dir=str(root))
    owner.INSTANTIATED_TEMPLATE_DIR_PATH=owner._TEMP_DIR.name
    atexit.register(owner._TEMP_DIR.cleanup)
    sys.path.append(owner._TEMP_DIR.name)
    source=Path(owner._TEMP_DIR.name)/(life.NAME+'.py');source.write_bytes(DATA)
    spec=importlib.util.spec_from_file_location(life.NAME,source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);sys.modules[life.NAME]=module
    files={name:dict(alias=str(Path(mod.__file__).absolute()),file=c.identity(mod.__file__)) for name,mod in
        [('torch',torch),*[(name,sys.modules[name]) for name in paths],(life.NAME,module)]}
    return owner,source,files


@pytest.fixture
def modules(tmp_path,monkeypatch):
    names=('torch',life.GENERATOR,life.TEMPLATE,life.NAME)
    old={name:sys.modules.get(name) for name in names};old_path=list(sys.path)
    monkeypatch.setattr(life,'BYTES',len(DATA));monkeypatch.setattr(life,'SHA256',hashlib.sha256(DATA).hexdigest())
    owner,source,records=install_fixture(tmp_path);original_td=owner._TEMP_DIR
    out=tmp_path/'report';out.mkdir()
    try:yield owner,source,records,out
    finally:
        original_td.cleanup();sys.path[:]=old_path
        for name,value in old.items():
            if value is None:sys.modules.pop(name,None)
            else:sys.modules[name]=value


def test_real_server_generated_identity_is_pinned():
    assert life.BYTES==2355 and life.SHA256=='8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8'


def test_child_scope_cleanup_keeps_durable_identity(modules):
    owner,source,records,out=modules
    persistent,generated=life.capture(records,out)
    assert source.is_file() and life.NAME in records and life.NAME not in persistent
    owner._TEMP_DIR.cleanup();assert not source.exists()
    c.verify_tree({'loaded_files':persistent,'process_modules':generated})
    ec.verify_loaded_files(persistent);life.verify(generated,persistent,out)
    assert c.verify(generated[life.NAME]['snapshot']).read_bytes()==DATA


def test_independent_process_exit_and_parent_validation(tmp_path,monkeypatch):
    monkeypatch.setattr(life,'BYTES',len(DATA));monkeypatch.setattr(life,'SHA256',hashlib.sha256(DATA).hexdigest())
    out=tmp_path/'child';out.mkdir()
    code=f"""import sys,hashlib,json
from pathlib import Path
sys.path[:0]=[{str(HERE/'test')!r},{str(HERE)!r}]
import test_module_lifetime as t
import raw_env_lifetime as life
life.BYTES=len(t.DATA);life.SHA256=hashlib.sha256(t.DATA).hexdigest()
out=Path({str(out)!r});owner,source,records=t.install_fixture(out)
persistent,generated=life.capture(records,out)
(out/'report.json').write_text(json.dumps(dict(loaded_files=persistent,process_modules=generated)))
"""
    p=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    r=c.read(out/'report.json');origin=Path(r['process_modules'][life.NAME]['original_realpath'])
    assert not origin.exists() and not origin.parent.exists()
    life.verify(r['process_modules'],r['loaded_files'],out)


@pytest.mark.parametrize('change',['generated-content','wrong-owner','missing-origin','producer-drift'])
def test_unproven_live_origin_is_blocked(modules,change):
    owner,source,records,out=modules
    if change=='generated-content':source.write_bytes(DATA+b'changed')
    elif change=='wrong-owner':owner._TEMP_DIR='not an owner'
    elif change=='missing-origin':source.unlink()
    else:Path(records[life.GENERATOR]['file']['path']).write_text('changed')
    with pytest.raises((ValueError,FileNotFoundError,AttributeError)):life.capture(records,out)


@pytest.mark.parametrize('change',['snapshot','producer','ordinary-missing','unknown-process-module'])
def test_parent_does_not_skip_other_integrity_failures(modules,change):
    owner,source,records,out=modules;persistent,generated=life.capture(records,out);owner._TEMP_DIR.cleanup()
    if change=='snapshot':Path(generated[life.NAME]['snapshot']['path']).write_bytes(b'')
    elif change=='producer':Path(persistent[life.GENERATOR]['file']['path']).write_bytes(b'')
    elif change=='ordinary-missing':Path(persistent['torch']['file']['path']).unlink()
    else:generated['unknown_module']=generated[life.NAME]
    with pytest.raises((ValueError,FileNotFoundError)):
        c.verify_tree({'loaded_files':persistent,'process_modules':generated})
        ec.verify_loaded_files(persistent);life.verify(generated,persistent,out)


def test_only_reviewed_random_paths_are_removed_from_comparison(modules,tmp_path):
    owner,source,records,out=modules
    persistent,left=life.capture(records,out)
    second=tmp_path/'second';second.mkdir()
    right={name:dict(item) for name,item in left.items()}
    for item in right.values():
        item['owner_temp_root']='/different/child';item['original_realpath']='/different/child/'+life.NAME+'.py'
        item['capture_pid']+=1;item['snapshot']=dict(item['snapshot'],path='/other/evidence.snapshot')
    assert life.comparison(left)==life.comparison(right)
    right[life.NAME]['observed_sha256']='0'*64
    assert life.comparison(left)!=life.comparison(right)


def test_previous_environment_sources_require_exact_git_blob(tmp_path):
    if not shutil.which('git'):pytest.skip('Git CLI integration unavailable in this Docker; archive contract covered by fixture')
    paths=renewal.git_bytes('ls-tree','-r','--name-only',renewal.BASE,'--',renewal.PREFIX).decode().splitlines()
    records=[]
    for rel in paths:
        blob=renewal.git_bytes('cat-file','blob',renewal.BASE+':'+rel)
        records.append(dict(path=str(c.REPO/rel),bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest()))
    sealed=renewal.archive_previous_sources(records,tmp_path/'valid')
    assert len(sealed)==len(records);c.verify_tree(sealed)
    records[0]['sha256']='0'*64
    with pytest.raises(ValueError,match='immutable Git blob'):renewal.archive_previous_sources(records,tmp_path/'wrong')


def test_source_archive_contract_without_git_dependency(tmp_path,monkeypatch):
    rel=renewal.PREFIX+'/contract.py';blob=b'# CPU source history fixture\n'
    def git(*args):
        if args[0]=='ls-tree':return (rel+'\n').encode()
        assert args==('cat-file','blob',renewal.BASE+':'+rel)
        return blob
    monkeypatch.setattr(renewal,'git_bytes',git)
    row=dict(path=str(c.REPO/rel),bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest())
    result=renewal.archive_previous_sources([row],tmp_path/'good');c.verify_tree(result)
    assert c.verify(result[0]['archive']).read_bytes()==blob
    for bad in (dict(row,bytes=0),dict(row,sha256='0'*64)):
        with pytest.raises(ValueError,match='immutable Git blob'):renewal.archive_previous_sources([bad],tmp_path/'bad')
    with pytest.raises(ValueError,match='inventory'):renewal.archive_previous_sources([],tmp_path/'missing')


def test_renewal_keeps_assets_and_rechecks_admission(tmp_path,monkeypatch):
    old=dict(status=renewal.STATES['admit'],continuation_sources=renewal.atc_sources(),environment_sources=[],frozen={'opaque':'unchanged'},predecessors={'opaque':'unchanged'})
    p=tmp_path/'old.json';c.write(p,old);original=p.read_bytes()
    monkeypatch.setattr(renewal,'archive_previous_sources',lambda items,root:[])
    calls=[];monkeypatch.setattr(renewal,'bound_admission',lambda view:calls.append(view['frozen']))
    new=renewal.renew(old,p,tmp_path)
    assert p.read_bytes()==original and old['environment_sources']==[]
    assert new['frozen']==old['frozen'] and new['predecessors']==old['predecessors']
    assert calls==[old['frozen']] and new['environment_sources']==ec.sources()
    assert new['environment_transition']['previous_admission']==c.identity(p)
    old['continuation_sources']=[]
    with pytest.raises(ValueError,match='ATC continuation'):renewal.renew(old,p,tmp_path)
