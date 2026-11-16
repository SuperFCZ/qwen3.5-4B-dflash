"""CPU environment-contract fixtures. No fake package is installed for export."""
import base64
import copy
import importlib.metadata as md
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import raw_env_contract as ec
import raw_env_worker as worker
import run_environment as driver
c=ec.c


@pytest.fixture
def package(tmp_path,monkeypatch):
    site=tmp_path/'site';pkg=site/'transformers';pkg.mkdir(parents=True)
    init=pkg/'__init__.py';init.write_text('__version__ = "5.13.1"\n')
    api=pkg/'initialization.py';api.write_text('# CPU record fixture only\n')
    info=site/'transformers-5.13.1.dist-info';info.mkdir()
    (info/'METADATA').write_text('Name: transformers\nVersion: 5.13.1\n')
    (info/'RECORD').write_text(''.join(f'{p.relative_to(site)},sha256={base64.urlsafe_b64encode(bytes.fromhex(c.sha(p))).decode().rstrip("=")},{p.stat().st_size}\n' for p in (init,api)))
    dist=md.PathDistribution(info);monkeypatch.setattr(ec.metadata,'distribution',lambda name:dist)
    module=SimpleNamespace(__file__=str(init),__path__=[str(pkg)],__version__='5.13.1')
    expected=dict(file=c.identity(init),version='5.13.1')
    return module,expected,api


def test_package_origin_version_and_record(package):
    module,expected,api=package
    assert ec.check_package(module,ec.distribution_info('transformers'),expected)['version']=='5.13.1'
    assert ec.installed_record('transformers','transformers/initialization.py')['file']==c.identity(api)
    assert ec.literal_version(module.__file__)=='5.13.1'


@pytest.mark.parametrize('kind',['version','origin','path-extension','historical-hash'])
def test_mixed_or_shadowed_package_is_blocked(package,tmp_path,kind):
    module,expected,_=package;dist=ec.distribution_info('transformers')
    if kind=='version':module.__version__='4.57.6'
    elif kind=='origin':
        other=tmp_path/'transformers.py';other.write_text(Path(module.__file__).read_text());module.__file__=str(other)
    elif kind=='path-extension':module.__path__.append(str(tmp_path))
    else:expected['file']['sha256']='0'*64
    with pytest.raises(ValueError):ec.check_package(module,dist,expected)


@pytest.mark.parametrize('kind',['changed','missing-record'])
def test_initialization_stub_or_unrecorded_file_is_blocked(package,kind):
    _,_,api=package
    if kind=='changed':api.write_text('')
    else:
        record=api.parent.parent/'transformers-5.13.1.dist-info/RECORD';record.write_text('')
    with pytest.raises(ValueError):ec.installed_record('transformers','transformers/initialization.py')


def test_auxiliary_source_is_outside_existing_experiment_lock():
    assert all(HERE not in Path(f['path']).parents for f in c.sources())
    assert {Path(f['path']).name for f in ec.sources()} >= {'raw_env_contract.py','raw_env_worker.py','run_environment.py'}


def test_existing_source_drift_blocks_reuse_before_build_or_probes(tmp_path):
    root=tmp_path/'old';(root/'precheck').mkdir(parents=True)
    c.write(root/'precheck/report.json',dict(status=ec.experiment.STATES['precheck'],sources=[],device=0,cann_root='/selected'))
    with pytest.raises(ValueError,match='identity drift: sources'):ec.previous(root)


def test_relative_historical_version_is_read_as_data_not_executed(tmp_path):
    p=tmp_path/'init.py';p.write_text('__version__="5.13.1"\nraise RuntimeError("must not execute")\n')
    assert ec.literal_version(p)=='5.13.1'
    p.write_text('__version__=unknown_call()')
    with pytest.raises(ValueError):ec.literal_version(p)


def test_missing_initialization_cannot_prepare_any_interface(package,tmp_path,monkeypatch):
    module,expected,_=package
    monkeypatch.setattr(ec,'historical_environment',lambda e:dict(modules={'transformers':expected},interpreter_hint=str(Path(sys.prefix)/'bin/python')))
    monkeypatch.setattr(ec,'direct_imports',lambda base:[])
    monkeypatch.setattr(ec,'loaded_files',lambda:{})
    calls=[]
    def importing(name):
        calls.append(name)
        if name=='transformers':return module
        raise ModuleNotFoundError("No module named 'transformers.initialization'")
    monkeypatch.setattr(worker.importlib,'import_module',importing)
    out=tmp_path/'output';req=tmp_path/'request.json'
    c.write(req,dict(output=str(out),evidence={'H':{'native_base':'/not-imported'}}))
    assert worker.main(req)==1
    r=c.read(out/'report.json')
    assert r['status']=='BLOCKED' and r['interfaces']=='NOT_PREPARED' and r['npu']=='NOT_RUN'
    assert 'initialization' in r['error'] and calls==['transformers','transformers.initialization']


def test_imported_module_alias_drift_is_blocked(tmp_path):
    real=tmp_path/'one.py';real.write_text('x=1');alias=tmp_path/'selected.py';alias.symlink_to(real)
    records={'example':dict(alias=str(alias),file=c.identity(real))};ec.verify_loaded_files(records)
    copyfile=tmp_path/'copy.py';copyfile.write_bytes(real.read_bytes());alias.unlink();alias.symlink_to(copyfile)
    with pytest.raises(ValueError,match='retargeted'):ec.verify_loaded_files(records)


def test_export_requires_successful_matching_env_report(tmp_path):
    path=tmp_path/'bad.json';c.write(path,dict(status='BLOCKED',error='missing module'))
    with pytest.raises(ValueError,match='not PASS'):driver.checked_environment(path,{'references':{}})
    with pytest.raises(SystemExit):driver.parse(['export','--source-root','/old','--output-dir','/new'])


def test_no_export_command_after_failed_import_gate(tmp_path,monkeypatch):
    old=tmp_path/'old';old.mkdir();out=tmp_path/'new';env=tmp_path/'check/environment.json';env.parent.mkdir();c.write(env,{})
    prior=dict(references={},status='CPU_FIXTURE',current={})
    monkeypatch.setattr(ec,'previous',lambda root:prior)
    monkeypatch.setattr(driver,'checked_environment',lambda path,previous:{})
    def failed(*args):raise ValueError('missing dependency')
    monkeypatch.setattr(driver,'check_workers',failed)
    command=Mock(side_effect=AssertionError('Export must not run'));monkeypatch.setattr(c,'command',command)
    args=SimpleNamespace(source_root=old,output_dir=out,environment_report=env,phase='export')
    assert driver.main(args)==1;command.assert_not_called()
    r=c.read(out/'report.json');assert r['export']=='NOT_RUN' and r['npu']=='NOT_RUN'


def test_environment_snapshot_does_not_merge_incompatible_imports():
    x=dict(python={'prefix':'/one'},packages={},loaded_files={},operators=[])
    y=copy.deepcopy(x);y['python']['prefix']='/two'
    assert ec.snapshot(x)!=ec.snapshot(y)
