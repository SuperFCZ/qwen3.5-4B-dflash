"""The interface producer has two explicit bases; cwd is never a candidate."""
import copy
import sys
from pathlib import Path

import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from split_atc import c
from raw_atc_interface_identity import bind_record,canonical_worker_report


@pytest.fixture
def interface_case(tmp_path):
    worker=tmp_path/'export/B/c16';worker.mkdir(parents=True)
    native=tmp_path/'native';native.mkdir()
    lock=worker/'source-lock.json';c.write(lock,dict(status='PASS',scope='CPU audit report'))
    air=native/'air-manifest-w8a16-chunk.json';c.write(air,dict(status='PASS',scope='CPU Native AIR fixture'))
    dep=native/'deployment.json';c.write(dep,dict(air_manifest=dict(c.identity(air),path=air.name)))
    iface=dict(status='PASS',source_lock=dict(c.identity(lock),path=lock.name),native_air=dict(c.identity(air),path=air.name))
    c.write(worker/'draft-interface.json',iface)
    model=dict(output_root=str(worker),interfaces=iface,other={'unchanged':True})
    report=worker/'report.json';c.write(report,model)
    evidence=dict(H=dict(native_air=c.identity(air),native_deployment=c.identity(dep)))
    return report,model,evidence


def test_correct_two_roots_and_original_files_are_unchanged(interface_case,tmp_path,monkeypatch):
    report,model,evidence=interface_case
    paths=[report,report.parent/'draft-interface.json',report.parent/'source-lock.json',Path(evidence['H']['native_air']['path']),Path(evidence['H']['native_deployment']['path'])]
    before={p:p.read_bytes() for p in paths};original=copy.deepcopy(model)
    monkeypatch.chdir(tmp_path)
    bound=canonical_worker_report(model,report,report.parent,evidence)
    assert model==original
    assert bound['interfaces']['source_lock']==c.identity(report.parent/'source-lock.json')
    assert bound['interfaces']['native_air']==evidence['H']['native_air']
    c.verify_tree(bound)
    assert canonical_worker_report(bound,report,report.parent,evidence)==bound
    assert all(p.read_bytes()==data for p,data in before.items())
    assert bound['interface_identity_binding']['worker_root']!=bound['interface_identity_binding']['native_air_root']


@pytest.mark.parametrize('field',['source_lock','native_air'])
def test_missing_referenced_file_is_blocked(interface_case,field):
    report,model,evidence=interface_case
    target=report.parent/'source-lock.json' if field=='source_lock' else Path(evidence['H']['native_air']['path'])
    target.unlink()
    with pytest.raises(FileNotFoundError):canonical_worker_report(model,report,report.parent,evidence)


@pytest.mark.parametrize('kind',['bytes','sha','declared-bytes','declared-sha'])
def test_size_and_sha_are_not_refreshed_to_current_file(interface_case,kind):
    report,model,evidence=interface_case;target=report.parent/'source-lock.json'
    if kind=='bytes':target.write_bytes(target.read_bytes()+b' ')
    elif kind=='sha':target.write_bytes(b'['+target.read_bytes()[1:])
    elif kind=='declared-bytes':model['interfaces']['source_lock']['bytes']+=1
    else:model['interfaces']['source_lock']['sha256']='0'*64
    with pytest.raises(ValueError):canonical_worker_report(model,report,report.parent,evidence)


@pytest.mark.parametrize('missing',[False,True])
def test_matching_cwd_decoy_never_substitutes_worker_file(interface_case,tmp_path,monkeypatch,missing):
    report,model,evidence=interface_case;target=report.parent/'source-lock.json'
    cwd=tmp_path/'cwd';cwd.mkdir();(cwd/target.name).write_bytes(target.read_bytes());monkeypatch.chdir(cwd)
    if missing:target.unlink()
    else:target.write_bytes(b'wrong file at actual historical root')
    with pytest.raises((ValueError,FileNotFoundError)):canonical_worker_report(model,report,report.parent,evidence)


def test_conflicting_worker_roots_block_even_when_both_files_match(interface_case,tmp_path):
    report,model,evidence=interface_case;other=tmp_path/'other-worker';other.mkdir()
    (other/'source-lock.json').write_bytes((report.parent/'source-lock.json').read_bytes())
    with pytest.raises(ValueError,match='ambiguous Worker'):canonical_worker_report(model,report,other,evidence)


def test_conflicting_native_air_contexts_block_even_with_same_bytes(interface_case,tmp_path):
    report,model,evidence=interface_case;air=Path(evidence['H']['native_air']['path'])
    other=tmp_path/'other-native';other.mkdir();copyfile=other/air.name;copyfile.write_bytes(air.read_bytes())
    dep=other/'deployment.json';c.write(dep,dict(air_manifest=dict(c.identity(copyfile),path=copyfile.name)))
    evidence['H']['native_deployment']=c.identity(dep)
    with pytest.raises(ValueError,match='ambiguous interface'):canonical_worker_report(model,report,report.parent,evidence)


def test_unknown_relative_field_stays_blocked(interface_case):
    report,model,evidence=interface_case
    model['unknown_reference']=copy.deepcopy(model['interfaces']['source_lock'])
    bound=canonical_worker_report(model,report,report.parent,evidence)
    with pytest.raises(ValueError,match='explicitly bound absolute'):c.verify_tree(bound)


def test_same_basename_at_worker_root_cannot_replace_native_air(interface_case):
    report,model,evidence=interface_case;native=Path(evidence['H']['native_air']['path'])
    decoy=report.parent/native.name;decoy.write_bytes(native.read_bytes())
    model['interfaces']['native_air']=c.identity(decoy)
    with pytest.raises(ValueError,match='declared owner'):canonical_worker_report(model,report,report.parent,evidence)


def test_producer_report_and_inline_interfaces_must_agree(interface_case):
    report,model,evidence=interface_case
    producer=c.read(report.parent/'draft-interface.json');producer['status']='BLOCKED';c.write(report.parent/'draft-interface.json',producer)
    with pytest.raises(ValueError,match='producer report'):canonical_worker_report(model,report,report.parent,evidence)


def test_relative_traversal_is_not_an_alternate_base(interface_case):
    report,model,_=interface_case;record=copy.deepcopy(model['interfaces']['source_lock']);record['path']='../source-lock.json'
    with pytest.raises(ValueError,match='unexpected relative'):bind_record(record,[report.parent/'source-lock.json'],'source_lock')
