"""Historical H references use their declared root, never the caller's cwd."""
import copy
import sys
from pathlib import Path

import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import raw_common as c
import raw_evidence as ev

REPORT_FILES={
    'native_boundary':'native-base-preflight.json',
    'draft_interface':'draft-interface-preflight.json',
    'provenance':'preflight.json',
    'preflight_imports':'preflight-imports.json',
    'coordinator_process':'coordinator-process.json',
}


@pytest.fixture
def historical(tmp_path):
    root=tmp_path/'H';root.mkdir()
    args=dict(native_base='/frozen/native',c_release='/frozen/c.json',factory_config='/frozen/factory.json',
        output_dir=str(root),draft_only=True,full_draft=False)
    source_lock=dict(path='/frozen/native/SOURCE_LOCK.json',sha256='locked',verified_file_count=205)
    refs={}
    for field,name in REPORT_FILES.items():
        payload=dict(kind=field,status='PASS')
        if field=='native_boundary':payload.update(accepted_args=args,identity=source_lock)
        c.write(root/name,payload)
        refs[field]=dict(c.identity(root/name),path=name)
    state=dict(args=args,path_args=['native_base','c_release','factory_config','output_dir'],
        native_boundary=copy.deepcopy(refs['native_boundary']),report=dict(refs,source_lock=source_lock),
        opp=dict(install='/frozen/opp'))
    c.write(root/'worker-input.json',state)
    return root,state


def test_all_six_h_references_are_canonical_without_changing_originals(historical,tmp_path,monkeypatch):
    root,state=historical;before=copy.deepcopy(state)
    originals={p:p.read_bytes() for p in root.iterdir()}
    elsewhere=tmp_path/'unrelated-cwd';elsewhere.mkdir();monkeypatch.chdir(elsewhere)
    normalized=ev.canonical_historical_state(root,state)
    assert state==before
    assert len(ev.H_STATE_REFERENCES)==6
    for key,name in REPORT_FILES.items():
        assert normalized['report'][key]==c.identity(root/name)
    assert normalized['native_boundary']==normalized['report']['native_boundary']
    for key in ('args','path_args','opp'):assert normalized[key]==before[key]
    assert normalized['report']['source_lock']==before['report']['source_lock']
    # The existing build/worker entry gate verifies this serialized evidence;
    # its args and the actual Native Boundary accepted_args remain unchanged.
    evidence=dict(H=dict(state=normalized,worker_input=c.identity(root/'worker-input.json')))
    p=tmp_path/'new-evidence.json';c.write(p,evidence)
    restored=c.read(p);c.verify_tree(restored)
    boundary=c.read(c.verify(restored['H']['state']['native_boundary']))
    assert boundary['accepted_args']==before['args']
    assert boundary['identity']==before['report']['source_lock']
    assert ev.canonical_historical_state(root,normalized)==normalized  # idempotent
    assert {p:p.read_bytes() for p in root.iterdir()}==originals


@pytest.mark.parametrize('name',REPORT_FILES.values())
def test_each_missing_h_target_is_rejected(historical,name):
    root,state=historical;(root/name).unlink()
    with pytest.raises(FileNotFoundError):ev.canonical_historical_state(root,state)


@pytest.mark.parametrize('kind',['bytes','same-size-sha','declared-bytes','declared-sha'])
def test_file_or_declared_identity_drift_is_rejected(historical,kind):
    root,state=historical;p=root/'native-base-preflight.json'
    if kind=='bytes':p.write_bytes(p.read_bytes()+b' ')
    elif kind=='same-size-sha':
        raw=p.read_bytes();p.write_bytes(b'['+raw[1:])
    elif kind=='declared-bytes':state['native_boundary']['bytes']+=1
    else:state['native_boundary']['sha256']='0'*64
    with pytest.raises(ValueError,match='changed'):ev.canonical_historical_state(root,state)


@pytest.mark.parametrize('missing',[False,True])
def test_matching_cwd_decoy_cannot_replace_wrong_or_missing_h_target(historical,tmp_path,monkeypatch,missing):
    root,state=historical;p=root/'native-base-preflight.json'
    cwd=tmp_path/'cwd';cwd.mkdir();(cwd/p.name).write_bytes(p.read_bytes());monkeypatch.chdir(cwd)
    if missing:p.unlink()
    else:p.write_bytes(b'wrong H file')
    with pytest.raises((ValueError,FileNotFoundError)):ev.canonical_historical_state(root,state)


def test_canonical_evidence_still_rejects_later_tampering(historical):
    root,state=historical;normalized=ev.canonical_historical_state(root,state)
    c.verify_tree(normalized)
    p=root/'coordinator-process.json';p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(ValueError,match='identity changed'):c.verify_tree(normalized)


def test_unbound_relative_identity_rejected_even_with_matching_cwd_file(historical,monkeypatch):
    root,state=historical;monkeypatch.chdir(root)
    with pytest.raises(ValueError,match='explicitly bound absolute path'):
        c.verify_tree(state['native_boundary'])


def test_known_h_reference_cannot_escape_its_bound_root(historical,tmp_path):
    root,state=historical
    outside=tmp_path/'outside.json';outside.write_bytes((root/'native-base-preflight.json').read_bytes())
    state['native_boundary']=dict(c.identity(outside),path='../outside.json')
    with pytest.raises(ValueError,match='escapes H'):ev.canonical_historical_state(root,state)


def test_unknown_relative_record_is_not_assigned_an_implicit_root(historical,monkeypatch):
    root,state=historical
    state['report']['unknown_scope']=copy.deepcopy(state['native_boundary'])
    normalized=ev.canonical_historical_state(root,state)
    monkeypatch.chdir(root)
    with pytest.raises(ValueError,match='explicitly bound absolute path'):c.verify_tree(normalized)
