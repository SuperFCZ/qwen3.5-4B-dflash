"""CPU filesystem fixtures only; no SDK libraries or historical files changed."""
import copy
import sys
from pathlib import Path

import pytest

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import raw_common as c
import raw_evidence as ev


@pytest.fixture
def installed(tmp_path):
    root=tmp_path/'head/opp';folder=root/'vendors/customize/op_impl/ai_core/tbe/op_tiling'
    real=folder/'lib/linux/aarch64/libcust_opmaster_rt2.0.so'
    real.parent.mkdir(parents=True);real.write_bytes(b'CPU_DSO_FIXTURE_NOT_ELF')
    alias=folder/'liboptiling.so';alias.symlink_to('lib/linux/aarch64/'+real.name)
    records=[c.identity(real),dict(c.identity(real),path=str(alias))]
    report=dict(status='PASS',release_only=True,install_root=str(root),libraries=records)
    return report,alias,real


def test_original_false_positive_and_bound_alias_roundtrip(installed,tmp_path):
    report,alias,real=installed;original=copy.deepcopy(report)
    # Reproduces the former exact-dictionary comparison: only path differs.
    assert c.identity(alias)!=report['libraries'][1]
    assert all(c.identity(alias)[k]==report['libraries'][1][k] for k in ('bytes','sha256'))
    with pytest.raises(ValueError,match='identity changed'):c.verify(report['libraries'][1])
    bound=ev.bind_head_libraries(report)
    assert report==original and len(bound)==2
    assert bound[0]['file']==bound[1]['file']==c.identity(real)
    assert bound[1]['alias']==str(alias) and bound[1]['links']==[dict(path=str(alias),target='lib/linux/aarch64/'+real.name)]
    p=tmp_path/'new-binding.json';c.write(p,bound);c.verify_tree(c.read(p))


@pytest.mark.parametrize('kind',['bytes','sha','missing','broken-alias'])
def test_missing_or_changed_library_never_reaccepted(installed,kind):
    report,alias,real=installed
    if kind=='bytes':real.write_bytes(real.read_bytes()+b'x')
    elif kind=='sha':real.write_bytes(b'X'+real.read_bytes()[1:])
    elif kind=='missing':real.unlink()
    else:alias.unlink();alias.symlink_to('missing.so')
    with pytest.raises((ValueError,FileNotFoundError)):ev.bind_head_libraries(report)


def test_alias_without_frozen_canonical_counterpart_blocks(installed):
    report,_,_=installed;report['libraries']=report['libraries'][1:]
    with pytest.raises(ValueError,match='frozen canonical'):ev.bind_head_libraries(report)


def test_conflicting_expected_identity_is_not_deduplicated(installed):
    report,_,_=installed
    report['libraries'].append(dict(report['libraries'][0],sha256='0'*64))
    with pytest.raises(ValueError,match='conflicting'):ev.bind_head_libraries(report)


def test_alias_counterpart_hash_must_agree(installed):
    report,_,_=installed;report['libraries'][1]['sha256']='0'*64
    with pytest.raises(ValueError,match='frozen canonical'):ev.bind_head_libraries(report)


def test_alias_escape_rejected_even_if_hash_and_frozen_record_match(installed,tmp_path):
    report,alias,real=installed;outside=tmp_path/'outside.so';outside.write_bytes(real.read_bytes())
    alias.unlink();alias.symlink_to(outside)
    report['libraries']=[report['libraries'][1],c.identity(outside)]
    with pytest.raises(ValueError,match='escapes OPP'):ev.bind_head_libraries(report)


@pytest.mark.parametrize('kind',['content','target','chain'])
def test_recursive_recheck_detects_drift_after_binding(installed,kind):
    report,alias,real=installed;bound=ev.bind_head_libraries(report)
    if kind=='content':real.write_bytes(real.read_bytes()+b'x')
    elif kind=='target':
        copyfile=real.with_name('same-bytes.so');copyfile.write_bytes(real.read_bytes())
        alias.unlink();alias.symlink_to(copyfile)
    else:
        # Same final target and bytes, different link chain must still fail.
        intermediate=alias.with_name('middle.so');intermediate.symlink_to(real)
        alias.unlink();alias.symlink_to(intermediate.name)
    with pytest.raises(ValueError):c.verify_tree({'bound':bound})


def test_parent_directory_symlink_is_in_the_binding(installed):
    report,alias,real=installed
    parent_alias=real.parent.parent/'arm-alias';parent_alias.symlink_to(real.parent.name,target_is_directory=True)
    other=parent_alias/real.name
    report['libraries'].append(dict(c.identity(real),path=str(other)))
    bound=ev.bind_head_libraries(report)
    assert bound[-1]['links']==[dict(path=str(parent_alias),target=real.parent.name)]
    c.verify_tree(bound)


def test_historical_release_not_air_or_h_joint_runtime(installed,tmp_path):
    report,_,_=installed
    release=tmp_path/'release.json';c.write(release,dict(installed_opp=report))
    state=dict(args={'c_release':str(release)},provenance=dict(records={'c_release':c.identity(release)},c_installed_opp=copy.deepcopy(report)),
        opp={'install':str(tmp_path/'H/joint/opp')})
    original=copy.deepcopy(state);saved=release.read_bytes()
    binding=ev.head_release_evidence(tmp_path,state)
    assert binding['role']=='HISTORICAL_C_RELEASE_SOURCE_PACKAGE_NOT_H_JOINT_RUNTIME_OPP'
    assert state==original and release.read_bytes()==saved
    c.verify_tree(binding)
    bad=copy.deepcopy(state);bad['provenance']['c_installed_opp']['libraries']=[]
    with pytest.raises(ValueError,match='provenance differs'):ev.head_release_evidence(tmp_path,bad)
    release.write_bytes(saved+b' ')
    with pytest.raises(ValueError,match='linked file changed'):ev.head_release_evidence(tmp_path,state)
