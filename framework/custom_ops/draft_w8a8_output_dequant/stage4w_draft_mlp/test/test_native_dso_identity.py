"""CPU /proc maps fixtures, not ELF loading, NPU execution or numerical proof."""
import copy
import os
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import real_mlp_common as c
import real_mlp_dso_identity as d


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.fixture
def setup(tmp_path):
    cann = tmp_path/'cann'
    vendors = [put(cann/p, ('vendor fixture '+p).encode()) for p in d.VENDOR_PATHS]
    opp = tmp_path/'private-opp'
    api = put(opp/'vendors/customize/op_api/lib/libcust_opapi.so', b'B-NZ private API')
    host = put(opp/'vendors/customize/op_impl/libcust_opmaster_rt2.0.so', b'B-NZ private Host')
    proto = put(opp/'vendors/customize/op_proto/libcust_opsproto_rt2.0.so', b'B-NZ private proto')
    alias = host.parent/'liboptiling.so'; alias.symlink_to(host)
    runner = put(tmp_path/'bin/normal_runner', b'frozen normal runner')
    build = dict(opp=str(opp), runner=c.identity(runner), installed=dict(api=c.identity(api),
        host=c.identity(host), host_alias=str(alias), all_files=[c.identity(x) for x in (api,host,proto)]))
    maps_file(tmp_path,vendors)
    binding = d.bind_native_runtime(cann,build,proc=tmp_path)
    return cann, vendors, build, binding


def maps_file(tmp_path, paths):
    lines=[]
    for i, p in enumerate(paths):
        st=p.stat()
        lines.append(f'{i:04x}-ffff r-xp 00000000 {os.major(st.st_dev):x}:{os.minor(st.st_dev):x} {st.st_ino} {p}')
    path=tmp_path/'maps';path.write_text('\n'.join(lines)+'\n')
    st=paths[0].stat()
    (tmp_path/'mountinfo').write_text(f'1 0 {os.major(st.st_dev)}:{os.minor(st.st_dev)} / / rw - fixturefs fixture rw\n')
    (tmp_path/'ns').mkdir(exist_ok=True)
    ns=tmp_path/'ns/mnt'
    if not ns.is_symlink():ns.symlink_to('mnt:[42]')
    return path


def test_three_exact_normal_vendors_import_and_preflight(setup,tmp_path):
    cann, vendors, _, binding=setup
    imported=d.mapped_runtime(cann,binding,maps_file(tmp_path,vendors[:2]))
    full=d.mapped_runtime(cann,binding,maps_file(tmp_path,vendors+[vendors[0]]))
    assert len(imported)==2 and len(full)==3
    d.compare_loaded(imported,full)
    d.compare_loaded(full,full,complete=True)
    for row in full:
        assert row['classification']=='EXACT_SELECTED_CANN_VENDOR'
        assert row['name']==Path(row['alias']).name
        assert row['realpath']==row['file']['path']
        assert len(row['file']['sha256'])==64 and row['file']['bytes']>0
        assert row['loading_identity']['inode']==Path(row['realpath']).stat().st_ino


def test_known_vendor_internal_symlink_canonical_mapping(setup,tmp_path):
    cann, vendors, build, _=setup
    alias=vendors[0];real=cann/'aarch64-linux/lib64/actual_vendor.so'
    put(real,alias.read_bytes());alias.unlink();alias.symlink_to(real)
    binding=d.bind_native_runtime(cann,build,proc=cann.parent)
    assert binding['vendor_libraries'][0]['symlink_chain']
    for path in (alias,real):
        rows=d.mapped_runtime(cann,binding,maps_file(tmp_path,[path]))
        assert rows[0]['file']==c.identity(real)


@pytest.mark.parametrize('location',['selected','external','unknown_alias'])
def test_unknown_vendor_paths_block(setup,tmp_path,location):
    cann, vendors, _, binding=setup
    root=cann if location!='external' else tmp_path/'other'
    unknown=root/'opp/vendors/other/libunknown.so'
    if location=='unknown_alias':
        unknown.parent.mkdir(parents=True);unknown.symlink_to(vendors[0])
    else:put(unknown,b'unknown vendor')
    with pytest.raises(ValueError,match='unknown vendor DSO') as error:
        d.mapped_runtime(cann,binding,maps_file(tmp_path,[unknown]))
    assert str(unknown) in str(error.value)


@pytest.mark.parametrize('mode',['path','alias','copy','runner','proto_copy'])
def test_private_bnz_paths_aliases_copies_and_runner_block(setup,tmp_path,mode):
    cann, _, build, binding=setup
    path=c.verify(build['installed']['api'])
    if mode=='alias':
        alias=tmp_path/'libalias.so';alias.symlink_to(path);path=alias
    if mode=='copy':path=put(tmp_path/'libcopied.so',path.read_bytes())
    if mode=='proto_copy':path=put(tmp_path/'librenamed.so',c.verify(build['installed']['all_files'][2]).read_bytes())
    if mode=='runner':path=c.verify(build['runner'])
    with pytest.raises(ValueError,match='private B-NZ') as error:
        d.mapped_runtime(cann,binding,maps_file(tmp_path,[path]))
    assert str(path) in str(error.value)


def test_private_copy_at_an_approved_vendor_path_is_not_allowed(setup):
    cann, vendors, build, _=setup
    vendors[2].write_bytes(c.verify(build['installed']['api']).read_bytes())
    with pytest.raises(ValueError,match='private B-NZ identity/copy'):
        d.bind_native_runtime(cann,build,proc=cann.parent)


@pytest.mark.parametrize('before_bind',[False,True])
def test_approved_vendor_symlink_escape_blocks(setup,tmp_path,before_bind):
    cann, vendors, build, binding=setup
    outside=put(tmp_path/'outside/libvendor.so',b'other installation')
    vendors[0].unlink();vendors[0].symlink_to(outside)
    with pytest.raises(ValueError,match='escapes allowed root') as error:
        if before_bind:d.bind_native_runtime(cann,build,proc=cann.parent)
        else:d.mapped_runtime(cann,binding,maps_file(tmp_path,[vendors[0]]))
    assert str(vendors[0]) in str(error.value)


def test_symlink_out_and_back_still_blocks(setup,tmp_path):
    cann, vendors, build, _=setup
    actual=put(cann/'lib64/vendor.so',vendors[0].read_bytes())
    outside=tmp_path/'outside.so';outside.symlink_to(actual)
    vendors[0].unlink();vendors[0].symlink_to(outside)
    with pytest.raises(ValueError,match='hop escapes'):
        d.bind_native_runtime(cann,build,proc=cann.parent)


@pytest.mark.parametrize('which',['vendor','private','runner','alias'])
def test_frozen_identity_drift_blocks(setup,tmp_path,which):
    cann, vendors, build, binding=setup
    mapped=maps_file(tmp_path,vendors)
    if which=='vendor':vendors[1].write_bytes(b'changed CANN library')
    elif which=='private':c.verify(build['installed']['host']).write_bytes(b'changed private library')
    elif which=='runner':c.verify(build['runner']).write_bytes(b'changed runner')
    else:
        alias=Path(build['installed']['host_alias']);alias.unlink();alias.symlink_to(c.verify(build['installed']['api']))
    with pytest.raises(ValueError,match='identity|drift|changed'):
        d.mapped_runtime(cann,binding,mapped)


def test_allowlist_cannot_be_expanded_in_a_report(setup,tmp_path):
    cann, vendors, _, binding=setup
    changed=copy.deepcopy(binding);changed['vendor_libraries'].append(changed['vendor_libraries'][0])
    with pytest.raises(ValueError,match='inventory changed'):
        d.mapped_runtime(cann,changed,maps_file(tmp_path,vendors))


def test_missing_exact_vendor_blocks_without_fallback(setup):
    cann, vendors, build, _=setup;vendors[0].unlink()
    with pytest.raises(ValueError,match='missing runtime path'):
        d.bind_native_runtime(cann,build,proc=cann.parent)


def test_nonvendor_library_identity_and_smoke_benchmark_drift(setup,tmp_path):
    cann, _, _, binding=setup
    lib=put(tmp_path/'system/libfixture.so',b'ordinary system library')
    alias=tmp_path/'libalias.so';alias.symlink_to(lib)
    mapped=maps_file(tmp_path,[alias])
    before=d.mapped_runtime(cann,binding,mapped)
    assert before[0]['classification']=='RECORDED_NON_VENDOR_DSO'
    assert before[0]['file']==c.identity(lib) and before[0]['alias']==str(alias)
    lib.write_bytes(b'drifted ordinary library')
    after=d.mapped_runtime(cann,binding,mapped)
    with pytest.raises(ValueError,match='differs from Smoke'):d.compare_loaded(after,before)
    other=put(tmp_path/'system/libnew.so',b'unknown new DSO at benchmark')
    extra=d.mapped_runtime(cann,binding,maps_file(tmp_path,[other]))
    with pytest.raises(ValueError,match='differs from Smoke'):d.compare_loaded(extra,before)
    with pytest.raises(ValueError,match='inventory differs'):d.compare_loaded([],before,complete=True)


def test_another_ascend_installation_blocks(setup,tmp_path):
    cann, _, _, binding=setup
    elsewhere=put(tmp_path/'other-cann/libother.so',b'other toolkit')
    alias=cann/'lib64/libother.so';alias.parent.mkdir();alias.symlink_to(elsewhere)
    with pytest.raises(ValueError,match='another Ascend installation') as error:
        d.mapped_runtime(cann,binding,maps_file(tmp_path,[alias]))
    assert str(alias) in str(error.value) and str(elsewhere) in str(error.value)


@pytest.mark.parametrize('change',['inode','device','duplicate'])
def test_actual_loaded_file_identity_not_only_path(setup,tmp_path,change):
    cann, vendors, _, binding=setup
    mapped=maps_file(tmp_path,[vendors[0]])
    parts=mapped.read_text().split(maxsplit=5)
    parts[4]=str(int(parts[4])+1)
    if change=='device':parts[3]='ffff:ffff'
    wrong=' '.join(parts)
    mapped.write_text((mapped.read_text() if change=='duplicate' else '')+wrong)
    with pytest.raises(ValueError,match='device/inode|mapping identity'):
        d.mapped_runtime(cann,binding,mapped)


def test_selected_root_is_bound(setup,tmp_path):
    _, vendors, _, binding=setup
    other=tmp_path/'other';other.mkdir()
    with pytest.raises(ValueError,match='CANN root changed'):
        d.mapped_runtime(other,binding,maps_file(tmp_path,vendors))
