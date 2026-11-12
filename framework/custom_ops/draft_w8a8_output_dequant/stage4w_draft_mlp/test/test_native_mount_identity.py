"""Mount translations are CPU contracts; mocked witnesses never prove a server."""
import copy
import errno
import os
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import real_mlp_dso_identity as d
import real_mlp_common as c
import test_native_dso_identity as base
from test_native_dso_identity import maps_file, put


@pytest.fixture
def setup(tmp_path):
    return base.setup.__wrapped__(tmp_path)


def overlay_case(tmp_path, setup, monkeypatch, permission=True):
    cann,vendors,build,binding=setup
    real=vendors[0];path=maps_file(tmp_path,[real])
    st=real.stat();parts=path.read_text().split(maxsplit=5);parts[3]='08:11'
    path.write_text(' '.join(parts))
    (tmp_path/'mountinfo').write_text(f'1 0 {os.major(st.st_dev)}:{os.minor(st.st_dev)} / / rw - overlay overlay rw,lowerdir=/unavailable/lower\n')
    binding=d.bind_native_runtime(cann,build,proc=tmp_path)
    (tmp_path/'map_files').mkdir()
    link=tmp_path/'map_files'/parts[0];link.symlink_to(real)
    actual_open=os.open
    def open_file(p,*args,**kwargs):
        if permission and Path(p)==link:raise PermissionError(errno.EPERM,'fixture permission denied')
        return actual_open(p,*args,**kwargs)
    monkeypatch.setattr(os,'open',open_file)
    witness=dict(loaded=[8,17,st.st_ino],bytes=st.st_size,sha256=c.identity(real)['sha256'],flags='MAP_PRIVATE_PROT_READ_NO_EXEC_NO_WRITE')
    monkeypatch.setattr(d,'probe_readonly_mapping',lambda *args:dict(witness))
    return cann,binding,path,real,witness,link


@pytest.mark.parametrize('permission',[True,False])
def test_proven_overlay_translation_keeps_limitations(setup,tmp_path,monkeypatch,permission):
    cann,binding,path,real,_,_=overlay_case(tmp_path,setup,monkeypatch,permission)
    rows=d.mapped_runtime(cann,binding,path);p=rows[0]['mount_proof']
    assert p['status']=='OVERLAY_TRANSLATION_PROVEN_BY_READONLY_MMAP'
    assert p['map_files_open']==('UNAVAILABLE' if permission else 'PATH_VIEW_ONLY_NOT_BACKING_IDENTITY')
    if permission:assert p['map_files_errno']==errno.EPERM and 'map_files_content' not in p
    else:assert p['map_files_content']['sha256']==c.identity(real)['sha256']
    assert rows[0]['loading_identity']['device_major']==8
    assert rows[0]['file']==c.identity(real)


def test_direct_map_files_fd_proof(setup,tmp_path,monkeypatch):
    cann,binding,path,real,_,link=overlay_case(tmp_path,setup,monkeypatch,False)
    actual_fstat=os.fstat
    from types import SimpleNamespace
    def fstat(fd):
        st=actual_fstat(fd)
        return SimpleNamespace(st_dev=os.makedev(8,17),st_ino=st.st_ino,st_size=st.st_size,st_mtime_ns=st.st_mtime_ns,st_ctime_ns=st.st_ctime_ns)
    monkeypatch.setattr(os,'fstat',fstat)
    monkeypatch.setattr(d,'probe_readonly_mapping',lambda *a:pytest.fail('direct FD proof needs no fallback'))
    result=d.mapped_runtime(cann,binding,path)[0]['mount_proof']
    assert result['status']=='MAP_FILES_FD_CONTENT_MATCH'
    assert result['map_files_content']['sha256']==c.identity(real)['sha256']


@pytest.mark.parametrize('bad',['inode','unknown_fs','mount_device','probe_device','probe_inode','probe_content','no_probe','readlink','deleted','fd_content'])
def test_unproved_or_corrupt_translation_blocks(setup,tmp_path,monkeypatch,bad):
    cann,binding,path,real,witness,link=overlay_case(tmp_path,setup,monkeypatch,bad!='fd_content')
    if bad=='inode':
        p=path.read_text().split(maxsplit=5);p[4]=str(int(p[4])+1);path.write_text(' '.join(p))
    elif bad=='unknown_fs':(tmp_path/'mountinfo').write_text((tmp_path/'mountinfo').read_text().replace(' - overlay ',' - ext4 '))
    elif bad=='mount_device':(tmp_path/'mountinfo').write_text('1 0 99:99 / / rw - overlay overlay rw\n')
    elif bad=='probe_device':witness['loaded'][0]=99
    elif bad=='probe_inode':witness['loaded'][2]+=1
    elif bad=='probe_content':witness['sha256']='0'*64
    elif bad=='no_probe':
        def fail(*a):raise ValueError('probe evidence unavailable')
        monkeypatch.setattr(d,'probe_readonly_mapping',fail)
    elif bad=='readlink':link.unlink();link.symlink_to(put(tmp_path/'elsewhere.so',b'other'))
    elif bad=='deleted':link.unlink();link.symlink_to(str(real)+' (deleted)')
    elif bad=='fd_content':monkeypatch.setattr(d,'fd_digest',lambda fd:dict(bytes=real.stat().st_size,sha256='0'*64))
    if bad in ('unknown_fs','mount_device'):
        # Bind this invalid starting environment to exercise the per-file proof,
        # separately from the mount-drift tests below.
        binding['mount_environment']=d.mount_snapshot(tmp_path)[0]
    with pytest.raises(ValueError) as error:d.mapped_runtime(cann,binding,path)
    assert str(real) in str(error.value)


def test_smoke_benchmark_mount_contract_drift(setup,tmp_path,monkeypatch):
    cann,binding,path,_,_,_=overlay_case(tmp_path,setup,monkeypatch)
    before=d.mapped_runtime(cann,binding,path)
    (tmp_path/'mountinfo').write_text((tmp_path/'mountinfo').read_text().replace('rw,lowerdir','ro,lowerdir'))
    with pytest.raises(ValueError,match='frozen mount environment changed'):d.mapped_runtime(cann,binding,path)
    fresh=copy.deepcopy(binding);fresh['mount_environment']=d.mount_snapshot(tmp_path)[0]
    after=d.mapped_runtime(cann,fresh,path)
    with pytest.raises(ValueError,match='differs from Smoke'):d.compare_loaded(after,before)
    ns=tmp_path/'ns/mnt';ns.unlink();ns.symlink_to('mnt:[43]')
    fresh['mount_environment']=d.mount_snapshot(tmp_path)[0]
    after2=d.mapped_runtime(cann,fresh,path)
    with pytest.raises(ValueError,match='differs from Smoke'):d.compare_loaded(after2,after)


def test_maps_changes_during_probe_blocks(setup,tmp_path,monkeypatch):
    cann,binding,path,_,witness,_=overlay_case(tmp_path,setup,monkeypatch)
    def changed(*args):path.write_text('');return witness
    monkeypatch.setattr(d,'probe_readonly_mapping',changed)
    with pytest.raises(ValueError,match='VMA changed'):d.mapped_runtime(cann,binding,path)


def test_mount_changes_during_scan_blocks(setup,tmp_path,monkeypatch):
    cann,binding,path,_,witness,_=overlay_case(tmp_path,setup,monkeypatch)
    def changed(*args):
        p=tmp_path/'mountinfo';p.write_text(p.read_text().replace('rw,lowerdir','ro,lowerdir'))
        return witness
    monkeypatch.setattr(d,'probe_readonly_mapping',changed)
    with pytest.raises(ValueError,match='mount namespace/table changed'):d.mapped_runtime(cann,binding,path)


def test_deleted_original_mapping_blocks(setup,tmp_path):
    cann,vendors,_,binding=setup
    path=maps_file(tmp_path,[vendors[0]])
    path.write_text(path.read_text().rstrip()+' (deleted)\n')
    with pytest.raises(ValueError,match='deleted DSO mapping'):d.mapped_runtime(cann,binding,path)


def test_vendor_and_private_filters_precede_mount_exception(setup,tmp_path,monkeypatch):
    cann,binding,path,_,_,_=overlay_case(tmp_path,setup,monkeypatch)
    private=Path(binding['private_bnz']['libraries'][0]['path'])
    for p in (private,put(cann/'opp/vendors/unknown/libx.so',b'unknown')):
        parts=path.read_text().split(maxsplit=5);parts[4]=str(p.stat().st_ino);parts[5]=str(p)+'\n';path.write_text(' '.join(parts))
        with pytest.raises(ValueError,match='private B-NZ|unknown vendor'):d.mapped_runtime(cann,binding,path)


def test_mount_parser_longest_prefix_and_escapes(tmp_path):
    (tmp_path/'ns').mkdir();(tmp_path/'ns/mnt').symlink_to('mnt:[42]')
    (tmp_path/'mountinfo').write_text('1 0 0:69 / / rw - overlay overlay rw\n2 1 8:17 /root /a\\040b ro - ext4 /dev/x ro\n')
    env,rows=d.mount_snapshot(tmp_path)
    assert d.covering_mount(Path('/a b/libx.so'),rows)['device']==[8,17]
    assert d.covering_mount(Path('/abc/libx.so'),rows)['fstype']=='overlay'
    assert len(env['mountinfo_sha256'])==64
    with pytest.raises(ValueError,match='ambiguous'):d.covering_mount(Path('/a b/libx.so'),rows+copy.deepcopy(rows[1:]))


@pytest.mark.skipif(sys.platform!='linux',reason='real read-only Linux mmap witness, no NPU')
def test_live_readonly_probe_no_exec_and_cleanup(tmp_path):
    p=put(tmp_path/'probe.so',b'CPU read-only data\n'*1024).resolve()
    with p.open('rb') as f:
        result=d.probe_readonly_mapping(f.fileno(),p,Path('/proc/self'))
        assert result['sha256']==c.identity(p)['sha256']
        assert result['bytes']==p.stat().st_size
        assert result['flags']=='MAP_PRIVATE_PROT_READ_NO_EXEC_NO_WRITE'
    # No residual full-file read-only probe mappings after return.
    assert not any(r['path']==str(p) for r in d.map_rows(Path('/proc/self/maps').read_text()).values())
