"""Synthetic ELF and filesystem fixtures only; not a Driver/NPU runtime test."""
import struct,sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import raw_probe_runtime_identity as rt


def elf(soname='libascend_hal.so'):
 strings=(b'\0'+soname.encode()+b'\0').ljust(128,b'\0');data=bytearray(512)
 data[:64]=b'\x7fELF'+bytes([2,1,1])+bytes(9)+struct.pack('<HHIQQQIHHHHHH',3,183,1,0,64,448,0,64,56,2,64,1,0)
 struct.pack_into('<IIQQQQQQ',data,64,1,5,0,0x10000,0x10000,512,512,4096)
 struct.pack_into('<IIQQQQQQ',data,120,2,6,176,0x10000+176,0x10000+176,64,64,8)
 for i,(tag,value) in enumerate([(5,0x10000+288),(10,128),(14,1),(0,0)]):struct.pack_into('<qQ',data,176+i*16,tag,value)
 data[288:416]=strings;return bytes(data)


@pytest.fixture
def tree(tmp_path):
 root=tmp_path.resolve();cann=root/'Ascend/cann-9.0.0';driver=root/'Ascend/driver';system=root/'usr/lib'
 for path in (cann/'lib64',driver/'lib64/driver',system):path.mkdir(parents=True)
 sdk=cann/'lib64/libnnopbase.so';sdk.write_bytes(b'bound SDK fixture')
 real=driver/'lib64/driver/libascend_hal.so.1';real.write_bytes(elf())
 hal=real.with_name('libascend_hal.so');hal.symlink_to(real.name)
 libc=system/'libc.so.6';libc.write_bytes(b'OS fixture')
 log=root/'host-ldd.log';log.write_text(f'linux-vdso.so.1 (0x1)\nlibnnopbase.so => {sdk} (0x2)\nlibascend_hal.so => {hal} (0x3)\nlibc.so.6 => {libc} (0x4)\n')
 return dict(root=root,cann=cann,driver=driver,system=system,sdk=sdk,real=real,hal=hal,libc=libc,log=log)


def bind(t,history=()):return rt.collect(t['log'],t['cann'],history,t['driver'],(t['system'],))


def test_classification_and_new_binding(tree):
 r=bind(tree)
 assert len(r['sdk_libraries'])==len(r['driver_libraries'])==len(r['system_libraries'])==1
 hal=r['driver_libraries'][0]
 assert hal['file']==rt.identity(tree['real']) and hal['soname']=='libascend_hal.so'
 assert hal['real_path']==str(tree['real']) and hal['symlink_chain']==[dict(path=str(tree['hal']),target=tree['real'].name,absolute_target=str(tree['real']))]
 assert r['driver_binding']['status']=='BOUND_INDEPENDENT_DRIVER_IDENTITY'
 assert r['driver_binding']['version']['status']=='UNRESOLVED'
 assert r['driver_binding']['historical_comparison']['binary_status']=='UNRESOLVED_NO_HISTORICAL_DRIVER_BINARY_IDENTITY'
 rt.verify_runtime(r)


@pytest.mark.parametrize('key',['real','sdk','libc'])
def test_sha_drift_rejected(tree,key):
 r=bind(tree);tree[key].write_bytes(tree[key].read_bytes()+b'change')
 with pytest.raises(ValueError):rt.verify_runtime(r)


def test_driver_symlink_escape(tree):
 other=tree['root']/'other.so';other.write_bytes(elf())
 tree['hal'].unlink();tree['hal'].symlink_to(other)
 with pytest.raises(ValueError,match='escapes'):bind(tree)


def test_directory_symlink_and_retarget(tree):
 alias=tree['driver']/'alias';alias.symlink_to(tree['real'].parent,target_is_directory=True)
 tree['log'].write_text(f'libascend_hal.so => {alias}/libascend_hal.so (0x3)\n')
 r=bind(tree);assert len(r['driver_libraries'][0]['symlink_chain'])==2
 alias.unlink();alias.symlink_to(tree['root'],target_is_directory=True)
 with pytest.raises((ValueError,FileNotFoundError)):rt.verify_runtime(r)


def test_identical_bytes_changed_alias_rejected(tree):
 r=bind(tree);copy=tree['real'].with_name('other-version');copy.write_bytes(elf())
 tree['hal'].unlink();tree['hal'].symlink_to(copy)
 with pytest.raises(ValueError):rt.verify_runtime(r)


@pytest.mark.parametrize('where',['Ascend/unconfirmed','Ascend/driver/other'])
def test_unknown_ascend_or_driver_library(tree,where):
 p=tree['root']/where/'libunknown.so';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(elf('libunknown.so'))
 tree['log'].write_text(f'libunknown.so => {p} (0x1)\n')
 with pytest.raises(ValueError):bind(tree)


def test_unknown_driver_soname(tree):
 tree['real'].write_bytes(elf('libunexpected.so'))
 with pytest.raises(ValueError,match='SONAME'):bind(tree)


def test_toolkit_cannot_import_driver_or_other_sdk(tree):
 tree['sdk'].unlink();tree['sdk'].symlink_to(tree['real'])
 with pytest.raises(ValueError,match='escapes'):bind(tree)


def test_hal_not_sdk_identity(tree):
 p=tree['cann']/'lib64/libascend_hal.so';p.write_bytes(elf())
 tree['log'].write_text(f'libascend_hal.so => {p} (0x2)\n')
 with pytest.raises(ValueError,match='HAL must bind'):bind(tree)


def history(t,content):
 p=t['root']/'history.json';rt.v.write(p,content)
 return [dict(label='preflight',report=rt.identity(p),content=content)]


def test_machine_wrapped_version_and_binary_match(tree):
 version=tree['driver']/'version.info';version.write_text('Version=24.1.0\ninnerversion=build-42\n')
 h=history(tree,dict(machine=dict(version_files=[dict(file=rt.identity(version),text=version.read_text())],driver_libraries=[rt.identity(tree['real'])])))
 r=bind(tree,h);cmp=r['driver_binding']['historical_comparison']
 assert cmp['binary_status']==cmp['version_status']=='MATCHED'
 assert r['driver_binding']['version']['files'][0]['version_fields']==dict(Version='24.1.0',innerversion='build-42')
 version.write_text('Version=25.0.0\n')
 with pytest.raises(ValueError):rt.verify_runtime(r)
 with pytest.raises(ValueError):bind(tree,h)


def test_version_record_is_not_binary_history(tree):
 version=tree['driver']/'version.info';version.write_text('Version=24.1.0\n')
 r=bind(tree,history(tree,dict(machine=dict(version_files=[dict(file=rt.identity(version),text='Version=24.1.0')]))))
 assert r['driver_binding']['historical_comparison']['version_status']=='MATCHED'
 assert r['driver_binding']['historical_comparison']['binary_status'].startswith('UNRESOLVED')


def test_unparseable_version_unresolved(tree):
 (tree['driver']/'version.info').write_text('no version field supplied\n')
 assert bind(tree)['driver_binding']['version']['status']=='UNRESOLVED'


def test_version_escape_and_missing_elf_rejected(tree):
 p=tree['root']/'version.info';p.write_text('Version=1\n');(tree['driver']/'version.info').symlink_to(p)
 with pytest.raises(ValueError):bind(tree)
 (tree['driver']/'version.info').unlink();tree['real'].write_bytes(b'not ELF')
 with pytest.raises(ValueError):bind(tree)


def test_unresolved_and_loader_lines(tree):
 tree['log'].write_text('libascend_hal.so => not found\n')
 with pytest.raises(ValueError):bind(tree)
 loader=tree['system']/'ld-linux-aarch64.so.1';loader.write_bytes(b'loader fixture')
 tree['log'].write_text(f'{loader} (0x1)\n')
 r=bind(tree);assert r['system_libraries'][0]['file']==rt.identity(loader)
 assert r['driver_binding']['status']=='NO_DRIVER_LIBRARY_IN_HOST_LDD'


def test_historical_hal_hash_mismatch_rejected(tree):
 h=history(tree,dict(machine=dict(driver_libraries=[rt.identity(tree['real'])])))
 tree['real'].write_bytes(elf()+b'changed')
 with pytest.raises(ValueError):bind(tree,h)


def test_host_library_hash_guard_still_active(tree):
 import run_raw_probe
 host=tree['root']/'private-host.so';host.write_bytes(elf('libcust_opmaster_rt2.0.so'))
 item=dict(core_sdk_binding=dict(host_library=rt.identity(host)))
 host.write_bytes(host.read_bytes()+b'changed')
 # Artifact verifier must fail on the recorded Host identity before accepting any gate.
 with pytest.raises(ValueError):run_raw_probe.verify_artifacts(item,{})


def test_history_sources_are_explicit_bound_reports(tree):
 p=tree['root']/'preflight.json';b=tree['root']/'bounded.json'
 rt.v.write(p,dict(machine=dict(version_files=[])));rt.v.write(b,dict(machine=dict(driver='UNAVAILABLE')))
 rows=rt.history_records(dict(preflight=rt.identity(p),frozen=dict(bounded=rt.identity(b))))
 assert [r['label'] for r in rows]==['preflight','frozen_machine']
 assert [r['report'] for r in rows]==[rt.identity(p),rt.identity(b)]
