"""Read-only ELF/packaging fixtures, not real CANN or runtime support evidence."""
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
from unittest.mock import patch
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import artifact_recheck as recheck
from compile_raw import check_artifacts,HOST_LINK_TARGET
from w8_common import identity,write


def host(rpath='/build/lib',needed='libc.so.6'):
 strings=b'\0'+needed.encode()+b'\0libcust_opmaster_rt2.0.so\0'
 needed_off=1;soname_off=len(needed)+2;rpath_off=len(strings)
 strings+=rpath.encode()+b'\0';strings=strings.ljust(128,b'\0')
 assert len(strings)==128
 data=bytearray(512)
 data[:64]=b'\x7fELF'+bytes([2,1,1])+bytes(9)+struct.pack('<HHIQQQIHHHHHH',3,183,1,0,64,448,0,64,56,2,64,1,0)
 struct.pack_into('<IIQQQQQQ',data,64,1,5,0,0x10000,0x10000,512,512,4096)
 struct.pack_into('<IIQQQQQQ',data,120,2,6,176,0x10000+176,0x10000+176,96,96,8)
 for i,(tag,value) in enumerate([(5,0x10000+288),(10,128),(1,needed_off),(14,soname_off),(29,rpath_off),(0,0)]):
  struct.pack_into('<qQ',data,176+i*16,tag,value)
 data[288:416]=strings
 return bytes(data)


def kernel():
 return b'\x7fELF'+bytes([2,1,1])+bytes(9)+struct.pack('<HHIQQQIHHHHHH',1,0x1234,1,0,0,64,0,64,0,0,64,1,0)+bytes(64)


def test_elf_rpath_only_is_diagnostic_not_acceptance(tmp_path):
 a=tmp_path/'a';b=tmp_path/'b';a.write_bytes(host());b.write_bytes(host(''))
 r=recheck.compare_host(a,b)
 assert r['program_headers_equal'] and r['needed_soname_equal'] and r['dynamic_entries_equal']
 assert not r['pt_load_equal'] and r['rpath_string_only_candidate']
 assert r['classification']=='UNRESOLVED_PACKAGING_TRANSFORM'
 assert r['build']['dynamic_strings'][-1]['value']=='/build/lib'
 assert r['package']['dynamic_strings'][-1]['value']==''


def test_non_rpath_load_changes_and_dependency_changes_not_hidden(tmp_path):
 a=tmp_path/'a';b=tmp_path/'b';a.write_bytes(host());data=bytearray(host());data[430]=9;b.write_bytes(data)
 r=recheck.compare_host(a,b);assert not r['rpath_string_only_candidate'] and not r['pt_load_equal']
 b.write_bytes(host(needed='libx.so.6'));r=recheck.compare_host(a,b)
 assert not r['needed_soname_equal'] and not r['rpath_string_only_candidate']


def test_bad_dynamic_bounds_fail_closed(tmp_path):
 p=tmp_path/'bad';data=bytearray(host());struct.pack_into('<Q',data,184,0x999999);p.write_bytes(data)
 with pytest.raises(ValueError,match='not mapped'):recheck.host_elf(p)


def make_run(tmp_path):
 old=tmp_path/'old';project=old/'CustomOp'
 k=project/'build_out/op_kernel/binary/ascend310p/d_flash_w8_a8_accumulator/DFlashW8A8Accumulator_fixture.o'
 a=project/'build_out/op_host/libcust_opmaster_rt2.0.so';k.parent.mkdir(parents=True);a.parent.mkdir(parents=True)
 k.write_bytes(kernel());a.write_bytes(host());(a.parent/'liboptiling.so').symlink_to(HOST_LINK_TARGET)
 op=project/'build_out/_CPack_Packages/Linux/External/custom_opp_fixture.run/packages/vendors/customize/op_impl/ai_core/tbe/op_tiling'
 b=op/HOST_LINK_TARGET;b.parent.mkdir(parents=True);b.write_bytes(host(''));(op/'liboptiling.so').symlink_to(HOST_LINK_TARGET)
 install=project/'build_out/op_host/cmake_install.cmake'
 install.write_text('file(RPATH_CHANGE\n  FILE "libcust_opmaster_rt2.0.so"\n  OLD_RPATH "/build/lib"\n  NEW_RPATH "")\nif(CMAKE_INSTALL_DO_STRIP)\n execute_process(COMMAND "/usr/bin/strip" "library")\nendif()\n')
 r=dict(scope='RAW_ND_K128_COMPILE_ONLY',shape=[16,128,64],input_count=2,installation='NOT_RUN',npu='NOT_RUN',
  commands={s:dict(status='COMPLETED',returncode=0) for s in ('msopgen','build')})
 with pytest.raises(ValueError):check_artifacts(project,r)
 write(old/'compile-report.json',r)
 args=SimpleNamespace(report=old/'compile-report.json',report_sha256=identity(old/'compile-report.json')['sha256'],
    build_host_sha256=identity(a)['sha256'],package_host_sha256=identity(b)['sha256'],output_dir=tmp_path/'out')
 return args,[k,a,b,install,args.report]


def test_readonly_recheck_preserves_kernel_and_requires_transform_proof(tmp_path):
 args,files=make_run(tmp_path);before=[identity(p) for p in files]
 with patch('subprocess.run',side_effect=AssertionError('must not execute any tool or build')):
  assert recheck.main(args)==2
 assert [identity(p) for p in files]==before
 r=json.loads((args.output_dir/'artifact-recheck.json').read_text())
 assert r['status']=='ARTIFACT_RECHECK_UNRESOLVED' and r['kernel_status']=='VERIFIED_UNCHANGED'
 assert r['rechecked_artifacts']['artifact_checks']['host_runtime']['status']=='FAILED'
 assert r['host_comparison']['rpath_string_only_candidate']
 assert r['source_review']=='UNRESOLVED' and r['npu']==r['installation']=='NOT_RUN' and not r['recompiled']
 evidence=r['packaging_evidence']['files'][0]
 assert evidence['source']==before[3] and any('RPATH_CHANGE' in x['text'] for x in evidence['excerpts'])


@pytest.mark.parametrize('field',['report_sha256','build_host_sha256','package_host_sha256'])
def test_recheck_rejects_tampered_identities(tmp_path,field):
 args,_=make_run(tmp_path);setattr(args,field,'0'*64)
 assert recheck.main(args)==2
 r=json.loads((args.output_dir/'artifact-recheck.json').read_text())
 assert r['status']=='BLOCKED_ARTIFACT_RECHECK'


def test_failed_original_build_does_not_become_recheck_success(tmp_path):
 args,_=make_run(tmp_path);r=json.loads(args.report.read_text());r['commands']['build']['returncode']=1;write(args.report,r)
 args.report_sha256=identity(args.report)['sha256']
 assert recheck.main(args)==2
 r=json.loads((args.output_dir/'artifact-recheck.json').read_text());assert 'not completed successfully' in r['error']
