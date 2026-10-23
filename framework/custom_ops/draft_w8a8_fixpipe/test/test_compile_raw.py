"""Compile-only orchestration fixtures. No CANN/device assertions."""
import json
from pathlib import Path
import subprocess
import sys
import struct
from unittest.mock import patch
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import compile_raw as probe
from w8_common import identity,write
from w8_run import build_inputs


def elf(host=True,machine=None):
    header=b'\x7fELF'+bytes([2,1,1])+bytes(9)
    header+=struct.pack('<HHIQQQIHHHHHH',3 if host else 1,machine if machine is not None else (183 if host else 0x1234),
                        1,0,0,64,0,64,0,0,64,1,0)
    return header+bytes(64)


def artifacts(root,host_name='libcust_opmaster_rt2.0.so'):
    k=root/'build_out/op_kernel/binary/ascend310p/d_flash_w8_a8_accumulator/DFlashW8A8Accumulator_fixture.o'
    h=root/'build_out/op_host'/host_name
    k.parent.mkdir(parents=True);h.parent.mkdir(parents=True)
    k.write_bytes(elf(False));h.write_bytes(elf(True));return k,h


def test_compile_only_generates_one_raw_op_and_never_installs(tmp_path):
 calls=[]
 def command(argv,log,env,report,key,timeout=1800):
  argv=list(map(str,argv));calls.append(argv);log.write_text('fixture only')
  project=tmp_path/'CustomOp'
  if argv[0]=='msopgen':
   (project/'op_host').mkdir(parents=True);(project/'op_kernel').mkdir()
   (project/'build.sh').write_text('# fixture, never executed\n')
  else:
   assert argv[0]=='bash' and argv[-1]==str(project)
   p=project/'build_out/op_kernel/binary/ascend310p/d_flash_w8_a8_accumulator/DFlashW8A8Accumulator_fixture.o'
   p.parent.mkdir(parents=True);p.write_bytes(elf(False))
   h=project/'build_out/op_host/libcust_opmaster_rt2.0.so';h.parent.mkdir();h.write_bytes(elf(True))
   (project/'build_out/custom_opp_fixture.run').write_bytes(b'not installed')
 r=dict(source_review='UNRESOLVED',installation='NOT_RUN',npu='NOT_RUN',host_tiling_runtime='NOT_RUN')
 with patch.object(probe,'run_build_command',side_effect=command):probe.compile_candidate(tmp_path,tmp_path/'sdk',r)
 assert len(calls)==2 and calls[0][-1]=='DFlashW8A8Accumulator'
 assert all('--install-path' not in ' '.join(c) and 'runner' not in ' '.join(c) for c in calls)
 assert len(json.loads((tmp_path/'operators.json').read_text()))==1
 assert r['status']=='COMPILE_COMPLETED_REVIEW_UNRESOLVED' and r['source_review']=='UNRESOLVED'
 assert r['installation']==r['npu']==r['host_tiling_runtime']=='NOT_RUN'
 assert not (tmp_path/'opp').exists() and not (tmp_path/'review-raw.json').exists()


def test_no_objects_cannot_be_compile_success(tmp_path):
 def command(argv,log,env,report,key,timeout=1800):
  if argv[0]=='msopgen':
   project=tmp_path/'CustomOp';(project/'op_host').mkdir(parents=True);(project/'op_kernel').mkdir()
   (project/'build.sh').write_text('# fixture')
 with patch.object(probe,'run_build_command',side_effect=command):
  with pytest.raises(ValueError,match='artifact verification failed: kernel, host'):probe.compile_candidate(tmp_path,tmp_path,{})


def test_preflight_identity_and_candidate_gate_without_fabricated_review(tmp_path):
 sdk=tmp_path/'sdk';sdk.mkdir();tool=sdk/'msopgen';tool.write_text('fixture')
 version=sdk/'version.info';version.write_text('9.0.0');snap=tmp_path/'version.snapshot';snap.write_bytes(version.read_bytes())
 _,files,_,_=build_inputs(True,'nd',{})
 p=dict(raw_only=True,status='UNRESOLVED',collection='FIXED_FILES_COLLECTED',cann_root=str(sdk),
        sources=[identity(f) for f in files.values()],files=[dict(source=identity(version),snapshot=identity(snap))],
        tools={'msopgen':dict(file=identity(tool),returncode=2,stderr='unrecognized arguments: --version')})
 path=tmp_path/'preflight.json';write(path,p);sha=identity(path)['sha256']
 with patch.object(probe.shutil,'which',return_value=str(tool)):
  assert probe.validate_preflight(path,sha,sdk)['status']=='UNRESOLVED'
  with pytest.raises(ValueError,match='SHA256 mismatch'):probe.validate_preflight(path,'0'*64,sdk)
  p['sources']=[];write(path,p)
  with pytest.raises(ValueError,match='not bound'):probe.validate_preflight(path,identity(path)['sha256'],sdk)
  p['sources']=[identity(f) for f in files.values()];write(path,p)
  version.write_text('changed')
  with pytest.raises(ValueError,match='identity changed'):probe.validate_preflight(path,identity(path)['sha256'],sdk)


def test_entry_has_no_install_smoke_or_review_options():
 r=subprocess.run([sys.executable,'-B',str(ROOT/'compile_raw.py'),'--help'],capture_output=True,text=True)
 assert r.returncode==0 and '--preflight-sha256' in r.stdout and '--review' not in r.stdout
 subprocess.run(['bash','-n',str(ROOT/'compile_raw.sh')],check=True)


@pytest.mark.parametrize('name',['libcust_opmaster_rt2.0.so','liboptiling.so','libcust_optiling.so'])
def test_real_host_names_and_canonical_location(tmp_path,name):
 k,h=artifacts(tmp_path,name);r={}
 packaged=tmp_path/'build_out/_CPack_Packages/vendor'/name;packaged.parent.mkdir(parents=True);packaged.write_bytes(h.read_bytes())
 probe.check_artifacts(tmp_path,r)
 assert r['artifact_checks']['kernel']['verified_count']==1
 host=r['artifact_checks']['host'];assert host['discovered_count']==2 and host['canonical_count']==host['verified_count']==1
 assert r['host_libraries']==[identity(h)] and r['kernel_objects']==[identity(k)]
 assert next(x for x in host['candidates'] if x['status']=='VERIFIED')['elf']['machine']==183


@pytest.mark.parametrize('missing',['kernel','host'])
def test_missing_groups_report_independently(tmp_path,missing):
 k,h=artifacts(tmp_path);(k if missing=='kernel' else h).unlink();r={}
 with pytest.raises(ValueError,match='failed: '+missing):probe.check_artifacts(tmp_path,r)
 assert r['artifact_checks'][missing]['discovered_count']==0
 assert r['artifact_checks']['host' if missing=='kernel' else 'kernel']['status']=='VERIFIED'


@pytest.mark.parametrize('kind',['empty','not_elf','wrong_machine','wrong_type','truncated','escaped','wrong_directory'])
def test_invalid_host_not_accepted(tmp_path,kind):
 project=tmp_path/'CustomOp';k,h=artifacts(project)
 if kind=='empty':h.write_bytes(b'')
 elif kind=='not_elf':h.write_bytes(b'not elf'*30)
 elif kind=='wrong_machine':h.write_bytes(elf(True,62))
 elif kind=='wrong_type':h.write_bytes(elf(False,183))
 elif kind=='truncated':h.write_bytes(elf(True)[:80])
 elif kind=='escaped':
  outside=tmp_path/'external.so';outside.write_bytes(elf());h.unlink();h.symlink_to(outside)
 else:
  data=h.read_bytes();h.unlink();(project/'liboptiling.so').write_bytes(data)
 r={}
 with pytest.raises(ValueError,match='failed: host'):probe.check_artifacts(project,r)
 assert r['artifact_checks']['host']['status']=='FAILED' and r['artifact_checks']['kernel']['status']=='VERIFIED'


def test_command_returncode_and_postbuild_failure_are_distinct(tmp_path):
 r={}
 with pytest.raises(RuntimeError,match='exited 7'):
  probe.run_build_command([sys.executable,'-c','print("compiler fixture error");raise SystemExit(7)'],tmp_path/'build.log',{},r,'build')
 assert r['commands']['build']['returncode']==7
 assert 'compiler fixture error' in (tmp_path/'build.log').read_text()
 r['cann_compile']='RUNNING';probe.record_failure(r,RuntimeError('build failed'))
 assert r['status']=='COMPILE_FAILED_REVIEW_UNRESOLVED'
 r={'cann_compile':'BUILD_RETURNED_ZERO','commands':{'build':{'returncode':0}}}
 probe.record_failure(r,ValueError('missing Host library'))
 assert r['status']=='ARTIFACT_CHECK_FAILED_REVIEW_UNRESOLVED' and r['cann_compile']=='BUILD_RETURNED_ZERO'


def test_successful_command_retains_zero_and_log_identity(tmp_path):
 r={};probe.run_build_command([sys.executable,'-c','print("fixture complete")'],tmp_path/'build.log',{},r,'build')
 assert r['commands']['build']['returncode']==0 and r['commands']['build']['log_identity']==identity(tmp_path/'build.log')
