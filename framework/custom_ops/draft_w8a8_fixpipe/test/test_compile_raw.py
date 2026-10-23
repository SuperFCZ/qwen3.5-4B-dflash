"""Compile-only orchestration fixtures. No CANN/device assertions."""
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import compile_raw as probe
from w8_common import identity,write
from w8_run import build_inputs


def test_compile_only_generates_one_raw_op_and_never_installs(tmp_path):
 calls=[]
 def command(argv,log,env,timeout=1800):
  argv=list(map(str,argv));calls.append(argv);log.write_text('fixture only')
  project=tmp_path/'CustomOp'
  if argv[0]=='msopgen':
   (project/'op_host').mkdir(parents=True);(project/'op_kernel').mkdir()
   (project/'build.sh').write_text('# fixture, never executed\n')
  else:
   assert argv[0]=='bash' and argv[-1]==str(project)
   p=project/'op_kernel/DFlashW8A8Accumulator.o';p.write_bytes(b'fixture object')
   (project/'libcust_optiling.so').write_bytes(b'fixture host')
   (project/'build_out').mkdir();(project/'build_out/custom_opp_fixture.run').write_bytes(b'not installed')
 r=dict(source_review='UNRESOLVED',installation='NOT_RUN',npu='NOT_RUN',host_tiling_runtime='NOT_RUN')
 with patch.object(probe,'command',side_effect=command):probe.compile_candidate(tmp_path,tmp_path/'sdk',r)
 assert len(calls)==2 and calls[0][-1]=='DFlashW8A8Accumulator'
 assert all('--install-path' not in ' '.join(c) and 'runner' not in ' '.join(c) for c in calls)
 assert len(json.loads((tmp_path/'operators.json').read_text()))==1
 assert r['status']=='COMPILE_COMPLETED_REVIEW_UNRESOLVED' and r['source_review']=='UNRESOLVED'
 assert r['installation']==r['npu']==r['host_tiling_runtime']=='NOT_RUN'
 assert not (tmp_path/'opp').exists() and not (tmp_path/'review-raw.json').exists()


def test_no_objects_cannot_be_compile_success(tmp_path):
 def command(argv,log,env,timeout=1800):
  if argv[0]=='msopgen':
   project=tmp_path/'CustomOp';(project/'op_host').mkdir(parents=True);(project/'op_kernel').mkdir()
   (project/'build.sh').write_text('# fixture')
 with patch.object(probe,'command',side_effect=command):
  with pytest.raises(ValueError,match='without raw kernel object'):probe.compile_candidate(tmp_path,tmp_path,{})


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
