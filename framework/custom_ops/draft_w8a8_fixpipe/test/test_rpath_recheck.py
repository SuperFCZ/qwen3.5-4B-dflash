"""CPU provenance/replay fixtures only; no CANN, library load or NPU run."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import rpath_recheck as replay
from w8_common import identity,write
from test_artifact_recheck import host,kernel
from compile_raw import check_artifacts,HOST_LINK_TARGET
from artifact_recheck import host_elf


def fixture(tmp_path):
 old=tmp_path/'old';project=old/'CustomOp'
 k=project/'build_out/op_kernel/binary/ascend310p/d_flash_w8_a8_accumulator/DFlashW8A8Accumulator_fixture.o'
 a=project/'build_out/op_host/libcust_opmaster_rt2.0.so';k.parent.mkdir(parents=True);a.parent.mkdir(parents=True)
 k.write_bytes(kernel());a.write_bytes(host('/sdk/lib64:'));(a.parent/'liboptiling.so').symlink_to(HOST_LINK_TARGET)
 root=project/'build_out/_CPack_Packages/Linux/External/custom_opp_fixture.run/packages/vendors/customize/op_impl/ai_core/tbe/op_tiling'
 b=root/HOST_LINK_TARGET;b.parent.mkdir(parents=True);b.write_bytes(host(''));(root/'liboptiling.so').symlink_to(HOST_LINK_TARGET)
 dest='${CMAKE_INSTALL_PREFIX}/packages/vendors/customize/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64'
 script=project/'build_out/op_host/cmake_install.cmake'
 script.write_text(f'''file(INSTALL DESTINATION "{dest}" TYPE SHARED_LIBRARY FILES "{a}")
if(EXISTS "$ENV{{DESTDIR}}{dest}/{a.name}")
 file(RPATH_CHANGE FILE "$ENV{{DESTDIR}}{dest}/{a.name}" OLD_RPATH "/sdk/lib64:" NEW_RPATH "")
endif()
''')
 cmake=tmp_path/'cmake';cmake.write_text('CPU fixture tool, never executed')
 pf=tmp_path/'preflight.json';write(pf,dict(cann_root='/sdk',tools={'cmake':dict(file=identity(cmake))}))
 r=dict(scope='RAW_ND_K128_COMPILE_ONLY',shape=[16,128,64],input_count=2,installation='NOT_RUN',npu='NOT_RUN',
        preflight=identity(pf),commands={s:dict(status='COMPLETED',returncode=0) for s in ('msopgen','build')})
 with pytest.raises(ValueError):check_artifacts(project,r)
 rp=old/'compile-report.json';write(rp,r)
 args=SimpleNamespace(report=rp,report_sha256=identity(rp)['sha256'],build_host_sha256=identity(a)['sha256'],
   package_host_sha256=identity(b)['sha256'],install_script_sha256=identity(script)['sha256'],output_dir=tmp_path/'out')
 return args,project,a,b,script,cmake


def fake_cmake(argv,**kwargs):
 # Model only the allowed path-string edit; never copy bytes from the package oracle.
 assert len(argv)==4 and argv[2]=='-P'
 path=Path(argv[1].split('=',1)[1]);info,data=host_elf(path);data=bytearray(data)
 for s in info['dynamic_strings']:
  if s['name'] in ('RPATH','RUNPATH'):data[s['file_offset']:s['file_offset']+s['bytes']]=bytes(s['bytes'])
 path.write_bytes(data)
 return SimpleNamespace(returncode=0)


def test_exact_replay_preserves_originals_and_accepts_only_compile(tmp_path):
 args,project,a,b,script,tool=fixture(tmp_path);before=[identity(p) for p in (a,b,script,args.report)]
 with patch.object(replay.subprocess,'run',side_effect=fake_cmake) as run:
  assert replay.main(args)==0
 assert run.call_count==1 and [identity(p) for p in (a,b,script,args.report)]==before
 r=json.loads((args.output_dir/'compile-report.json').read_text())
 assert r['status']=='COMPILE_COMPLETED_REVIEW_UNRESOLVED' and r['source_review']=='UNRESOLVED'
 assert r['npu']==r['installation']==r['host_tiling_runtime']=='NOT_RUN' and not r['recompiled']
 assert r['rpath_replay']['exact_sha_match'] and r['rpath_replay']['replay']['sha256']==identity(b)['sha256']
 assert r['rpath_replay']['rule']['script']==before[2]
 assert r['kernel_status']=='VERIFIED_UNCHANGED'
 assert (args.output_dir/'replay/replay.cmake').read_text().count('file(')==1


@pytest.mark.parametrize('fault',['conflict','missing','wrong_source','wrong_target','wrong_old','wrong_new','script_hash','alias_escape','library_changed'])
def test_provenance_failure_blocks_replay(tmp_path,fault):
 args,project,a,b,script,tool=fixture(tmp_path)
 text=script.read_text()
 if fault=='conflict':text+=next(line for line in text.splitlines() if 'RPATH_CHANGE' in line)+'\n'
 elif fault=='missing':text='file(INSTALL DESTINATION "x" TYPE SHARED_LIBRARY FILES "y")\n'
 elif fault=='wrong_source':text=text.replace(str(a),'/wrong/library.so')
 elif fault=='wrong_target':text=text.replace('/packages/vendors/','/other/vendors/')
 elif fault=='wrong_old':text=text.replace('/sdk/lib64:','/wrong:')
 elif fault=='wrong_new':text=text.replace('NEW_RPATH ""','NEW_RPATH "/injected"')
 elif fault=='script_hash':args.install_script_sha256='0'*64
 elif fault=='alias_escape':
  alias=b.parents[3]/'liboptiling.so';alias.unlink();alias.symlink_to(a)
 elif fault=='library_changed':a.write_bytes(host('/other:'))
 script.write_text(text)
 if fault!='script_hash':args.install_script_sha256=identity(script)['sha256']
 with patch.object(replay.subprocess,'run',side_effect=AssertionError('no replay on invalid provenance')) as run:
  assert replay.main(args)==1
 assert run.call_count==0
 r=json.loads((args.output_dir/'compile-report.json').read_text());assert r['status']=='ARTIFACT_CHECK_FAILED_REVIEW_UNRESOLVED'


@pytest.mark.parametrize('fault',['nonzero','mismatched_result','extra_load_change','changed_original'])
def test_replay_failure_cannot_be_accepted(tmp_path,fault):
 args,project,a,b,script,tool=fixture(tmp_path)
 if fault=='extra_load_change':
  data=bytearray(b.read_bytes());data[430]=7;b.write_bytes(data);args.package_host_sha256=identity(b)['sha256']
 def run(argv,**kwargs):
  if fault=='nonzero':return SimpleNamespace(returncode=1)
  if fault=='mismatched_result':return SimpleNamespace(returncode=0)
  result=fake_cmake(argv,**kwargs)
  if fault=='changed_original':a.write_bytes(host('/changed:'))
  return result
 with patch.object(replay.subprocess,'run',side_effect=run):assert replay.main(args)==1
 r=json.loads((args.output_dir/'compile-report.json').read_text());assert r['status']=='ARTIFACT_CHECK_FAILED_REVIEW_UNRESOLVED'


def test_comments_are_not_rules_and_conflicting_live_rules_rejected(tmp_path):
 args,project,a,b,script,tool=fixture(tmp_path)
 script.write_text('# file(RPATH_CHANGE FILE "fake" OLD_RPATH "bad" NEW_RPATH "bad")\n#[[ file(RPATH_CHANGE fake) ]]\n'+script.read_text())
 rule=replay.extract_rule(project,a,b,identity(script)['sha256'],Path('/sdk'))
 assert rule['old_rpath']=='/sdk/lib64:'
 assert rule['destination']['resolved']==str(b)


def test_quoted_command_names_cannot_supply_provenance():
 assert replay.file_commands('message("file" "(" "RPATH_CHANGE" ")")')==[]


def test_cmake_tool_change_and_timeout_fail_closed(tmp_path):
 args,project,a,b,script,tool=fixture(tmp_path);tool.write_text('changed')
 with patch.object(replay.subprocess,'run') as run:
  assert replay.main(args)==1
 assert run.call_count==0


def test_timeout_keeps_failed_report_and_log(tmp_path):
 args,*_=fixture(tmp_path)
 with patch.object(replay.subprocess,'run',side_effect=replay.subprocess.TimeoutExpired('fixture',60)):
  assert replay.main(args)==1
 r=json.loads((args.output_dir/'compile-report.json').read_text())
 assert r['rpath_replay']['status']=='FAILED' and 'log' in r['rpath_replay']


@pytest.mark.parametrize('change',['dependency','segment_mapping'])
def test_exact_bytes_alone_do_not_authorize_other_elf_changes(tmp_path,change):
 args,project,a,b,script,tool=fixture(tmp_path)
 data=bytearray(host('',needed='libx.so.6') if change=='dependency' else host(''))
 if change=='segment_mapping':data[68]=7 # PT_LOAD flags
 b.write_bytes(data);args.package_host_sha256=identity(b)['sha256']
 def forged_tool(argv,**kwargs):
  Path(argv[1].split('=',1)[1]).write_bytes(data);return SimpleNamespace(returncode=0)
 with patch.object(replay.subprocess,'run',side_effect=forged_tool):assert replay.main(args)==1
 r=json.loads((args.output_dir/'compile-report.json').read_text())
 assert 'mapping or dependency' in r['error']
