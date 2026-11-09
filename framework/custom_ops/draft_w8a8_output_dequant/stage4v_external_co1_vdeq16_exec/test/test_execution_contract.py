"""Compile the actual C++ JSON producers and feed their output to real Python gates.
No CANN, device context or Kernel is executed by this test harness.
"""
import copy,json,shutil,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from echo_execution_contract import validate_execution
import run_echo as run


@pytest.fixture(scope='module')
def producer(tmp_path_factory):
 root=tmp_path_factory.mktemp('execution-producer');src=root/'main.cpp';exe=root/'producer'
 src.write_text('''#include "execution_report.h"
#include <iostream>
#include <fstream>
#include <string>
int main(int argc,char**argv){try{
 if(argc!=4)return 2;std::string mode=argv[1];auto bytes=std::stoull(argv[2]);auto device=std::stoi(argv[3]);
 if(mode=="host")echo_execution_report::Host(std::cout,bytes,device);
 else if(mode=="echo")echo_execution_report::Echo(std::cout,bytes,device);
 else {std::ofstream closed;if(mode=="bad-host")echo_execution_report::Host(closed,bytes,device);else echo_execution_report::Echo(closed,bytes,device);}
 return 0;}catch(const std::exception&){return 7;}}
''')
 subprocess.run([shutil.which('c++'),'-std=c++17','-I'+str(HERE/'runner'),str(src),'-o',str(exe)],check=True,capture_output=True)
 def emit(mode='host',device=0,workspace=4195328):
  result=subprocess.run([str(exe),mode,str(workspace),str(device)],check=True,capture_output=True,text=True)
  return json.loads(result.stdout)
 return exe,emit


@pytest.mark.parametrize('host',[True,False])
@pytest.mark.parametrize('device,workspace',[(0,4195328),(2,8389632)])
def test_real_cpp_producer_matches_all_python_fields(producer,host,device,workspace):
 _,emit=producer;record=emit('host' if host else 'echo',device,workspace)
 assert validate_execution(record,device,workspace,host)==record
 assert record['status']==('HOST_TILING_ONLY' if host else 'ECHO_EXECUTION_MATCH')
 assert set(record)==({'status','npu','workspace_bytes','device'} if host else {'status','npu','device','workspace_bytes','repetitions','guards','readonly','poison','repeat_equal','l0c_guard','vdeq16','cpu_fallback'})


@pytest.mark.parametrize('host',[True,False])
def test_every_field_missing_or_wrong_is_rejected(producer,host):
 _,emit=producer;r=emit('host' if host else 'echo')
 for key,value in r.items():
  missing=copy.deepcopy(r);missing.pop(key)
  with pytest.raises(ValueError):validate_execution(missing,0,4195328,host)
  bad=copy.deepcopy(r);bad[key]=not value if type(value)is bool else value+1 if type(value)is int else 'WRONG'
  with pytest.raises(ValueError):validate_execution(bad,0,4195328,host)
  bad=copy.deepcopy(r);bad[key]=int(value) if type(value)is bool else float(value) if type(value)is int else None
  with pytest.raises(ValueError):validate_execution(bad,0,4195328,host)
 with pytest.raises(ValueError):validate_execution(dict(r,extra='unreviewed'),0,4195328,host)


def test_old_status_and_in_progress_are_not_success(producer):
 _,emit=producer
 for status in ('HOST_ONLY','RUNNING','FORMAL_HOST_FEASIBLE','BLOCKED'):
  r=emit();r['status']=status
  with pytest.raises(ValueError):validate_execution(r,0,4195328,True)
 for npu in ('ATTEMPTED','COMPLETED'):
  r=emit();r['npu']=npu
  with pytest.raises(ValueError):validate_execution(r,0,4195328,True)


@pytest.mark.parametrize('mode',['bad-host','bad-echo'])
def test_cpp_write_failure_is_not_success(producer,mode):
 exe,_=producer
 assert subprocess.run([str(exe),mode,'4195328','0'],capture_output=True).returncode==7


def test_real_runner_uses_tested_producers():
 text=(HERE/'runner/runner.cpp').read_text()
 assert '#include "execution_report.h"' in text
 assert text.count('echo_execution_report::Host(out,bytes,r.device)')==1
 assert text.count('echo_execution_report::Echo(out,previous,r.device)')==1
 assert text.index('echo_execution_report::Host')<text.index('r.workspace.Allocate')<text.index('Check(aclnnDFlashExternalCo1EchoProbe(')<text.index('echo_execution_report::Echo')
 assert 'HOST_ONLY' not in text


def test_execute_and_verify_host_use_same_contract(tmp_path,monkeypatch,producer):
 _,emit=producer;record=emit()
 # Mock only the resource/SDK checks outside this focused status regression.
 # Assert those gates still run; no hardware evidence is produced.
 calls=[]
 monkeypatch.setattr(run,'artifacts',lambda *a:calls.append('artifacts'))
 monkeypatch.setattr(run,'bind_capture',lambda row,proof,ex:(calls.append('bind_capture') or {},{}))
 monkeypatch.setattr(run,'verify_serialization_record',lambda *a:calls.append('serialization'))
 monkeypatch.setattr(run,'compare_callbacks',lambda rows:calls.append('callbacks'))
 def command(argv,log,*rest):
  assert argv[-1]=='host';log.write_text('CPU contract fixture, NOT device evidence')
  run.c.write(log.parent/'execution.json',record)
  run.c.write(log.parent/'callback-0.json',{'fixture':True})
  for suffix in ('cube','op','cube-payload'):(log.parent/f'callback-0-{suffix}.bin').write_bytes(b'fixture')
 monkeypatch.setattr(run.build,'command',command)
 binary=tmp_path/'runner';binary.write_bytes(b'not-executed')
 r=dict(data={},artifacts=dict(runner=run.c.identity(binary),opp=str(tmp_path/'opp')),evidence=dict(accepted_workspace=4195328),device=0)
 args=SimpleNamespace(device=0,cann_root=tmp_path/'sdk')
 run.execute(args,tmp_path,r,host=True)
 assert r['npu']=='NOT_RUN' and all(k in calls for k in ('artifacts','bind_capture','serialization','callbacks'))
 monkeypatch.setattr(run.c,'validate_callback',lambda *a:calls.append('resource'))
 run.verify_host(r);assert 'resource' in calls
 bad=dict(record,status='HOST_ONLY');run.c.write(tmp_path/'run/execution.json',bad)
 # Rebind fixture identity so this specifically tests the status, not SHA rejection.
 r['files']=[run.c.identity(p) for p in sorted((tmp_path/'run').iterdir()) if p.is_file()]
 with pytest.raises(ValueError):run.verify_host(r)


@pytest.mark.parametrize('stage,expected,npu',[('source','SOURCE_REVIEWED','NOT_RUN'),('cpu','CPU_TESTS_PASSED','NOT_RUN'),('build','PRIVATE_OPP_BUILT','NOT_RUN'),('host','FORMAL_HOST_FEASIBLE','NOT_RUN'),('echo','ECHO_NPU_MATCH','COMPLETED')])
def test_stage_statuses_are_distinct_from_runner_status(tmp_path,stage,expected,npu):
 assert run.STATUS[stage]==expected
 proof=dict(sources=[],evidence={},device=0)
 p=tmp_path/'report.json';run.c.write(p,dict(proof,status=expected,phase='complete',npu=npu))
 assert run.prior(p,expected,proof)['status']==expected
 wrong=dict(proof,status='HOST_TILING_ONLY' if stage=='host' else 'ECHO_EXECUTION_MATCH',phase='complete',npu=npu)
 run.c.write(p,wrong)
 with pytest.raises(ValueError):run.prior(p,expected,proof)


def test_build_producer_success_status_matches_driver():
 import ast
 tree=ast.parse((HERE/'echo_build.py').read_text())
 states=[k.value.value for node in ast.walk(tree) if isinstance(node,ast.Call)
         and isinstance(node.func,ast.Attribute) and node.func.attr=='update'
         for k in node.keywords if k.arg=='status' and isinstance(k.value,ast.Constant)]
 assert states==[run.STATUS['build']]
