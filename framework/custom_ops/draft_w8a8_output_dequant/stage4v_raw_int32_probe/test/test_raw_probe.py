"""CPU-only contract tests; no hardware safety or arithmetic attestation."""
import sys,subprocess,shutil,copy
from pathlib import Path
import numpy as np
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import raw_probe_common as c
import run_raw_probe as run

def fixture():
 fields={k:0 for k in c.v.p.FIELDS}
 fields.update(usedCoreNum=1,singleCoreM=16,singleCoreN=256,singleCoreK=2560,baseM=16,baseN=256,baseK=128,depthA1=20,depthB1=10,stepM=1,stepN=1,stepKa=20,stepKb=5,dbL0A=2,dbL0B=2,dbL0C=1,transLength=174080,shareL1Size=368640,shareL0CSize=16384,M=16,N=256,Ka=2560,Kb=2560)
 return dict(fields=fields,capacity=dict(UB=262144,L1=1048576,L0A=65536,L0B=65536,L0C=262144),scratch=202240,system_workspace_bytes=4096)

def test_source_budget_layout():
 assert c.audit()['ub']['total']==228864
 indices=[(n//16)*256+m*16+n%16 for m in range(16) for n in range(256)]
 assert sorted(indices)==list(range(4096))
 assert all((nb*256+m*16)*4%32==0 and (m*256+nb*16)*4%32==0 for nb in range(16) for m in range(16))
 assert 'SetDequantType' not in (HERE/'op_host/host_common.h').read_text()

@pytest.mark.parametrize('stage',['host','build','bounded'])
def test_cli(stage):
 args=[stage,'--output-dir','/tmp/out','--frozen-root','/tmp/frozen','--raw-host-report','/tmp/raw.json']
 if stage!='host':
  with pytest.raises(SystemExit):run.parse(args)
  args+=['--host-report','/tmp/host/report.json']
 if stage=='bounded':
  with pytest.raises(SystemExit):run.parse(args)
  args+=['--build-report','/tmp/build/report.json']
 assert run.parse(args).stage==stage

@pytest.mark.parametrize('order',["import run_raw_probe", "import run_stage4i; import run_raw_probe", "import raw_host_dependency_identity; import run_raw_probe"])
def test_imports(order):
 script=f"import sys;sys.path[:0]=[{str(HERE)!r},{str(HERE.parent/'stage4v_diagnostics')!r},{str(HERE.parent/'stage4i')!r}];{order};assert run_raw_probe.c.HERE.name=='stage4v_raw_int32_probe'"
 subprocess.run([sys.executable,'-B','-c',script],check=True,capture_output=True)

def test_compare_all_elements_and_failure():
 a=np.arange(-2048,2048,dtype=np.int64).reshape(16,256);raw=a.astype('<i4').tobytes()
 r=c.compare(a,[raw,raw]);assert r['status']=='MATCH' and r['repeats'][0]['mismatches']==0
 for m,n in [(0,0),(15,255)]+c.KEYS:
  b=a.copy();b[m,n]+=1
  r=c.compare(a,[b.astype('<i4').tobytes(),raw]);assert r['status']=='MISMATCH' and r['repeats'][0]['mismatches']==1
  assert r['repeats'][0]['first']==dict(row=m,channel=n,cpu=int(a[m,n]),npu=int(b[m,n]))
 assert c.compare(a,[raw])['status']=='MISMATCH'
 with pytest.raises(ValueError):c.compare(a,[b''])
 # Poison is far outside the mathematical INT32 bound: unwritten output cannot pass.
 assert c.compare(a,[bytes([127])*16384,bytes([128])*16384])['repeats'][0]['mismatches']==4096


def test_resource_log_contract():
 t=fixture();policy=dict(aic=8,aiv=8,all=7,selected=7,arch=2002)
 lines=['STAGE4RAW_CORE_POLICY aic=8 aiv=8 all=7 selected=7','STAGE4RAW_ARCH_PROBE arch=2002','STAGE4RAW_T1_CONTRACT match=1 effectiveSharedUb=0','STAGE4RAW_SCHEDULE coreLimit=1 blocks=1 cores=7 system=4096 arena=4096 stride=5120 user=5120 total=9216 mode=0']
 text='\n'.join(lines+[f'STAGE4RAW_T1 {k}={v}' for k,v in t['fields'].items()])
 assert run.plan(text,t,policy)['usedCoreNum']==1
 for before,after in [('blocks=1','blocks=8'),('transLength=174080','transLength=8192'),('selected=7','selected=8'),('total=9216','total=8192'),('match=1','match=0')]:
  with pytest.raises(ValueError):run.plan(text.replace(before,after),t,policy)
 assert 'p.get_transLength()==174080' in c.contract_header(t)
 changed=copy.deepcopy(t);changed['scratch']=226816
 with pytest.raises(ValueError):c.contract_header(changed)


def test_identity_tamper(tmp_path):
 p=tmp_path/'file';p.write_bytes(b'original');record=c.identity(p);assert c.verify(record)==p
 p.write_bytes(b'tampered')
 with pytest.raises(ValueError):c.verify(record)


def test_wrong_predecessor():
 from types import SimpleNamespace
 args=SimpleNamespace(device=0)
 for status in ('BLOCKED','RAW_HOST_CANDIDATE_FEASIBLE_CONTEXT_PENDING','BUILT_NPU_NOT_RUN'):
  with pytest.raises(ValueError):run.verify_gate(dict(status=status),args,{},'FORMAL_HOST_FEASIBLE')


def test_compiled_kernel_cpu_double(tmp_path):
 compiler=shutil.which('c++');assert compiler
 exe=tmp_path/'kernel-model'
 subprocess.run([compiler,'-std=c++17','-O2','-I'+str(HERE/'test/cpu_include'),str(HERE/'test/kernel_cpu.cpp'),'-o',str(exe)],check=True,capture_output=True)
 for fault in range(6):
  r=subprocess.run([str(exe),str(fault)],check=True,capture_output=True,text=True)
  assert ('CPU_SOURCE_MODEL_MATCH' if fault==0 else 'CPU_REJECTED_BEFORE_GM') in r.stdout


def test_raw_resource_rejects_changes():
 t=fixture();platform=dict(soc='Ascend310P3',arch=2002,platform_source='NO_ARGUMENT_MANAGER_ON_SELECTED_DEVICE',capacity=t['capacity'],system_workspace_bytes=4096)
 row=dict(fields=t['fields'],name='RAW_INT32_N256',requested_n=256,requested_base_n=256,scratch=202240,c_dtype='INT32',c_position='VECIN',c_format='NZ',scale_input=False,setters={},setup_accepted=True,get_tiling_rc=0)
 assert c.raw_resources.validate(platform,row)['status']=='RAW_RESOURCE_CHECKS_PASS'
 for key,value in [('transLength',202272),('stepKa',21),('depthB1',9),('shareL1Size',368641),('dbL0B',3),('usedCoreNum',2),('baseN',128)]:
  bad=copy.deepcopy(row);bad['fields'][key]=value
  assert c.raw_resources.validate(platform,bad)['status']=='BLOCKED'
 for key,value in [('L0B',65535),('L1',368639),('L0C',16384),('UB',228863)]:
  bad=copy.deepcopy(platform);bad['capacity'][key]=value
  assert c.raw_resources.validate(bad,row)['status']=='BLOCKED'


def test_make_target_binding_and_alias(tmp_path):
 from raw_probe_dependencies import collect,verify_binding
 project=tmp_path/'CustomOp';build=project/'build_out';target=build/'op_host/CMakeFiles/cust_optiling.dir';target.mkdir(parents=True)
 src=project/'op_host/d_flash_full_k_int32_probe.cpp';src.parent.mkdir();src.write_text('// source')
 obj=target/'d_flash_full_k_int32_probe.cpp.o';obj.write_bytes(b'object')
 cann=tmp_path/'cann';cann.mkdir();header=cann/'platform_ascendc.h';header.write_text('// reviewed')
 alias=cann/'alias';alias.mkdir();(alias/header.name).symlink_to(header)
 (build/'CMakeCache.txt').write_text('CMAKE_GENERATOR:INTERNAL=Unix Makefiles\n')
 (target/'DependInfo.cmake').write_text(f'set(CMAKE_DEPENDS_CHECK_CXX "{src}" "{obj}")')
 rel=obj.relative_to(build)
 (target/'depend.make').write_text(f'{rel}: {src} {alias/header.name}\n')
 (target/'depend.internal').write_text(f'{rel}\n {src}\n {alias/header.name}\n')
 record=collect(project,cann,c.identity(header));assert record['route']=='CMAKE_MAKE_IMPLICIT_DEPENDENCIES'
 verify_binding(record)
 header.write_text('// modified')
 with pytest.raises(ValueError):verify_binding(record)
 header.write_text('// reviewed')
 (target/'depend.internal').write_text(f'{rel}\n {src}\n')
 with pytest.raises(ValueError):collect(project,cann,c.identity(header))
 (target/'depend.internal').write_text(f'{rel}\n {src}\n {alias/header.name}\n')
 (target/'DependInfo.cmake').write_text(f'set(CMAKE_DEPENDS_CHECK_CXX "{src}" "{target}/wrong.cpp.o")')
 with pytest.raises(ValueError):collect(project,cann,c.identity(header))


def test_no_launch_or_timing_in_host_branch():
 text=(HERE/'runner/runner.cpp').read_text()
 assert 'RunMeasuredCalls' not in text and 'Timing timing' not in text
 assert text.index('if(hostOnly){std::ofstream') < text.index('Check(aclnnDFlashFullKInt32Probe(')
 assert 'DFLASH_STAGE4RAW_EXPECTED_WORKSPACE' in text
 assert text.index('WriteBytes(folder+"/actual-')<text.index('if(out!=expected)')
 assert 'else if(out!=first)' in text
 assert text.count('CheckNamedGuards(')==2


def test_poison_cannot_equal_any_valid_dot():
 assert (2560*127*127)==41290240
 for byte in (127,128):
  value=int(np.frombuffer(bytes([byte])*4,'<i4')[0])
  assert abs(value)>41290240
 assert 'repeat?0x80:0x7f' in (HERE/'runner/runner.cpp').read_text()

@pytest.mark.parametrize('host',[True,False])
def test_receiver_failure_preserves_evidence_and_execution_state(tmp_path,monkeypatch,host):
 from types import SimpleNamespace
 args=SimpleNamespace(cann_root=tmp_path/'cann',device=0)
 runner=tmp_path/'runner';runner.write_bytes(b'CPU_MOCK_NOT_DEVICE')
 oracle=np.arange(4096,dtype=np.int64).reshape(16,256)
 r=dict(data=run.data(tmp_path/'data',bytes(40960),bytes(655360),oracle),artifacts=dict(opp=str(tmp_path/'opp'),runner=c.identity(runner)),evidence=dict(t1=fixture(),policy=dict(aic=8,aiv=8,all=7,selected=7,arch=2002)))
 def failure(argv,log,*unused):
  log.write_text('deliberate CPU test failure')
  c.write(log.parent/'execution.json',dict(status='RUNNING',npu='NOT_RUN' if host else 'ATTEMPTED'))
  if not host:
   actual=oracle.astype('<i4');actual[0,84]+=1;(log.parent/'actual-0.bin').write_bytes(actual.tobytes())
  raise RuntimeError('receiver failed')
 monkeypatch.setattr(run,'command',failure)
 with pytest.raises(RuntimeError):run.run_receiver(args,tmp_path,r,host,9216)
 assert r['npu']==('NOT_RUN' if host else 'ATTEMPTED')
 assert r['receiver_files']
 if not host:
  assert r['integer_comparison']['repeats'][0]['first']['channel']==84
  assert r['integer_comparison']['status']=='MISMATCH'
  assert c.verify(r['outputs'][0]).read_bytes()!=oracle.astype('<i4').tobytes()


def test_report_field_count_and_raw_abi():
 import json
 op=json.loads((HERE/'operator.json').read_text())[0]
 assert op['op']=='DFlashFullKInt32Probe'
 assert [x['type'] for x in op['input_desc']]==[['int8'],['int8']]
 assert op['output_desc'][0]['type']==['int32'] and not op.get('attr')
 assert len(c.v.p.FIELDS)==27
