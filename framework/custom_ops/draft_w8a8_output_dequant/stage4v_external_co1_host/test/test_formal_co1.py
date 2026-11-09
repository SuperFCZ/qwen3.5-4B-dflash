"""Synthetic callback records/bytes. CPU checks are not Formal Host measurements."""
import json,struct,subprocess,sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import co1_host_common as c
import run_formal_host as run


def fixture():
 names=list(c.v.p.FIELDS)+list(c.BATCH)+list(c.AUX_ZERO)
 p={k:0 for k in names};p.update(usedCoreNum=1,singleCoreM=16,singleCoreN=256,singleCoreK=2560,baseM=16,baseN=256,baseK=128,depthA1=20,depthB1=10,stepM=1,stepN=1,stepKa=20,stepKb=5,dbL0A=2,dbL0B=2,dbL0C=1,transLength=174080,shareL1Size=368640,shareL0CSize=16384,M=16,N=256,Ka=2560,Kb=2560)
 policy=dict(aic=8,aiv=8,all=7,selected=7,arch=2002)
 proof=dict(field_names=names,schema=dict(bytes=200,fields=[dict(name=n,offset=i*4) for i,n in enumerate(names)]),policy=policy)
 row=dict(status='FORMAL_CALLBACK_ACCEPTED',context='REGISTERED_FORMAL_TILING_CALLBACK',kernel='COMPILE_ONLY',npu='NOT_RUN',error='',get_tiling_rc=0,fields=p,
  setters={k:0 for k in ('SetAType','SetBType','SetCType','SetBiasType','SetBias','SetShape','SetOrgShape','SetFixSplit','SetBufferSpace')},
  capacity=dict(UB=262144,L1=1048576,L0A=65536,L0B=65536,L0C=262144),core=policy,block_dim=1,system_workspace_bytes=4096,arena_bytes=4096,requested_workspace_bytes=9216,
  checks={k:True for k in ('shape','scope','steps','depths','L1','db','L0AB','external_L0C','UB','scratch','auxiliary_consumers','Workspace')},
  external_l0c=dict(bytes=16384,start=0,alignment=1024,effective_shared=0,allocation='PLANNED_NOT_EXECUTED'),
  ub=dict(scratch=202240,echo=16384,scale=2048,fp16=8192,guards=2048,reserve=8192,total=239104,effective_shared=0))
 return row,proof,dict(workspace_bytes=9216)


def binary(row,proof):return struct.pack('<50i',*[row['fields'][k] for k in proof['field_names']])


def test_valid_synthetic_record_and_complete_schema():
 r,p,e=fixture();out=c.validate_callback(r,binary(r,p),p,e)
 assert out['status']=='CONDITIONAL_ARITHMETIC_CHECKS_PASS' and out['ub']['total']==239104
 header=c.generated_header(p['field_names'])
 assert all(('p.get_'+k+'()') in header for k in p['field_names'])
 assert header.count('out<<"\\"')+header.count('out<<",\\"')==50
 assert 'SetDequantType' not in (HERE/'op_host/host_common.h').read_text()


@pytest.mark.parametrize('field,value',[('baseN',128),('singleCoreK',128),('usedCoreNum',2),('depthB1',11),('stepKa',21),('transLength',202272),('dbL0B',3),('shareL1Size',368641),('isBias',1),('depthAL1CacheUB',1),('ALayoutInfoN',1),('batchM',2),('mxTypePara',1)])
def test_tiling_rejection_without_fallback(field,value):
 r,p,e=fixture();r['fields'][field]=value
 with pytest.raises(ValueError):c.validate_callback(r,binary(r,p),p,e)


def test_serialized_and_logged_fields_must_match():
 r,p,e=fixture();raw=binary(r,p);r['fields']['shareUbSize']=2048
 with pytest.raises(ValueError):c.validate_callback(r,raw,p,e)
 r,p,e=fixture();r['fields'].pop('BatchNum')
 with pytest.raises(ValueError):c.validate_callback(r,raw,p,e)


@pytest.mark.parametrize('name,amount',[('L0C',16383),('L0B',65535),('UB',239103),('L1',368639)])
def test_actual_capacity_limits(name,amount):
 r,p,e=fixture();r['capacity'][name]=amount
 with pytest.raises(ValueError):c.validate_callback(r,binary(r,p),p,e)


def test_driver_request_and_core_policy_cannot_drift():
 r,p,e=fixture();e['workspace_bytes']=9215
 with pytest.raises(ValueError):c.validate_callback(r,binary(r,p),p,e)
 r,p,e=fixture();r['core']=dict(aic=8,aiv=8,all=8,selected=8,arch=2002)
 with pytest.raises(ValueError):c.validate_callback(r,binary(r,p),p,e)
 r,p,e=fixture();r['external_l0c']['start']=1024
 with pytest.raises(ValueError):c.validate_callback(r,binary(r,p),p,e)


def test_failed_gettiling_and_missing_checks_rejected():
 r,p,e=fixture();r['get_tiling_rc']=-1
 with pytest.raises(ValueError):c.validate_callback(r,binary(r,p),p,e)
 r,p,e=fixture();r['checks'].pop('Workspace')
 with pytest.raises(ValueError):c.validate_callback(r,binary(r,p),p,e)


@pytest.mark.parametrize('stage',['source','build','host'])
def test_cli_gates(stage):
 argv=[stage,'--review-report','/tmp/review.json','--output-dir','/tmp/new']
 if stage!='source':
  with pytest.raises(SystemExit):run.parse(argv)
  argv+=['--source-report','/tmp/source.json']
 if stage=='host':
  with pytest.raises(SystemExit):run.parse(argv)
  argv+=['--build-report','/tmp/build.json']
 assert run.parse(argv).stage==stage


def test_no_compute_or_device_execution_route():
 text=(HERE/'runner/runner.cpp').read_text();kernel=(HERE/'op_kernel/d_flash_external_co1_host_probe.cpp').read_text()
 assert 'aclnnDFlashExternalCo1HostProbeGetWorkspaceSize(' in text
 for forbidden in ('aclnnDFlashExternalCo1HostProbe(', 'aclrtCreateStream(', 'aclrtLaunchKernel(', 'aclrtSynchronizeStream(', 'aclrtMemcpy('):assert forbidden not in text
 for forbidden in ('Matmul','DataCopy(','GET_TILING_DATA','SetFlag<','__gm__'):assert forbidden not in kernel
 op=json.loads((HERE/'operator.json').read_text())[0]
 assert op['op']=='DFlashExternalCo1HostProbe' and len(op['input_desc'])==3
 assert not op.get('attr')


def test_imports_and_cli_help():
 subprocess.run([sys.executable,'-B',str(HERE/'run_formal_host.py'),'--help'],check=True,capture_output=True)


def test_source_identity_tamper(tmp_path):
 p=tmp_path/'host.cpp';p.write_text('original');item=c.identity(p);p.write_text('changed')
 with pytest.raises(ValueError):c.verify(item)


def test_callback_uses_real_context_and_saves_before_resource_reject():
 s=(HERE/'op_host/host_common.h').read_text()
 assert 'PlatformAscendC platform(c->GetPlatformInfo())' in s
 assert 'PlatformAscendCManager' not in s and 'ContextFaker' not in s
 assert 'TPosition::CO1,CubeFormat::NZ,DataType::DT_INT32' in s
 assert s.index('Co1Write(r.stem+"-cube.bin"')<s.index('if(!r.ok())')
 assert 'O_CREAT|O_EXCL|O_WRONLY|O_NOFOLLOW' in s


def test_only_host_stages_exist():
 for forbidden in ('bounded','smoke','benchmark','execute'):
  with pytest.raises(SystemExit):run.parse([forbidden,'--review-report','/tmp/r','--output-dir','/tmp/x'])


def test_wrong_stage_status_cannot_reach_host(tmp_path):
 r=dict(sources=[],evidence={},device=0)
 for status in ('BLOCKED','SOURCE_REVIEWED','FORMAL_CO1_HOST_FEASIBLE'):
  path=tmp_path/(status+'.json');c.write(path,dict(status=status,phase='complete',sources=[],evidence={},device=0,npu='NOT_RUN'))
  with pytest.raises(ValueError):run.prior(path,'PRIVATE_OPP_COMPILED',r)
