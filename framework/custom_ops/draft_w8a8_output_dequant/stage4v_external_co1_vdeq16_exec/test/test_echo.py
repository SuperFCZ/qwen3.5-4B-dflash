import sys,json,struct,subprocess,shutil
from pathlib import Path
import numpy as np
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import echo_common as c


def fixture():
 names=c.read(HERE.parent/'stage4v_external_co1_host/test/fixtures/tcube-schema.json')['fields']
 p={x['name']:0 for x in names};p.update(usedCoreNum=1,M=16,N=256,Ka=2560,Kb=2560,singleCoreM=16,singleCoreN=256,singleCoreK=2560,baseM=16,baseN=256,baseK=128,depthA1=20,depthB1=10,stepM=1,stepN=1,stepKa=20,stepKb=5,dbL0A=2,dbL0B=2,dbL0C=1,shareL1Size=368640,shareL0CSize=16384,transLength=174080)
 return dict(fields=p,capacity=dict(UB=262144,L1=1048576,L0A=65536,L0B=65536,L0C=262144),core=dict(aic=8,aiv=8,all=7,selected=7,arch=2002),system_workspace_bytes=2097152,requested_workspace_bytes=4195328)


def test_compile_contract_full50():
 f=fixture();h=c.generated_header(list(f['fields']),f)
 assert all('p.get_'+k+'()==' in h for k in f['fields'])
 assert 'EchoExpectedWorkspace=4195328ULL' in h
 assert 'EchoWireMatches' in h


def test_no_vdeq_development_in_echo_source():
 k=(HERE/'op_kernel/d_flash_external_co1_echo_probe.cpp').read_text()
 for s in ('SetQuantVector','DeqScale::VDEQ16','Cast(','MatmulType<TPosition::VECIN'):assert s not in k
 for s in ('TBuf<TPosition::CO1>','Iterate(false,co1)','Fence<HardEvent::M_V>','Fence<HardEvent::V_M>','mm.End()'):assert s in k
 assert 'meta.SetValue(7,0)' in k


def test_address_and_alias_coverage():
 assert sorted((n//16)*256+m*16+n%16 for m in range(16) for n in range(256))==list(range(4096))
 assert 202240+16384+2048+8192+2048+8192==239104
 assert 0+512<=512 and 512+256<=8192
 assert 16384%1024==0


def test_integer_comparison_is_all_elements():
 a=np.arange(4096,dtype='<i4').reshape(16,256);b=a.copy();b[7,235]+=1
 assert c.compare_echo(a.tobytes(),[a.tobytes(),a.tobytes()])['status']=='MATCH'
 r=c.compare_echo(a.tobytes(),[b.tobytes(),a.tobytes()]);assert r['status']=='MISMATCH' and r['repeats'][0]['mismatches']==1


def test_abi():
 op=json.loads((HERE/'operator.json').read_text())[0]
 assert op['op']=='DFlashExternalCo1EchoProbe'
 assert [x['type'] for x in op['output_desc']]==[['int32'],['uint64']]


def test_cpu_kernel_source_model(tmp_path):
 exe=tmp_path/'model';subprocess.run([shutil.which('c++'),'-std=c++17','-O2','-I'+str(HERE/'test/cpu_include'),str(HERE/'test/kernel_model.cpp'),'-o',str(exe)],check=True,capture_output=True)
 for fault in (0,1,2,3,4,5,6,9):subprocess.run([str(exe),str(fault)],check=True,capture_output=True)


def test_witness_and_no_stage_bypass():
 from run_echo import witness,parse
 words=[0x434f314543484f31,1,0,0,16384,230912,1,0]+[0]*8+[1]*8+[0]*8
 assert witness(struct.pack('<32Q',*words))['status']=='VALID'
 for index in (0,2,3,4,5,6,7,8):
  bad=words.copy();bad[index]=99
  assert witness(struct.pack('<32Q',*bad))['status']=='FAILED'
 bad=words.copy();bad[16]=0;assert witness(struct.pack('<32Q',*bad))['status']=='FAILED'
 with pytest.raises(SystemExit):parse(['echo','--output-dir','/tmp/new','--co1-root','/tmp/c','--raw-root','/tmp/r','--frozen-root','/tmp/f'])


def test_all_guard_bits(tmp_path):
 exe=tmp_path/'guards';subprocess.run([shutil.which('c++'),'-std=c++17','-O2','-I'+str(HERE/'test/cpu_include'),str(HERE/'test/guard_model.cpp'),'-o',str(exe)],check=True,capture_output=True)
 assert 'CPU_ONLY_ALL_512_WORDS_32_BITS_DETECTED' in subprocess.run([str(exe)],check=True,capture_output=True,text=True).stdout


@pytest.mark.parametrize('stage,needed',[('source',[]),('cpu',['source']),('build',['source','cpu']),('host',['source','cpu','build']),('echo',['source','cpu','build','host'])])
def test_cli_gate_reports(stage,needed):
 from run_echo import parse
 args=[stage,'--output-dir','/tmp/new','--co1-root','/tmp/c','--raw-root','/tmp/r','--frozen-root','/tmp/f']
 for key in needed:
  with pytest.raises(SystemExit):parse(args)
  args+=['--'+key+'-report','/tmp/'+key+'.json']
 assert parse(args).stage==stage


def test_vdeq_entry_has_no_execute_branch():
 text=(HERE/'run_echo.py').read_text()
 assert "VDEQ16 development/execution gate closed" in text
 assert text.index("if a.stage=='vdeq':")<text.index("r['data']=c.prepare")
 assert 'GetQueueEndAddress' not in (HERE/'op_kernel/d_flash_external_co1_echo_probe.cpp').read_text()


def test_prior_cannot_promote_executed_status_to_host(tmp_path):
 from run_echo import prior
 p=tmp_path/'host.json';r=dict(sources=[],evidence={},device=0)
 c.write(p,dict(status='FORMAL_HOST_FEASIBLE',phase='complete',sources=[],evidence={},device=0,npu='COMPLETED'))
 with pytest.raises(ValueError):prior(p,'FORMAL_HOST_FEASIBLE',r)
