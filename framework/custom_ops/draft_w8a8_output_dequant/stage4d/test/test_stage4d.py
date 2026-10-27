"""CPU source/control tests only. No CANN/NPU or safety-equivalence claim."""
import copy,json,os,shutil,subprocess,sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import pair_common as c
from guard_variant import materialize,transform,verify_audit,FROZEN,KERNEL,SCAN,POLICIES
from run_stage4d import profile_delta


def test_exact_single_source_difference_and_failure_closed(tmp_path):
 audit=materialize(tmp_path/'sources',c.identity,c.write);verify_audit(audit,c.verify)
 a=c.verify(audit['variants']['A']['kernel']).read_bytes();b=c.verify(audit['variants']['B']['kernel']).read_bytes()
 assert a==(FROZEN/KERNEL).read_bytes() and b==a.replace(SCAN.encode(),b'',1)
 assert audit['group_scan_success_path_getvalue_calls_removed']==3891200
 for token in ('PipeBarrier<','SetFlag<','WaitFlag<','FetchEventID','mm.SetQuantVector','mm.End','mm.Iterate'):
  assert a.count(token.encode())==b.count(token.encode())
 with pytest.raises(ValueError):transform(a.replace(SCAN.encode(),b''),'B')
 with pytest.raises(ValueError):transform(a+SCAN.encode(),'B')
 path=Path(audit['variants']['B']['source_dir'])/'op_host/host_common.h';path.write_text(path.read_text()+'\n// tamper\n')
 with pytest.raises(ValueError):verify_audit(audit,c.verify)


def test_summary_never_mixes_profile_or_wrong_safety():
 runs=[]
 for i,row in enumerate(c.ORDER):
  for pos,v in enumerate(row):
   stats={k:c.statistics_ms([10. if v=='A' else 5.]*30,30) for k in ('execute_sync','prepare')}
   runs.append(dict(round=i,position=pos,variant=v,status='SCOPED_GOLDEN_MATCH',profiled=False,formal_latency_evidence=True,safety_protocol=POLICIES[v],timing=stats))
 r=c.summarize(runs);assert r['speedup_A_over_B']==2 and r['delta_A_minus_B_ms']==5
 assert len(r['variants']['B']['execute_sync']['samples_ms'])==90 and 'WEAKER' in r['safety']
 bad=copy.deepcopy(runs);bad[1]['profiled']=True
 with pytest.raises(ValueError):c.summarize(bad)
 bad=copy.deepcopy(runs);bad[1]['safety_protocol']=POLICIES['A']
 with pytest.raises(ValueError):c.summarize(bad)


def test_profile_delta_preserves_columns_and_unequal_coverage():
 def raw(value,count):return dict(counter_rows=[dict(file=dict(path='/profile/op_summary_20261027111111.csv'),fields={'Task Duration(us)':str(value),'aic_scalar_time(us)':str(value/2)}) for _ in range(count)])
 r=profile_delta(dict(A=dict(raw=raw(100,38)),B=dict(raw=raw(40,38))))
 assert all(x['status']=='RAW_SAME_COLUMN_COMPARISON' for x in r['columns'])
 assert r['critical_path']=='UNRESOLVED' and r['equivalent_vector_guard']=='NOT_RUN'
 r=profile_delta(dict(A=dict(raw=raw(100,38)),B=dict(raw=raw(40,37))))
 assert all(x['status'].startswith('UNRESOLVED') for x in r['columns'])

@pytest.fixture(scope='module')
def executable(tmp_path_factory):
 compiler=shutil.which('c++')
 if not compiler:pytest.skip('no local compiler')
 root=tmp_path_factory.mktemp('bench-cpu');exe=root/'runner'
 r=subprocess.run([compiler,'-std=c++17','-O1','-I'+str(HERE.parent/'stage4c/test/cpu_include'),'-I'+str(c.common.QUANT/'test'),str(HERE/'test/runner.cpp'),str(HERE.parent/'stage4c/test/cpu_api.cpp'),'-o',str(exe)],capture_output=True,text=True)
 assert r.returncode==0,r.stderr
 return exe


def inputs(folder):
 folder.mkdir()
 for name,size in [('a_i8.bin',40960),('b_i8.bin',49807360),('quant_u64_doc_candidate.bin',3112960),('golden_fp16.bin',622592)]:
  with (folder/name).open('wb') as f:f.truncate(size)
 return dict(files={'golden_fp16.bin':c.identity(folder/'golden_fp16.bin')})


@pytest.mark.parametrize('timed',[False,True])
def test_real_runner_continuous_schedule_has_no_timed_copies(executable,tmp_path,timed):
 folder=tmp_path/'case';proof=inputs(folder);env=os.environ.copy();env['DFLASH_STAGE4B_EXPECTED_CORE_POLICY']='8,8,7,7,2002'
 argv=[str(executable),'0',str(folder),'0','A']+(['--benchmark'] if timed else [])
 with (folder/'runner.log').open('w') as log:r=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode==0,(folder/'runner.log').read_text()
 ex=json.loads((folder/'execution.json').read_text());stats=c.check_execution(ex,0,'A',timed)
 assert len(c.baseline.verify_outputs(proof,folder,timed))==(4 if timed else 2)
 assert ex['prepare_calls']==(38 if timed else 2) and ex['verify_calls']==(4 if timed else 2)
 if timed:
  assert len(stats['execute_sync']['samples_ms'])==30 and len(stats['prepare']['samples_ms'])==30
  bad=copy.deepcopy(ex);bad['timing']['per_timed_call_readback']=True
  with pytest.raises(ValueError):c.check_execution(bad,0,'A',True)
  bad=copy.deepcopy(ex);bad['timing']['timed_tail_checked']=False
  with pytest.raises(ValueError):c.check_execution(bad,0,'A',True)
  (folder/'postcheck.bin').write_bytes(bytes([1])*622592)
  with pytest.raises(ValueError,match='Golden'):c.baseline.verify_outputs(proof,folder,True)


def test_bad_precheck_cannot_enter_timing(executable,tmp_path):
 folder=tmp_path/'bad';inputs(folder);env=os.environ.copy();env.update(DFLASH_STAGE4B_EXPECTED_CORE_POLICY='8,8,7,7,2002',TEST_BAD_GOLDEN='1')
 with (folder/'runner.log').open('w') as log:r=subprocess.run([str(executable),'0',str(folder),'0','A','--benchmark'],env=env,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode!=0
 ex=json.loads((folder/'execution.json').read_text());assert ex['status']=='FAIL' and ex['timing']['execute_sync']['samples_ms']==[]
 assert not (folder/'benchmark-last.bin').exists()


def test_timed_sdk_logging_is_reported_not_silently_muted(executable,tmp_path):
 folder=tmp_path/'logged';inputs(folder);env=os.environ.copy();env.update(DFLASH_STAGE4B_EXPECTED_CORE_POLICY='8,8,7,7,2002',TEST_TIMED_LOG='1')
 with (folder/'runner.log').open('w') as log:r=subprocess.run([str(executable),'0',str(folder),'0','A','--benchmark'],env=env,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode!=0 and 'reject formal timing' in (folder/'runner.log').read_text()
 ex=json.loads((folder/'execution.json').read_text());assert ex['status']=='FAIL' and len(ex['timing']['execute_sync']['samples_ms'])==30
 assert (folder/'benchmark-last.bin').exists() and not (folder/'postcheck.bin').exists()



def test_runner_B_reports_weaker_coverage(executable,tmp_path):
 folder=tmp_path/'B';proof=inputs(folder);env=os.environ.copy();env['DFLASH_STAGE4B_EXPECTED_CORE_POLICY']='8,8,7,7,2002'
 with (folder/'runner.log').open('w') as log:r=subprocess.run([str(executable),'0',str(folder),'0','B'],env=env,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode==0,(folder/'runner.log').read_text()
 ex=json.loads((folder/'execution.json').read_text());c.check_execution(ex,0,'B',False)
 assert ex['safety_protocol']=='DIAGNOSTIC_ONLY_WEAKER_GUARD_COVERAGE' and ex['kernel_guards_unchanged'] is False
 with pytest.raises(ValueError):c.check_execution(ex,0,'A',False)
 with pytest.raises(ValueError):c.baseline.check_execution(ex,0,0,False)


def test_actual_kernel_group_read_count_and_weaker_transient_detection(tmp_path):
 compiler=shutil.which('c++')
 if not compiler:pytest.skip('no compiler')
 audit=materialize(tmp_path/'sources',c.identity,c.write)
 include=tmp_path/'include';shutil.copytree(FROZEN/'test/cpu_include',include)
 header=include/'kernel_operator.h';text=header.read_text().replace('inline int fault=0,targetGroup=19;','inline int fault=0,targetGroup=19;inline size_t guardReads=0;')
 text=text.replace('T GetValue(size_t i)const{Require(i<size);return data[i];}','T GetValue(size_t i)const{Require(i<size);++guardReads;return data[i];}')
 header.write_text(text)
 lib=include/'lib/matmul_intf.h';text=lib.read_text().replace('if(fault==6)registeredWorkspace[-1]=0;', '''
  // Simulate transient corruption restored in the next group, or corruption retained to tile end.
  if(fault==1&&iterations%20==1)w.data[-1]=0;
  if(fault==1&&iterations%20==2)w.data[-1]=0x5a;
  if(fault==2&&iterations%20==0)w.data[-1]=0;
 ''');lib.write_text(text)
 results={}
 for variant in ('A','B'):
  cpp=tmp_path/(variant+'.cpp');cpp.write_text('#include <iostream>\n#include "'+audit['variants'][variant]['kernel']['path']+'"\n'+r'''
int main(int argc,char**argv){
 using namespace AscendC;
 std::vector<int8_t>a(40960),b(49807360);std::vector<uint64_t>s(389120);std::vector<half>parts(304*20*1024),out(311296);
 for(size_t t=0;t<304;++t)for(size_t g=0;g<20;++g)for(size_t n=0;n<1024;++n)parts[(t*20+g)*1024+n]=half((g+1)/64.0f);
 std::memset(out.data(),0x7f,622592);activationBase=a.data();weightBase=b.data();scaleBase=s.data();partialBase=parts.data();outputBase=out.data();
 blockNum=1;blockIdx=0;fault=std::stoi(argv[1]);std::vector<uint8_t>workspace(5120);workspaceBase=workspace.data();TestTiling td{239104,0,1,4096,4096,5120,0};
 d_flash_w8_a8_full_gate((uint8_t*)a.data(),(uint8_t*)b.data(),(uint8_t*)s.data(),(uint8_t*)out.data(),workspace.data(),(uint8_t*)&td);
 bool matched=true;for(auto v:out)if(v!=half(210.0f/64.0f))matched=false;
 std::cout<<guardReads<<" "<<matched<<" "<<iterations<<" "<<copies<<"\n";
}
''')
  exe=tmp_path/variant;r=subprocess.run([compiler,'-std=c++17','-O1','-I'+str(include),str(cpp),'-o',str(exe)],capture_output=True,text=True)
  assert r.returncode==0,r.stderr
  results[variant]={}
  for fault in (0,1,2):
   r=subprocess.run([str(exe),str(fault)],capture_output=True,text=True,check=True)
   results[variant][fault]=list(map(int,r.stdout.split()))
 assert results['A'][0][0]-results['B'][0][0]==3891200
 assert results['A'][0][1:]==results['B'][0][1:]==[1,6080,304*64]
 assert results['A'][1][1]==0 and results['B'][1][1]==1 # B is observably weaker; never label equivalent
 assert results['A'][2][1]==results['B'][2][1]==0 # persistent corruption still rejected at tile end


def test_same_resource_gate_rejects_tiling_or_workspace_drift():
 from run_stage4d import same_resources
 a=dict(plan={'blocks':7},workspace_bytes=100,kernel_configuration='same returned Tiling')
 same_resources([a,dict(a)])
 for field,value in [('plan',{'blocks':6}),('workspace_bytes',101),('kernel_configuration','different transLength')]:
  b=dict(a);b[field]=value
  with pytest.raises(ValueError,match='differs'):same_resources([a,b])
