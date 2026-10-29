"""CPU doubles exercise exact source edits and control flow; no CANN/NPU claims."""
import copy,json,os,shutil,subprocess,sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import experiment_common as c
from guard_variants import (d_guard,FROZEN,KERNEL,materialize,verify_audit,transform,TILE_SCAN,ARENA_CALL,POLICIES,REMOVED)
from run_stage4e import profile_delta,same_resources

def materialized(tmp_path):
    accepted=tmp_path/'accepted_b.cpp';accepted.write_bytes(d_guard.transform((FROZEN/KERNEL).read_bytes(),'B'))
    return materialize(tmp_path/'sources',c.identity,c.write,c.identity(accepted))

def test_exact_diff_and_bound_originals(tmp_path):
    audit=materialized(tmp_path);verify_audit(audit,c.verify)
    b0=c.verify(audit['variants']['B0']['kernel']).read_bytes()
    assert b0==d_guard.transform((FROZEN/KERNEL).read_bytes(),'B')
    assert transform(b0,'B1')==b0.replace(TILE_SCAN.encode(),b'',1)
    assert transform(b0,'B2').count(ARENA_CALL.encode())==1 # initial check retained
    assert transform(b0,'B2').count(TILE_SCAN.encode())==1
    for v in POLICIES:
        assert 'if(!CheckGuard(p0FinalRaw.Get<uint32_t>(),2048)||!CheckGuard(castRaw.Get<uint32_t>(),4096))return;' in transform(b0,v).decode()
    with pytest.raises(ValueError):transform(b0.replace(TILE_SCAN.encode(),b''),'B1')
    with pytest.raises(ValueError):transform(b0+TILE_SCAN.encode(),'B2')
    path=Path(audit['variants']['B1']['source_dir'])/'op_host/host_common.h';path.write_text(path.read_text()+'// tamper')
    with pytest.raises(ValueError):verify_audit(audit,c.verify)

def test_wrong_baseline_and_coverage_rejected(tmp_path):
    bad=tmp_path/'bad.cpp';bad.write_bytes((FROZEN/KERNEL).read_bytes())
    with pytest.raises(ValueError,match='identity'):materialize(tmp_path/'bad-src',c.identity,c.write,c.identity(bad))
    audit=copy.deepcopy(materialized(tmp_path));audit['coverage']['B1']=dict(audit['coverage']['B1'],tile_five_ub_scans=True)
    with pytest.raises(ValueError,match='coverage'):verify_audit(audit,c.verify)


def test_statistics_rotation_and_profile_separation():
    runs=[]
    for i,row in enumerate(c.ORDER):
        for pos,v in enumerate(row):
            stats={k:c.statistics_ms([{'B0':4.,'B1':3.,'B2':3.5}[v]]*30,30) for k in ('execute_sync','prepare')}
            runs.append(dict(round=i,position=pos,variant=v,status='SCOPED_GOLDEN_MATCH',profiled=False,formal_latency_evidence=True,safety_protocol=POLICIES[v],timing=stats))
    summary=c.summarize(runs);assert summary['contrasts']['B1']['delta_B0_minus_variant_ms']==1
    assert all(len(x['execute_sync']['samples_ms'])==90 for x in summary['variants'].values())
    for field,value in [('profiled',True),('safety_protocol','BASELINE_GROUP_GUARDS'),('position',2)]:
        bad=copy.deepcopy(runs);bad[0][field]=value
        with pytest.raises(ValueError):c.summarize(bad)

def test_profile_columns_and_unavailable_not_guessed():
    def raw(n):return dict(counter_rows=[dict(file=dict(path='/prof/op_summary_20261029121212.csv'),fields={'Task Duration(us)':'100','aic_scalar_time(us)':'50','Machine Id':'0'}) for _ in range(n)])
    profiles={v:dict(raw=raw(38)) for v in POLICIES}
    result=profile_delta(profiles)
    assert len(result['columns'])==4 and result['non_guard_scalar_time']=='UNRESOLVED'
    assert result['availability']['B0']['mte2_time']=='UNAVAILABLE'
    profiles['B2']['raw']=raw(37)
    assert all(x['status'].startswith('UNRESOLVED') for x in profile_delta(profiles)['columns'] if x['variant']=='B2')

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
 argv=[str(executable),'0',str(folder),'0','B0']+(['--benchmark'] if timed else [])
 with (folder/'runner.log').open('w') as log:r=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode==0,(folder/'runner.log').read_text()
 ex=json.loads((folder/'execution.json').read_text());stats=c.check_execution(ex,0,'B0',timed)
 assert len(c.baseline.verify_outputs(proof,folder,timed))==(4 if timed else 2)
 assert ex['prepare_calls']==(38 if timed else 2) and ex['verify_calls']==(4 if timed else 2)
 if timed:
  assert len(stats['execute_sync']['samples_ms'])==30 and len(stats['prepare']['samples_ms'])==30
  bad=copy.deepcopy(ex);bad['timing']['per_timed_call_readback']=True
  with pytest.raises(ValueError):c.check_execution(bad,0,'B0',True)
  bad=copy.deepcopy(ex);bad['timing']['timed_tail_checked']=False
  with pytest.raises(ValueError):c.check_execution(bad,0,'B0',True)
  (folder/'postcheck.bin').write_bytes(bytes([1])*622592)
  with pytest.raises(ValueError,match='Golden'):c.baseline.verify_outputs(proof,folder,True)


def test_bad_precheck_cannot_enter_timing(executable,tmp_path):
 folder=tmp_path/'bad';inputs(folder);env=os.environ.copy();env.update(DFLASH_STAGE4B_EXPECTED_CORE_POLICY='8,8,7,7,2002',TEST_BAD_GOLDEN='1')
 with (folder/'runner.log').open('w') as log:r=subprocess.run([str(executable),'0',str(folder),'0','B0','--benchmark'],env=env,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode!=0
 ex=json.loads((folder/'execution.json').read_text());assert ex['status']=='FAIL' and ex['timing']['execute_sync']['samples_ms']==[]
 assert not (folder/'benchmark-last.bin').exists()


def test_timed_sdk_logging_is_reported_not_silently_muted(executable,tmp_path):
 folder=tmp_path/'logged';inputs(folder);env=os.environ.copy();env.update(DFLASH_STAGE4B_EXPECTED_CORE_POLICY='8,8,7,7,2002',TEST_TIMED_LOG='1')
 with (folder/'runner.log').open('w') as log:r=subprocess.run([str(executable),'0',str(folder),'0','B0','--benchmark'],env=env,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode!=0 and 'reject formal timing' in (folder/'runner.log').read_text()
 ex=json.loads((folder/'execution.json').read_text());assert ex['status']=='FAIL' and len(ex['timing']['execute_sync']['samples_ms'])==30
 assert (folder/'benchmark-last.bin').exists() and not (folder/'postcheck.bin').exists()



@pytest.mark.parametrize('variant',['B1','B2'])
def test_runner_reports_exact_weaker_coverage(executable,tmp_path,variant):
 folder=tmp_path/variant;inputs(folder);env=os.environ.copy();env['DFLASH_STAGE4B_EXPECTED_CORE_POLICY']='8,8,7,7,2002'
 with (folder/'runner.log').open('w') as log:r=subprocess.run([str(executable),'0',str(folder),'0',variant],env=env,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode==0,(folder/'runner.log').read_text()
 ex=json.loads((folder/'execution.json').read_text());c.check_execution(ex,0,variant,False)
 assert ex['safety_protocol']==POLICIES[variant] and ex['kernel_guards_unchanged'] is False
 with pytest.raises(ValueError):c.check_execution(ex,0,'B0',False)
 with pytest.raises(ValueError):c.baseline.check_execution(ex,0,0,False)



def test_real_kernel_control_flow_faults_and_sync_unchanged(tmp_path):
    compiler=shutil.which('c++')
    if not compiler:pytest.skip('no compiler')
    audit=materialized(tmp_path)
    include=tmp_path/'include';shutil.copytree(FROZEN/'test/cpu_include',include)
    header=include/'kernel_operator.h';s=header.read_text()
    s=s.replace('inline int fault=0,targetGroup=19;','inline int fault=0,targetGroup=19;inline size_t guardReads=0,barriers=0,fetches=0,sets=0,waits=0,arenaCopies=0;')
    s=s.replace('T GetValue(size_t i)const{Require(i<size);return data[i];}','T GetValue(size_t i)const{Require(i<size);++guardReads;return data[i];}')
    s=s.replace('int FetchEventID(HardEvent){return 0;}','int FetchEventID(HardEvent){++fetches;return 0;}')
    s=s.replace('void PipeBarrier(){}','void PipeBarrier(){++barriers;}').replace('void SetFlag(int){}','void SetFlag(int){++sets;}').replace('void WaitFlag(int){}','void WaitFlag(int){++waits;}')
    s=s.replace('DataCopy(LocalTensor<T>d,GlobalTensor<T>s,size_t n){Require','DataCopy(LocalTensor<T>d,GlobalTensor<T>s,size_t n){++arenaCopies;Require')
    header.write_text(s)
    lib=include/'lib/matmul_intf.h';s=lib.read_text().replace('if(fault==6)registeredWorkspace[-1]=0;', '''
        if(fault==1)w.data[-1]=0; // persistent scratch guard corruption
        if(fault==2)registeredWorkspace[-1]=0; // internal Arena, not outer allocation guard
        if(fault==3)reinterpret_cast<uint8_t*>(out.data)[-1]=0; // p1 guard; final p0/cast do not cover it
    ''');lib.write_text(s)
    results={}
    for v in POLICIES:
        cpp=tmp_path/(v+'.cpp');cpp.write_text('#include <iostream>\n#include "'+audit['variants'][v]['kernel']['path']+'"\n'+r'''
int main(int argc,char**argv){
 using namespace AscendC;
 std::vector<int8_t>a(40960),b(49807360);std::vector<uint64_t>s(389120);std::vector<half>parts(304*20*1024),out(311296);
 for(size_t t=0;t<304;++t)for(size_t g=0;g<20;++g)for(size_t n=0;n<1024;++n)parts[(t*20+g)*1024+n]=half((g+1)/64.0f);
 std::memset(out.data(),0x7f,622592);activationBase=a.data();weightBase=b.data();scaleBase=s.data();partialBase=parts.data();outputBase=out.data();
 blockNum=1;blockIdx=0;fault=std::stoi(argv[1]);std::vector<uint8_t>workspace(5120);workspaceBase=workspace.data();TestTiling td{239104,0,1,4096,4096,5120,0};
 d_flash_w8_a8_full_gate((uint8_t*)a.data(),(uint8_t*)b.data(),(uint8_t*)s.data(),(uint8_t*)out.data(),workspace.data(),(uint8_t*)&td);
 bool matched=true;for(auto v:out)if(v!=half(210.0f/64.0f))matched=false;
 std::cout<<guardReads<<" "<<matched<<" "<<iterations<<" "<<copies<<" "<<barriers<<" "<<fetches<<" "<<sets<<" "<<waits<<" "<<arenaCopies<<"\n";
}
''')
        exe=tmp_path/v;r=subprocess.run([compiler,'-std=c++17','-O1','-I'+str(include),str(cpp),'-o',str(exe)],capture_output=True,text=True)
        assert r.returncode==0,r.stderr
        results[v]={}
        for fault in (0,1,2,3):
            r=subprocess.run([str(exe),str(fault)],capture_output=True,text=True,check=True)
            results[v][fault]=list(map(int,r.stdout.split()))
    for v in POLICIES:
        assert results[v][0][1:]==results['B0'][0][1:] # exact transfer/event/barrier counts, no synchronization deletion
        assert results['B0'][0][0]-results[v][0][0]==REMOVED[v]
    assert results['B0'][0][1:4]==[1,6080,304*64]
    for fault in (1,3):
        assert results['B0'][fault][1]==results['B2'][fault][1]==0 and results['B1'][fault][1]==1
    assert results['B0'][2][1]==results['B1'][2][1]==0 and results['B2'][2][1]==1


def test_resource_drift_rejects():
    a=dict(plan={'blocks':7},workspace_bytes=100,kernel_configuration='same returned Tiling')
    same_resources([a,dict(a),dict(a)])
    for field,value in [('plan',{'blocks':6}),('workspace_bytes',101),('kernel_configuration','different transLength')]:
        b=dict(a);b[field]=value
        with pytest.raises(ValueError,match='differs'):same_resources([a,b])

@pytest.mark.parametrize('stage',['smoke','benchmark','profile'])
def test_driver_three_variants_and_phase_separation(tmp_path,monkeypatch,stage):
    from types import SimpleNamespace
    import run_stage4e as driver
    import build_variants
    args=SimpleNamespace(stage=stage,output_dir=tmp_path/'out',device=0,cann_root=tmp_path/'cann',stage4d_root=tmp_path/'d',
        stage4c_root=tmp_path/'c',stage4b_root=tmp_path/'b',pair_build_report=None,smoke_report=tmp_path/'smoke/report.json',
        benchmark_report=tmp_path/'benchmark/report.json')
    args.smoke_report.parent.mkdir();args.smoke_report.write_text('{}')
    args.benchmark_report.parent.mkdir();args.benchmark_report.write_text('{}')
    # Avoid profile's automatic real build lookup; this test isolates execution scheduling.
    args.pair_build_report=tmp_path/'build/build.json';args.pair_build_report.parent.mkdir();args.pair_build_report.write_text('{}')
    monkeypatch.setattr(driver,'sources',lambda:[])
    monkeypatch.setattr(driver,'evidence',lambda _:dict(stage4b={'opp':str(tmp_path/'opp')}))
    monkeypatch.setattr(driver.c4,'machine',lambda *a:{})
    monkeypatch.setattr(driver,'accept_build',lambda *a:{})
    monkeypatch.setattr(driver,'validate_build',lambda *a:None)
    monkeypatch.setattr(driver,'accepted_report',lambda *a:{'runs':[]})
    monkeypatch.setattr(driver,'collect',lambda *a:dict(status='RAW_OPERATOR_METRICS_FOUND',counter_rows=[]))
    calls=[]
    def invoke(args,folder,r,b,v,timed,profile=False):
        calls.append((v,timed,profile))
        return dict(status='SCOPED_GOLDEN_MATCH',variant=v,profiled=profile,formal_latency_evidence=timed and not profile,
            safety_protocol=POLICIES[v],timing={k:c.statistics_ms([4.]*30,30) for k in ('execute_sync','prepare')},
            plan={'blocks':7},kernel_configuration='same',workspace_bytes=100,profiler={'sha256':'same'})
    monkeypatch.setattr(driver,'invoke',invoke)
    assert driver.main(args)==0
    report=json.loads((args.output_dir/(stage+'-report.json')).read_text())
    if stage=='benchmark':
        assert calls==[(v,True,False) for row in c.ORDER for v in row]
        assert report['status']=='STAGE4E_CONTRASTS_MEASURED_DIAGNOSTIC_ONLY'
    else:
        assert calls==[(v,stage=='profile',stage=='profile') for v in POLICIES]
    assert report['stage4f'].startswith('NOT_RUN')
    # The builder really resolves re; avoids repeating Stage4D's missing-import regression.
    assert build_variants.re.findall(r'=>\s+(/\S+)\s+\(', 'lib.so => /sdk/lib.so (0x0)')==['/sdk/lib.so']


def test_benchmark_blocked_without_accepted_smoke(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import run_stage4e as driver
    args=SimpleNamespace(stage='benchmark',output_dir=tmp_path/'out',device=0,cann_root=tmp_path/'cann',stage4d_root=tmp_path/'d',
        stage4c_root=tmp_path/'c',stage4b_root=tmp_path/'b',pair_build_report=None,smoke_report=None,benchmark_report=None)
    monkeypatch.setattr(driver,'sources',lambda:[])
    monkeypatch.setattr(driver,'evidence',lambda _:dict(stage4b={'opp':str(tmp_path/'opp')}))
    monkeypatch.setattr(driver.c4,'machine',lambda *a:{})
    monkeypatch.setattr(driver,'accept_build',lambda *a:{})
    monkeypatch.setattr(driver,'invoke',lambda *a:pytest.fail('must not run before accepted smoke'))
    assert driver.main(args)==1
    r=json.loads((args.output_dir/'benchmark-report.json').read_text())
    assert r['status']=='FAILED' and r['performance']=='NOT_RUN' and r['npu']=='NOT_RUN'
