"""CPU protocol and state-model tests only; no CANN/NPU proof."""
import copy,json,os,shutil,subprocess,sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import f_common as c
from workspace_variant import FROZEN,KERNEL,POLICIES,CALL,OLD_COMMENT,NEW_COMMENT,hoist,materialize,verify_audit,d
from run_stage4f import profile_delta,same_resources

def materialized(tmp_path):
    p=tmp_path/'accepted_b.cpp';p.write_bytes(d.transform((FROZEN/KERNEL).read_bytes(),'B'))
    return materialize(tmp_path/'sources',c.identity,c.write,c.identity(p))

def test_only_statement_moves_and_baselines_are_exact(tmp_path):
    audit=materialized(tmp_path);verify_audit(audit,c.verify)
    raw=(FROZEN/KERNEL).read_bytes()
    assert c.verify(audit['variants']['A0']['kernel']).read_bytes()==raw
    assert c.verify(audit['variants']['B0']['kernel']).read_bytes()==d.transform(raw,'B')
    for pair in ('A','B'):
        a=c.verify(audit['variants'][pair+'0']['kernel']).read_text()
        b=c.verify(audit['variants'][pair+'1']['kernel']).read_text()
        assert a.replace(CALL,'')==b.replace('    mm.SetLocalWorkspace(scratch);\n','').replace(NEW_COMMENT,OLD_COMMENT)
        for token in ('PipeBarrier<','SetFlag<','WaitFlag<','CheckGuard(','CheckArena(','mm.End','mm.SetQuantVector','mm.GetTensorC','mm.Iterate','Add('):
            assert a.count(token)==b.count(token)
    with pytest.raises(ValueError):hoist(raw+CALL.encode())
    with pytest.raises(ValueError):hoist(raw.replace(b'Duplicate(sum,0.0f,1024)',b'Duplicate(sum,1.0f,1024)'))
    p=Path(audit['variants']['A1']['source_dir'])/'op_host/host_common.h';p.write_text(p.read_text()+'// changed')
    with pytest.raises(ValueError):verify_audit(audit,c.verify)

def test_separate_pairs_and_exact_protocol():
    runs=[]
    for i,order in enumerate(c.ORDER):
        for j,v in enumerate(order):
            stats={k:c.statistics_ms([{'A0':12.,'A1':11.,'B0':4.,'B1':3.9}[v]]*30,30) for k in ('execute_sync','prepare')}
            runs.append(dict(round=i,position=j,variant=v,core_limit=0,status='SCOPED_GOLDEN_MATCH',profiled=False,formal_latency_evidence=True,safety_protocol=POLICIES[v],timing=stats))
    report=c.summarize(runs)
    assert set(report['pairs'])=={'A','B'} and report['pairs']['A']['delta_baseline_minus_candidate_ms']==1
    assert all(len(s['execute_sync']['samples_ms'])==90 for s in report['variants'].values())
    for key,value in [('profiled',True),('core_limit',2),('safety_protocol',POLICIES['B0'])]:
        bad=copy.deepcopy(runs);bad[0][key]=value
        with pytest.raises(ValueError):c.summarize(bad)

def test_profile_never_crosses_guard_protocols():
    def raw(n):return dict(counter_rows=[dict(file=dict(path='/prof/op_summary_20261029121212.csv'),fields={'Task Duration(us)':'100','aic_scalar_time(us)':'50'}) for _ in range(n)])
    profiles={v:dict(raw=raw(38)) for v in POLICIES};r=profile_delta(profiles)
    assert {(x['baseline'],x['variant']) for x in r['columns']}=={('A0','A1'),('B0','B1')}
    profiles['B1']['raw']=raw(37)
    assert all(x['status'].startswith('UNRESOLVED') for x in profile_delta(profiles)['columns'] if x['variant']=='B1')

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
 argv=[str(executable),'0',str(folder),'0','A0']+(['--benchmark'] if timed else [])
 with (folder/'runner.log').open('w') as log:r=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode==0,(folder/'runner.log').read_text()
 ex=json.loads((folder/'execution.json').read_text());stats=c.check_execution(ex,0,'A0',timed)
 assert len(c.baseline.verify_outputs(proof,folder,timed))==(4 if timed else 2)
 assert ex['prepare_calls']==(38 if timed else 2) and ex['verify_calls']==(4 if timed else 2)
 if timed:
  assert len(stats['execute_sync']['samples_ms'])==30 and len(stats['prepare']['samples_ms'])==30
  bad=copy.deepcopy(ex);bad['timing']['per_timed_call_readback']=True
  with pytest.raises(ValueError):c.check_execution(bad,0,'A0',True)
  bad=copy.deepcopy(ex);bad['timing']['timed_tail_checked']=False
  with pytest.raises(ValueError):c.check_execution(bad,0,'A0',True)
  (folder/'postcheck.bin').write_bytes(bytes([1])*622592)
  with pytest.raises(ValueError,match='Golden'):c.baseline.verify_outputs(proof,folder,True)


def test_bad_precheck_cannot_enter_timing(executable,tmp_path):
 folder=tmp_path/'bad';inputs(folder);env=os.environ.copy();env.update(DFLASH_STAGE4B_EXPECTED_CORE_POLICY='8,8,7,7,2002',TEST_BAD_GOLDEN='1')
 with (folder/'runner.log').open('w') as log:r=subprocess.run([str(executable),'0',str(folder),'0','A0','--benchmark'],env=env,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode!=0
 ex=json.loads((folder/'execution.json').read_text());assert ex['status']=='FAIL' and ex['timing']['execute_sync']['samples_ms']==[]
 assert not (folder/'benchmark-last.bin').exists()


def test_timed_sdk_logging_is_reported_not_silently_muted(executable,tmp_path):
 folder=tmp_path/'logged';inputs(folder);env=os.environ.copy();env.update(DFLASH_STAGE4B_EXPECTED_CORE_POLICY='8,8,7,7,2002',TEST_TIMED_LOG='1')
 with (folder/'runner.log').open('w') as log:r=subprocess.run([str(executable),'0',str(folder),'0','A0','--benchmark'],env=env,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode!=0 and 'reject formal timing' in (folder/'runner.log').read_text()
 ex=json.loads((folder/'execution.json').read_text());assert ex['status']=='FAIL' and len(ex['timing']['execute_sync']['samples_ms'])==30
 assert (folder/'benchmark-last.bin').exists() and not (folder/'postcheck.bin').exists()



@pytest.mark.parametrize('variant',['A1','B0','B1'])
def test_runner_reports_exact_weaker_coverage(executable,tmp_path,variant):
 folder=tmp_path/variant;inputs(folder);env=os.environ.copy();env['DFLASH_STAGE4B_EXPECTED_CORE_POLICY']='8,8,7,7,2002'
 with (folder/'runner.log').open('w') as log:r=subprocess.run([str(executable),'0',str(folder),'0',variant],env=env,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode==0,(folder/'runner.log').read_text()
 ex=json.loads((folder/'execution.json').read_text());c.check_execution(ex,0,variant,False)
 assert ex['safety_protocol']==POLICIES[variant] and ex['kernel_guards_unchanged'] is variant.startswith('A')
 with pytest.raises(ValueError):c.check_execution(ex,0,'A0',False)
 with pytest.raises(ValueError):c.baseline.check_execution(ex,0,0,False)




@pytest.mark.parametrize('limit',[0,2])
def test_untimed_core_limits(executable,tmp_path,limit):
    folder=tmp_path/str(limit);proof=inputs(folder);env=os.environ.copy();env['DFLASH_STAGE4B_EXPECTED_CORE_POLICY']='8,8,7,7,2002'
    with (folder/'runner.log').open('w') as log:r=subprocess.run([str(executable),'0',str(folder),str(limit),'A1'],env=env,stdout=log,stderr=subprocess.STDOUT)
    assert r.returncode==0,(folder/'runner.log').read_text()
    ex=json.loads((folder/'execution.json').read_text());c.check_execution(ex,0,'A1',False,limit)
    assert len(c.baseline.verify_outputs(proof,folder,False))==2


def test_sdk_hash_mismatch_fails_without_collecting_source(tmp_path,monkeypatch):
    from f_common import sdk_binding
    fake=tmp_path/'contract';fake.mkdir();sdk=tmp_path/'cann';sdk.mkdir();p=sdk/'fixed.h';p.write_text('test-only')
    contract=dict(files=[dict(relative='fixed.h',sha256=c.identity(p)['sha256'],lines=[[1,1]])])
    (fake/'sdk-contract.json').write_text(json.dumps(contract));monkeypatch.setattr(c,'HERE',fake)
    assert sdk_binding(sdk)['status']=='APPROVED_IDENTITIES_MATCHED_NOT_NPU_VERIFIED'
    p.write_text('changed')
    with pytest.raises(ValueError,match='identity'):sdk_binding(sdk)


def test_kernel_binding_survives_twenty_ends_and_rebinds_next_tile(tmp_path):
    compiler=shutil.which('c++')
    if not compiler:pytest.skip('no compiler')
    audit=materialized(tmp_path);include=tmp_path/'include';shutil.copytree(FROZEN/'test/cpu_include',include)
    h=include/'kernel_operator.h';s=h.read_text().replace('inline int fault=0,targetGroup=19;', 'inline int fault=0,targetGroup=19;inline size_t bindings=0,lastBoundIter=0,barriers=0,fetches=0,sets=0,waits=0,guardReads=0;')
    s=s.replace('int FetchEventID(HardEvent){return 0;}','int FetchEventID(HardEvent){++fetches;return 0;}').replace('void PipeBarrier(){}','void PipeBarrier(){++barriers;}').replace('void SetFlag(int){}','void SetFlag(int){++sets;}').replace('void WaitFlag(int){}','void WaitFlag(int){++waits;}').replace('return data[i];','++guardReads;return data[i];')
    h.write_text(s)
    p=include/'lib/matmul_intf.h';s=p.read_text()
    s=s.replace('w=v;hasW=true;', 'w=v;hasW=true;++bindings;lastBoundIter=iterations;')
    s=s.replace('active=true;++iterations;', 'if(iterations%20==0)Require(lastBoundIter==iterations);active=true;++iterations;')
    s=s.replace('hasA=hasB=hasS=hasW=false;', 'hasA=hasB=hasS=false;if(fault==8)hasW=false;')
    # SDK evidence says bindings persist; counterfactual invalidation must be detected.
    p.write_text(s);results={}
    for v in POLICIES:
        cpp=tmp_path/(v+'.cpp');cpp.write_text('#include <iostream>\n#include "'+audit['variants'][v]['kernel']['path']+'"\n'+r'''
int main(int argc,char**argv){using namespace AscendC;
 std::vector<int8_t>a(40960),b(49807360);std::vector<uint64_t>s(389120);std::vector<half>parts(304*20*1024),out(311296);
 for(size_t t=0;t<304;++t)for(size_t g=0;g<20;++g)for(size_t n=0;n<1024;++n)parts[(t*20+g)*1024+n]=half((g+1)/64.0f);
 std::memset(out.data(),0x7f,622592);activationBase=a.data();weightBase=b.data();scaleBase=s.data();partialBase=parts.data();outputBase=out.data();
 fault=std::stoi(argv[1]);blockNum=std::stoi(argv[2]);uint32_t mode=argc>3?std::stoi(argv[3]):0;std::vector<uint8_t>workspace(blockNum*5120);workspaceBase=workspace.data();TestTiling td{239104,mode,blockNum,4096,4096,5120,0};
 try {for(blockIdx=0;blockIdx<blockNum;++blockIdx){ubBytes=0;d_flash_w8_a8_full_gate((uint8_t*)a.data(),(uint8_t*)b.data(),(uint8_t*)s.data(),(uint8_t*)out.data(),workspace.data(),(uint8_t*)&td);Require(ubBytes==253952);}}
 catch(const std::exception&){return 9;}
 bool matched=true;for(auto x:out)if(x!=half(mode?mode/64.0f:210.0f/64.0f))matched=false;
 std::cout<<bindings<<" "<<matched<<" "<<iterations<<" "<<ends<<" "<<copies<<" "<<barriers<<" "<<fetches<<" "<<sets<<" "<<waits<<" "<<guardReads<<"\n";
}
''')
        exe=tmp_path/v;r=subprocess.run([compiler,'-std=c++17','-O1','-I'+str(include),str(cpp),'-o',str(exe)],capture_output=True,text=True)
        assert r.returncode==0,r.stderr
        results[v]={}
        for blocks in (1,2,7):
            r=subprocess.run([str(exe),'0',str(blocks)],capture_output=True,text=True,check=True);results[v][blocks]=list(map(int,r.stdout.split()))
            assert results[v][blocks][0]==(304 if v.endswith('1') else 6080)
            assert results[v][blocks][1:5]==[1,6080,6080,19456]
        for mode in (1,20):
            partial=subprocess.run([str(exe),'0','7',str(mode)],capture_output=True,text=True,check=True)
            assert list(map(int,partial.stdout.split()))[1:5]==[1,6080,6080,19456]
        r=subprocess.run([str(exe),'8','1'],capture_output=True,text=True)
        assert r.returncode==(9 if v.endswith('1') else 0)
    for pair in ('A','B'):
        for blocks in (1,2,7):assert results[pair+'0'][blocks][1:]==results[pair+'1'][blocks][1:]

def test_equal_binary_is_reported_not_silently_credited():
    variants={v:dict(installed=dict(kernels=[dict(binary=dict(sha256='same'))])) for v in POLICIES}
    r=c.binary_summary(variants)
    assert all(x['exact_sha_equal'] and x['status']=='IDENTICAL_BINARY_NO_COMPILED_CHANGE' and x['optimization_success']=='NOT_ESTABLISHED' for x in r.values())
    variants['A1']['installed']['kernels'][0]['binary']['sha256']='different'
    assert c.binary_summary(variants)['A']['status']=='DIFFERENT_BINARY_NOT_PROOF_OF_INSTRUCTION_SAVING'

def test_resource_scope_accepts_two_limits_but_not_drift():
    rows=[dict(core_limit=i,plan={'blocks':i or 7},workspace_bytes=100+i,kernel_configuration='same') for i in (0,0,2,2)]
    same_resources(rows)
    rows[-1]['workspace_bytes']+=1
    with pytest.raises(ValueError):same_resources(rows)

@pytest.mark.parametrize('stage',['smoke','benchmark','profile'])
def test_driver_exact_correctness_order_and_separate_pairs(tmp_path,monkeypatch,stage):
    from types import SimpleNamespace
    import run_stage4f as driver
    args=SimpleNamespace(stage=stage,output_dir=tmp_path/'out',device=0,cann_root=tmp_path/'cann',stage4e_root=tmp_path/'e',
        stage4d_root=tmp_path/'d',stage4c_root=tmp_path/'c',stage4b_root=tmp_path/'b',
        pair_build_report=tmp_path/'build/report.json',smoke_report=tmp_path/'smoke/report.json',benchmark_report=tmp_path/'bench/report.json')
    for p in (args.pair_build_report,args.smoke_report,args.benchmark_report):p.parent.mkdir();p.write_text('{}')
    monkeypatch.setattr(driver,'sources',lambda:[])
    monkeypatch.setattr(driver,'evidence',lambda _:dict(stage4b={'opp':str(tmp_path/'opp')}))
    monkeypatch.setattr(driver,'sdk_binding',lambda _:dict(status='CPU_FIXTURE'))
    monkeypatch.setattr(driver.c4,'machine',lambda *a:{})
    monkeypatch.setattr(driver,'accept_build',lambda *a:{})
    monkeypatch.setattr(driver,'validate_build',lambda *a:None)
    monkeypatch.setattr(driver,'accepted_report',lambda *a:{'runs':[]})
    monkeypatch.setattr(driver,'collect',lambda *a:dict(status='RAW_OPERATOR_METRICS_FOUND',counter_rows=[]))
    calls=[]
    def invoke(args,folder,r,b,v,timed,profile=False,limit=0):
        calls.append((v,limit,timed,profile))
        return dict(status='SCOPED_GOLDEN_MATCH',variant=v,core_limit=limit,profiled=profile,formal_latency_evidence=timed and not profile,
            safety_protocol=POLICIES[v],timing={k:c.statistics_ms([4.]*30,30) for k in ('execute_sync','prepare')},
            plan={'blocks':limit or 7},kernel_configuration='same',workspace_bytes=100+limit,profiler={'sha256':'same'})
    monkeypatch.setattr(driver,'invoke',invoke)
    assert driver.main(args)==0
    if stage=='smoke':assert calls==[(v,l,False,False) for v,l in c.SMOKE_ORDER]
    if stage=='benchmark':assert calls==[(v,0,True,False) for row in c.ORDER for v in row]
    if stage=='profile':assert calls==[(v,0,True,True) for v in POLICIES]


def test_failed_smoke_blocks_all_timing(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import run_stage4f as driver
    args=SimpleNamespace(stage='benchmark',output_dir=tmp_path/'out',device=0,cann_root=tmp_path/'cann',stage4e_root=tmp_path/'e',
        stage4d_root=tmp_path/'d',stage4c_root=tmp_path/'c',stage4b_root=tmp_path/'b',pair_build_report=None,smoke_report=None,benchmark_report=None)
    monkeypatch.setattr(driver,'sources',lambda:[])
    monkeypatch.setattr(driver,'evidence',lambda _:dict(stage4b={'opp':str(tmp_path/'opp')}))
    monkeypatch.setattr(driver,'sdk_binding',lambda _:dict(status='CPU_FIXTURE'))
    monkeypatch.setattr(driver.c4,'machine',lambda *a:{})
    monkeypatch.setattr(driver,'accept_build',lambda *a:{})
    monkeypatch.setattr(driver,'invoke',lambda *a:pytest.fail('must not execute before accepted Smoke'))
    assert driver.main(args)==1
    r=json.loads((args.output_dir/'benchmark-report.json').read_text())
    assert r['npu']=='NOT_RUN' and r['performance']=='NOT_RUN'
