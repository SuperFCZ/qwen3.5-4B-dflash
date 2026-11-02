"""CPU-only timing and actual source guard-lifetime fault injection."""
import copy,json,os,shutil,subprocess,sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import l_common as l
from profile_rows import collect

@pytest.fixture(scope='module',params=l.NAMES)
def runner(request,tmp_path_factory):
 n=request.param;width='N128' if n.startswith('N128') else 'N64';root=tmp_path_factory.mktemp(n);exe=root/'runner';stub=HERE.parent/'stage4k/test'/width
 p=subprocess.run([shutil.which('c++'),'-std=c++17','-O1','-I'+str(stub/'cpu_include'),'-I'+str(l.common.QUANT/'test'),str(HERE/n/'runner.cpp'),str(stub/'cpu_api.cpp'),'-o',str(exe)],capture_output=True,text=True)
 assert p.returncode==0,p.stderr
 return n,exe

@pytest.mark.parametrize('fault',[None,'TEST_BAD_GOLDEN','TEST_TIMED_LOG'])
def test_four_runner_protocol_and_fail_closed(runner,tmp_path,fault):
 n,exe=runner;folder=tmp_path/'case';folder.mkdir()
 for file,size in [('a_i8.bin',40960),('b_i8.bin',49807360),('quant_u64_doc_candidate.bin',3112960),('golden_fp16.bin',622592)]:
  with (folder/file).open('wb') as f:f.truncate(size)
 env=os.environ.copy();env['DFLASH_'+('STAGE4J' if n.startswith('N128') else 'STAGE4B')+'_EXPECTED_CORE_POLICY']='8,8,7,7,2002'
 if fault:env[fault]='1'
 with (folder/'runner.log').open('w') as log:p=subprocess.run([str(exe),'0',str(folder),'0','--benchmark'],stdout=log,stderr=subprocess.STDOUT,env=env)
 ex=json.loads((folder/'execution.json').read_text())
 if fault:
  assert p.returncode!=0 and ex['status']=='FAIL' and not (folder/'postcheck.bin').exists()
  if fault=='TEST_BAD_GOLDEN':assert ex['timing']['execute_sync']['samples_ms']==[]
 else:
  assert p.returncode==0,(folder/'runner.log').read_text()
  stats=l.check_execution(ex,0,n,True);assert len(stats['execute_sync']['samples_ms'])==30
  assert ex['guard_coverage']==l.POLICY[n] and ex['kernel_guards_unchanged']==n.endswith('Full')
  bad=copy.deepcopy(ex);bad['guard_coverage']='FULL_GROUP_GUARDS' if n.endswith('Weak') else 'DIAGNOSTIC_ONLY_WEAKER_GUARD_COVERAGE'
  with pytest.raises(ValueError):l.check_execution(bad,0,n,True)

def test_exact_source_transform_and_audit(tmp_path):
 raw=(HERE.parent/'stage4j/op_kernel/d_flash_w8_a8_n128_gate.cpp').read_bytes();weak=l.transform(raw)
 assert raw.count(b'CheckGuard(')-weak.count(b'CheckGuard(')==5
 for op in (b'mm.End()',b'PipeBarrier<PIPE_ALL>',b'PipeBarrier<PIPE_V>',b'V_S',b'SetQuantVector',b'SetLocalWorkspace'):
  assert raw.count(op)==weak.count(op)
 a=l.materialize(tmp_path/'src');l.verify_audit(a)
 p=Path(a['files'][0]['generated']['path']);p.write_bytes(p.read_bytes()+b'bad')
 with pytest.raises(ValueError):l.verify_audit(a)
 with pytest.raises(ValueError):l.transform(weak)

def test_balanced_contrasts_no_profile_pooling():
 values=dict(N64Full=12,N128Full=7,N64Weak=4,N128Weak=3)
 rows=[dict(round=i,position=j,variant=n,safety_protocol=l.POLICY[n],profiled=False,formal_latency_evidence=True,status='FROZEN_GOLDEN_MATCH',timing={key:l.statistics_ms([values[n]]*30,30) for key in ('execute_sync','prepare')}) for i,order in enumerate(l.ORDER) for j,n in enumerate(order)]
 for position in range(4):assert {order[position] for order in l.ORDER}==set(l.NAMES)
 assert len({(a,b) for order in l.ORDER for a,b in zip(order,order[1:])})==12
 r=l.comparison(rows);assert r['n64_guard_net_ms']==8 and r['n128_guard_net_ms']==4 and r['interaction_ms']==4
 assert len(r['statistics']['N128Weak']['execute_sync']['samples_ms'])==120
 rows[0]['profiled']=True
 with pytest.raises(ValueError):l.comparison(rows)

def test_csv_source_type_unit_duplicates_remain_separate(tmp_path):
 text='Op Name,Task Type,Task Id,Core Id,Task Duration(us),scalar_time(us)\nDFlashW8A8N128Gate,AI_CORE,4,,100,80\nDFlashW8A8N128Gate,AI_CORE,4,0,90,70\n'
 (tmp_path/'op_summary.csv').write_text(text);(tmp_path/'duplicate.csv').write_text(text)
 raw=collect(tmp_path,'DFlashW8A8N128Gate');rows=raw['row_review']['rows']
 assert len(rows)==4 and {r['row_type'] for r in rows}=={'TASK_TAGGED','CORE_TAGGED'}
 assert sum(r['duplicate_of'] is not None for r in rows)==2
 assert all(r['units']['scalar_time(us)']=='us' and r['aggregation']=='NOT_PERFORMED' for r in rows)
 assert not raw['formal_latency_evidence']

@pytest.mark.parametrize('weak,permanent,expected',[(False,False,False),(True,False,True),(True,True,False)])
def test_transient_between_groups_can_escape_weak_coverage(tmp_path,weak,permanent,expected):
 base=HERE.parent/'stage4j';inc=tmp_path/'include';shutil.copytree(base/'test/cpu_include',inc)
 p=inc/'lib/matmul_intf.h';s=p.read_text();s=s.replace('if(fault==6)registeredWorkspace[-1]=0;', 'if(g==3)w.data[-1]=0;\n  if(g==4)w.data[-1]=0x5a;'+('if(g==19)w.data[-1]=0;' if permanent else ''))
 p.write_text(s);raw=(base/'op_kernel/d_flash_w8_a8_n128_gate.cpp').read_bytes();(tmp_path/'kernel.cpp').write_bytes(l.transform(raw) if weak else raw)
 (tmp_path/'main.cpp').write_text(r'''
#include "kernel.cpp"
int main(){using namespace AscendC;std::vector<int8_t>a(40960),b(49807360);std::vector<uint64_t>s(389120);std::vector<half>parts(152*20*2048),out(311296);
std::vector<uint8_t>w(5120);activationBase=a.data();weightBase=b.data();scaleBase=s.data();partialBase=parts.data();outputBase=out.data();workspaceBase=w.data();blockIdx=0;blockNum=1;
TestTiling td{226816,0,0,1,4096,4096,5120,0};
d_flash_w8_a8_n128_gate((uint8_t*)a.data(),(uint8_t*)b.data(),(uint8_t*)s.data(),(uint8_t*)out.data(),w.data(),(uint8_t*)&td);
return copies==128?0:1;}
''')
 exe=tmp_path/'model';p=subprocess.run([shutil.which('c++'),'-std=c++17','-O1','-I'+str(inc),str(tmp_path/'main.cpp'),'-o',str(exe)],capture_output=True,text=True)
 assert p.returncode==0,p.stderr
 result=subprocess.run([str(exe)]);assert (result.returncode==0)==expected

def test_failed_or_incomplete_smoke_cannot_authorize_benchmark(tmp_path):
 import run_stage4l as run
 base=dict(sources=[],evidence={},active={},device=0,build_report={})
 p=tmp_path/'smoke.json'
 for status,rows in [('FAILED',[]),(run.STATES['smoke'],[])]:
  l.write(p,dict(sources=[],evidence={},device=0,build_report={},phase='complete',status=status,runs=rows))
  with pytest.raises(ValueError):run.prior(p,'smoke',base)
 with pytest.raises(ValueError):run.prior(None,'build',base)

def test_guard_mismatch_cannot_be_pooled_as_full():
 rows=[dict(round=i,position=j,variant=n,safety_protocol=l.POLICY[n],profiled=False,formal_latency_evidence=True,status='FROZEN_GOLDEN_MATCH',
 timing={key:l.statistics_ms([1]*30,30) for key in ('execute_sync','prepare')},plan={'width':128 if n.startswith('N128') else 64},workspace_bytes=100) for i,order in enumerate(l.ORDER) for j,n in enumerate(order)]
 bad=copy.deepcopy(rows);bad[2]['safety_protocol']='FULL_GROUP_GUARDS'
 with pytest.raises(ValueError):l.comparison(bad)
 bad=copy.deepcopy(rows);bad[-1]['workspace_bytes']=101
 with pytest.raises(ValueError):l.comparison(bad)

def test_reused_host_kernel_same_basename_have_separate_identity(tmp_path):
 src=tmp_path/'source';compiled=tmp_path/'compiled';records=[]
 for sub in ('op_host','op_kernel'):
  for base in (src,compiled):
   p=base/sub/'same.cpp';p.parent.mkdir(parents=True);p.write_text(sub)
  records.append(l.identity(compiled/sub/'same.cpp'))
 db=dict(source_audit=dict(variants=dict(B=dict(source_dir=str(src)))),variants=dict(B=dict(generated_sources=records)))
 l.verify_compiled_b(db)
 p=compiled/'op_host/same.cpp';p.write_text('wrong Host');db['variants']['B']['generated_sources'][0]=l.identity(p)
 with pytest.raises(ValueError,match='compiled source'):l.verify_compiled_b(db)
