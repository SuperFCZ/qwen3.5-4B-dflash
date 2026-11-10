import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import explore_numeric as n
import explore_common as c
import explore_protocol as p
import run_explore as run


def data(tile=0,repeat=0):
 x=np.full((16,19456),0x7f7f if repeat==0 else 0xffff,dtype='<u2')
 x[:,:256 if tile==0 else 19456]=0x3c00
 return x


def test_six_historical_locations_and_direction():
 g=data();a=g.copy()
 for m,k in n.KEYS:g[m,k]=0xbc01;a[m,k]=0xbc00
 r=n.observe(a.tobytes(),g.tobytes(),0,0);assert n.known_six(r).startswith('SIX_FROZEN')
 assert r['mismatch_count']==6 and r['max_ulp']==1
 r['differences'][0]['ulp']=2
 with pytest.raises(ValueError):n.known_six(r)


def test_only_numeric_difference_separates_from_safety():
 g=data(-1);a=g.copy();a[15,19455]=0x3bff
 r=n.observe(a.tobytes(),g.tobytes(),-1,0,a.tobytes())
 assert r['numeric_status']=='NUMERIC_MISMATCH_UNRESOLVED' and r['mismatch_count']==1
 assert r['active_elements']==311296 and r['differences'][0]['channel']==19455


@pytest.mark.parametrize('tile',[0,-1])
@pytest.mark.parametrize('bits',[0x7f7f,0xffff,0x7c00,0xfc00,0x7e01])
def test_nonfinite_and_unwritten_active_outputs_reject(tile,bits):
 g=data(tile);a=g.copy();a[3,9]=bits
 with pytest.raises(ValueError):n.observe(a.tobytes(),g.tobytes(),tile,0)


def test_inactive_scope_repeat_and_size():
 a=data();b=data(repeat=1)
 assert n.observe(b.tobytes(),b.tobytes(),0,1,a.tobytes())['mismatch_count']==0
 b[0,256]=0
 with pytest.raises(ValueError):n.observe(b.tobytes(),data(repeat=1).tobytes(),0,1)
 b=data(repeat=1);b[15,255]=0x3bff
 with pytest.raises(ValueError):n.observe(b.tobytes(),data(repeat=1).tobytes(),0,1,a.tobytes())
 with pytest.raises(ValueError):n.observe(a.tobytes()[:-2],a.tobytes(),0,0)


@pytest.fixture(scope='module')
def compiled(tmp_path_factory):
 root=tmp_path_factory.mktemp('cpp-numeric');src=root/'check.cpp';src.write_text('''
#include "output_contract.h"
#include <iterator>
std::vector<uint8_t> read(const char* p){std::ifstream f(p,std::ios::binary);return {std::istreambuf_iterator<char>(f),{}};}
int main(int argc,char**argv){try{auto a=read(argv[1]),g=read(argv[2]);std::vector<uint8_t> first;if(argc==7)first=read(argv[6]);auto r=fullk_explore::Inspect(a,g,first,std::stoll(argv[3]),std::stoul(argv[4]));fullk_explore::WriteNumeric(argv[5],r);return 0;}catch(...){return 3;}}
''')
 tool=shutil.which('c++');assert tool
 out=root/'check';subprocess.run([tool,'-std=c++17','-O2','-I',str(HERE/'runner'),str(src),'-o',str(out)],check=True,capture_output=True)
 return out


@pytest.mark.parametrize('tile',[0,-1])
def test_actual_cpp_numeric_contract_agrees_all_elements(compiled,tmp_path,tile):
 g=data(tile);a=g.copy()
 count=256 if tile==0 else 19456
 rng=np.random.default_rng(9)
 g[:,:count]=rng.integers(0,0x7c00,(16,count),dtype=np.uint16)
 # Include signs, subnormals and signed zero without creating nonfinite values.
 g[::2,:count]|=0x8000;a[:,:count]=g[:,:count]
 a[:,::127]=np.where((a[:,::127]&0x7fff)>0,a[:,::127]-1,a[:,::127])
 if tile==0:a[:,256:]=0x7f7f
 for name,z in [('a',a),('g',g)]: (tmp_path/name).write_bytes(z.tobytes())
 out=tmp_path/'numeric.json'
 result=subprocess.run([compiled,tmp_path/'a',tmp_path/'g',str(tile),str(0x7f7f),out],capture_output=True)
 assert result.returncode==0,result.stderr
 assert json.loads(out.read_text())==n.observe(a.tobytes(),g.tobytes(),tile,0)


def test_actual_cpp_rejects_poison_repeat_and_inactive(compiled,tmp_path):
 a=data();g=data()
 for name,z in [('g',g),('first',g)]: (tmp_path/name).write_bytes(z.tobytes())
 for coordinate,bits in [((0,0),0xffff),((0,256),0),((15,255),0x3bff)]:
  a=data();a[coordinate]=bits;(tmp_path/'a').write_bytes(a.tobytes())
  r=subprocess.run([compiled,tmp_path/'a',tmp_path/'g','0',str(0x7f7f),tmp_path/'out',tmp_path/'first'],capture_output=True)
  assert r.returncode!=0


@pytest.mark.parametrize('stage,flags',[
 ('source',[]),('build',['source']),('host',['source','build']),('tile0',['source','build','host']),('full',['source','build','host','tile']),('benchmark',['source','build','host','tile','full'])])
def test_cli_all_stages(stage,flags):
 argv=[stage,'--output-dir','/tmp/new']
 for name in flags:argv+=['--'+name+'-report','/tmp/'+name+'.json']
 assert run.parse(argv).phase==stage
 if flags:
  with pytest.raises(SystemExit):run.parse(argv[:-2])


def test_fresh_statistics_and_no_device_time_inference():
 rows=[]
 for i in range(3):
  t={k:n.stats([1.0+i/10]*30,30) for k in ('prepare','execute_sync')}
  rows.append(dict(round=i,tile=-1,profiled=False,validation=dict(status='EXPLORATORY_SAFETY_AND_DATA_COMPLETE',numeric_status='NUMERIC_MISMATCH_UNRESOLVED',workspace_bytes=16784384,timing=t)))
 r=p.summary(rows)
 assert r['status']=='EXPLORATORY_PERFORMANCE_MEASURED' and r['numeric_status']=='NUMERIC_MISMATCH_UNRESOLVED'
 assert r['statistics']['execute_sync']['count']==90 and r['statistics']['execute_sync']['median_ms']==1.1
 assert r['device_kernel_time'].startswith('UNAVAILABLE') and r['historical_reference_only']['fresh_paired'] is False
 rows[1]['profiled']=True
 with pytest.raises(ValueError):p.summary(rows)


@pytest.mark.parametrize('bad',[0,-1,True,float('inf'),float('nan')])
def test_bad_samples_reject(bad):
 with pytest.raises(ValueError):n.stats([bad]*30,30)


def test_no_kernel_build_or_install_and_normal_only_scope():
 assert not any((HERE/x).exists() for x in ('op_host','op_kernel','operator.json'))
 text=(HERE/'explore_common.py').read_text()
 assert 'msopgen' not in text and '--install-path' not in text
 runner=(HERE/'runner/runner.cpp').read_text()
 assert 'if(mode!=0||limit!=0||(tile!=0&&tile!=-1))' in runner
 assert 'out!=expected[repeatIndex]' not in runner
 assert 'fullk_explore::Inspect' in runner and 'CheckNamedGuards' in runner and 'read-only x/w changed' in runner
 assert 'EXPLORATORY_EXECUTION_SAFE' in runner and 'EXECUTION_COMPLETED_GOLDEN_MATCH' not in runner


def test_actual_continuous_scheduler_order_without_device(tmp_path):
 fake=tmp_path/'acl';fake.mkdir()
 (fake/'acl.h').write_text('''#pragma once
#include <cstddef>
constexpr int ACL_SUCCESS=0,ACL_MEM_MALLOC_HUGE_FIRST=0,ACL_MEMCPY_HOST_TO_DEVICE=0,ACL_MEMCPY_DEVICE_TO_HOST=1;
inline const char* aclGetRecentErrMsg(){return "MOCK";}
inline int aclrtFree(void*){return 1;}
inline int aclrtMalloc(void**,size_t,int){return 1;}
inline int aclrtMemset(void*,size_t,int,size_t){return 1;}
inline int aclrtMemcpy(void*,size_t,const void*,size_t,int){return 1;}
''')
 src=tmp_path/'schedule.cpp';src.write_text('''#include "runner_common.h"
#include <cassert>
using namespace draft_quant_test;
int main(){Timing t;t.continuous=true;t.warmup=5;t.repetitions=30;int prep=0,exec=0,poison=0,verify=0;
RunMeasuredCalls(t,[&](){++prep;},[&](){++exec;},[&](int){++poison;},[&](const std::string& file){++verify;if(verify==1)assert(exec==1&&file=="actual-0.bin");if(verify==2)assert(exec==2&&file=="actual-1.bin");if(verify==3)assert(exec==37&&poison==2&&file=="benchmark-last.bin");if(verify==4)assert(exec==38&&poison==3&&file=="postcheck.bin");});assert(prep==38&&exec==38&&verify==4&&poison==3&&t.execute.size()==30&&t.prepare.size()==30);}
''')
 exe=tmp_path/'schedule';subprocess.run(['c++','-std=c++17','-I',str(tmp_path),'-I',str(c.v.common.QUANT/'test'),str(src),'-o',str(exe)],check=True,capture_output=True);subprocess.run([exe],check=True)


def test_frozen_golden_only_failure_and_other_errors_stop():
 lines=[f'guard buffer={name} side={side} payload_bytes=1 status=INTACT changed_bytes=0' for name in ('a','b','scale','o','workspace') for side in ('prefix','suffix')]
 lines+=['FAIL: strict Stage4V Golden mismatch; no timing after failed precheck']
 log='\n'.join(lines)+'\n';ex=dict(status='FAIL',timing_complete=False,timing=dict(execute_sync=dict(samples_ms=[])))
 g=data();a=g.copy()
 for m,k in n.KEYS:g[m,k]=0xbc01;a[m,k]=0xbc00
 assert c.validate_historical_failure(log,ex,a.tobytes(),g.tobytes())['mismatch_count']==6
 for bad in (log+'FAIL: synchronize failed\n',log.replace('INTACT','CORRUPT',1),log.replace('guard buffer=workspace','guard buffer=missing'),log.replace('strict Stage4V Golden mismatch','read-only x/w changed')):
  with pytest.raises(ValueError):c.validate_historical_failure(bad,ex,a.tobytes(),g.tobytes())
 a[0,0]=0x3bff
 with pytest.raises(ValueError):c.validate_historical_failure(log,ex,a.tobytes(),g.tobytes())


@pytest.mark.parametrize('key,value',[('guards',False),('readonly',False),('repeat_equal',False),('cpu_fallback',True),('sync_completed',False),('finite_active_output',False),('kernel_guards_unchanged',False),('status','EXECUTION_COMPLETED_GOLDEN_MATCH'),('prepare_calls',37),('verify_calls',3)])
def test_execution_flags_reject_even_if_numeric_mismatch_allowed(tmp_path,monkeypatch,key,value):
 # Admission is mocked only to isolate the execution contract rejection, never to open a device gate.
 pl=dict(total=16784384,blocks=7,cores=7,core_policy={})
 monkeypatch.setattr(p.old_protocol,'plan',lambda *args:pl)
 proof=dict(workspace=16784384,t1={},policy={})
 adm=dict(status='ADMITTED_BEFORE_EXECUTION',workspace_bytes=16784384,declared_total=16784384,blocks=7,cores=7,core_policy={},host_proof_checked=True)
 ex=dict(status='EXPLORATORY_EXECUTION_SAFE',workspace_bytes=16784384,op='DFlashW8A8FullKChannel',soc='Ascend310P3',shape=[16,2560,19456],tile_count=76,group_count=1,guards=True,readonly=True,repeat_equal=True,repetitions=2,device=0,cpu_fallback=False,output_mode=0,core_limit=0,tile_id=-1,guard_coverage='FULL_K_TILE_384_WORD_THREE_BOUNDARIES',prepare_calls=38,verify_calls=4,poison_verified=True,poison_patterns=[127,255],timed_prepare_io=False,kernel_guards_unchanged=True,timed_log_checked=True,timed_log_stable=True,strict_golden_match=False,sync_completed=True,finite_active_output=True,inactive_poison_intact=True,active_coverage_checked=True,production='NOT_APPROVED',kernel_guard_observation='POISON_ON_FAILURE_AND_COMPLETE_FINITE_OUTPUT',reference_kind='CPU_FULL_K_REFERENCE')
 ex[key]=value
 for name,d in [('ex',ex),('adm',adm)]: (tmp_path/name).write_text(json.dumps(d))
 (tmp_path/'log').write_text('mock Host plan')
 row=dict(tile=-1,execution=c.identity(tmp_path/'ex'),admission=c.identity(tmp_path/'adm'),log=c.identity(tmp_path/'log'))
 with pytest.raises(ValueError,match='execution/safety'):p.validate(row,proof,0,True)


def test_benchmark_anchor_before_timing_and_no_golden_replacement():
 text=(HERE/'runner/runner.cpp').read_text()
 assert text.index('admitted=ReadBytes')<text.index('RunMeasuredCalls(')
 assert 'benchmark&&out!=admitted' in text
 assert text.index('WriteNumeric(')<text.index('benchmark&&out!=admitted')
 assert 'Inspect(out,expected[repeatIndex]' in text
 # Same shared schedule still calls check twice before warmup, tested above.


def test_wrong_private_api_resolution_rejected(tmp_path,monkeypatch):
 opp=tmp_path/'opp';opp.mkdir();bad=tmp_path/'elsewhere';bad.mkdir();lib=bad/'libcust_opapi.so';lib.write_bytes(b'not admitted')
 log=tmp_path/'ldd.log';log.write_text(f'libcust_opapi.so => {lib} (0x1)\n')
 proof=dict(normal=dict(opp=str(opp),installed=dict(api={},host={})))
 with pytest.raises(ValueError,match='outside normal OPP'):c.runtime(log,tmp_path,proof)


def test_predecessor_with_empty_runs_never_passes(tmp_path):
 r=dict(sources=[],evidence={},device=0,cann_root='/sdk',machine=dict(os={},version_files=[]))
 old=dict(r,status=run.STATES['full'],runs=[]);f=tmp_path/'report.json';f.write_text(json.dumps(old))
 with pytest.raises(ValueError,match='coverage incomplete'):run.prior(f,'full',r)
