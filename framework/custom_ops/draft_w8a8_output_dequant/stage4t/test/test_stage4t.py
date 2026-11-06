"""CPU evidence only: numerical protocol and explicitly weaker fault coverage."""
import copy,json,os,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import source_t as s
import t_common as c
import run_stage4t as run


def test_source_one_line_only_and_audit_tamper(tmp_path):
 old=(s.BASE/'op_kernel/d_flash_w8_a8_n256_gate.cpp').read_bytes()
 new=(HERE/'op_kernel/d_flash_w8_a8_n256_gate.cpp').read_bytes()
 assert new==s.transform(old) and old.count(s.OLD.encode())==1
 for token in (b'CheckGuard',b'PipeBarrier',b'SetFlag',b'WaitFlag',b'mm.End()',b'InitBuffer',b'SetLocalWorkspace',b'SetQuantVector',b'DataCopy'):
  assert new.count(token)==old.count(token)
 a=s.materialize(tmp_path/'src');s.verify_audit(a)
 assert a['coverage']=='WEAKER_GUARD_DIAGNOSTIC_ONLY'
 bad=copy.deepcopy(a);bad['coverage']='FULL'
 with pytest.raises(ValueError):s.verify_audit(bad)
 f=Path(a['files'][-1]['candidate']['path']);f.write_bytes(f.read_bytes()+b'\n')
 with pytest.raises(ValueError):s.verify_audit(a)
 with pytest.raises(ValueError):s.transform(new)

@pytest.fixture(scope='module')
def kernel_model(tmp_path_factory):
 sys.path.insert(0,str(s.BASE/'test'));import test_stage4p as p
 model=p.model.__wrapped__(tmp_path_factory);root,inc=model
 # Inject between the last Vector completion and GroupDelta; restore at next Iterate.
 # Pure CPU instrumentation, never included in OPPs. One group and one exact guard bit.
 h=inc/'kernel_operator.h';text=h.read_text()
 text=text.replace('struct TPipe {', '''inline std::vector<uint32_t*> guarded;
inline uint32_t* corrupt=nullptr;inline size_t groupEvents=0;inline bool injected=false;
inline int injectGroup=0,injectWord=0,injectBit=0,injectKind=0;
struct TPipe {''')
 text=text.replace('ubBytes+=bytes;', '''ubBytes+=bytes;
 for(size_t side=0;side<2;++side)for(size_t i=0;i<64;++i)guarded.push_back(reinterpret_cast<uint32_t*>(b.bytes.data())+(side?(bytes-256)/4:0)+i);''')
 text=text.replace('template<HardEvent>void WaitFlag(int){}', '''template<HardEvent E>void WaitFlag(int){
 if constexpr(E==HardEvent::V_S){if(iterations && groupEvents<iterations){
  groupEvents=iterations;
  if(injectKind && !injected && int(iterations-1)==injectGroup){corrupt=guarded.at(injectWord);*corrupt^=uint32_t(1)<<injectBit;injected=true;}
 }}
}''')
 h.write_text(text)
 h=inc/'lib/matmul_intf.h';text=h.read_text().replace('bool Iterate(bool acc){', 'bool Iterate(bool acc){if(injectKind==1&&corrupt){*corrupt^=uint32_t(1)<<injectBit;corrupt=nullptr;}')
 h.write_text(text)
 body=p.BODY.replace('fault=std::stoi(argv[5]);','fault=0;injectKind=std::stoi(argv[5]);injectGroup=std::stoi(argv[6]);injectWord=std::stoi(argv[7]);injectBit=std::stoi(argv[8]);')
 body=body.replace('Require(ubBytes==253952);','Require(ubBytes==253952);')
 body=body.replace('std::ofstream(argv[1]', 'std::cout<<iterations<<" "<<ends<<" "<<copies<<"\\n";\n std::ofstream(argv[1]')
 exes={}
 for name,file in [('baseline',s.BASE/'op_kernel/d_flash_w8_a8_n256_gate.cpp'),('candidate',HERE/'op_kernel/d_flash_w8_a8_n256_gate.cpp')]:
  src=root/(name+'.cpp');src.write_text('#include <iostream>\n#include "'+str(file)+'"\n'+body)
  out=root/name;cp=subprocess.run([shutil.which('c++'),'-std=c++17','-O2','-I'+str(inc),str(src),'-o',str(out)],capture_output=True,text=True)
  assert cp.returncode==0,cp.stderr;exes[name]=out
 return exes,p

@pytest.mark.parametrize('blocks,tile,mode',[(7,-1,0),(2,-1,0),(7,0,1),(7,37,20),(7,38,0),(7,75,20)])
def test_candidate_exact_output_and_schedule(kernel_model,tmp_path,blocks,tile,mode):
 exes,_=kernel_model;raw=[]
 for name,exe in exes.items():
  out=tmp_path/name;subprocess.run([str(exe),str(out),str(blocks),str(tile),str(mode),'0','0','0','0'],check=True,capture_output=True)
  raw.append(out.read_bytes())
 assert raw[0]==raw[1]
 assert len(raw[1])==622592

@pytest.mark.parametrize('group,kind',[(0,1),(9,1),(18,1),(19,1),(0,2),(19,2)])
def test_actual_cpp_transient_and_persistent_detection_timing(kernel_model,tmp_path,group,kind):
 exes,_=kernel_model
 for name,exe in exes.items():
  out=tmp_path/name
  cp=subprocess.run([str(exe),str(out),'1','0','0',str(kind),str(group),'639','31'],check=True,capture_output=True,text=True)
  iterations,ends,copies=map(int,cp.stdout.split())
  if name=='baseline':assert iterations==ends==group+1 and copies==0
  elif kind==1 and group<19:
   assert iterations==ends==20 and copies==256
   assert np.isfinite(np.frombuffer(out.read_bytes(),'<f2').reshape(16,19456)[:,:256]).all()
  else:assert iterations==ends==20 and copies==0
  if copies==0:assert out.read_bytes()==bytes([127])*622592


def test_exhaustive_single_bit_coverage_difference_cpu():
 # Pure boundary model: ALL640 positions x32 bits x20 groups. No claim of device fault sweep.
 words=np.full((640,32,640),0x5a39c671,dtype=np.uint32)
 for w in range(640):words[w,:,w]^=(np.uint32(1)<<np.arange(32,dtype=np.uint32))
 delta=np.bitwise_or.reduce(words^np.uint32(0x5a39c671),axis=-1)
 assert np.all(delta!=0)
 for g in range(20):
  full_detected_at=g
  weak_detected_at=19 # persistent corruption; no earlier group check
  assert full_detected_at<=weak_detected_at
  restored_before_tile_end=g<19
  if restored_before_tile_end:assert np.all(np.bitwise_or.reduce(words^words,axis=-1)==0)
 assert delta.size*20==409600

@pytest.fixture(scope='module',params=['baseline','candidate'])
def runner(request,tmp_path_factory):
 name=request.param;root=tmp_path_factory.mktemp(name);old=HERE.parent/'stage4k/test/N128';inc=root/'inc';shutil.copytree(old/'cpu_include',inc)
 text=(old/'cpu_api.cpp').read_text().replace('N128','N256').replace('n128','n256').replace('STAGE4J','STAGE4P');(root/'api.cpp').write_text(text)
 f=inc/'aclnn_d_flash_w8_a8_n128_gate.h';(inc/'aclnn_d_flash_w8_a8_n256_gate.h').write_text(f.read_text().replace('N128','N256'));f.unlink()
 out=root/'runner';cp=subprocess.run([shutil.which('c++'),'-std=c++17','-O1','-DSTAGE4T_DIAGNOSTIC='+str(int(name=='candidate')),'-I'+str(inc),'-I'+str(c.common.QUANT/'test'),str(HERE/'benchmark/runner.cpp'),str(root/'api.cpp'),'-o',str(out)],capture_output=True,text=True)
 assert cp.returncode==0,cp.stderr
 return name,out

@pytest.mark.parametrize('fault',[None,'TEST_BAD_GOLDEN','TEST_TIMED_LOG'])
def test_continuous_protocol_and_no_full_guard_label(runner,tmp_path,fault):
 name,exe=runner
 for file,size in [('a_i8.bin',40960),('b_i8.bin',49807360),('quant_u64_doc_candidate.bin',3112960),('golden_fp16.bin',622592)]:
  with (tmp_path/file).open('wb') as f:f.truncate(size)
 env=os.environ.copy();env['DFLASH_STAGE4P_EXPECTED_CORE_POLICY']='8,8,7,7,2002'
 if fault:env[fault]='1'
 with (tmp_path/'runner.log').open('w') as log:cp=subprocess.run([str(exe),'0',str(tmp_path),'0','--benchmark'],env=env,stdout=log,stderr=subprocess.STDOUT)
 ex=json.loads((tmp_path/'execution.json').read_text())
 if fault:assert cp.returncode!=0 and ex['status']=='FAIL';return
 assert cp.returncode==0 and ex['guard_coverage']==c.COVERAGE[name]
 assert len(c.check_execution(ex,0,name,True)['execute_sync']['samples_ms'])==30
 bad=dict(ex,guard_coverage=c.COVERAGE['candidate' if name=='baseline' else 'baseline'])
 with pytest.raises(ValueError):c.check_execution(bad,0,name,True)


def test_fail_closed_prior_and_diagnostic_only_comparison():
 rows=[dict(round=i,position=j,variant=n,status='FROZEN_GOLDEN_MATCH',profiled=False,formal_latency_evidence=True,plan={},workspace_bytes=123,
 timing={k:c.q.statistics_ms([2 if n=='baseline' else 1]*30,30) for k in ('execute_sync','prepare')}) for i,o in enumerate(c.ORDER) for j,n in enumerate(o)]
 result=c.comparison(rows)
 assert result['decision']=='DIAGNOSTIC_MEASUREMENT_ONLY_NOT_PRODUCTION' and result['net_implementation_delta_ms']==1
 with pytest.raises(ValueError):run.prior(None,'full',{}, {})
 rows[0]['profiled']=True
 with pytest.raises(ValueError):c.comparison(rows)
 assert 'faults' not in run.STATES
 assert 'DIAGNOSTIC' in run.STATES['full']

@pytest.mark.parametrize('stage',tuple(run.STATES))
def test_phase_cli_required_evidence(stage,tmp_path):
 args=[stage,'--output-dir',str(tmp_path/'out')]
 if stage!='build':
  with pytest.raises(SystemExit):run.parse(args)
  args+=['--build-report',str(tmp_path/'build.json')]
 if stage not in ('build','host-only'):
  with pytest.raises(SystemExit):run.parse(args)
  args+=['--prior-report',str(tmp_path/'prior.json')]
 parsed=run.parse(args)
 assert parsed.stage==stage


def test_bounded_prior_rejects_wrong_runner_before_numerical_admission(tmp_path):
 expected=tmp_path/'expected';expected.write_bytes(b'expected')
 other=tmp_path/'other';other.write_bytes(b'other')
 ident=c.identity(expected)
 r=dict(sources=[],evidence={},device=0,build_report={},guard_coverage=c.COVERAGE,machine=dict(os={},version_files=[]))
 old=dict(r,status=run.STATES['host-only'],phase='complete',runs=[dict(core_limit=1,mode=21,tile_id=-1,variant='candidate',runner=c.identity(other))])
 path=tmp_path/'host.json';c.write(path,old)
 b=dict(variants=dict(candidate=dict(build=dict(runner=ident))))
 with pytest.raises(ValueError,match='Runner binding'):run.prior(path,'host-only',r,b)
