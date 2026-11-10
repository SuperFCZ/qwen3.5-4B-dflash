import copy
import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import sg_transform as tr
import sg_common as s
import sg_protocol as pr
import run_guard_ab as run
GUARD=0x5a39c671


def delta(words):
 result=0
 for x in words:result|=int(x)^GUARD
 return result


def test_every_word_bit_all_checkpoints_and_no_cancellation():
 counts={}
 for name,words,points in [('UB',384,3),('Arena',256,2)]:
  values=[GUARD]*words;count=0
  for checkpoint in range(points):
   for i in range(words):
    for bit in range(32):
     values[i]^=1<<bit
     assert delta(values)==1<<bit and any(x!=GUARD for x in values)
     values[i]^=1<<bit;count+=1
  assert delta(values)==0;counts[name]=count
 assert counts==dict(UB=36864,Arena=16384)
 values=[GUARD]*128;values[0]^=1;values[-1]^=1
 assert delta(values)==1 and ((values[0]^GUARD)^(values[-1]^GUARD))==0


def test_random_multibit_exact_equivalence():
 rng=np.random.default_rng(33)
 for _ in range(200):
  x=rng.integers(0,2**32,128,dtype=np.uint32)
  assert (delta(x)==0)==all(int(v)==GUARD for v in x)
 assert delta([GUARD]*128)==0


def test_source_single_variable_and_repository_binding(tmp_path):
 run.verify_repository()
 original=(tr.BASE/'op_kernel'/ (tr.OLD_S+'.cpp')).read_text()
 candidate=tr.expected_files('candidate')['op_kernel/'+tr.NAMES['candidate'][1]+'.cpp'].decode()
 normalized=candidate.replace(tr.NAMES['candidate'][1],tr.OLD_S)
 assert normalized=='#define STAGE4V_FAULT 0\n'+tr.aggregate(original)
 assert tr.once(normalized,tr.AGG,tr.SCAN)=='#define STAGE4V_FAULT 0\n'+original
 for kind in tr.NAMES:
  audit=tr.materialize(tmp_path/kind,kind,s.identity,s.write);tr.verify_audit(audit,s.verify)
  f=s.verify(audit['files'][0]);f.write_bytes(f.read_bytes()+b'\n')
  with pytest.raises(ValueError):tr.verify_audit(audit,s.verify)


@pytest.fixture(scope='module')
def kernels(tmp_path_factory):
 # Reuse the existing bounded CPU Matmul/buffer model. Execute the actual new
 # source with only the entry name normalized; never emulate a hardware event.
 root=tmp_path_factory.mktemp('arena-source');paths={}
 for fault,kind in enumerate(('candidate','fault')):
  src=root/f'{kind}.cpp';raw=(HERE/kind/'op_kernel'/(tr.NAMES[kind][1]+'.cpp')).read_text()
  src.write_text(raw.replace(tr.NAMES[kind][1],tr.OLD_S));paths[fault]=src
 spec=importlib.util.spec_from_file_location('_guard_ab_old_cpu',tr.BASE/'test/test_stage4v.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 code=inspect.getsource(module.kernel_exes.__wrapped__)
 code=code.replace("@pytest.fixture(scope='module')\n",'').replace('def kernel_exes(tmp_path_factory):','def reused_builder(tmp_path_factory):')
 anchor="str(HERE/'op_kernel/d_flash_w8_a8_full_k_channel.cpp')"
 assert code.count(anchor)==1;code=code.replace(anchor,'str(paths[fault])')
 code=code.replace("'V_S,MTE2_S,S_MTE3'", "'V_S,MTE2_S,S_MTE3,MTE3_MTE2,MTE3_S'")
 ns=dict(module.__dict__,paths=paths);exec(compile(code,'reused_stage4v_cpu_builder','exec'),ns)
 return ns['reused_builder'](tmp_path_factory)


@pytest.mark.parametrize('mode',list(range(2,8)))
def test_actual_fault_source_sweeps_and_aborts(kernels,tmp_path,mode):
 out=tmp_path/'out';cp=subprocess.run([str(kernels[1]),str(out),'1','0',str(mode),'0'],capture_output=True,text=True,timeout=120)
 assert cp.returncode==0,cp.stderr
 assert out.read_bytes()==pr.fault_expected(mode,0)
 assert int(cp.stdout)==(0 if mode in (3,6) else 1)


@pytest.mark.parametrize('blocks,tile',[(7,-1),(2,-1),(7,0)])
def test_actual_normal_source_numerics_layout_and_budget(kernels,tmp_path,blocks,tile):
 out=tmp_path/'out';cp=subprocess.run([str(kernels[0]),str(out),str(blocks),str(tile),'0','0'],capture_output=True,text=True)
 assert cp.returncode==0,cp.stderr
 actual=np.fromfile(out,'<f2').reshape(16,19456)
 factor=sum((k%5-2)*(k%3-1) for k in range(2560))
 for t in range(76):
  if tile>=0 and t!=tile:assert np.all(actual[:,t*256:(t+1)*256].view('<u2')==0x7f7f);continue
  dot=np.array([[factor*(m%3-1)*(t%3-1)*(n%5-2) for n in range(256)] for m in range(16)],np.int64)
  scale=np.array([1/(1<<(n%4+3)) for n in range(256)],'<f4')
  assert actual[:,t*256:(t+1)*256].tobytes()==(dot.astype('<f4')*scale).astype('<f2').tobytes()


@pytest.mark.parametrize('fault',[1,2])
def test_actual_normal_guard_and_arena_corruption_stops_output(kernels,tmp_path,fault):
 out=tmp_path/'out';subprocess.run([str(kernels[0]),str(out),'1','0','0',str(fault)],check=True,capture_output=True)
 assert out.read_bytes()==bytes([127])*622592


@pytest.mark.parametrize('phase',list(run.STATES))
def test_cli_gates(phase,tmp_path):
 order=['source','build','host','tile','full','fault'];needed=list(run.STATES).index(phase)
 argv=[phase,'--output-dir',str(tmp_path/'new')]
 if needed:
  with pytest.raises(SystemExit):run.parse(argv)
 for key in order[:needed]:argv+=['--'+key+'-report',str(tmp_path/key)]
 assert run.parse(argv).phase==phase


def test_paired_new_samples_threshold_and_fault_exclusion():
 rows=[]
 for i,order in enumerate(pr.ORDER):
  for j,name in enumerate(order):
   values=[1.1 if name=='baseline' else 1.]*30
   rows.append(dict(round=i,position=j,variant=name,profiled=False,validation=dict(status='BASELINE_BITS_AND_SAFETY_MATCH',plan=dict(total=16784384),timing={k:pr.stats(values,30) for k in ('execute_sync','prepare')})))
 result=pr.paired(rows);assert result['decision']=='GO_FURTHER_REVIEW' and result['statistics']['candidate']['execute_sync']['count']==90
 assert result['numeric_status']=='NUMERIC_MISMATCH_UNRESOLVED'
 low=copy.deepcopy(rows)
 for row in low:
  if row['variant']=='baseline':row['validation']['timing']['execute_sync']=pr.stats([1.02]*30,30)
 assert pr.paired(low)['decision']=='NO_GO_NO_MATERIAL_IMPROVEMENT'
 rows[-1]['variant']='fault'
 with pytest.raises(ValueError):pr.paired(rows)


def test_no_ub_or_sync_change_and_fault_not_timed():
 a=tr.aggregate((tr.BASE/'op_kernel'/(tr.OLD_S+'.cpp')).read_text())
 assert 'Cast(' not in a and 'Compare' not in a and 'Reduce' not in a
 assert a.count('InitBuffer(')==3
 assert a.count('mm.Iterate(false)')==1 and a.count('mm.GetTensorC(partial,0,true)')==1
 assert 'if(benchmark)' in (HERE/'fault/runner/runner.cpp').read_text()


@pytest.fixture(scope='module')
def candidate_runner(tmp_path_factory):
 spec=importlib.util.spec_from_file_location('_guard_ab_cpu_runner',tr.BASE/'test/test_stage4v.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 code=inspect.getsource(m.runner.__wrapped__).replace("@pytest.fixture(scope='module')\n",'').replace('def runner(tmp_path_factory):','def build_receiver(tmp_path_factory):')
 code=code.replace("str(HERE/'runner/runner.cpp')",'str(candidate_source)')
 code=code.replace(tr.OLD,tr.NAMES['candidate'][0]).replace(tr.OLD_S,tr.NAMES['candidate'][1])
 ns=dict(m.__dict__,candidate_source=HERE/'candidate/runner/runner.cpp');exec(compile(code,'reused_cpu_receiver','exec'),ns)
 return ns['build_receiver'](tmp_path_factory)


@pytest.mark.parametrize('problem',[None,'TEST_TIMED_LOG','TEST_OUTPUT_GUARD','TEST_READONLY'])
def test_actual_candidate_receiver_keeps_safety_and_timing(candidate_runner,tmp_path,problem):
 import os,json
 for name,size in [('a_i8.bin',40960),('b_i8.bin',49807360),('quant_u64_doc_candidate.bin',155648),('expected-0.bin',622592),('expected-1.bin',622592),('admitted_full.bin',622592)]:
  with (tmp_path/name).open('wb') as f:f.truncate(size)
 env=os.environ.copy();env['DFLASH_STAGE4V_EXPECTED_CORE_POLICY']='8,8,7,7,2002';env['FULLK_EXPLORE_WORKSPACE_BYTES']='39936' # CPU mock platform only
 if problem:env[problem]='1'
 with (tmp_path/'runner.log').open('w') as log:cp=subprocess.run([str(candidate_runner),'0',str(tmp_path),'0','0','-1','--benchmark'],stdout=log,stderr=subprocess.STDOUT,env=env)
 ex=json.loads((tmp_path/'execution.json').read_text())
 if problem:assert cp.returncode!=0 and ex['status']=='FAIL';return
 assert cp.returncode==0,(tmp_path/'runner.log').read_text()
 assert ex['op']==tr.NAMES['candidate'][0] and ex['guard_coverage_unchanged'] and ex['arena_detector']=='XOR_OR_U32'
 assert not ex['kernel_guards_unchanged'] and ex['status']=='EXPLORATORY_EXECUTION_SAFE'
 assert ex['prepare_calls']==38 and ex['verify_calls']==4 and len(ex['timing']['execute_sync']['samples_ms'])==30


def test_float_numeric_cast_would_miss_guard_low_bits():
 assert np.float32(GUARD)==np.float32(GUARD^1)
 assert delta([GUARD^1]+[GUARD]*127)!=0


def test_same_checkpoint_transient_cannot_be_hidden_at_tile_end():
 words=[GUARD]*128
 for checkpoint in range(2):
  words[127]^=1<<31
  assert delta(words)!=0 # failure must be returned at this site
  words[127]^=1<<31
  assert delta(words)==0 # later healthy scan does not erase the earlier failure
