import json
from pathlib import Path
import sys
import numpy as np
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import pair_evidence as e
import pair_contract as c
import run_pair as run


def rows():
 result=[]
 for i,order in enumerate(c.ORDER):
  for j,name in enumerate(order):
   values=[.8 if name=='FP16' else .5]*30
   v=dict(status='FP16_INDEPENDENT_REFERENCE_AND_SAFETY_MATCH' if name=='FP16' else 'BASELINE_BITS_AND_SAFETY_MATCH',workspace_bytes=0 if name=='FP16' else 16784384,core_count='UNAVAILABLE_INTERNAL_ACLNN' if name=='FP16' else 7,
    timing={k:c.stats(values,30) for k in ('prepare','execute_sync')})
   result.append(dict(round=i,position=j,scheme=name,profiled=False,validation=v))
 return result


def packing():return dict(layout=dict(permutation_ms=30.,prepare_wall_ms=90.))


def test_fresh_pair_order_counts_ratios_and_independent_workspaces():
 out=c.comparison(rows(),packing())
 assert out['status']=='BNZ_FP16_FRESH_PAIRED_MEASURED'
 assert out['fp16_over_w8a8_nz']==pytest.approx(1.6)
 assert out['w8a8_nz_over_fp16']==pytest.approx(.625)
 assert len(out['round_fp16_over_w8a8_nz'])==3 and out['historical_samples_used'] is False
 assert all(x['execute_sync']['count']==90 for x in out['statistics'].values())
 assert sum(x['execute_sync']['count'] for x in out['statistics'].values())==180
 assert out['workspace_by_scheme']==dict(FP16=0,W8A8_NZ=16784384)
 assert out['numeric_status_w8']=='NUMERIC_MISMATCH_UNRESOLVED'
 assert out['kernel_device_time'].startswith('UNAVAILABLE')


@pytest.mark.parametrize('change',['order','count','profile','safety','workspace','scheme','nan'])
def test_bad_measurements_reject(change):
 r=rows()
 if change=='order':r[0],r[1]=r[1],r[0]
 elif change=='count':r.pop()
 elif change=='profile':r[0]['profiled']=True
 elif change=='safety':r[0]['validation']['status']='UNVALIDATED'
 elif change=='workspace':r[0]['validation']['workspace_bytes']=100
 elif change=='scheme':r[0]['scheme']='N128'
 elif change=='nan':r[0]['validation']['timing']['execute_sync']['samples_ms'][0]=float('nan')
 with pytest.raises(ValueError):c.comparison(r,packing())


def test_no_positive_saving_no_fake_amortization():
 r=rows()
 for row in r:
  if row['scheme']=='W8A8_NZ':row['validation']['timing']['execute_sync']=c.stats([1.0]*30,30)
 result=c.comparison(r,packing())
 assert result['amortization_permutation']['break_even_calls'] is None
 assert result['production']=='NOT_APPROVED'


def test_numeric_contract_is_original_and_not_w8_bitwise():
 assert e.rr.POLICY==dict(element_atol=.002,element_rtol=.002,relative_l2_max=.001,finite_required=True,adaptive_tolerance=False)
 ref=np.ones((16,19456),dtype='<f8')
 assert e.rr.check(ref,ref.astype('<f2'))['status']=='INDEPENDENT_CPU_NUMERICS_PASS'
 for x in (ref*.99,ref+.01,ref*np.nan):
  with pytest.raises(ValueError):e.rr.check(ref,x)
 assert 'NUMERIC_MISMATCH_UNRESOLVED'==c.comparison(rows(),packing())['numeric_status_w8']


def test_actual_operator_name_comes_from_header_and_definition(tmp_path):
 opp=tmp_path/'normal/opp';opp.mkdir(parents=True)
 header=tmp_path/'api.h';header.write_text('int aclnnDFlashFullKBNzGetWorkspaceSize(); int aclnnDFlashFullKBNz();')
 definition=opp.parent/'operator.json';definition.write_text(json.dumps([dict(op='DFlashFullKBNz')]))
 installed=dict(header=e.identity(header),kernels=[],api={},host={})
 b=dict(variants=dict(candidate=dict(build=dict(opp=str(opp),installed=installed))))
 result=e.resolve_operator(b)
 assert result['op']=='DFlashFullKBNz' and result['execute_symbol']=='aclnnDFlashFullKBNz'
 definition.write_text(json.dumps([dict(op='DFlashFullKBnz')]))
 with pytest.raises(ValueError):e.resolve_operator(b)
 definition.write_text(json.dumps([dict(op='DFlashFullKBNz')]))
 header.write_text('int aclnnDFlashFullKBNzGetWorkspaceSize(); int aclnnDFlashFullKBNz(); int aclnnOtherGetWorkspaceSize();')
 installed['header']=e.identity(header)
 with pytest.raises(ValueError):e.resolve_operator(b)


@pytest.mark.parametrize('phase,needs',[('source',[]),('build',['source']),('smoke',['source','build']),('benchmark',['source','build','smoke'])])
def test_all_cli_gates(phase,needs,tmp_path):
 argv=[phase,'--output-dir',str(tmp_path/'new')]
 if needs:
  with pytest.raises(SystemExit):run.parse(argv)
 for key in needs:argv+=['--'+key+'-report',str(tmp_path/key)]
 assert run.parse(argv).phase==phase


def test_native_fresh_anchor_requires_independent_validation(tmp_path,monkeypatch):
 e0=dict(native=dict(build=dict(runner={})),native_extra='unused')
 with pytest.raises(ValueError):run.prior(None,'smoke',{})
 f=tmp_path/'report';f.write_text(json.dumps(dict(status='UNVALIDATED')))
 with pytest.raises(ValueError):run.prior(f,'smoke',dict(sources=[],evidence=e0,device=0,cann_root='/sdk'))


def test_normal_variant_and_physical_packing_required(tmp_path):
 bound=dict(bnz=dict(build=dict(variants=dict(candidate=dict(build=dict(runner={'path':'normal'})))),packing=dict(packed_b={}),t1={}))
 row=dict(runner={'path':'fault'},variant='fault',mode=2,limit=1,tile=0,profiled=False)
 with pytest.raises(ValueError):c.w8_validate(row,bound,0,True)


def test_no_new_kernel_runner_or_mutation_entrypoints():
 assert not any((HERE/n).exists() for n in ('op_host','op_kernel','operator.json','runner','CMakeLists.txt'))
 src=(HERE/'run_pair.py').read_text()
 assert 'msopgen' not in src and '--install-path' not in src
 assert 'build_native(' not in src
 assert not any('/stage4v_bnz_fp16_paired/' in x['path'] for x in e.nz.sources())


def test_native_uses_independent_reference_before_repeat_anchor(tmp_path):
 ref=tmp_path/'reference.bin';np.ones((16,19456),dtype='<f8').tofile(ref)
 runner=tmp_path/'runner';runner.write_bytes(b'CPU TEST ID ONLY')
 out=np.full((16,19456),np.float16(1.0009765625),'<f2')
 files=[]
 for i in range(2):
  f=tmp_path/f'actual-{i}.bin';out.tofile(f);files.append(e.identity(f))
 ex=dict(status='PASS',runtime='AscendCL native ACLNN',cpu_fallback=False,input_readonly=True,guards_intact=True,repetitions=2,m=16,k=2560,n=19456,device_id=0,op='aclnnMm',cube_math_type=0,dtype='float16',weight_layout='ND_KN',soc_name='Ascend310P3',independent_cpu_reference=True,poison_verified=True,timed_log_stable=True,prepare_calls=2,verify_calls=2,workspace_bytes=0,tracked_device_allocation_bytes=100,
 timing=dict(status='NOT_RUN',warmup=0,repetitions=0,execute_sync=dict(samples_ms=[]),prepare=dict(samples_ms=[])))
 f=tmp_path/'execution.json';f.write_text(json.dumps(ex))
 row=dict(scheme='FP16',runner=e.identity(runner),profiled=False,execution=e.identity(f),outputs=files)
 proof=dict(native=dict(build=dict(runner=e.identity(runner)),data=dict(files={'reference_f64.bin':e.identity(ref)})))
 result=c.native_validate(row,proof,0,False)
 assert result['numerics'][0]['relative_l2']<=.001
 bad=np.full((16,19456),1.01,'<f2');bad.tofile(tmp_path/'actual-0.bin');bad.tofile(tmp_path/'actual-1.bin')
 row['outputs']=[e.identity(tmp_path/f'actual-{i}.bin') for i in range(2)]
 with pytest.raises(ValueError):c.native_validate(row,proof,0,False)


def test_failure_preserves_files_and_cannot_produce_measurements(tmp_path,monkeypatch):
 class Args:device=0;cann_root=Path('/sdk')
 monkeypatch.setattr(run,'verify_build',lambda *args:None)
 def fail(*args,**kwargs):
  folder=args[1];folder.mkdir();(folder/'runner.log').write_text('CPU synthetic failure')
  raise RuntimeError('numerical admission failed')
 monkeypatch.setattr(e.rr,'invoke_native',fail)
 bound=dict(bnz=dict(evidence={},packing={},t1={}),native=dict(build={},data={}))
 r=dict(output_dir=str(tmp_path),evidence=bound)
 with pytest.raises(RuntimeError):run.invoke(Args(),tmp_path/'run',r,{},'FP16',False)
 assert 'validation' not in r['runs'][0]
 assert r['runs'][0]['retained_files'][0]['sha256']
 assert 'comparison' not in r
