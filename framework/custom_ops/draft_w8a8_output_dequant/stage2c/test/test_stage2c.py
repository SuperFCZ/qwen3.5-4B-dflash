"""CPU fixture regressions only. No Stage1/2B/2C device execution."""
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import s2c_evidence as evidence
import s2c_reference as ref
import run_stage2c as run
from s2b_common import identity,write,verify_tree,stage2a
from w8_reference import nz_pack


def arrays():
 x=((np.arange(16*2560)%63-31)/32).astype('<f2').reshape(16,2560);x[0,-1]=3
 nk=((np.arange(64*128)*37)%256-128).astype('i1').reshape(64,128)
 sw=np.linspace(.0003,.003,64).astype('<f2')
 return x,nk,sw


@pytest.fixture
def frozen(tmp_path):
 x,nk,sw=arrays();folder=tmp_path/'frozen';folder.mkdir();files={}
 for name,dt,shape in [('x.bin','<f2',(16,2560)),('q_nk.bin','i1',(19456,2560)),('w_nz.bin','i1',(80,1216,16,32)),('s_gn.bin','<f2',(20,19456))]:
  p=folder/name;a=np.memmap(p,mode='w+',dtype=dt,shape=shape)
  if name=='x.bin':a[:]=x
  elif name=='q_nk.bin':a[:64,:128]=nk
  elif name=='w_nz.bin':a[:4,:4]=nz_pack(nk.T.copy())
  else:a[0,:64]=sw
  a.flush();del a;files[name]=identity(p)
 sentinel=folder/'full-k-golden';sentinel.write_bytes(b'not a K128 reference; must never be sliced')
 return dict(files=files,watched={**files,'golden':identity(sentinel)},full_k_golden=identity(sentinel),identity={'CPU_FIXTURE_ONLY':True})


def test_global_x_scale_rne_clipping_and_zero():
 x,nk,sw=arrays();x[:]=0;x[0,-1]=127;x[0,:6]=[.5,1.5,2.5,-.5,-1.5,-2.5]
 before=[a.tobytes() for a in (x,nk,sw)];a,d=ref.derive(x,nk,sw)
 assert d['activation']['scale']==1 and a['a_i8.bin'][0,:6].tolist()==[0,2,2,0,-2,-2]
 assert d['activation']['scale_scope']==[16,2560] and d['activation']['clipped_ratio']==0
 assert [a.tobytes() for a in (x,nk,sw)]==before
 with pytest.raises(ValueError,match='full FP16'):ref.derive(x[:,:128],nk,sw)
 x[:]=0;a,d=ref.derive(x,nk,sw)
 assert d['activation']['scale']==1 and d['activation']['all_zero'] and not a['activation_full_i8.bin'].any()


def test_effective_weight_half_rounding_and_original_partial_reference():
 x,nk,sw=arrays();sw[0]=np.float16(1.0009765625)
 a,d=ref.derive(x,nk,sw)
 expected=b''.join(struct.pack('<e',int(nk[n,k])*float(sw[n])) for n in range(64) for k in range(128))
 assert a['weight_effective_fp16.bin'].tobytes()==expected
 x2=x.copy();x2[:,128:]=127
 b,_=ref.derive(x2,nk,sw)
 assert a['w8a16_k128_fp16.bin'].tobytes()==b['w8a16_k128_fp16.bin'].tobytes()
 assert a['a_i8.bin'].tobytes()!=b['a_i8.bin'].tobytes() # full-X max changes quantization
 assert not np.array_equal(a['ideal_fp16.bin'].view('<u2'),a['w8a16_k128_fp16.bin'].view('<u2'))


def test_scale_error_distribution_zero_denominators_and_codec_loss():
 r=ref.distribution(np.array([0,1,2.]),np.array([0,.5,2.]))
 assert r['zero_reference_count']==1 and r['changed']==1 and r['absolute']['max']==.5
 assert r['relative_on_nonzero_reference']['count']==2 and r['relative_on_nonzero_reference']['max']==.5
 a,d=ref.derive(*arrays())
 assert d['scale_error']['effective_vs_combined_fp32']['changed']>0
 assert d['scale_error']['effective_vs_combined_fp32']['signed_max']<=0
 assert len(d['scale_error']['per_channel'])==64
 assert np.array_equal(a['scale_f32.bin'],(d['activation']['scale']*arrays()[2].astype(np.float64)).astype('<f4'))


def test_prepare_preserves_frozen_identity_and_three_references(frozen,tmp_path):
 before=copy.deepcopy(frozen);case=ref.prepare(tmp_path/'data',frozen)
 a=ref.verify_case(case)
 assert case['files']['x_full_fp16.bin']['sha256']==frozen['files']['x.bin']['sha256']
 assert case['nz_roundtrip']=='BIT_EXACT' and case['group']==0
 result=ref.compare(case,[a['doc_candidate_fp16.bin'].tobytes()]*2)
 assert result['status']=='CANDIDATE_BITWISE_MATCH' and result['npu_vs_candidate'][0]['bit_mismatches']==0
 assert 'relative_l2_error' in result['npu_vs_w8a16_k128'][0]
 assert result['npu_vs_w8a16_k128'][0]['bit_mismatches']>0
 assert json.loads(json.dumps(result,allow_nan=False))==result
 verify_tree(before)


def test_one_bit_or_repeat_drift_cannot_pass(frozen,tmp_path):
 case=ref.prepare(tmp_path/'data',frozen);gold=Path(case['files']['doc_candidate_fp16.bin']['path']).read_bytes()
 bad=bytearray(gold);bad[0]^=1
 r=ref.compare(case,[bytes(bad)]*2)
 assert r['status']=='FAIL_NUMERICAL' and r['npu_vs_candidate'][0]['bit_mismatches']==1
 assert not ref.compare(case,[gold,bytes(bad)])['repeat_equal']


def test_mutated_golden_or_slice_cannot_be_rebound(frozen,tmp_path):
 case=ref.prepare(tmp_path/'data',frozen);path=Path(case['files']['doc_candidate_fp16.bin']['path'])
 data=bytearray(path.read_bytes());data[0]^=1;path.write_bytes(data)
 with pytest.raises(ValueError,match='identity'):ref.verify_case(case)
 case['files']['doc_candidate_fp16.bin']=identity(path)
 with pytest.raises(ValueError,match='model/reference mismatch'):ref.verify_case(case)


def test_bad_nz_inverse_blocks_before_execution(frozen,tmp_path):
 p=Path(frozen['files']['w_nz.bin']['path'])
 with p.open('r+b') as f:b=f.read(1);f.seek(0);f.write(bytes([b[0]^1]))
 frozen['files']['w_nz.bin']=identity(p)
 with pytest.raises(ValueError,match='NZ inverse'):ref.prepare(tmp_path/'data',frozen)


def ex():
 return dict(status='EXECUTION_COMPLETED_NUMERICS_UNCHECKED',op=evidence.common.NAME,soc='Ascend310P3',shape=[16,128,64],
  guards=True,readonly=True,repeat_equal=True,repetitions=2,device=0,cpu_fallback=False,layout='nd',input_count=3,
  poison_verified=True,poison_patterns=[127,255],timing=dict(status='NOT_RUN',warmup=0,repetitions=0))


@pytest.fixture
def stage2b(tmp_path):
 root=tmp_path/'stage2b';root.mkdir();cpu=tmp_path/'stage2a';stage2a.make_cases(cpu)
 upstream=dict(stage2a_manifest=identity(cpu/'manifest.json'),stage2a_case_files=evidence.common.checked_cases(cpu/'manifest.json'))
 cann=tmp_path/'sdk';pf=tmp_path/'pf.json';write(pf,dict(cann_root=str(cann)))
 upstream['stage1']=dict(preflight=identity(pf))
 runner=tmp_path/'accepted_runner';runner.write_text('CPU fixture, not executable')
 built=dict(status='STAGE2B_BUILT_NPU_NOT_RUN',phase='complete',settings=evidence.SETTINGS,codec_status=evidence.CODEC,
   sources=evidence.common.sources(),device=0,upstream=upstream,opp=str(tmp_path/'opp'),installed={},runner=identity(runner))
 (root/'build').mkdir();bp=root/'build/report.json';write(bp,built)
 for group,names in evidence.common.GROUPS.items():
  folder=root/group;folder.mkdir();cases=evidence.reference.prepare(folder/'data',upstream,group);entries=[]
  for case in cases:
   d=Path(case['files']['a_i8.bin']['path']).parent;write(d/'execution.json',ex());payload=(d/'doc_candidate_fp16.bin').read_bytes()
   outs=[]
   for i in range(2):p=d/f'actual-{i}.bin';p.write_bytes(payload);outs.append(identity(p))
   entries.append(dict(name=case['name'],case=identity(d/'case.json'),execution=identity(d/'execution.json'),outputs=outs,
      comparison=evidence.reference.compare(case,[payload]*2)))
  r=dict(status='STAGE2B_'+group.upper()+'_MATCH',phase='complete',group=group,build_report=identity(bp),device=0,
     upstream=upstream,settings=evidence.SETTINGS,sources=built['sources'],codec_status=evidence.CODEC,performance='NOT_RUN',cases=entries)
  if group!='basic':r['basic_proof']=identity(root/'basic/report.json')
  write(folder/'report.json',r)
 args=SimpleNamespace(build_report=bp,basic_report=root/'basic/report.json',zero_report=root/'zero/report.json',
   nonpower_report=root/'nonpower/report.json',device=0,cann_root=cann)
 return args,built


def test_reuse_validates_all_stage2b_groups_without_any_device_call(stage2b):
 args,built=stage2b
 with patch.object(evidence.stage2b,'installed',return_value={}),patch('subprocess.run',side_effect=AssertionError('must not rerun Stage1/2B')):
  actual,proof=evidence.validated_stage2b(args)
 assert actual==built and proof['stage1_rerun'] is False and proof['stage2b_rerun'] is False


@pytest.mark.parametrize('bad',['nonpower','zero','guard','device','build_identity'])
def test_invalid_stage2b_proof_rejected(stage2b,bad):
 args,built=stage2b
 path=args.nonpower_report if bad=='nonpower' else args.zero_report;r=json.loads(path.read_text())
 if bad in ('nonpower','zero'):r['status']='FAILED'
 elif bad=='guard':
  item=r['cases'][0]['execution'];p=Path(item['path']);value=json.loads(p.read_text());value['guards']=False;write(p,value)
  r['cases'][0]['execution']=identity(p)
 elif bad=='device':r['device']=1
 else:r['build_report']['sha256']='0'*64
 write(path,r)
 with patch.object(evidence.stage2b,'installed',return_value={}):
  with pytest.raises(ValueError):evidence.validated_stage2b(args)


def test_execute_uses_existing_runner_once_and_no_timing(frozen,tmp_path):
 root=tmp_path/'run';root.mkdir();case=ref.prepare(root/'data',frozen)
 runner=tmp_path/'runner';runner.write_text('CPU fixture only')
 built=dict(runner=identity(runner),opp=str(tmp_path/'opp'));args=SimpleNamespace(device=0,cann_root=tmp_path/'sdk');r={};calls=[]
 def command(argv,log,environment,report,label,timeout):
  calls.append(list(map(str,argv)));write(log.parent/'execution.json',ex())
  log.write_text('STAGE2B_PLAN arch=2002 M=16 K=128 N=64 blockDim=1 HostC=DT_FLOAT16 dequant=TENSOR single=16/64/128 base=16/64/128')
  for i in range(2):(log.parent/f'actual-{i}.bin').write_bytes(Path(case['files']['doc_candidate_fp16.bin']['path']).read_bytes())
 with patch.object(run.stage2b,'command',side_effect=command):run.execute(args,root,r,built,case)
 assert len(calls)==1 and calls[0]==[str(runner), '0',str(root/'execute')]
 assert r['status']=='STAGE2C_REAL_K128_MATCH'


def test_no_build_performance_or_scope_expansion_cli(tmp_path):
 subprocess.run(['bash','-n',str(HERE/'run.sh')],check=True)
 for extra in (['build'],['--group','1'],['--k','2560'],['--repetitions','30']):
  result=subprocess.run([sys.executable,'-B',str(HERE/'run_stage2c.py'),'--output-dir',str(tmp_path/'out'),*extra],capture_output=True,text=True)
  assert result.returncode==2


def test_prepare_only_never_calls_runner(frozen,tmp_path):
 args=SimpleNamespace(output_dir=tmp_path/'prepare',cann_root=tmp_path/'sdk',stage2b_root=tmp_path/'old2b',
   a51_evidence=tmp_path/'a51/suite.json',prefetch_evidence=tmp_path/'pair/suite.json',device=0,prepare_only=True,
   build_report=tmp_path/'old2b/build/report.json',basic_report=tmp_path/'old2b/basic/report.json',
   zero_report=tmp_path/'old2b/zero/report.json',nonpower_report=tmp_path/'old2b/nonpower/report.json')
 built={'opp':str(tmp_path/'old2b/opp'),'installed':{}}
 with patch.object(run,'validated_stage2b',return_value=(built,{'fixture_only':True})),\
      patch.object(run,'frozen_source',return_value=frozen),patch.object(run.stage2b,'installed',return_value={}),\
      patch.object(run.stage2b,'command',side_effect=AssertionError('prepare-only must not execute')) as command:
  assert run.main(args)==0
 assert command.call_count==0
 r=json.loads((args.output_dir/'report.json').read_text())
 assert r['status']=='CPU_REFERENCES_READY_NPU_NOT_RUN' and r['npu']=='NOT_RUN'
 assert r['stage1_rerun']==r['stage2b_rerun']==False and r['performance']=='NOT_RUN'


def test_guard_failure_cannot_be_numeric_success(frozen,tmp_path):
 root=tmp_path/'run';root.mkdir();case=ref.prepare(root/'data',frozen)
 runner=tmp_path/'runner';runner.write_text('fixture')
 built=dict(runner=identity(runner),opp=str(tmp_path/'opp'));args=SimpleNamespace(device=0,cann_root=tmp_path/'sdk')
 def command(argv,log,environment,r,label,timeout):
  bad=ex();bad['guards']=False;write(log.parent/'execution.json',bad);log.write_text('guard failed')
 with patch.object(run.stage2b,'command',side_effect=command):
  with pytest.raises(ValueError,match='execution/guard'):run.execute(args,root,{},built,case)
