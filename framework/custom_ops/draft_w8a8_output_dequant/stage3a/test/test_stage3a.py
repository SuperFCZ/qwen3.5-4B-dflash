"""CPU fixtures only; no device or Native OM execution."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import s3a_reference as ref
import s3a_evidence as evidence
import run_stage3a as run
import native_semantics as native
from w8_reference import nz_pack
from w8_common import identity,write


def arrays():
 x=((np.arange(16*2560)%63-31)/32).astype('<f2').reshape(16,2560);x[0,-1]=3
 nk=((np.arange(64*2560)*37)%256-128).astype('i1').reshape(64,2560)
 sw=np.linspace(.0003,.004,20*64).astype('<f2').reshape(20,64)
 return x,nk,sw


@pytest.fixture
def frozen(tmp_path):
 x,nk,sw=arrays();folder=tmp_path/'frozen';folder.mkdir();files={}
 for name,dt,shape in [('x.bin','<f2',(16,2560)),('q_nk.bin','i1',(19456,2560)),('w_nz.bin','i1',(80,1216,16,32)),('s_gn.bin','<f2',(20,19456))]:
  p=folder/name;a=np.memmap(p,mode='w+',dtype=dt,shape=shape)
  if name=='x.bin':a[:]=x
  elif name=='q_nk.bin':a[:64,:]=nk
  elif name=='w_nz.bin':
   for g in range(20):a[4*g:4*g+4,:4]=nz_pack(nk[:,128*g:128*(g+1)].T.copy())
  else:a[:,:64]=sw
  a.flush();del a;files[name]=identity(p)
 return files


def test_activation_computed_once_full_x_and_group_slices():
 x,nk,sw=arrays();x[:]=.5;x[0,-1]=127
 with patch.object(ref,'activation',wraps=ref.activation) as quant:
  aq,q,groups,refs,_=ref.references(x,nk,sw)
 assert quant.call_count==1 and q['scale']==1 and aq[0,0]==0 and aq[0,-1]==127
 assert len(groups)==20 and [g['index'] for g in groups]==list(range(20))
 for g in groups:
  lo,hi=g['k_range'];assert np.array_equal(g['a'],aq[:,lo:hi]);assert np.array_equal(g['b'],nk[:,lo:hi].T)
  assert np.array_equal(g['scale'],sw[g['index']].astype('<f4'))


def test_fp16_partial_rounding_is_not_used_in_ideal_or_effective_sum():
 x=np.zeros((16,2560),'<f2');w=np.zeros((64,2560),'i1');sc=np.ones((20,64),'<f2')
 x[:,:16]=127;x[:,16]=17;w[:,:17]=1 # 2049 -> half 2048
 x[:,128:144]=127;x[:,144]=16;w[:,128:145]=-1 # -2048
 _,_,groups,refs,_=ref.references(x,w,sc)
 assert (refs['ideal_f32']==1).all() and (refs['effective_f32']==1).all() and (refs['w8a16_f32']==1).all()
 assert (refs['partial_f32']==0).all() and (refs['partial_f16']==0).all()
 effect=ref.effect_report(refs)
 assert effect['scale_encoding']['fp32']['max_abs']==0 and effect['group_fp16_rounding']['fp32']['max_abs']==1


def test_prepare_all_nz_groups_and_reference_integrity(frozen,tmp_path):
 prepared=ref.prepare(tmp_path/'cpu',frozen);refs=ref.verify_prepared(prepared)
 assert len(prepared['cases'])==20 and prepared['group_order']==list(range(20))
 assert prepared['scale_error']['changed']>0
 case=json.loads(Path(prepared['cases'][19]['path']).read_text());assert case['k_range']==[2432,2560]
 p=Path(prepared['references']['effective_f32']['path']);data=bytearray(p.read_bytes());data[0]^=1;p.write_bytes(data)
 prepared['references']['effective_f32']=identity(p)
 with pytest.raises(ValueError,match='full-K reference mismatch'):ref.verify_prepared(prepared)


def test_nonzero_group_nz_corruption_blocks(frozen,tmp_path):
 p=Path(frozen['w_nz.bin']['path'])
 with p.open('r+b') as f:f.seek(4*11*1216*16*32);x=f.read(1);f.seek(-1,1);f.write(bytes([x[0]^1]))
 frozen['w_nz.bin']=identity(p)
 with pytest.raises(ValueError,match='group 11 NZ'):ref.prepare(tmp_path/'out',frozen)


def test_saved_partial_aggregation_checks_all_groups_and_order(tmp_path):
 _,_,groups,refs,_=ref.references(*arrays());cases=[]
 for g in groups:
  p=tmp_path/f'{g["index"]}.bin';g['partial_f16'].tofile(p)
  cases.append(dict(group_index=g['index'],outputs=[identity(p)]*2))
 sums,halves=ref.accumulate_saved(cases)
 assert all(s.tobytes()==refs['partial_f32'].tobytes() for s in sums)
 assert all(s.tobytes()==refs['partial_f16'].tobytes() for s in halves)
 for bad in (cases[:-1],list(reversed(cases)),cases[:19]+[cases[0]]):
  with pytest.raises(ValueError,match='20 unique ordered'):ref.accumulate_saved(bad)


def desc(shape):return {'name':['y'],'dtype':['DT_FLOAT16'],'shape':[{'dim':shape}],'layout':['ND']}


def graph():
 attrs=[{'key':[k],'value':[{('b' if type(v) is bool else 'i'):[v]}]} for k,v in dict(antiquant_group_size=128,inner_precise=0,transpose_x=False,transpose_weight=True).items()]
 return [dict(name='x',type='Data',inputs=[],output_desc=[desc([16,2560])]),
  dict(name='wq',type='WeightQuantBatchMatmulV2',inputs=['x:0','w:0','s:0','','',''],output_desc=[desc([16,19456])],attributes=attrs),
  dict(name='output',type='NetOutput',inputs=['wq:0'])]


def test_native_graph_requires_direct_linear_output_and_explicit_attributes():
 nodes=graph();assert native.pure_linear_graph(nodes,'wq')['output_shape']==[16,19456]
 changed=copy.deepcopy(nodes);changed[1]['inputs'][3]='bias:0'
 with pytest.raises(native.Mismatch,match='bias'):native.pure_linear_graph(changed,'wq')
 changed=copy.deepcopy(nodes);changed.insert(-1,dict(name='post',type='Silu',inputs=['wq:0']));changed[-1]['inputs']=['post:0']
 with pytest.raises(native.Mismatch,match='post-projection'):native.pure_linear_graph(changed,'wq')
 changed=copy.deepcopy(nodes);changed[1]['attributes']=[]
 with pytest.raises(native.Unknown,match='missing WeightQuant'):native.pure_linear_graph(changed,'wq')
 changed=copy.deepcopy(nodes);changed[1]['output_desc'][0]['shape'][0]['dim']=[16,64]
 with pytest.raises(native.Mismatch,match='not full-K'):native.pure_linear_graph(changed,'wq')


def test_native_missing_metadata_is_optional_not_candidate_failure(tmp_path):
 args=SimpleNamespace(a51_evidence=tmp_path/'missing.json',prefetch_evidence=tmp_path/'missing2.json')
 result=native.audit(args,{})
 assert result['native_semantics_status']=='UNRESOLVED' and result['native_comparison_status']=='SKIPPED'
 assert result['reason']=='NATIVE_COMPARISON_SKIPPED_SEMANTICS_UNRESOLVED'


def execution():
 return dict(status='EXECUTION_COMPLETED_NUMERICS_UNCHECKED',op='DFlashW8A8OutputDequant',soc='Ascend310P3',shape=[16,128,64],guards=True,readonly=True,
    repeat_equal=True,repetitions=2,device=0,cpu_fallback=False,layout='nd',input_count=3,poison_verified=True,poison_patterns=[127,255],
    timing=dict(status='NOT_RUN',warmup=0,repetitions=0))


def fake_runner(calls,bad_group=None,guard_group=None):
 def command(argv,log,env,r,label,timeout):
  calls.append(label);g=int(label.split('-')[1]);ex=execution()
  if guard_group==g:ex['guards']=False
  write(log.parent/'execution.json',ex)
  log.write_text('STAGE2B_PLAN arch=2002 M=16 K=128 N=64 blockDim=1 HostC=DT_FLOAT16 dequant=TENSOR single=16/64/128 base=16/64/128')
  data=bytearray((log.parent/'doc_candidate_fp16.bin').read_bytes())
  if bad_group==g:data[0]^=1
  for i in range(2):(log.parent/f'actual-{i}.bin').write_bytes(data)
 return command


def test_twenty_groups_run_and_numeric_failure_does_not_hide_later_groups(frozen,tmp_path):
 prepared=ref.prepare(tmp_path/'cpu',frozen);root=tmp_path/'npu';root.mkdir();runner=tmp_path/'runner';runner.write_text('CPU fixture')
 args=SimpleNamespace(device=0,cann_root=tmp_path/'sdk');b=dict(runner=identity(runner),opp=str(tmp_path/'opp'));r={};calls=[]
 with patch.object(run.e2c.stage2b,'command',side_effect=fake_runner(calls,bad_group=5)):
  run.execute_groups(args,root,r,b,prepared)
 assert len(calls)==20 and r['groups'][5]['status']=='FAIL_NUMERICAL'
 assert sum(g['status']=='PASS' for g in r['groups'])==19
 assert all(len(g['outputs'])==2 for g in r['groups'])


def test_guard_failure_stops_device_calls_not_reported_as_native_skip(frozen,tmp_path):
 prepared=ref.prepare(tmp_path/'cpu',frozen);root=tmp_path/'npu';root.mkdir();runner=tmp_path/'runner';runner.write_text('fixture')
 r={};calls=[]
 with patch.object(run.e2c.stage2b,'command',side_effect=fake_runner(calls,guard_group=1)):
  with pytest.raises(ValueError):run.execute_groups(SimpleNamespace(device=0,cann_root=tmp_path/'sdk'),root,r,dict(runner=identity(runner),opp=str(tmp_path/'opp')),prepared)
 assert len(calls)==2 and r['groups'][1]['status']=='FAIL_EXECUTION'
 assert all(g['status']=='NOT_RUN_AFTER_EXECUTION_FAILURE' for g in r['groups'][2:])


@pytest.mark.parametrize('native_state',['UNRESOLVED','MISMATCH'])
def test_native_audit_never_blocks_candidate_suite(frozen,tmp_path,native_state):
 old=tmp_path/'stage2c';case=ref.r2c.prepare(old,dict(files=frozen));runner=tmp_path/'runner';runner.write_text('fixture')
 b=dict(runner=identity(runner),opp=str(tmp_path/'opp'),installed={})
 proof=dict(input_files=frozen,case=identity(old/'case.json'))
 args=SimpleNamespace(output_dir=tmp_path/'out',stage2c_report=tmp_path/'c/report.json',stage2b_root=tmp_path/'b',cann_root=tmp_path/'sdk',
  a51_evidence=tmp_path/'a51/suite.json',prefetch_evidence=tmp_path/'pair/suite.json',device=0,prepare_only=False,cpu_report=None,
  build_report=tmp_path/'b/build/report.json',basic_report=tmp_path/'b/basic/report.json',zero_report=tmp_path/'b/zero/report.json',nonpower_report=tmp_path/'b/nonpower/report.json')
 calls=[]
 audit=dict(native_semantics_status=native_state,native_comparison_status='SKIPPED',reason='fixture optional Native evidence')
 with patch.object(run,'baseline',return_value=(b,proof,{})),patch.object(run,'audit',return_value=audit),\
      patch.object(run.e2c.stage2b,'installed',return_value={}),patch.object(run.e2c.stage2b,'command',side_effect=fake_runner(calls)):
  assert run.main(args)==0
 result=json.loads((args.output_dir/'report.json').read_text())
 assert result['status']=='STAGE3A_20_GROUP_MATCH' and result['summary']==dict(passed=20,failed=0,not_run=0)
 assert result['native_semantics_status']==native_state and result['native_comparison_status']=='SKIPPED'
 assert len(calls)==20 and result['aggregation']['repeat_equal']
 assert result['performance']=='NOT_RUN' and result['fused_kernel']=='NOT_IMPLEMENTED'


def test_native_compares_only_final_full_k_n64(tmp_path):
 p=tmp_path/'native.bin';gold=np.arange(16*19456,dtype=np.float32).reshape(16,19456)%100;gold.astype('<f2').tofile(p)
 n=dict(native_semantics_status='VERIFIED',golden=identity(p))
 result=run.native_compare(n,{'final':gold[:,:64].astype('<f2')})
 assert result['comparisons']['final']['bit_mismatches']==0
 assert run.native_compare({'native_semantics_status':'UNRESOLVED'}, {}) is None


def test_native_complete_metadata_graph_and_golden_chain(tmp_path):
 # All legacy artifact validators are injected CPU fixtures; no real Native claim.
 import a2_common,run_a2,fp16_gate_evidence
 root=tmp_path/'native';root.mkdir()
 def record(p):
  item=identity(p);return {**item,'path':p.name}
 node_attrs='\n'.join('attr { key: "'+k+'" value { '+('b: '+str(v).lower() if type(v) is bool else 'i: '+str(v))+' } }'
   for k,v in dict(antiquant_group_size=128,inner_precise=0,transpose_x=False,transpose_weight=True).items())
 pb=root/'dynamo.pbtxt';pb.write_text('''op { name: "x" type: "Data" output_desc { name:"y" dtype:DT_FLOAT16 shape { dim:16 dim:2560 } layout:"ND" } }
op { name:"wq" type:"WeightQuantBatchMatmulV2" input:"x:0" input:"w:0" input:"s:0"
 output_desc { name:"y" dtype:DT_FLOAT16 shape { dim:16 dim:19456 } layout:"ND" }
'''+node_attrs+''' }
op { name:"out" type:"NetOutput" input:"wq:0" }
''')
 af=root/'model.air';af.write_bytes(b'CPU fixture AIR');om=root/'model.om';om.write_bytes(b'CPU fixture OM')
 layout=dict(nodes=[dict(name='wq',weight='w:0',scale='s:0',group_size=128,scale_layout='GN',scale_shape=[20,19456],weight_shape=[19456,2560],weight_layout='NK',weight_format='FRACTAL_NZ',transpose_weight=True)])
 graph=dict(air=record(af),payload_files=[record(pb)],runtime_input_abi={'weight_quant_layout':layout},metadata={},input_names=['x'],output_names=['y'])
 air=root/'air.json';write(air,dict(status='PASS',graphs=[graph]))
 dep=root/'deployment.json';write(dep,dict(status='PASS',graphs=[{**graph,'atc_command':['atc','--precision_mode=must_keep_origin_dtype']}]))
 case=dict(name='layer-0-gate_up',layer=0,projection='gate_up',group_size=128)
 compiled=dict(name=case['name'],air_manifest=record(air),deployment=record(dep))
 nm=root/'native.json';write(nm,dict(status='PASS',cases=[compiled],bundle_sha256='fixture'))
 bundle=root/'bundle.json';write(bundle,dict(cases=[case]))
 gold=[]
 for i in range(4):
  p=root/f'gold-{i}.bin';p.write_bytes(bytes(622592));gold.append(identity(p))
 gs=root/'suite.json';write(gs,dict(cases=[dict(name=case['name'],executions={'native_om':dict(status='PASS',runtime='AscendCL native OM',io_validated=True,input_readonly=True,guards_intact=True,m=16,k=2560,n=19456,repetitions=2)},comparison={'custom_vs_native_om':dict(bitwise_equal=True,finite=True)})]))
 a51=root/'a51.json';a51.write_text('{}');pair=root/'pair.json';pair.write_text('{}')
 inputs={'fixture':identity(bundle)}
 source=dict(files=inputs,source=dict(bundle=str(bundle),native=str(nm),golden=str(gs)),identity=dict(golden=gold,native=dict(om=identity(om),deployment=identity(dep),air_manifest=identity(air))))
 args=SimpleNamespace(a51_evidence=a51,prefetch_evidence=pair)
 with patch.object(fp16_gate_evidence,'accepted',return_value=source),patch.object(a2_common,'load_bundle',return_value={'cases':[case]}),\
      patch.object(a2_common,'case_shape',return_value=(16,2560,19456)),patch.object(run_a2,'validate_om_manifest',side_effect=lambda *a,**k:dict(cases=[compiled],bundle_sha256='fixture')),\
      patch.object(a2_common,'validate_native_graph',return_value=None):
  result=native.audit(args,inputs)
  assert result['native_semantics_status']=='VERIFIED',result.get('error',result)
  assert all(v['status']=='PASS' for v in result['checks'].values())
  assert result['native_comparison_status']=='SKIPPED' # no accumulated NPU result yet
  graph['payload_files']=[];write(air,dict(status='PASS',graphs=[graph]))
  compiled['air_manifest']=record(air)
  result=native.audit(args,inputs)
  assert result['native_semantics_status']=='UNRESOLVED' and 'pbtxt' in result['error']


def test_cpu_only_then_npu_reuses_frozen_cpu_report(frozen,tmp_path):
 old=tmp_path/'stage2c';ref.r2c.prepare(old,dict(files=frozen));runner=tmp_path/'runner';runner.write_text('fixture')
 b=dict(runner=identity(runner),opp=str(tmp_path/'opp'),installed={});proof=dict(input_files=frozen,case=identity(old/'case.json'))
 args=SimpleNamespace(output_dir=tmp_path/'cpu-run',stage2c_report=tmp_path/'c/report.json',stage2b_root=tmp_path/'b',cann_root=tmp_path/'sdk',
  a51_evidence=tmp_path/'a51/suite.json',prefetch_evidence=tmp_path/'pair/suite.json',device=0,prepare_only=True,cpu_report=None,
  build_report=tmp_path/'b/build/report.json',basic_report=tmp_path/'b/basic/report.json',zero_report=tmp_path/'b/zero/report.json',nonpower_report=tmp_path/'b/nonpower/report.json')
 audit=dict(native_semantics_status='UNRESOLVED',native_comparison_status='SKIPPED',reason='fixture')
 with patch.object(run,'baseline',return_value=(b,proof,{})),patch.object(run,'audit',return_value=audit),patch.object(run.e2c.stage2b,'installed',return_value={}):
  with patch.object(run.e2c.stage2b,'command',side_effect=AssertionError('CPU-only must not launch')):assert run.main(args)==0
  cpu=args.output_dir/'report.json';before=identity(cpu)
  args.cpu_report=cpu;args.prepare_only=False;args.output_dir=tmp_path/'npu-run';calls=[]
  with patch.object(run.e2c.stage2b,'command',side_effect=fake_runner(calls)):assert run.main(args)==0
 assert len(calls)==20 and identity(cpu)==before
 r=json.loads((args.output_dir/'report.json').read_text());assert r['cpu_report']==before and r['summary']['passed']==20
 assert not (args.output_dir/'data').exists() # references were reused, not regenerated


def test_stage2c_candidate_proof_does_not_require_native_artifacts(frozen,tmp_path):
 old=tmp_path/'old2c';case=ref.r2c.prepare(old,dict(files=frozen));payload=(old/'doc_candidate_fp16.bin').read_bytes()
 outs=[]
 for i in range(2):p=old/f'actual-{i}.bin';p.write_bytes(payload);outs.append(identity(p))
 write(old/'execution.json',execution())
 proof={'fixture_stage2b':True};built={'fixture_build':True}
 absent=dict(path=str(tmp_path/'missing-native.om'),bytes=1,sha256='0'*64)
 f=dict(files=frozen,identity=dict(name='layer-0-gate_up',capture_files=frozen,native={'om':absent},golden=[absent]),full_k_golden=absent)
 r=dict(stage='2C',status='STAGE2C_REAL_K128_MATCH',phase='complete',shape=[16,128,64],original_shape=[16,2560,19456],
    npu='MATCHED_SINGLE_REAL_K128_ONLY',layer=0,projection='gate_up',group=0,k_range=[0,128],n_range=[0,64],device=0,
    codec_status=ref.CODEC,performance='NOT_RUN',sources=ref.e2c.sources(),stage2b_evidence=proof,case=identity(old/'case.json'),
    execution=identity(old/'execution.json'),outputs=outs,comparison=ref.r2c.compare(case,[payload]*2),frozen_evidence=f)
 rp=tmp_path/'2c-report.json';write(rp,r)
 with patch.object(evidence.e2c,'validated_stage2b',return_value=(built,proof)):
  b,p,n=evidence.baseline(SimpleNamespace(stage2c_report=rp,device=0))
 assert b==built and p['input_files']==frozen and n['full_k_golden']==absent


def test_missing_native_descriptor_is_unknown_not_invented():
 nodes=graph();nodes[1]['output_desc'][0].pop('layout')
 with pytest.raises(native.Unknown,match='dtype/layout'):native.pure_linear_graph(nodes,'wq')


def test_cli_has_no_build_or_performance_options(tmp_path):
 import subprocess
 subprocess.run(['bash','-n',str(HERE/'run.sh')],check=True)
 for flags in (['build'],['--repetitions','30'],['--group-count','10']):
  r=subprocess.run([sys.executable,'-B',str(HERE/'run_stage3a.py'),'--output-dir',str(tmp_path/'out'),*flags],capture_output=True,text=True)
  assert r.returncode==2
