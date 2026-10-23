"""CPU fixtures do not constitute SDK or device compatibility evidence."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import w8_common as common
import w8_reference as ref
import w8_sdk as sdk
import w8_run as run

# Test adapter only; no CANN encoding/default is asserted by this fixture.
CODEC=dict(float_field='low32_ieee_float32',constant_u64=0,stored_mantissa_bits=23,
 effective_mantissa_bits=23,coefficient_rounding='truncate_low_mantissa_bits',
 product_rounding='fp32_then_fp16_rne',overflow='ieee',subnormal='preserve',
 sign='positive_finite_scale_signed_accumulator',vector_storage='64_contiguous_u64')


def test_integer_oracle_nz_and_signed_boundaries():
 a=np.full((16,128),-128,dtype='i1');b=np.full((128,64),-128,dtype='i1')
 assert (ref.dot(a,b)==2097152).all()
 b[:,1]=127;assert (ref.dot(a,b)[:,1]==-2080768).all()
 b=((np.arange(8192)*73)%256-128).astype('i1').reshape(128,64)
 nz=ref.nz_pack(b)
 assert nz.shape==(4,4,16,32)
 for k in (0,31,32,127):
  for n in (0,15,16,63):assert nz[k//32,n//16,n%16,k%32]==b[k,n]
 with pytest.raises(ValueError):ref.dot(a.astype('f2'),b)


def test_activation_entire_x_rne_zero_and_readonly():
 x=np.zeros((16,2560),dtype='<f2');x[0,-1]=127
 x[0,:6]=[.5,1.5,2.5,-.5,-1.5,-2.5];before=x.tobytes()
 q,scale,info=ref.activation(x)
 assert scale==1 and q[0,:6].tolist()==[0,2,2,0,-2,-2]
 assert q[0,-1]==127 and info['scale_scope']==[16,2560] and info['clipped_ratio']==0
 assert before==x.tobytes()
 x[:]=0;q,scale,info=ref.activation(x)
 assert scale==1 and info['all_zero'] and not q.any()
 with pytest.raises(ValueError):ref.activation(x[:,:128])
 x[0,0]=np.inf
 with pytest.raises(ValueError):ref.activation(x)


def test_reviewed_encoding_precision_and_conversion_are_distinct():
 scale=np.full(64,1.0009,np.float32)
 c={**CODEC,'effective_mantissa_bits':10}
 words,effective=ref.encode_scales(scale,c)
 assert np.array_equal(words.astype('<u4'),scale.view('<u4'))
 assert (effective==1).all()
 acc=np.full((16,64),1000,np.int64)
 assert not np.array_equal(ref.convert(acc,effective,c),ref.convert(acc,scale,c))
 c={**c,'stored_mantissa_bits':10,'constant_u64':1<<40}
 words,_=ref.encode_scales(scale,c);assert (words>>40==1).all()
 assert (words.astype('<u4')==np.float32(1).view('<u4')).all()
 for bad in ({},{**CODEC,'constant_u64':1},{**CODEC,'effective_mantissa_bits':24}):
  with pytest.raises(ValueError):ref.encode_scales(scale,bad)
 for value in (0,-1,np.inf,np.nan):
  with pytest.raises(ValueError):ref.encode_scales(np.full(64,value),CODEC)
 a=np.ones((16,64),np.int64)
 assert (ref.convert(a,np.full(64,2**-24),CODEC).view('<u2')==1).all()
 assert (ref.convert(a,np.full(64,2**-24),{**CODEC,'subnormal':'flush_signed_zero'}).view('<u2')==0).all()
 assert (ref.convert(a,np.full(64,131008),CODEC).view('<u2')==0x7c00).all()
 assert (ref.convert(a,np.full(64,131008),{**CODEC,'overflow':'saturate_finite'}).view('<u2')==0x7bff).all()


def outputs(case):
 raw=np.fromfile(case['files']['acc_int64.bin']['path'],'<i8').astype('<i4').tobytes()
 quant=Path(case['files']['encoded_ref.bin']['path']).read_bytes()
 return [raw]*2,[quant]*2


def test_eight_cases_strict_bits_repeat_and_cross_semantics(tmp_path):
 cases=ref.synthetic(tmp_path/'data',CODEC)
 assert len({c['name'] for c in cases})==8
 for c in cases:
  raw,quant=outputs(c);r=ref.compare_case(c,raw,quant)
  assert r['status']=='W8A8_REFERENCE_MATCH'
  assert json.loads(json.dumps(r,allow_nan=False))==r
 c=cases[0];raw,quant=outputs(c);bad=bytearray(quant[0]);bad[0]^=1
 r=ref.compare_case(c,raw,[bytes(bad)]*2)
 assert r['status']=='FAIL_NUMERICAL' and r['encoded'][0]['bit_mismatches']==1
 assert json.loads(json.dumps(r))==r
 assert ref.compare_case(c,raw,[quant[0],bytes(bad)])['repeat_equal'] is False
 bad=bytearray(raw[0]);bad[0]^=1
 assert ref.compare_case(c,[bytes(bad)]*2,quant)['accumulator'][0]['mismatches']==1
 # A different W8A16 reference never changes this W8A8 gate.
 f=tmp_path/'w8a16.bin';np.full((16,64),3,dtype='<f2').tofile(f)
 c['files']['w8a16_k128_ref.bin']=common.identity(f)
 r=ref.compare_case(c,raw,quant)
 assert r['status']=='W8A8_REFERENCE_MATCH' and r['w8a8_vs_w8a16']['npu'][0]['bit_mismatches']>0


def audit_fixture(tmp_path):
 cann=tmp_path/'sdk';cann.mkdir();(cann/'version.info').write_text('Version=9.0.0\n')
 (cann/'fixture.h').write_text('// CPU citation fixture only, not hardware evidence\n')
 out=tmp_path/'audit';out.mkdir()
 with patch.object(sdk.shutil,'which',return_value=None):p=sdk.preflight(cann,out,[Path('fixture.h')])
 assert p['status']=='UNRESOLVED'
 r=json.loads((out/'review-template.json').read_text());f=p['files'][-1]
 citation=dict(path=f['source']['path'],sha256=f['source']['sha256'],start=1,end=1,
               quote='// CPU citation fixture only, not hardware evidence')
 for claim in [*r['common'].values(),r['layouts']['nd']]:
  claim.update(status='SUPPORTED',citations=[citation])
 r.update(status='SOURCE_REVIEWED',reviewer='CPU fixture only',settings={'host_scaled_c_dtype':'DT_INT32','codec':CODEC})
 review=tmp_path/'review.json';common.write(review,r)
 return out/'preflight.json',review,r,cann


def test_sdk_claims_no_automatic_support_tampering_and_layout_gate(tmp_path):
 p,r,review,cann=audit_fixture(tmp_path)
 sdk.reviewed(p,r,'nd')
 with pytest.raises(ValueError,match='UNRESOLVED: nz'):sdk.reviewed(p,r,'nz')
 review['common']['integer_cube']['status']='UNSUPPORTED';common.write(r,review)
 with pytest.raises(ValueError,match='BLOCKED_UNSUPPORTED'):sdk.reviewed(p,r,'nd')
 review['common']['integer_cube']['status']='SUPPORTED'
 review['common']['integer_cube']['citations'][0]['quote']='invented';common.write(r,review)
 with pytest.raises(ValueError,match='quote differs'):sdk.reviewed(p,r,'nd')
 (cann/'fixture.h').write_text('changed')
 with pytest.raises(ValueError,match='identity changed'):sdk.reviewed(p,r,'nd')


def test_fixed_source_collection_missing_extra_and_escape(tmp_path):
 cann=tmp_path/'sdk';cann.mkdir();(cann/'version.info').write_text('9.0.0')
 out=tmp_path/'audit';out.mkdir();outside=tmp_path/'outside';outside.write_text('private')
 (cann/'escape.h').symlink_to(outside)
 with patch.object(sdk.shutil,'which',return_value=None):p=sdk.preflight(cann,out,[Path('escape.h')])
 assert p['status']=='BLOCKED_PREFLIGHT' and 'escapes' in p['error']
 assert 'rglob' not in (ROOT/'w8_sdk.py').read_text()


def execution(mode='raw'):
 return dict(status='EXECUTION_COMPLETED_NUMERICS_UNCHECKED',op=common.OPS[mode][0],soc='Ascend310P3',
  shape=[16,128,64],guards=True,readonly=True,repeat_equal=True,repetitions=2,device=0,cpu_fallback=False,
  layout='nd',raw_accumulator=mode=='raw',timing=dict(status='NOT_RUN',warmup=0,repetitions=0))


def test_execution_gates_and_prior_stage_cannot_be_fabricated(tmp_path):
 run.check_execution(execution(),'raw','nd',0)
 for k,v in [('guards',False),('readonly',False),('raw_accumulator',False),('repeat_equal',False),
             ('shape',[16,256,64]),('timing',dict(status='PASS',warmup=5,repetitions=30))]:
  ex=execution();ex[k]=v
  with pytest.raises(ValueError):run.check_execution(ex,'raw','nd',0)
 with pytest.raises(ValueError,match='required prior'):run.require_proof(None,'SYNTHETIC_ND_PASS',{})
 p=tmp_path/'proof.json';common.write(p,dict(status='SYNTHETIC_ND_PASS',phase='complete',settings={},dataset='synthetic',cases=[]))
 with pytest.raises(ValueError,match='all unique'):run.require_proof(p,'SYNTHETIC_ND_PASS',{})


def test_reference_and_input_sha_rejected_after_mutation(tmp_path):
 c=ref.synthetic(tmp_path/'data',CODEC)[0];raw,quant=outputs(c)
 Path(c['files']['a.bin']['path']).write_bytes(bytes(2048))
 with pytest.raises(ValueError,match='identity changed'):ref.compare_case(c,raw,quant)


def test_raw_failure_stops_before_quant(tmp_path):
 built=dict(status='BUILT_NOT_NPU_VERIFIED',cann_compile='BUILD_COMPLETED_NPU_NOT_RUN',layout='nd',settings={'codec':CODEC},preflight={},review={},opp=str(tmp_path/'opp'),installed={})
 p=tmp_path/'build.json';common.write(p,built)
 args=SimpleNamespace(build_report=p,layout='nd',dataset='synthetic')
 r=dict(settings={'codec':CODEC},preflight={},review={});modes=[]
 def invoke(case,mode,root,b,args):
  modes.append(mode);path=tmp_path/'incorrect_raw';path.write_bytes(bytes(4096))
  return dict(outputs=[common.identity(path)]*2)
 with patch.object(run,'installed',return_value={}),patch.object(run,'invoke',side_effect=invoke):
  with pytest.raises(ValueError,match='accumulator mismatch'):run.smoke(args,tmp_path,r)
 assert modes==['raw']


@pytest.mark.parametrize("raw_only",[False,True])
def test_cpp_runner_syntax_only(tmp_path,raw_only):
 compiler=shutil.which('c++')
 if not compiler:pytest.skip('no local C++ compiler')
 (tmp_path/'acl').mkdir()
 (tmp_path/'acl/acl.h').write_text('''#pragma once
#include <cstddef>
#include <cstdint>
using aclrtStream=void*;using aclDataType=int;using aclFormat=int;
constexpr int ACL_SUCCESS=0,ACL_MEM_MALLOC_HUGE_FIRST=0,ACL_MEMCPY_HOST_TO_DEVICE=1,ACL_MEMCPY_DEVICE_TO_HOST=2,ACL_FLOAT16=1,ACL_INT8=2,ACL_INT32=3,ACL_UINT64=4,ACL_FORMAT_ND=2,ACL_FORMAT_FRACTAL_NZ=29;
const char* aclGetRecentErrMsg();const char* aclrtGetSocName();
int aclrtMalloc(void**,size_t,int);int aclrtFree(void*);int aclrtMemset(void*,size_t,int,size_t);int aclrtMemcpy(void*,size_t,const void*,size_t,int);
int aclInit(const char*);int aclFinalize();int aclrtSetDevice(int);int aclrtResetDevice(int);int aclrtCreateStream(aclrtStream*);int aclrtDestroyStream(aclrtStream);int aclrtSynchronizeStream(aclrtStream);
''')
 decl='''#pragma once
struct aclTensor;struct aclOpExecutor;
aclTensor* aclCreateTensor(const int64_t*,uint64_t,int,const int64_t*,int64_t,int,const int64_t*,uint64_t,void*);
int aclDestroyTensor(const aclTensor*);
'''
 inputs='const aclTensor*,const aclTensor*,'+('' if raw_only else 'const aclTensor*,')
 for op,snake in (dict(raw=common.OPS['raw']) if raw_only else common.OPS).values():
  (tmp_path/('aclnn_'+snake+'.h')).write_text(decl+f'''int aclnn{op}GetWorkspaceSize({inputs}aclTensor*,uint64_t*,aclOpExecutor**);
int aclnn{op}(void*,uint64_t,aclOpExecutor*,aclrtStream);
''')
 result=subprocess.run([compiler,'-std=c++17','-fsyntax-only','-I'+str(tmp_path),'-I'+str(common.QUANT/'test'),str(ROOT/('test/raw_runner.cpp' if raw_only else 'test/runner.cpp'))],capture_output=True,text=True)
 assert result.returncode==0,result.stderr


def test_independent_kernel_no_weight_dequant_or_timing():
 source=(ROOT/'op_kernel/kernel_common.h').read_text()
 assert 'SetQuantVector(scaleGm)' in source and 'int8_t' in source and 'int32_t' in source
 assert not run.re.search(r'\b(?:Cast|Brcb|Mul|Muls|AscendAntiQuant|SetAntiQuantVector)\s*[<(]',source)
 assert 'W8A8_B_NZ' in source
 source=(ROOT/'test/runner.cpp').read_text()
 assert 'RunMeasuredCalls' in source and 'CheckNamedGuards' in source and 'Timing timing;' in source
 assert 'r.s.buffer.CopyOut()!=s' in source
 subprocess.run(['bash','-n',str(ROOT/'run.sh')],check=True)


def test_frozen_slice_uses_whole_activation_and_never_full_k_golden(tmp_path):
 sys.path.insert(0,str(common.QUANT/'fp16_compare'))
 import fp16_gate_evidence
 files={}
 specs={'x.bin':('<f2',(16,2560)),'q_nk.bin':('i1',(19456,2560)),
        'w_nz.bin':('i1',(80,1216,16,32)),'s_gn.bin':('<f2',(20,19456))}
 b=((np.arange(8192)*7)%255-127).astype('i1').reshape(128,64)
 for name,(dt,shape) in specs.items():
  a=np.memmap(tmp_path/name,mode='w+',dtype=dt,shape=shape)
  if name=='x.bin':a[:,:128]=.5;a[0,2559]=127
  elif name=='q_nk.bin':a[:64,:128]=b.T
  elif name=='w_nz.bin':a[:4,:4]=ref.nz_pack(b)
  else:a[0,:64]=.5
  a.flush();del a
  files[name]=common.identity(tmp_path/name)
 golden=tmp_path/'full-k-golden';golden.write_bytes(b'Never read as a K128 output')
 watched={**files,'full_golden':common.identity(golden)}
 src=dict(files=files,watched=watched,identity={'fixture_only':True})
 with patch.object(fp16_gate_evidence,'accepted',return_value=src):
  case=ref.frozen(tmp_path/'slice',CODEC,None,None)[0]
 assert case['extra']['activation']['scale']==1
 assert not np.fromfile(case['files']['a.bin']['path'],'i1').any() # .5 -> even zero
 assert np.array_equal(np.fromfile(case['files']['b_nd.bin']['path'],'i1').reshape(128,64),b)
 assert np.array_equal(np.fromfile(case['files']['scale_ideal_f32.bin']['path'],'<f4'),np.full(64,.5))
 assert golden.read_bytes()==b'Never read as a K128 output'
 assert 'w8a16_k128_ref.bin' in case['files'] and case['watched']==watched


def test_frozen_stage_requires_same_build_synthetic_proof(tmp_path):
 built=dict(status='BUILT_NOT_NPU_VERIFIED',cann_compile='BUILD_COMPLETED_NPU_NOT_RUN',layout='nd',
            settings={'codec':CODEC},preflight={},review={},opp=str(tmp_path/'opp'),installed={})
 p=tmp_path/'build.json';common.write(p,built)
 args=SimpleNamespace(build_report=p,layout='nd',dataset='frozen',synthetic_proof=None,device=0)
 with patch.object(run,'installed',return_value={}),patch.object(ref,'frozen') as load:
  with pytest.raises(ValueError,match='required prior NPU'):run.smoke(args,tmp_path,dict(settings={'codec':CODEC},preflight={},review={}))
  load.assert_not_called()


def test_successful_proof_is_recomputed_and_layout_or_sdk_mix_rejected(tmp_path):
 cases=ref.synthetic(tmp_path/'data',CODEC);entries=[]
 for case in cases:
  raw,quant=outputs(case);entry=dict(name=case['name'],case_file=common.identity(Path(case['files']['a.bin']['path']).parent/'case.json'))
  for mode,values in [('raw',raw),('quant',quant)]:
   folder=tmp_path/'execute'/case['name']/mode;folder.mkdir(parents=True)
   paths=[]
   for i,value in enumerate(values):
    path=folder/f'actual-{i}.bin';path.write_bytes(value);paths.append(common.identity(path))
   common.write(folder/'execution.json',execution(mode))
   entry[mode]=dict(outputs=paths,execution_file=common.identity(folder/'execution.json'))
  entry['comparison']=ref.compare_case(case,raw,quant);entries.append(entry)
 report=dict(status='SYNTHETIC_ND_PASS',phase='complete',layout='nd',dataset='synthetic',device=0,
             settings={'codec':CODEC},cases=entries,preflight={'fixture':True},build_report={'fixture':True})
 p=tmp_path/'proof.json';common.write(p,report)
 run.require_proof(p,'SYNTHETIC_ND_PASS',report['settings'],0,report['preflight'],report['build_report'])
 with pytest.raises(ValueError,match='SDK preflight differs'):run.require_proof(p,'SYNTHETIC_ND_PASS',report['settings'],0,{'wrong':True})
 report['cases'][0]['comparison']['encoded'][0]['bit_mismatches']=1;common.write(p,report)
 with pytest.raises(ValueError,match='numerical evidence inconsistent'):run.require_proof(p,'SYNTHETIC_ND_PASS',report['settings'])


def test_inf_vs_finite_is_a_strict_one_ulp_failure():
 r=ref.metrics(np.array([np.inf],dtype='<f2'),np.array([65504],dtype='<f2'))
 assert r['bit_mismatches']==1 and r['max_ulp']==1 and r['max_abs'] is None


def test_new_stage_directory_can_be_sibling_of_readonly_review(tmp_path):
 review=tmp_path/'review.json';review.write_text('{}')
 out=common.fresh(tmp_path/'build-nd',[review])
 assert out.is_dir() and review.read_text()=='{}'
 with pytest.raises((ValueError,FileExistsError)):common.fresh(out,[review])
 with pytest.raises(ValueError,match='overlaps'):common.fresh(tmp_path,[review])


def test_raw_review_does_not_need_codec_fp16_or_fixpipe(tmp_path):
 cann=tmp_path/'sdk';cann.mkdir();(cann/'version.info').write_text('9.0.0')
 (cann/'raw_fixture.h').write_text('// CPU raw source fixture; not hardware evidence\n')
 out=tmp_path/'preflight';out.mkdir()
 with patch.object(sdk.shutil,'which',return_value=None):p=sdk.preflight(cann,out,[Path('raw_fixture.h')],raw_only=True)
 review=json.loads((out/'review-template.json').read_text())
 assert set(review['common'])==set(sdk.RAW_CLAIMS) and 'codec' not in review['settings']
 assert 'layouts' not in review and not any('fixpipe' in str(x).lower() for x in sdk.RAW_ASC_FILES)
 f=p['files'][-1];citation=dict(path=f['source']['path'],sha256=f['source']['sha256'],start=1,end=1,
                              quote='// CPU raw source fixture; not hardware evidence')
 for c in review['common'].values():c.update(status='SUPPORTED',citations=[citation])
 review.update(status='SOURCE_REVIEWED',reviewer='CPU fixture only')
 rp=tmp_path/'review.json';common.write(rp,review)
 with patch.object(ref,'check_codec',side_effect=AssertionError('raw must not inspect codec')):
  sdk.reviewed(out/'preflight.json',rp,'nd',raw_only=True)
 with pytest.raises(ValueError,match='scope mismatch'):sdk.reviewed(out/'preflight.json',rp,'nd')
 review['common']['nd_io']['status']='UNRESOLVED';common.write(rp,review)
 with pytest.raises(ValueError,match='UNRESOLVED: nd_io'):sdk.reviewed(out/'preflight.json',rp,'nd',True)
 review['common']['nd_io']['status']='UNSUPPORTED';common.write(rp,review)
 with pytest.raises(ValueError,match='BLOCKED_UNSUPPORTED'):sdk.reviewed(out/'preflight.json',rp,'nd',True)
 (cann/'raw_fixture.h').write_text('changed')
 with pytest.raises(ValueError,match='identity changed'):sdk.reviewed(out/'preflight.json',rp,'nd',True)


def test_raw_build_plan_has_one_op_two_inputs_no_quant_sources():
 definition,files,config,ops=run.build_inputs(True,'nd',sdk.RAW_SETTINGS)
 assert len(definition)==len(ops)==1 and definition[0]['op']=='DFlashW8A8Accumulator'
 assert [x['name'] for x in definition[0]['input_desc']]==['a','b']
 assert definition[0]['output_desc'][0]['type']==['int32'] and config is None
 assert len(files)==4
 for p in files.values():
  text=p.read_text()
  for token in ('SetQuantVector','SetDequantType','W8A8_SCALED_C_DTYPE','DT_FLOAT16','DT_UINT64','DFlashW8A8Fixpipe'):
   assert token not in text
 assert 'DataType::DT_INT32' in files['op_host/host_common.h'].read_text()
 with pytest.raises(ValueError,match='ND K128 only'):run.build_inputs(True,'nz',{})


def test_raw_oracle_repeat_mutation_and_no_scale_files(tmp_path):
 import w8_raw
 cases=w8_raw.prepare(tmp_path/'data');assert len(cases)==6
 for c in cases:
  assert set(c['files'])=={'a.bin','b_nd.bin','acc_int64.bin'}
  expected=np.fromfile(c['files']['acc_int64.bin']['path'],'<i8')
  out=expected.astype('<i4').tobytes();r=w8_raw.compare(c,[out,out])
  assert r['status']=='RAW_INT32_MATCH' and json.loads(json.dumps(r))==r
  if c['name']=='max_positive':assert (expected==2097152).all()
  if c['name']=='max_negative':assert (expected==-2080768).all()
 c=cases[0];out=np.fromfile(c['files']['acc_int64.bin']['path'],'<i8').astype('<i4').tobytes()
 bad=bytearray(out);bad[0]^=1
 r=w8_raw.compare(c,[out,bytes(bad)])
 assert r['status']=='FAIL_NUMERICAL' and not r['repeat_equal'] and r['comparisons'][1]['mismatches']==1
 with pytest.raises(ValueError,match='4096'):w8_raw.compare(c,[out[:1],out])
 Path(c['files']['a.bin']['path']).write_bytes(bytes(2048))
 with pytest.raises(ValueError,match='identity changed'):w8_raw.compare(c,[out,out])


def test_raw_smoke_stops_on_first_wrong_integer(tmp_path):
 import w8_raw
 args=SimpleNamespace(raw_only=True,layout='nd',device=0,build_report=tmp_path/'build.json')
 built=dict(status='BUILT_NOT_NPU_VERIFIED',raw_only=True,layout='nd',settings=sdk.RAW_SETTINGS,
            preflight={},review={},opp=str(tmp_path/'opp'),installed={},cann_compile='fixture')
 common.write(args.build_report,built);r=dict(settings=sdk.RAW_SETTINGS,preflight={},review={});calls=[]
 def invoke(case,mode,root,b,args):
  calls.append((case['name'],mode));p=tmp_path/'wrong';p.write_bytes(bytes(4096));return dict(outputs=[common.identity(p)]*2)
 with patch.object(run,'installed',return_value={}),patch.object(run,'invoke',side_effect=invoke):
  with pytest.raises(ValueError,match='raw INT32 mismatch'):run.raw_smoke(args,tmp_path,r)
 assert calls==[('signed_dense','raw')] and r['cases'][0]['comparison']['status']=='FAIL_NUMERICAL'


@pytest.mark.parametrize('extra',[['--layout','nz'],['--dataset','frozen']])
def test_raw_cli_rejects_automatic_later_stages(tmp_path,extra):
 result=subprocess.run([sys.executable,'-B',str(ROOT/'w8_run.py'),'smoke','--raw-only','--output-dir',str(tmp_path/'out'),*extra],capture_output=True,text=True)
 assert result.returncode==2 and 'only synthetic ND K128' in result.stderr


def test_raw_success_does_not_start_conversion_or_performance(tmp_path):
 args=SimpleNamespace(raw_only=True,layout='nd',device=0,build_report=tmp_path/'build.json')
 built=dict(status='BUILT_NOT_NPU_VERIFIED',raw_only=True,layout='nd',settings=sdk.RAW_SETTINGS,
            preflight={},review={},opp=str(tmp_path/'opp'),installed={},cann_compile='CPU_FIXTURE_NOT_CANN')
 common.write(args.build_report,built);r=dict(settings=sdk.RAW_SETTINGS,preflight={},review={});calls=[]
 def invoke(case,mode,root,b,args):
  calls.append(mode);root.mkdir(parents=True)
  p=root/'actual';np.fromfile(case['files']['acc_int64.bin']['path'],'<i8').astype('<i4').tofile(p)
  return dict(outputs=[common.identity(p)]*2)
 with patch.object(run,'installed',return_value={}),patch.object(run,'invoke',side_effect=invoke):run.raw_smoke(args,tmp_path,r)
 assert calls==['raw']*6 and r['status']=='RAW_INT32_ND_PASS'
 assert r['output_conversion']==r['performance']==r['nz']=='NOT_RUN'
