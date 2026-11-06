import copy,sys,json
from fractions import Fraction
from pathlib import Path
import numpy as np
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import stage4v_rounding_models as m
import raw_resources as resources
from evidence import v


def test_binary16_round_trip_every_finite_encoding():
 for bits in range(65536):
  if bits&0x7c00==0x7c00:continue
  assert m.half(m.decode(bits,5,10),negative_zero=bool(bits&0x8000))==bits


def test_direct_rounding_near_midpoints_subnormals_and_overflow():
 mid=Fraction(2051,2048)
 x=mid-Fraction(1,2**30)
 assert m.half(x)==0x3c01
 assert m.half(m.decode(m.f32(x),8,23))==0x3c02
 assert m.half(-x)==0xbc01 and m.half(m.decode(m.f32(-x),8,23))==0xbc02
 assert m.half(Fraction(1,2**25))==0 and m.half(Fraction(3,2**25))==2
 assert m.half(65520)==0x7c00 and m.half(65520,'rtz')==0x7bff
 assert m.half(-65520,'rup')==0xfbff
 assert m.half(-65520,'rdn')==0xfc00
 assert m.half(Fraction(1),'odd')==0x3c00
 assert m.half(Fraction(1)+Fraction(1,4096),'odd')==0x3c01


def test_fp32_int_conversion_matches_ieee_numpy_in_scope():
 for d in (-41290240,-16777219,-16777217,-3,-1,0,1,3,16777217,16777219,41290240):
  assert m.f32(d)==int(np.array(d,'<f4').view('<u4'))


def test_f32_products_reconstruct_original_contract():
 rng=np.random.default_rng(31)
 for _ in range(1000):
  d=int(rng.integers(-2000000,2000000));bits=int(0x35800000+int(rng.integers(0,1024))*8192)
  scale=np.array(bits,'<u4').view('<f4');target=(np.float32(d)*scale).astype('<f2')
  assert m.predict(d,bits)['A_original']==int(target.view('<u2'))


def test_documented_word_decode_mcb_is_not_sign46():
 fields=m.word_fields((1<<46)|0x3f800000)
 assert fields['mcb']==0 and fields['n_raw']==0 and fields['n_documented_value']==1
 assert fields['multiplier']==1 and fields['sign_bit46']==1 and fields['offset_raw']==0
 assert m.word_fields((1<<36)|0x3f800000)['mcb']==1


def synthetic_six():
 # Independently constructed mathematical counterexamples, NOT user/device data.
 points=[(-536015,0x35b5a000),(-1512863,0x35fa2000),(-1416085,0x35f56000),(-1737205,0x35a38000),(-1696255,0x35844000),(-388205,0x35a6c000)]
 dots=np.zeros((16,256),np.int64);words=np.full(256,(1<<46)|0x3f800000,'<u8');actual=np.zeros((16,256),'<u2');expected=actual.copy()
 for n,(d,b) in enumerate(points):
  dots[0,n]=d;words[n]=np.uint64((1<<46)|b);p=m.predict(d,b)
  expected[0,n]=p['A_original'];actual[0,n]=p['C_exact_product']
  assert p['B_even_toward_zero']==actual[0,n]
 dots[0,10]=1;actual[0,10]=expected[0,10]=0x3c00 # correct positive odd discriminator
 return dots,words,actual,expected


def test_all4096_no_cherrypicking_and_nonidentifiability():
 args=synthetic_six();report,rows,bad=m.analyze(*args)
 assert len(rows)==4096 and len(bad)==6
 assert report['original_mismatches']==6 and report['original_match_count']==4090
 assert report['original_distribution']['negative_odd']==6 and report['original_max_ulp']==1 and report['original_toward_zero']==6
 b=report['models']['B_even_toward_zero'];assert b['explained_original']==6 and b['new_errors_on_original_matches']==1 and not b['matches_entire_tile']
 assert report['models']['C_exact_product']['matches_entire_tile']
 assert any('C_exact_product' in x and 'D_negative_odd_plus1' in x for x in report['indistinguishable_on_entire_tile'])
 assert report['midpoint_statistics']['original_mismatches']['fp32_created_midpoints']==6
 assert report['benchmark_admission']=='PROHIBITED'
 changed=list(args);changed[-1]=args[-1].copy();changed[-1][0,10]^=np.uint16(1)
 with pytest.raises(ValueError,match='expected'):m.analyze(*changed)


def test_even_toward_zero_is_not_twos_complement_lsb_clear():
 p=m.predict(-3,0x3f800000)
 assert p['B_even_toward_zero']==m.half(-2)
 assert p['D_clear_twos_complement_lsb']==m.half(-4)
 assert m.predict(3,0x3f800000)['B_even_toward_zero']==m.half(2)


def test_nearest_midpoint_at_power_of_two_and_zero():
 x=m.midpoint_info(1,0x3f800000)
 assert x['lower_magnitude_bits']=='3bff' and x['upper_magnitude_bits']=='3c00'
 assert x['midpoint_magnitude']<1
 assert m.midpoint_info(0,0x3f800000)['exact_midpoint'] is False


def raw_fixture():
 # Synthetic other fields; test budgets, never a real platform report.
 f={k:0 for k in v.p.FIELDS};f.update(usedCoreNum=1,singleCoreM=16,singleCoreN=256,singleCoreK=2560,baseM=16,baseN=256,baseK=128,Ka=2560,Kb=2560,M=16,N=256,
 depthA1=20,depthB1=10,stepM=1,stepN=1,stepKa=20,stepKb=5,dbL0A=2,dbL0B=2,dbL0C=1,shareL1Size=368640,shareL0CSize=16384,transLength=34816)
 platform=dict(soc='Ascend310P3',arch=2002,platform_source='NO_ARGUMENT_MANAGER_ON_SELECTED_DEVICE',capacity=dict(UB=262144,L1=1048576,L0A=65536,L0B=65536,L0C=262144),system_workspace_bytes=2097152)
 row=dict(name='RAW_INT32_N256',requested_n=256,requested_base_n=256,scratch=202240,c_dtype='INT32',c_position='VECIN',c_format='NZ',scale_input=False,setters={},setup_accepted=True,get_tiling_rc=0,fields=f)
 return platform,row


def test_raw_budget_and_no_fp16_scale_gate():
 p,r=raw_fixture();result=resources.validate(p,r)
 assert result['status']=='RAW_RESOURCE_CHECKS_PASS'
 assert result['ub']['total']==228864 and result['ub']['scale_buffer']==0
 assert result['layout']['output_bytes']==16384 and result['npu']=='NOT_RUN'
 assert result['tiling_context'].startswith('NOT_RUN')
 for field,value in [('singleCoreK',128),('dbL0B',3),('shareL1Size',368672),('depthB1',11),('transLength',202272)]:
  wrong=copy.deepcopy(r);wrong['fields'][field]=value
  assert resources.validate(p,wrong)['status']=='BLOCKED'
 wrong=copy.deepcopy(r);wrong['setters']['SetDequantType']=0
 assert resources.validate(p,wrong)['status']=='BLOCKED'
 wrong=copy.deepcopy(p);wrong['capacity']['UB']=228863
 assert resources.validate(wrong,r)['status']=='BLOCKED'


def test_no_opp_runner_or_compute_kernel_generated():
 assert not (HERE/'operator.json').exists() and not (HERE/'op_kernel').exists() and not (HERE/'runner').exists()
 cpp=(HERE/'raw_host/probe.cpp').read_text()
 assert 'SetCType(TPosition::VECIN,CubeFormat::NZ,DataType::DT_INT32)' in cpp
 assert 'mm.SetDequantType(' not in cpp
 assert 'aclrtLaunch' not in cpp and 'aclnn' not in cpp
 contract=json.loads((HERE/'sdk-review.json').read_text())
 assert contract['hardware_vdeq_rounding']=='UNRESOLVED'


def test_frozen_loader_reads_strided_rows_and_rejects_changed_expected(tmp_path,monkeypatch):
 # File-layout test only: resource/build validation is mocked, not claimed PASS.
 import evidence as e
 root=tmp_path/'frozen';root.mkdir();folder=root/'bounded/l0-t0-m0';folder.mkdir(parents=True)
 a=np.zeros((16,2560),'i1');a[:,0]=np.arange(16)-8
 b=np.memmap(root/'b.bin',mode='w+',dtype='i1',shape=(76,2560,256));b[:]=0;b[0,0]=np.arange(256)%127-63;b.flush()
 dot=np.zeros((16,19456),'<i4');dot[:,:256]=np.outer(a[:,0].astype(np.int32),b[0,0].astype(np.int32))
 eff=np.ones(19456,'<f4');words=np.full((76,256),(1<<46)|0x3f800000,'<u8');gold=dot.astype('<f2')
 files={};maps={}
 for name,array in [('a_i8.bin',a),('dot_i32.bin',dot),('effective_f32.bin',eff),('quant_u64_doc_candidate.bin',words),('golden_fp16.bin',gold)]:
  array.tofile(root/name);files[name]=e.identity(root/name);maps[name]=array
 files['b_i8.bin']=e.identity(root/'b.bin');maps['b_i8.bin']=b
 case=root/'case.json';e.write(case,dict(files=files));monkeypatch.setattr(e.ref,'verify_case',lambda _:maps)
 expected=np.full((16,19456),0x7f7f,'<u2');expected[:,:256]=gold[:,:256].view('<u2')
 expected.tofile(folder/'expected-0.bin');actual=expected.copy();actual[15,255]^=np.uint16(1);actual.tofile(folder/'actual-0.bin')
 (folder/'runner.log').write_text('fixture')
 e.write(folder/'execution.json',dict(status='FAIL'))
 policy=dict(aic=8,aiv=8,all=7,selected=7,arch=2002)
 e.write(folder/'resource-admission.json',dict(status='ADMITTED_BEFORE_EXECUTION',blocks=7,declared_total=4096,workspace_bytes=4096,core_policy=policy))
 monkeypatch.setattr(e,'plan',lambda *args:dict(blocks=7,total=4096,core_policy=policy))
 monkeypatch.setattr(e,'verify_build',lambda *args:None)
 common=dict(sources=[],evidence=dict(formal_core_policy=policy),device=0,policy={})
 report=dict(common,status='BLOCKED',case=e.identity(case),t1={})
 for name,status in [('prepare','STAGE4V_CPU_PREPARED_QUALITY_UNRESOLVED'),('host','STAGE4V_FULL_K_HOST_FEASIBLE'),('build','STAGE4V_BUILT_NPU_NOT_RUN')]:
  file=root/(name+'.json');e.write(file,dict(common,status=status,phase='complete'));report[name+'_report']=e.identity(file)
 diag=folder/'failure-diagnostics.json';e.write(diag,dict(mode=0,limit=0,tile=0,outputs=[dict(file=e.identity(folder/'actual-0.bin'))]));report['failure_diagnostics']=[e.identity(diag)]
 e.write(root/'bounded/report.json',report)
 data=e.load(root)
 assert data['actual'].shape==(16,256) and data['actual'][15,255]==actual[15,255]
 assert np.array_equal(data['oracle'],dot[:,:256])
 expected[0,0]^=np.uint16(1);expected.tofile(folder/'expected-0.bin')
 with pytest.raises(ValueError,match='expected-0'):e.load(root)


def test_arithmetic_cause_is_not_inferred_from_matching_model():
 _,words,_,_=synthetic_six()
 p=m.predict(-536015,int(words[0]&np.uint64(0xffffffff)))
 assert p['C_exact_product']==p['B_even_toward_zero']!=p['A_original']
 # The exact-product path kept the integer unchanged; a dot+1 fit cannot prove an error in it.
 assert m.decode(m.f32(-536015),8,23)==-536015
