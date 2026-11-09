"""CPU acceptance of equations and rejection of missing dependency edges only."""
import sys,subprocess
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import co1_contract as c
from co1_audit import formal_capability


def fixture():
 p=dict(baseM=16,baseN=256,baseK=128,singleCoreM=16,singleCoreN=256,singleCoreK=2560,usedCoreNum=1,depthA1=20,depthB1=10,stepM=1,stepN=1,stepKa=20,stepKb=5,dbL0A=2,dbL0B=2,dbL0C=1,shareL1Size=368640,shareL0CSize=16384,shareUbSize=2048,transLength=174080,isBias=0,shareMode=0,Ka=2560,Kb=2560,M=16,N=256)
 cap=dict(UB=262144,L1=1048576,L0A=65536,L0B=65536,L0C=262144)
 return p,cap


def test_budget_is_conditional_not_hardware_proof():
 p,cap=fixture();r=c.conditional_budget(p,cap,4096)
 assert r['status']=='CONDITIONAL_ARITHMETIC_CHECKS_PASS'
 assert r['ub']['total']==239104 and r['ub']['reserve']==8192
 assert r['l0c']['effective_sdk_shared']==0 and r['l0c']['reported_shared']==16384
 assert r['shared_ub']==dict(reported=2048,effective=0,reason='dav_2002 getter')
 assert r['formal_host']=='NOT_PROVEN_BY_THIS_CALCULATION'


@pytest.mark.parametrize('key,value',[('baseN',128),('singleCoreK',128),('usedCoreNum',2),('depthB1',11),('stepKa',21),('shareL1Size',368641),('dbL0B',3),('transLength',202272),('isBias',1)])
def test_bad_tiling_rejected(key,value):
 p,cap=fixture();p[key]=value
 assert c.conditional_budget(p,cap,4096)['status']=='BLOCKED'


@pytest.mark.parametrize('key,value',[('L0C',16383),('L0B',65535),('UB',239103),('L1',368639)])
def test_capacity_rejected(key,value):
 p,cap=fixture();cap[key]=value
 assert c.conditional_budget(p,cap,4096)['status']=='BLOCKED'


def test_unknown_or_misaligned_l0c_is_not_assumed_safe():
 p,cap=fixture()
 for offset in (32,1024,-1):assert c.conditional_budget(p,cap,4096,offset)['status']=='BLOCKED'
 p.pop('transLength')
 with pytest.raises(ValueError):c.conditional_budget(p,cap,4096)


def test_coverage_4096_and_n16_scale_mapping():
 nz=[(n//16)*256+m*16+n%16 for m in range(16) for n in range(256)]
 assert sorted(nz)==list(range(4096))
 for nb in range(16):
  assert nb*256*4%1024==0 and nb*16*8%32==0
  assert (nb+1)*256*4<=16384 and (nb+1)*128<=2048


def test_explicit_completion_and_lifetime():
 assert c.validate_trace(['allocate','mmad','M_V','echo','vdeq','consumer_complete','end','release'])=='CPU_DEPENDENCY_MODEL_ONLY'


@pytest.mark.parametrize('trace',[
 ['allocate','mmad','echo'],
 ['allocate','mmad','end','echo'],
 ['allocate','mmad','M_V','vdeq'],
 ['allocate','mmad','M_V','echo','release'],
 ['allocate','mmad','M_V','echo','vdeq','mmad'],
 ['allocate','mmad','M_V','echo','vdeq','consumer_complete','mmad','M_V','consumer_complete'],
 ['allocate','mmad','M_V','echo','vdeq','consumer_complete','release','echo'],
])
def test_absent_event_or_premature_reuse_fails(trace):
 with pytest.raises(ValueError):c.validate_trace(trace)


def test_formal_fields_never_synthesized():
 r=formal_capability();assert r['status']=='BLOCKED'
 assert 'ContextFaker/manual context with guessed resources' in r['rejected_substitutes']
 assert 'fields' not in r
 assert c.decision()['kernel']=='NOT_CREATED' and c.decision()['external_co1_formal_host']=='NOT_RUN'


def test_static_interfaces_not_device_kernel():
 text=(HERE/'local_check/interfaces.cpp').read_text()
 assert '__global__' not in text and 'extern "C"' not in text
 assert 'HardEvent::M_V' in text and 'HardEvent::V_M' in text
 assert not (HERE/'op_kernel').exists() and not (HERE/'runner').exists()


def test_identity_tamper(tmp_path):
 f=tmp_path/'sdk.h';f.write_text('fixed');item=c.identity(f);f.write_text('modified')
 with pytest.raises(ValueError):c.verify(item)


def test_cli():
 subprocess.run([sys.executable,'-B',str(HERE/'run_review.py'),'--help'],check=True,capture_output=True)


def test_prior_saved_fields_are_redecoded(tmp_path):
 import struct
 from co1_audit import prior_evidence
 root=tmp_path/'prior';(root/'review').mkdir(parents=True);(root/'tiling').mkdir()
 names=['shareUbSize','transLength','shareL1Size']+[f'extra{i}' for i in range(47)]
 schema=dict(bytes=200,fields=[dict(name=n,offset=i*4) for i,n in enumerate(names)])
 c.write(root/'schema.json',schema)
 left=[2048,174080,368640]+[0]*47;right=[0,174080,368640]+[0]*47
 (root/'fp.bin').write_bytes(struct.pack('<50i',*left));(root/'raw.bin').write_bytes(struct.pack('<50i',*right))
 c.write(root/'fp-host.json',dict(tiling_binary=c.identity(root/'fp.bin')))
 c.write(root/'raw-host.json',dict(tiling_binary=c.identity(root/'raw.bin')))
 sdk=dict(files=[]);review=dict(status='SOURCE_REVIEWED',sources=[],sdk=sdk,serializer=dict(schema=c.identity(root/'schema.json')))
 c.write(root/'review/report.json',review)
 result=dict(field_count=50,fields=[dict(field=n,fp16=a,raw_int32=b) for n,a,b in zip(names,left,right)],formal_fields_missing=names[27:],launch_contexts={})
 c.write(root/'comparison.json',result)
 stage=dict(status='SAVED_TILING_COMPARED_FORMAL_SCOPE_PARTIAL',review_report=c.identity(root/'review/report.json'),sources=[],sdk=sdk,tiling_comparison=c.identity(root/'comparison.json'),evidence=dict(frozen=dict(host=c.identity(root/'fp-host.json')),raw_standalone=c.identity(root/'raw-host.json')))
 c.write(root/'tiling/report.json',stage)
 out=prior_evidence(root,dict(files=[]))
 assert out['user_reported_only_share_ub_confirmed'] is True
 assert out['formal_50']=='BLOCKED_NOT_RECORDED'
 assert out['memory_interpretation']['effective_share_ub_dav2002']==dict(fp16=0,raw=0)
 (root/'raw.bin').write_bytes(bytes(200))
 with pytest.raises(ValueError):prior_evidence(root,dict(files=[]))
