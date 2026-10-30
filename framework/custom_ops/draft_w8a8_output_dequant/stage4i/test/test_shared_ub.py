"""No device execution: reviewed consumption policy and accounting regression."""
import json,sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from resources import validate
import run_stage4i as driver


def case(n,bn,scratch,trans,shared):
 platform=dict(platform_source='NO_ARGUMENT_MANAGER_ON_SELECTED_DEVICE',soc='Ascend310P3',arch=2002,aic=8,aiv=8,all=7,
  capacity=dict(UB=262144,L1=1048576,L0A=65536,L0B=65536,L0C=262144),system_workspace_bytes=4096)
 p=dict(usedCoreNum=1,singleCoreM=16,singleCoreN=n,singleCoreK=128,baseM=16,baseN=bn,baseK=128,
  depthA1=1,depthB1=1 if bn==n else 2,dbL0A=2,dbL0B=2,dbL0C=1,transLength=trans,
  stepM=1,stepN=1,stepKa=1,stepKb=1,shareMode=0,shareL1Size=0,shareL0CSize=0,shareUbSize=shared,isBias=0)
 name='N64' if n==64 else 'T1' if bn==128 else 'T2'
 return platform,dict(name=name,requested_n=n,requested_base_n=bn,scratch=scratch,setup_accepted=True,get_tiling_rc=0,fields=p)

@pytest.mark.parametrize('n,bn,scratch,trans,shared',[(64,64,239104,8192,512),(128,128,226816,16384,1024),(128,64,226816,16384,512)])
def test_reported_host_values_are_kept_but_not_double_charged(n,bn,scratch,trans,shared):
 platform,row=case(n,bn,scratch,trans,shared);r=validate(platform,row);d=r['details']
 assert r['status']=='HOST_RESOURCE_CHECKS_PASS'
 assert row['fields']['shareUbSize']==d['reported_shareUbSize']==shared
 assert d['effective_sdk_reserved_ub']==d['extra_shared_ub']==0
 assert d['total_ub']==262144 and d['reserve']==8192
 assert d['coefficient_region_in_scratch']==[trans,trans+bn*8]
 assert d['coefficient_logical_bound']==[trans,trans+n*8]
 # The scalar TPipe address rule: first reservation end=start+effective, End
 # restores that end. Changing raw metadata alone must not advance dav_2002 UB.
 before=16*n*2+512
 assert before+d['effective_sdk_reserved_ub']==before
 assert d['total_ub']!=scratch+d['user_ub']+2560+8192+shared

@pytest.mark.parametrize('fault',['wrong_arch','wrong_mode','negative','int32_overflow','coefficient_overflow','unaligned_offset','insufficient_ub','scratch_changed'])
def test_unaccounted_or_overflow_is_rejected(fault):
 platform,row=case(128,128,226816,16384,1024)
 if fault=='wrong_arch':platform['arch']=2201
 elif fault=='wrong_mode':row['fields']['shareMode']=1
 elif fault=='negative':row['fields']['shareUbSize']=-1
 elif fault=='int32_overflow':row['fields']['shareUbSize']=2**31
 elif fault=='coefficient_overflow':row['fields']['transLength']=226816-512
 elif fault=='unaligned_offset':row['fields']['transLength']=16385
 elif fault=='insufficient_ub':platform['capacity']['UB']=262143
 else:row['scratch']+=1024
 assert validate(platform,row)['status']=='BLOCKED'


def test_consumption_binding_checks_server_hash_and_root(tmp_path):
 cann=tmp_path/'cann';cann.mkdir();header=cann/'accessor.h';header.write_text('CPU fixture, not SDK text')
 lib=cann/'tiling.a';lib.write_bytes(b'fixture-library')
 h=driver.identity(header)['sha256'];lh=driver.identity(lib)['sha256']
 contract=dict(arch=2002,files=[dict(relative='accessor.h',local_sha256=h,server_sha256=h,lines=[[1,1]])],
  local_host_library=dict(relative='tiling.a',server_sha256=lh))
 cp=tmp_path/'contract.json';cp.write_text(json.dumps(contract))
 report=driver.bind_shared_ub(cann,cp)
 assert report['host_library_identity']=='EXACT_MATCH_RECORDED_SERVER_LIBRARY'
 lib.write_bytes(b'different local library')
 assert driver.bind_shared_ub(cann,cp)['host_library_identity']=='LOCAL_ONLY_NOT_SERVER_VERIFIED'
 header.write_text('changed')
 with pytest.raises(ValueError,match='identity mismatch'):driver.bind_shared_ub(cann,cp)
 outside=tmp_path/'outside.h';outside.write_text('CPU fixture, not SDK text');header.unlink();header.symlink_to(outside)
 with pytest.raises(ValueError,match='identity mismatch'):driver.bind_shared_ub(cann,cp)


def test_no_silent_local_only_header_admission(tmp_path):
 cann=tmp_path/'cann';cann.mkdir();h=cann/'accessor.h';h.write_text('fixture')
 cp=tmp_path/'contract.json';cp.write_text(json.dumps(dict(arch=2002,files=[dict(relative='accessor.h',local_sha256=driver.identity(h)['sha256'],server_sha256='unknown',lines=[[1,1]])])))
 with pytest.raises(ValueError,match='identity mismatch'):driver.bind_shared_ub(cann,cp)
