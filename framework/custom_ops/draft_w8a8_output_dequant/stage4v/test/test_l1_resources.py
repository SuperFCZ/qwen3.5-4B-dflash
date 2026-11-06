"""Server-reported L1 subset; other fields in integration fixtures are synthetic."""
import copy,json,shutil,subprocess,sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import resources_v as resource
import v_common as common
from test_stage4v import fixture_tiling


def reported_l1():
 # From the user-supplied310P3 Host query; isBias/shareMode are the fixed Stage4V scope.
 return dict(baseM=16,baseN=256,baseK=128,singleCoreM=16,singleCoreN=256,singleCoreK=2560,
             depthA1=20,depthB1=10,stepM=1,stepN=1,stepKa=20,stepKb=5,
             shareL1Size=368640,shareMode=0,isBias=0)


def test_reported_l1_no_step_or_shared_pool_double_count():
 fields=reported_l1();checks,detail=resource.l1_contract(fields,1048576)
 assert all(checks.values())
 assert detail['a1_bytes']==40960 and detail['b1_bytes']==327680
 assert detail['allocation_bytes']==detail['reported_shareL1Size']==368640
 assert (detail['a_step_batches'],detail['b_step_batches'])==(1,2)
 wrong=2048*20*1*20+32768*10*1*5+368640
 assert wrong==2826240 and wrong>1048576
 # Every NORMAL allocation branch consumes exactly depth basic blocks.
 for basic,depth in ((2048,20),(32768,10)):
  full=depth*basic;partial=(depth-2)*basic+2*basic
  assert full==partial

@pytest.mark.parametrize('key,value',[
 ('depthA1',0),('depthA1',-1),('depthA1',19),('depthA1',40),('depthB1',11),('depthB1',15),
 ('stepKa',0),('stepKa',21),('stepKa',19),('stepKb',0),('stepKb',3),('stepKb',4),
 ('stepM',2),('stepN',2),('shareL1Size',368608),('shareL1Size',368672),
 ('shareL1Size',369664),('shareL1Size',0),('isBias',1),('shareMode',1),('baseN',128),('depthB1',True)])
def test_invalid_or_unexplained_l1_rejected(key,value):
 fields=reported_l1();fields[key]=value
 checks,_=resource.l1_contract(fields,1048576)
 assert not all(checks.values())


def test_real_capacity_limit_not_bypassed_by_matching_share():
 fields=reported_l1()
 assert not resource.l1_contract(fields,368639)[0]['L1_capacity']
 assert resource.l1_contract(fields,368640)[0]['L1_capacity']
 fields.update(depthB1=20,stepKb=10,shareL1Size=696320)
 assert not resource.l1_contract(fields,524288)[0]['L1_capacity']


def test_full_validator_other_resource_checks_are_preserved():
 platform,row=fixture_tiling();row['fields'].update(reported_l1())
 valid=resource.validate(platform,row)
 assert valid['status']=='HOST_RESOURCE_CHECKS_PASS'
 d=valid['details'];assert d['l1_conservative_bound']==368640
 assert d['requested_scratch']==202240 and d['reserve']==8192 and d['l0b_bound']==65536
 for key,value in [('dbL0A',3),('dbL0B',3),('dbL0C',3),('transLength',202240)]:
  bad=copy.deepcopy(row);bad['fields'][key]=value
  assert resource.validate(platform,bad)['status']=='BLOCKED'
 bad=copy.deepcopy(platform);bad['capacity']['L1']=368639
 assert resource.validate(bad,row)['status']=='BLOCKED'


def test_formal_host_exact_contract_and_same_l1_formula(tmp_path):
 platform,row=fixture_tiling();row['fields'].update(reported_l1())
 fields={k:row['fields'][k] for k in common.p.FIELDS}
 t1=dict(fields=fields,capacity=platform['capacity'],system_workspace_bytes=4096,scratch=202240)
 (tmp_path/'contract.h').write_text(common.contract_header(t1))
 # Compile the real generated admission header. No SDK or device is mocked as passed.
 members='\n'.join(f'int {k}={v}; int get_{k}()const{{return {k};}}' for k,v in fields.items())
 code='''#include <cstdint>
#include <cstdlib>
#include "contract.h"
struct T{MEMBERS};
int main(int argc,char**argv){T p;int mode=std::atoi(argv[1]);
 if(mode==1)++p.shareL1Size;if(mode==2)--p.depthA1;if(mode==3)++p.stepKb;
 bool same=S4VAcceptedT1(p,262144,1048576,65536,65536,262144,4096,202240);
 uint64_t a=16*128,bbytes=256*128;
 return same && a*p.get_depthA1()+bbytes*p.get_depthB1()<=1048576 ? 0 : 1;}
'''.replace('MEMBERS',members)
 (tmp_path/'t.cpp').write_text(code)
 cp=subprocess.run([shutil.which('c++'),'-std=c++17',str(tmp_path/'t.cpp'),'-o',str(tmp_path/'t')],capture_output=True,text=True)
 assert cp.returncode==0,cp.stderr
 for mode in range(4):
  run=subprocess.run([str(tmp_path/'t'),str(mode)],capture_output=True)
  assert run.returncode==(0 if mode==0 else 1)
 host=(HERE/'op_host/host_common.h').read_text()
 assert 'a*p.get_depthA1()+bbytes*p.get_depthB1()>l1' in host
 assert host.index('S4VAcceptedT1(')<host.index('a*p.get_depthA1()+bbytes*p.get_depthB1()>l1')


def test_sdk_evidence_has_runtime_normal_allocator_and_pool_bindings():
 contract=json.loads((HERE/'sdk-contract.json').read_text())
 ids={x['id'] for x in contract['sources']}
 assert {'l1_input_params','l1_input_init','l1_normal_allocator','l1_selector','l1_tpipe_base','l1_tpipe_alloc','l1_cfg'}<=ids
 assert 'LOCAL_ONLY_NOT_SERVER_VERIFIED' in {x['identity_status'] for x in contract['sources']}
