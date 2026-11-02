"""Synthetic CPU data and Host stubs; no platform query or NPU result claim."""
import json,os,re,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import o_resources as res
import o_layout as layout
import run_stage4o as run

def fixture(name='N256'):
 p=dict(platform_source='NO_ARGUMENT_MANAGER_ON_SELECTED_DEVICE',soc='Ascend310P3',arch=2002,aic=8,aiv=8,all=8,capacity=dict(UB=262144,L1=1048576,L0A=65536,L0B=65536,L0C=262144),system_workspace_bytes=4096)
 f=dict(usedCoreNum=1,singleCoreM=16,singleCoreN=256,singleCoreK=128,baseM=16,baseN=256,baseK=128,depthA1=1,depthB1=1,dbL0A=2,dbL0B=2,dbL0C=1,transLength=32768,stepM=1,stepN=1,stepKa=1,stepKb=1,shareMode=0,shareL1Size=0,shareL0CSize=0,shareUbSize=2048,isBias=0)
 return p,dict(name='N256',requested_n=256,requested_base_n=256,scratch=202240,setup_accepted=True,get_tiling_rc=0,fields=f)

def test_budget_scale_l0b_boundary_and_core_reference():
 p,r=fixture();v=res.validate(p,r)
 assert v['status']=='HOST_RESOURCE_CHECKS_PASS' and v['details']['total_ub']==262144
 assert v['details']['scale_storage_bound']==2048 and v['details']['l0b_bound']==65536
 assert v['details']['reported_shareUbSize']==2048 and v['details']['effective_sdk_reserved_ub']==0
 policy=dict(aic=8,aiv=8,all=7,selected=7,arch=2002)
 assert layout.core_consistency(p,policy) and policy['selected']==7
 p['all']=6;assert not layout.core_consistency(p,policy)

@pytest.mark.parametrize('key,value',[('baseN',128),('singleCoreN',128),('dbL0B',3),('transLength',202240),('shareMode',1),('shareUbSize',-1),('stepN',100),('depthA1',0),('isBias',1)])
def test_no_tiling_fallback_or_resource_relaxation(key,value):
 p,r=fixture();r['fields'][key]=value;assert res.validate(p,r)['status']=='BLOCKED'

def test_rejected_setup_and_sdk_rc_block():
 p,r=fixture();r.update(setup_accepted=False,get_tiling_rc=None,fields=None)
 assert res.decide([res.validate(p,r)],True)['status']=='STAGE4O_N256_BLOCKED'
 p,r=fixture();r['get_tiling_rc']=-1;assert res.validate(p,r)['status']=='BLOCKED'
 p,r=fixture();r['scratch']+=256;assert res.validate(p,r)['status']=='BLOCKED'
 p,r=fixture();p['capacity']['UB']=131072;assert res.validate(p,r)['status']=='BLOCKED'

def test_all_76_tiles_layout_and_boundaries():
 info=layout.symbolic();assert info['validated_tile_groups']==1520 and info['gate_up_boundary']['tile']==38
 n,k=19456,2560
 # Broadcasted synthetic NK with channel/K structure, not an all-zero mapping.
 q=np.broadcast_to((np.arange(n,dtype=np.int32)%251-125).astype('i1')[:,None],(n,k))
 nz=q.reshape(n//16,16,k//32,32).transpose(2,0,1,3)
 sw=np.broadcast_to(np.float16(.125),(20,n));b=np.broadcast_to(q[:,0].reshape(152,128)[:,None,None,:],(152,20,128,128))
 encoded=layout.n.common.stage2a.encode_doc_candidate(np.full(64,.03125,'<f4'))[0][0]
 words=np.full((152,20,128),encoded,'<u8');src=dict(q=q,nz=nz,sw=sw,sa=.25)
 for t in range(76):
  packed,code=layout.verify_tile(src,b,words,t)
  assert packed.shape==(20,128,256) and code.shape==(20,256)
 bad=words.copy();bad[151,19,127]^=np.uint64(8192)
 with pytest.raises(ValueError,match='tile=75 group=19'):layout.verify_tile(src,b,bad,75)

def test_no_compute_or_fallback_in_actual_source():
 text=(HERE/'probe.cpp').read_text()
 assert 'query(*p,argv[2],"N256",256,256,202240)' in text
 assert len(re.findall(r'query\(\*p,',text))==1
 assert 'PlatformAscendCManager::GetInstance()' in text
 assert not re.search(r'\b(?:aclnn\w+|aclrtLaunch\w*|aclmdlExecute\w*|aclrtCreateStream|aclrtMalloc|__aicore__)\s*\(',text)
 assert 'NO_DEFAULT_PATH' in (HERE/'CMakeLists.txt').read_text()
 assert '${CMAKE_DL_LIBS}' in (HERE/'CMakeLists.txt').read_text()


def test_missing_fp16_evidence_is_optional_not_fabricated(tmp_path):
 from fp16_review import audit
 r=audit(tmp_path/'missing.json',{})
 assert r['status']=='UNRESOLVED' and r['fresh_paired_measurement']=='NOT_RUN'
 assert r['native_golden_semantics'].startswith('UNRESOLVED')


@pytest.fixture(scope='module')
def host_exe(tmp_path_factory):
 compiler=shutil.which('c++')
 if not compiler:pytest.skip('no local C++ compiler')
 out=tmp_path_factory.mktemp('stage4o-cpu');inc=out/'include'
 for d in ('acl','tiling/platform','tiling'):(inc/d).mkdir(parents=True,exist_ok=True)
 (inc/'acl/acl.h').write_text('''#pragma once
#include <cstdlib>
using aclError=int;constexpr int ACL_SUCCESS=0;
inline int aclInit(const char*){return 0;}inline int aclrtSetDevice(int){return 0;}
inline const char* aclrtGetSocName(){return std::getenv("BAD_SOC")?"WrongSoc":"Ascend310P3";}
inline int aclrtResetDevice(int){return 0;}inline int aclFinalize(){return 0;}
''')
 (inc/'tiling/platform/platform_ascendc.h').write_text('''#pragma once
#include <cstdint>
#include <cstdlib>
namespace platform_ascendc {
enum class CoreMemType {UB,L1,L0_A,L0_B,L0_C};enum class NpuArch {DAV_2002=2002,BAD=999};enum class SocVersion {FIXTURE=2};
struct PlatformAscendC {
 NpuArch GetCurNpuArch()const{return std::getenv("BAD_ARCH")?NpuArch::BAD:NpuArch::DAV_2002;}
 SocVersion GetSocVersion()const{return SocVersion::FIXTURE;}
 uint32_t GetCoreNumAic()const{return 8;}uint32_t GetCoreNumAiv()const{return 8;}uint32_t GetCoreNum()const{return 7;}
 uint64_t GetLibApiWorkSpaceSize()const{return 4096;}
 void GetCoreMemSize(CoreMemType t,uint64_t& v)const{v=t==CoreMemType::UB?262144:t==CoreMemType::L1?1048576:t==CoreMemType::L0_C?262144:65536;}
};
struct PlatformAscendCManager {static PlatformAscendC* GetInstance(){static PlatformAscendC p;return std::getenv("NULL_PLATFORM")?nullptr:&p;}};
}
''')
 getters=[]
 p,_=fixture();_,row=fixture();values=row['fields']
 names=set(re.findall(r'FIELD\((\w+)\)',(HERE/'probe.cpp').read_text()))-{'name'}
 for name in sorted(names):
  value='n' if name in ('N','singleCoreN') else 'bn' if name=='baseN' else str(values.get(name,0))
  getters.append(f'long long get_{name}()const{{return {value};}}')
 header='''#pragma once
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include "platform/platform_ascendc.h"
namespace optiling {struct TCubeTiling {int n=64,bn=64;
'''+ '\n'.join(getters)+'''
size_t GetDataSize()const{return 8;}void SaveToBuffer(void* p,size_t)const{std::memcpy(p,&n,4);std::memcpy(static_cast<char*>(p)+4,&bn,4);}
};}
namespace matmul_tiling {
enum class TPosition{GM,VECIN};enum class CubeFormat{ND,NZ};enum class DataType{DT_INT8,DT_FLOAT16,DT_INT32};enum class DequantType{TENSOR};
struct MatmulApiTiling {int n=0,bn=0;explicit MatmulApiTiling(platform_ascendc::PlatformAscendC&){}
 int SetAType(TPosition,CubeFormat,DataType){return 0;}int SetBType(TPosition,CubeFormat,DataType,bool){return 0;}
 int SetCType(TPosition,CubeFormat,DataType){return std::getenv("REJECT_C")?-1:0;}
 int SetBiasType(TPosition,CubeFormat,DataType){return 0;}void SetBias(bool){}
 int SetDequantType(DequantType){return 0;}void SetShape(int,int v,int){n=v;}void SetOrgShape(int,int,int){}
 int SetFixSplit(int,int v,int){bn=v;return std::getenv("REJECT_T2")&&n==128&&bn==64?-1:0;}
 void SetBufferSpace(int,int,int){}
 int GetTiling(optiling::TCubeTiling& t){t.n=n;t.bn=bn;return std::getenv("REJECT_T1")&&bn==256?-1:0;}
};}
'''
 (inc/'tiling/tiling_api.h').write_text(header)
 exe=out/'probe';r=subprocess.run([compiler,'-std=c++17','-I'+str(inc),str(HERE/'probe.cpp'),'-o',str(exe)],capture_output=True,text=True)
 assert r.returncode==0,r.stderr
 return exe


@pytest.mark.parametrize('flag',[None,'REJECT_T1','REJECT_C','BAD_SOC','NULL_PLATFORM'])
def test_host_only_real_source_stub_paths(host_exe,tmp_path,flag):
 env=dict(os.environ)
 if flag:env[flag]='1'
 p=subprocess.run([str(host_exe),'0',str(tmp_path)],env=env,capture_output=True,text=True)
 if flag in ('BAD_SOC','NULL_PLATFORM'):assert p.returncode!=0;return
 assert p.returncode==0,p.stderr
 result=json.loads((tmp_path/'host-query.json').read_text());assert len(result['cases'])==1
 row=result['cases'][0];check=res.validate(result,row)
 assert check['status']==('BLOCKED' if flag else 'HOST_RESOURCE_CHECKS_PASS')
 if flag=='REJECT_T1':assert row['get_tiling_rc']==-1 and row['fields'] is None

def test_query_failure_report_stops_before_resources(tmp_path,monkeypatch):
 from types import SimpleNamespace
 args=SimpleNamespace(stage4n_root=tmp_path/'history',fp16_report=tmp_path/'fp16'/'comparison.json',output_dir=tmp_path/'new',cann_root=tmp_path/'sdk',device=0)
 proof=dict(status='CPU_FROZEN_LAYOUT_MATCH',stage4n=dict(device=0,sdk=dict(status='test',files=[],versions=[],shared_ub_contract={})))
 monkeypatch.setattr(run,'real',lambda _:proof)
 monkeypatch.setattr(run,'bind',lambda _:proof['stage4n']['sdk'])
 monkeypatch.setattr(run.platform,'system',lambda:'Linux');monkeypatch.setattr(run.platform,'machine',lambda:'aarch64')
 import fp16_review
 monkeypatch.setattr(fp16_review,'audit',lambda *_:dict(status='UNRESOLVED'))
 calls=[]
 def cmd(argv,path,*a):
  calls.append(argv);path.write_text('synthetic compile failure');return dict(returncode=2,log=layout.identity(path))
 monkeypatch.setattr(run,'command',cmd)
 tool=shutil.which('true');monkeypatch.setattr(run.shutil,'which',lambda _:tool)
 assert run.main(args)==1
 r=json.loads((args.output_dir/'tiling-report.json').read_text())
 assert r['status']=='STAGE4O_N256_BLOCKED' and len(calls)==1
 assert r['npu_kernel_execution']=='NOT_RUN' and not (args.output_dir/'host-query.json').exists()

def test_dependency_checker_target_is_only_new_host():
 text=(HERE/'dependency_identity_o.py').read_text()
 assert 'CMakeFiles/stage4o_host.dir' in text and 'CMakeFiles/stage4i_host.dir' not in text
 assert 'reviewed platform header absent or identity mismatch' in text


def test_positive_unknown_gettiling_code_is_not_success():
 p,r=fixture();r['get_tiling_rc']=1;assert res.validate(p,r)['status']=='BLOCKED'
