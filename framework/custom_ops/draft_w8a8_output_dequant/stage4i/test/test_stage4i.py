"""Synthetic CPU tests, never a substitute for real Host or NPU evidence."""
import json,os,re,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from resources import CONFIGS,validate,decide
import layout


def fixture(name):
 n,bn,scratch,user=CONFIGS[name]
 platform=dict(platform_source='NO_ARGUMENT_MANAGER_ON_SELECTED_DEVICE',soc='Ascend310P3',arch=2002,aic=8,aiv=8,all=7,
  capacity=dict(UB=262144,L1=1048576,L0A=65536,L0B=65536,L0C=262144),system_workspace_bytes=4096)
 fields=dict(usedCoreNum=1,singleCoreM=16,singleCoreN=n,singleCoreK=128,baseM=16,baseN=bn,baseK=128,
  depthA1=2,depthB1=2,dbL0A=2,dbL0B=2,dbL0C=1,transLength=65536,stepM=1,stepN=1,stepKa=1,stepKb=1,
  shareMode=0,shareL1Size=0,shareL0CSize=0,shareUbSize=0,isBias=0)
 row=dict(name=name,requested_n=n,requested_base_n=bn,scratch=scratch,setup_accepted=True,get_tiling_rc=0,fields=fields)
 return platform,row

@pytest.mark.parametrize('name',list(CONFIGS))
def test_capacity_calculations_are_not_measurements(name):
 p,r=fixture(name);v=validate(p,r)
 assert v['status']=='HOST_RESOURCE_CHECKS_PASS'
 assert v['details']['total_ub']==262144
 assert v['details']['scale_storage_bound']==(512 if name=='N64' else 1024)
 assert v['details']['device_allocation']=='NOT_RUN'

@pytest.mark.parametrize('field,value',[('transLength',226816),('baseN',32),('singleCoreN',64),('usedCoreNum',2),('shareUbSize',1024),('shareMode',1),('dbL0B',999),('stepKb',None),('isBias',1)])
def test_bad_or_unknown_resources_block(field,value):
 p,r=fixture('T1');r['fields'][field]=value
 assert validate(p,r)['status']=='BLOCKED'


def test_sdk_rejection_not_overridden_and_decision_per_candidate():
 p,r=fixture('T2');r.update(setup_accepted=False,get_tiling_rc=None,fields=None)
 rejected=validate(p,r);assert rejected['status']=='BLOCKED'
 rows=[validate(*fixture(n)) for n in ('N64','T1')]+[rejected]
 report=decide(rows,True);assert report['feasible_candidates']==['T1']
 assert report['n128_kernel_compile']==report['npu_kernel_execution']=='NOT_RUN'
 assert decide(rows,False)['status']=='STAGE4I_N128_BLOCKED'
 rows[0]['status']='BLOCKED';assert decide(rows,True)['status']=='STAGE4I_N128_BLOCKED'


def test_real_capacity_not_assumed():
 p,r=fixture('T1');p['capacity']['UB']=128*1024
 assert validate(p,r)['status']=='BLOCKED'
 p,r=fixture('T1');p['platform_source']='CUSTOM_SOC';assert validate(p,r)['status']=='BLOCKED'


def test_layout_nz_stride_boundary_and_t2():
 r=layout.symbolic();assert r['tile_count']==152 and r['group_matmuls']==3040
 assert r['gate_up_boundary']==dict(column=9728,tile=76,aligned=True)
 assert r['B_bytes']==49807360 and r['scale_bytes']==3112960
 assert not r['activation_requantized'] and not r['new_golden_created']


def test_frozen_style_tile_repacking_and_corruption():
 n=256;k=2560;q=(np.arange(n*k,dtype=np.int32).reshape(n,k)%251-125).astype('i1')
 nz=q.reshape(n//16,16,k//32,32).transpose(2,0,1,3).copy()
 sw=(np.arange(20*n).reshape(20,n)%63+1).astype('<f2')/64
 b64=np.empty((4,20,128,64),'i1');words=np.empty((4,20,64),'<u8')
 for t in range(4):
  for g in range(20):
   b64[t,g]=q[t*64:(t+1)*64,g*128:(g+1)*128].T
   words[t,g]=layout.old.common.stage2a.encode_doc_candidate((sw[g,t*64:(t+1)*64].astype(np.float64)*0.25).astype('<f4'))[0]
 src=dict(q=q,nz=nz,sw=sw,sa=0.25)
 for t in (0,1):
  packed,scales=layout.verify_tile(src,b64,words,t)
  assert np.array_equal(packed[:,:,:64],b64[2*t]) and np.array_equal(scales[:,64:],words[2*t+1])
 changed=words.copy();changed[3,19,63]^=np.uint64(1)
 with pytest.raises(ValueError,match='Scale'):layout.verify_tile(src,b64,changed,1)
 nz[79,15,15,31]^=np.int8(1)
 with pytest.raises(ValueError,match='NZ/ND'):layout.verify_tile(src,b64,words,1)


def test_no_compute_kernel_or_custom_platform_route():
 s=(HERE/'probe.cpp').read_text()
 assert 'PlatformAscendCManager::GetInstance()' in s and 'p->GetCurNpuArch()' in s
 assert not re.search(r'\b(?:aclnn\w+|aclrtLaunch\w*|aclmdlExecute\w*|aclrtCreateStream|aclrtMalloc|__aicore__|__global__)\s*\(',s)
 assert not re.search(r'GetInstance\(\s*"',s)
 assert re.findall(r'add_executable\(([^\n]+)',(HERE/'CMakeLists.txt').read_text())==['stage4i_host probe.cpp)']

@pytest.fixture(scope='module')
def host_exe(tmp_path_factory):
 compiler=shutil.which('c++')
 if not compiler:pytest.skip('no local C++ compiler')
 out=tmp_path_factory.mktemp('stage4i-cpu');inc=out/'include'
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
 p,_=fixture('N64');_,row=fixture('N64');values=row['fields']
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
 int GetTiling(optiling::TCubeTiling& t){t.n=n;t.bn=bn;return std::getenv("REJECT_T1")&&bn==128?-1:0;}
};}
'''
 (inc/'tiling/tiling_api.h').write_text(header)
 exe=out/'probe';r=subprocess.run([compiler,'-std=c++17','-I'+str(inc),str(HERE/'probe.cpp'),'-o',str(exe)],capture_output=True,text=True)
 assert r.returncode==0,r.stderr
 return exe


def test_actual_host_source_control_flow_without_device(host_exe,tmp_path):
 r=subprocess.run([str(host_exe),'0',str(tmp_path)],capture_output=True,text=True)
 assert r.returncode==0,r.stderr
 result=json.loads((tmp_path/'host-query.json').read_text())
 assert [x['name'] for x in result['cases']]==['N64','T1','T2']
 for row in result['cases']:
  assert row['setters']['SetBias'] is None and row['serialized_bytes']==8
  assert validate(result,row)['status']=='HOST_RESOURCE_CHECKS_PASS'
 assert decide([validate(result,x) for x in result['cases']],True)['status']=='STAGE4I_N128_HOST_TILING_FEASIBLE'

@pytest.mark.parametrize('flag',['NULL_PLATFORM','BAD_ARCH','BAD_SOC'])
def test_platform_query_rejects_invalid_real_route(host_exe,tmp_path,flag):
 env=dict(os.environ);env[flag]='1';r=subprocess.run([str(host_exe),'0',str(tmp_path)],env=env,capture_output=True,text=True)
 assert r.returncode!=0 and 'STAGE4I_HOST_BLOCKED:' in r.stderr
 assert not (tmp_path/'N64-tiling.bin').exists()

@pytest.mark.parametrize('flag',['REJECT_C','REJECT_T1','REJECT_T2'])
def test_installed_sdk_failure_preserved_without_fallback(host_exe,tmp_path,flag):
 env=dict(os.environ);env[flag]='1';r=subprocess.run([str(host_exe),'0',str(tmp_path)],env=env,capture_output=True,text=True)
 assert r.returncode==0,r.stderr
 result=json.loads((tmp_path/'host-query.json').read_text());rows={x['name']:x for x in result['cases']}
 if flag=='REJECT_C':assert all(not x['setup_accepted'] and x['get_tiling_rc'] is None for x in rows.values())
 elif flag=='REJECT_T1':assert rows['T1']['get_tiling_rc']==-1 and rows['T1']['fields'] is None
 else:assert rows['T2']['setters']['SetFixSplit']==-1 and rows['T2']['get_tiling_rc'] is None


def test_compile_failure_writes_all_reports_and_never_queries(tmp_path,monkeypatch):
 from types import SimpleNamespace
 import run_stage4i as driver
 args=SimpleNamespace(output_dir=tmp_path/'new',cann_root=tmp_path/'sdk',stage4b_report=tmp_path/'old/report.json',device=0)
 monkeypatch.setattr(driver,'real',lambda _:dict(status='CPU_FROZEN_LAYOUT_MATCH'))
 monkeypatch.setattr(driver,'bind',lambda _:dict(status='TEST_SDK_BINDING'))
 monkeypatch.setattr(driver.platform,'system',lambda:'Linux');monkeypatch.setattr(driver.platform,'machine',lambda:'aarch64')
 monkeypatch.setattr(driver.shutil,'which',lambda _:sys.executable)
 calls=[]
 def command(argv,path,env,timeout):
  calls.append(list(map(str,argv)));path.write_text('CPU fixture configuration rejection')
  return dict(argv=calls[-1],returncode=1,log=driver.identity(path))
 monkeypatch.setattr(driver,'command',command)
 assert driver.main(args)==1
 assert len(calls)==1 and '-S' in calls[0] and str(HERE) in calls[0]
 for name in ('tiling-report.json','resource-report.json','layout-report.json','sdk-binding.json','comparison.md'):assert (args.output_dir/name).is_file()
 report=json.loads((args.output_dir/'tiling-report.json').read_text())
 assert report['status']=='STAGE4I_N128_BLOCKED' and report['npu_kernel_execution']=='NOT_RUN' and report['n128_kernel_compile']=='NOT_RUN'


def test_resource_missing_setter_target_not_replaced():
 p,row=fixture('T1');row['requested_base_n']=64
 assert validate(p,row)['status']=='BLOCKED'
 p,row=fixture('T2');row['get_tiling_rc']=-1
 assert validate(p,row)['status']=='BLOCKED'


def test_sdk_identity_binding_and_no_snapshots(tmp_path,monkeypatch):
 import run_stage4i as driver
 sdk=tmp_path/'sdk';sdk.mkdir();header=sdk/'platform.h';header.write_text('CPU fixture only')
 (sdk/'version.info').write_text('Version=9.0.0\n')
 monkeypatch.setattr(driver,'SDK',{'platform.h':(driver.identity(header)['sha256'],[[1,1]])})
 report=driver.bind(sdk);assert report['source_snapshots_saved'] is False
 header.write_text('changed')
 with pytest.raises(ValueError,match='identity'):driver.bind(sdk)


def test_timeout_is_not_success(tmp_path,monkeypatch):
 import run_stage4i as driver
 def timeout(*args,**kwargs):raise subprocess.TimeoutExpired(args[0],1)
 monkeypatch.setattr(driver.subprocess,'run',timeout)
 result=driver.command(['host-only-fixture'],tmp_path/'log.txt',{},1)
 assert result['status']=='TIMEOUT' and result['returncode'] is None

@pytest.mark.parametrize('relative',['share/info/asc-devkit/version.info','opp/version.info'])
def test_component_version_path_without_legacy_files(tmp_path,monkeypatch,relative):
 import run_stage4i as driver
 cann=tmp_path/'sdk';cann.mkdir();header=cann/'platform.h';header.write_text('CPU fixture only')
 monkeypatch.setattr(driver,'SDK',{'platform.h':(driver.identity(header)['sha256'],[[1,1]])})
 version=cann/relative;version.parent.mkdir(parents=True);version.write_text('Version=9.0.0\n')
 r=driver.bind(cann)
 assert r['versions'][0]['file']==driver.identity(version) and r['versions'][0]['cann_9_0_0']
 header.write_text('SDK header tampered')
 with pytest.raises(ValueError,match='SDK identity'):driver.bind(cann)


def test_version_directory_name_and_escape_never_prove_version(tmp_path,monkeypatch):
 import run_stage4i as driver
 cann=tmp_path/'cann-9.0.0';cann.mkdir();monkeypatch.setattr(driver,'SDK',{})
 with pytest.raises(ValueError,match='version unverified'):driver.bind(cann)
 p=cann/'opp/version.info';p.parent.mkdir();outside=tmp_path/'version.info';outside.write_text('Version=9.0.0\n');p.symlink_to(outside)
 with pytest.raises(ValueError,match='escapes'):driver.bind(cann)
 p.unlink();p.write_text('Version=8.5.0\n')
 with pytest.raises(ValueError,match='version unverified'):driver.bind(cann)


def test_actual_metadata_formats_and_three_source_agreement(tmp_path,monkeypatch):
 import run_stage4i as driver
 cann=tmp_path/'sdk';cann.mkdir();monkeypatch.setattr(driver,'SDK',{})
 # Minimal format fixtures, not copies of server files or private installation data.
 texts={'aarch64-linux/ascend_toolkit_install.info':'package_name=Ascend-cann-toolkit\nversion=9.0.0\ninnerversion=BUILD_TAG\ncompatible_version=[OLD_RELEASE]\n',
        'share/info/asc-devkit/version.info':'Version=9.0.0\nversion_dir=cann\nrequired_package_runtime_version="9.0"\n',
        'opp/version.info':'Version=9.0.0\nrequired_package_runtime_version=">=8.5"\n'}
 for rel,text in texts.items():
  p=cann/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
 result=driver.bind(cann)
 assert len(result['versions'])==3 and all(v['reported_version']=='9.0.0' for v in result['versions'])
 assert result['versions'][0]['version_field']['line']==2
 assert result['versions'][0]['package_field']['value']=='Ascend-cann-toolkit'
 (cann/'opp/version.info').write_text('Version=8.5.0\n')
 with pytest.raises(ValueError,match='conflicting/unsupported'):driver.bind(cann)

@pytest.mark.parametrize('text',[
 'required_package_runtime_version=9.0.0\n',
 'compatible_version=9.0.0\ninnerversion=9.0.0\n',
 '# Version=9.0.0\nversion_dir=cann-9.0.0\n',
 'Version=9.0.0\nversion=9.0.0\n',
 'Version=9.0.0\nversion=8.5.0\n',
])
def test_nonrelease_or_duplicate_fields_cannot_prove_version(text):
 import run_stage4i as driver
 with pytest.raises(ValueError,match='version unverified'):driver.version_fields(text,'opp/version.info')

@pytest.mark.parametrize('value',['9.0.0-beta','9.0.0.1','19.0.0','8.5.0'])
def test_exact_release_only(value):
 import run_stage4i as driver
 assert not driver.version_fields('Version='+value,'version.info')['cann_9_0_0']

@pytest.mark.parametrize('text',['version=9.0.0\n','package_name=other-component\nversion=9.0.0\n'])
def test_install_record_requires_toolkit_identity(text):
 import run_stage4i as driver
 with pytest.raises(ValueError,match='package'):driver.version_fields(text,'aarch64-linux/ascend_toolkit_install.info')
