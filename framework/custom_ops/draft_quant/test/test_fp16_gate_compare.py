"""Same-effective-weight/protocol CPU checks; hardware execution is NOT_RUN."""
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest

EX=Path(__file__).resolve().parents[1]/'fp16_compare';sys.path.insert(0,str(EX))
import fp16_gate_common as common
import experiment


def fixture(root,n=32,k=256):
    root.mkdir();q=(((np.arange(n*k)*31)%256)-128).astype('i1').reshape(n,k)
    scale=np.array([[1+g/8+i/1024 for i in range(n)] for g in range(k//128)],dtype='<f2')
    scale[0,0]=np.float16(1.0009765625);q[0,0]=3
    scale[1,1]=np.float16(2**-24)
    nz=q.reshape(n//16,16,k//32,32).transpose(2,0,1,3).copy()
    for name,value in (('q_nk.bin',q),('w_nz.bin',nz),('s_gn.bin',scale)):value.tofile(root/name)
    return {p.name:common.identity(p) for p in root.iterdir()},q,scale


def test_exact_effective_weights_signed_nz_rounding_and_transpose(tmp_path):
    files,q,s=fixture(tmp_path/'frozen');before={p:common.sha256(p) for p in (tmp_path/'frozen').iterdir()}
    result=common.generate_weights(files,tmp_path/'new',32,256)
    nk=np.fromfile(tmp_path/'new/w_fp16_nk.bin','<f2').reshape(32,256)
    kn=np.fromfile(tmp_path/'new/w_fp16_kn.bin','<f2').reshape(256,32)
    expected=b''.join(struct.pack('<e',int(q[n,k])*float(s[k//128,n])) for n in range(32) for k in range(256))
    assert nk.tobytes()==expected and nk.tobytes()==kn.T.copy().tobytes()
    assert result['npu_validation']=='NOT_RUN' and result['nz_inverse']=='BIT_EXACT'
    assert before=={p:common.sha256(p) for p in before}
    assert np.any(nk<0) and np.any(nk==0)


def test_tamper_wrong_nz_and_nonfinite_do_not_prepare(tmp_path):
    files,_,_=fixture(tmp_path/'frozen');p=Path(files['w_nz.bin']['path']);raw=bytearray(p.read_bytes());raw[100]^=1;p.write_bytes(raw)
    with pytest.raises(ValueError,match='identity'):common.generate_weights(files,tmp_path/'bad1',32,256)
    files['w_nz.bin']=common.identity(p)
    with pytest.raises(ValueError,match='NZ inverse'):common.generate_weights(files,tmp_path/'bad2',32,256)
    files,_,_=fixture(tmp_path/'other');scale=Path(files['s_gn.bin']['path']);s=np.fromfile(scale,'<f2');s[0]=np.inf;s.tofile(scale);files['s_gn.bin']=common.identity(scale)
    with pytest.raises(ValueError,match='finite'):common.generate_weights(files,tmp_path/'bad3',32,256)


def test_fp16_differences_are_visible_but_never_relax_w8():
    golden=struct.pack('<2H',0x3c00,0x0000);different=struct.pack('<2H',0x3c01,0x8000)
    result=common.numeric(golden,different,'fp16',(1,2,2))
    assert result['status']=='FP16_DIFFERENCE_REQUIRES_REVIEW' and result['bit_mismatches']==2
    for mode in ('off','raw'):
        with pytest.raises(ValueError,match='strict bitwise'):common.numeric(golden,different,mode,(1,2,2))
    with pytest.raises(ValueError,match='nonfinite'):common.numeric(golden,struct.pack('<2H',0x7c00,0),'fp16',(1,2,2))


def test_crossed_order_and_original_samples_not_sorted():
    assert len(common.ORDERS)==6 and len(set(common.ORDERS))==6
    for mode in common.MODES:
        assert [sum(order[pos]==mode for order in common.ORDERS) for pos in range(3)]==[2,2,2]
    values=[3.,1.,2.];s=common.stats(values)
    assert s['samples_ms']==values and s['median_ms']==2 and s['p95_ms']==3
    for bad in ([True],[0],[float('nan')]):
        with pytest.raises(ValueError):common.stats(bad)


def test_profiler_is_separate_raw_metrics_and_no_pipe_sum(tmp_path):
    (tmp_path/'task.csv').write_text('Op Name,Op Type,Task Duration(us),aic_mte2_time(us),aic_vec_time(us),aic_scalar_time(us),aic_mac_time(us)\nMatMul_0,MatMul,10,8,7,6,9\n')
    result=common.pipe_report(tmp_path)
    assert result['formal_latency'] is False and result['status']=='RAW_COLUMNS_FOUND'
    assert result['rows'][0]['metrics']['Task Duration(us)']=='10'
    assert 'FORBIDDEN' in result['pipe_time_addition'] and 'total_pipe_time' not in result


def test_runner_targets_native_not_w8_or_simulator():
    source=(EX/'native_mm.cpp').read_text();cmake=(EX/'CMakeLists.txt').read_text()
    assert 'aclnnMmGetWorkspaceSize(r.x.desc,r.w.desc,r.y.desc,0,' in source
    assert 'r.w.Make(K,N,&w)' in source and 'r.w.buffer.CopyOut()!=w' in source
    assert 'RunMeasuredCalls(timing,prepare,execute,poison,verify)' in source
    assert 'runtime_camodel' not in cmake and 'cust_opapi' not in cmake
    assert 'NO_DEFAULT_PATH' in cmake


def args(tmp_path):
    a51=tmp_path/'a51/suite.json';a51.parent.mkdir();a51.write_text('{}')
    pair=tmp_path/'pair/suite.json';pair.parent.mkdir();pair.write_text('{}')
    return SimpleNamespace(output_dir=tmp_path/'new',a51_evidence=a51,prefetch_evidence=pair,prepare_only=False,cann_root=tmp_path/'sdk',msprof='msprof',device_id=0)


def fake_source(tmp_path):
    path=tmp_path/'dummy';path.write_text('fixture');item=common.identity(path)
    return dict(watched={},identity={},files={},golden=item,modes={m:dict(runner=str(path),opp=str(tmp_path/m),config={}) for m in ('off','raw')})


def test_failed_w8_precheck_prevents_any_timing_or_profiler(tmp_path):
    a=args(tmp_path);src=fake_source(tmp_path);modes=[]
    def invoke(mode,folder,*rest):
        timed=rest[5];modes.append((mode,timed))
        if mode=='off':raise ValueError('W8 mismatch')
        return {'golden_comparison':{'status':'BITWISE_EQUAL'}},b'fixture'
    built=dict(runner=src['golden'],native_header=src['golden'],sources=[],native_libraries={})
    with patch.object(experiment,'accepted',return_value=src),patch.object(experiment,'generate_weights',return_value={'files':{}}),patch.object(experiment,'build_native',return_value=(Path(src['golden']['path']),built)),patch.object(experiment.shutil,'which',return_value=src['golden']['path']),patch.object(experiment,'command'),patch.object(experiment,'cpu_sample',return_value={}),patch.object(experiment,'invoke',side_effect=invoke):
        assert experiment.experiment(a)==1
    assert modes==[('fp16',False),('off',False)]
    r=json.loads((a.output_dir/'comparison.json').read_text())
    assert r['performance']=='NOT_RUN' and r['rounds']==[] and r['profiles']=={}


def test_prepare_only_cannot_become_hardware_pass(tmp_path):
    a=args(tmp_path);a.prepare_only=True;src=fake_source(tmp_path)
    with patch.object(experiment,'accepted',return_value=src),patch.object(experiment,'generate_weights',return_value={'files':{}}),patch.object(experiment,'build_native') as build,patch.object(experiment,'invoke') as run:
        assert experiment.experiment(a)==0;build.assert_not_called();run.assert_not_called()
    r=json.loads((a.output_dir/'comparison.json').read_text())
    assert r['status']=='PREPARED_NOT_RUN' and r['performance']=='NOT_RUN' and r['simulator']['fp16']=='NOT_RUN'


def test_all_prechecks_then_six_unprofiled_orders_then_separate_profiles(tmp_path):
    a=args(tmp_path);src=fake_source(tmp_path);calls=[]
    built=dict(runner=src['golden'],native_header=src['golden'],sources=[],native_libraries={})
    def invoke(mode,folder,*rest):
        timed=rest[5];profiled=len(rest)>6
        calls.append((mode,timed,profiled))
        result=dict(golden_comparison={'status':'BITWISE_EQUAL'},timing={'samples_ms':[9999. if profiled else 2.]*30})
        return result,b'same'
    with patch.object(experiment,'accepted',return_value=src),patch.object(experiment,'generate_weights',return_value={'files':{}}),patch.object(experiment,'build_native',return_value=(Path(src['golden']['path']),built)),patch.object(experiment.shutil,'which',return_value=src['golden']['path']),patch.object(experiment,'command'),patch.object(experiment,'cpu_sample',return_value={}),patch.object(experiment,'invoke',side_effect=invoke):
        assert experiment.experiment(a)==0
    assert calls[:3]==[(m,False,False) for m in common.MODES]
    assert calls[3:21]==[(m,True,False) for order in common.ORDERS for m in order]
    assert calls[21:]==[(m,True,True) for m in common.MODES]
    r=json.loads((a.output_dir/'comparison.json').read_text())
    assert all(x['median_ms']==2. and len(x['samples_ms'])==180 for x in r['timing_summary'].values())
    assert r['status']=='EXPERIMENT_COMPLETED_DIAGNOSTIC_ONLY' and r['simulator']['fp16']=='NOT_RUN'


def test_native_runner_cpp_syntax_with_api_declarations_only(tmp_path):
    # No linking/execution; these declarations test C++ syntax, NOT CANN ABI compatibility.
    import shutil,subprocess
    compiler=shutil.which('c++')
    if not compiler:pytest.skip('no local C++ compiler')
    (tmp_path/'acl').mkdir()
    (tmp_path/'acl/acl.h').write_text('''#pragma once
#include <cstddef>
#include <cstdint>
using aclrtStream=void*;
constexpr int ACL_SUCCESS=0,ACL_MEM_MALLOC_HUGE_FIRST=0,ACL_MEMCPY_HOST_TO_DEVICE=1,ACL_MEMCPY_DEVICE_TO_HOST=2,ACL_FLOAT16=1,ACL_FORMAT_ND=2;
const char* aclGetRecentErrMsg(); const char* aclrtGetSocName();
int aclrtMalloc(void**,size_t,int); int aclrtFree(void*); int aclrtMemset(void*,size_t,int,size_t); int aclrtMemcpy(void*,size_t,const void*,size_t,int);
int aclInit(const char*); int aclFinalize(); int aclrtSetDevice(int); int aclrtResetDevice(int); int aclrtCreateStream(aclrtStream*); int aclrtDestroyStream(aclrtStream); int aclrtSynchronizeStream(aclrtStream);
''')
    mm=tmp_path/'mm.h';mm.write_text('''#pragma once
struct aclTensor; struct aclOpExecutor;
aclTensor* aclCreateTensor(const int64_t*,uint64_t,int,const int64_t*,int64_t,int,const int64_t*,uint64_t,void*);
int aclDestroyTensor(const aclTensor*);
int aclnnMmGetWorkspaceSize(const aclTensor*,const aclTensor*,aclTensor*,int8_t,uint64_t*,aclOpExecutor**);
int aclnnMm(void*,uint64_t,aclOpExecutor*,aclrtStream);
''')
    result=subprocess.run([compiler,'-std=c++17','-fsyntax-only','-I'+str(tmp_path),'-I'+str(EX.parent/'test'),'-DDFLASH_NATIVE_MM_HEADER='+json.dumps(str(mm)),str(EX/'native_mm.cpp')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
