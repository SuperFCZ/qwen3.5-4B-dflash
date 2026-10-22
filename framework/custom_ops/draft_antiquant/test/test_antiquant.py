"""CPU contracts only. Synthetic SDK text is NOT a supported-hardware assertion."""
import copy
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import aq_common as common
import aq_sdk as sdk
import aq_prepare as prep
import aq_run as run


def collected(tmp_path,api=True):
    cann=tmp_path/'cann';cann.mkdir();(cann/'version.info').write_text('Version=9.0.0\n')
    (cann/'matmul_impl.h').write_text('#if __CCE_AICORE__ == 200\n'+
        ('// fixture only: SetAntiQuantVector\n' if api else '// no api\n')+'#endif\n')
    metadata=cann/'opp/weight_quant_batch_matmul_v3/config.json';metadata.parent.mkdir(parents=True)
    metadata.write_text('{"fixture_only": true}')
    with patch.object(sdk.shutil,'which',return_value=None):
        report,path=sdk.collect(cann,tmp_path/'audit')
    review=json.loads((path.parent/'review-template.json').read_text())
    file=next((f for f in report['files'] if f['source']['path'].endswith('matmul_impl.h')),None)
    review['reviewer']='CPU fixture, not device evidence';review['decision']='K128_SOURCE_REVIEWED'
    if file:
        for c in review['claims'].values():
            c.update(status='SUPPORTED',finding='test harness fixture only',citations=[dict(
                source_path=file['source']['path'],sha256=file['source']['sha256'],start=1,end=3,
                quote=Path(file['source']['path']).read_text().rstrip('\n'))])
    r=tmp_path/'review.json';common.write(r,review)
    return cann,report,path,review,r


def test_presence_never_automatically_supports_hardware(tmp_path):
    cann,report,path,review,r=collected(tmp_path)
    assert report['status']=='SDK_REVIEW_REQUIRED' and report['npu']=='NOT_RUN'
    assert report['cann_compile']=='NOT_RUN' and report['k256']=='NOT_RUN'
    assert any(f['source']['path'].endswith('weight_quant_batch_matmul_v3/config.json') for f in report['files'])
    template=json.loads((path.parent/'review-template.json').read_text())
    assert all(c['status']=='UNRESOLVED' for c in template['claims'].values())
    sdk.reviewed(path,r)
    for c in review['claims'].values():c['status']='UNRESOLVED'
    common.write(r,review)
    with pytest.raises(ValueError,match='SDK_REVIEW_REQUIRED'):sdk.reviewed(path,r)


def test_missing_api_and_wrong_version_do_not_pass(tmp_path):
    _,report,_,_,_=collected(tmp_path,False)
    assert report['status']=='BLOCKED_MISSING_API'
    other=tmp_path/'wrong';other.mkdir();(other/'version.info').write_text('Version=8.5.0')
    report,_=sdk.collect(other,tmp_path/'wrong-audit')
    assert report['status']=='BLOCKED_AUDIT' and report['cann_compile']=='NOT_RUN'


@pytest.mark.parametrize('tamper',['source','snapshot','audit','quote','range','hash'])
def test_review_tamper_rejected(tmp_path,tamper):
    cann,report,path,review,r=collected(tmp_path)
    file=next(f for f in report['files'] if f['source']['path'].endswith('matmul_impl.h'))
    if tamper in ('source','snapshot'):
        Path(file[tamper]['path']).write_text('changed')
    elif tamper=='audit':path.write_text(path.read_text()+'\n')
    else:
        c=review['claims']['chip']['citations'][0]
        if tamper=='quote':c['quote']='fake'
        elif tamper=='range':c['start']=0
        else:c['sha256']='0'*64
        common.write(r,review)
    with pytest.raises(ValueError):sdk.reviewed(path,r)


def test_unsupported_stops_before_build_even_with_other_unresolved_claims(tmp_path):
    cann,report,path,review,r=collected(tmp_path)
    review['claims']['chip']['status']='UNSUPPORTED'
    review['claims']['mixed_types']['status']='UNRESOLVED';common.write(r,review)
    a=SimpleNamespace(stage='build',output_dir=tmp_path/'build',cann_root=cann,
        audit=path,review=r,build_report=None,a51_evidence=tmp_path/'old-a/suite.json',
        prefetch_evidence=tmp_path/'old-p/suite.json')
    # review is alongside tmp_path, so use a separate report directory (never nest in review evidence).
    reviews=tmp_path/'review-dir';reviews.mkdir();r.rename(reviews/'review.json');a.review=reviews/'review.json'
    with patch.object(run,'command') as command:
        assert run.main(a)==1;command.assert_not_called()
    result=json.loads((a.output_dir/'report.json').read_text())
    assert result['status']=='BLOCKED_UNSUPPORTED' and result['npu']=='NOT_RUN'
    assert result['cann_compile']=='NOT_RUN'


def arrays():
    q=((np.arange(128*256)*37)%256-128).astype('i1').reshape(128,256)
    nz=q.reshape(8,16,8,32).transpose(2,0,1,3).copy()
    scale=np.full((2,128),.0078125,dtype='<f2');scale[0,0]=np.float16(1.0009765625)
    scale[0,1]=np.float16(2**-24)
    x=np.zeros((16,256),dtype='<f2');x[:,0]=1;x[:,127]=-1;x[:,128]=4
    return x,q,nz,scale


def test_slice_nz_signed_half_rounding_group_boundary_and_readonly():
    x,q,nz,s=arrays();before=[a.tobytes() for a in (x,q,nz,s)]
    result=prep.slice_arrays(x,q,nz,s)
    assert all(a.tobytes()==b for a,b in zip((x,q,nz,s),before))
    assert result['w_nz.bin'].shape==(4,4,16,32)
    assert np.array_equal(result['q_nk.bin'],q[:64,:128])
    expected=b''.join(struct.pack('<e',int(q[n,k])*float(s[0,n])) for k in range(128) for n in range(64))
    assert result['w_fp16_kn.bin'].tobytes()==expected
    assert result['offset.bin'].tobytes()==bytes(128)
    assert result['x.bin'].shape==(16,128)
    assert result['cpu_fp64.bin'].shape==(16,64)
    # Alter the unselected group: K128 must not accidentally consume it.
    x[:,128:]=88;s[1,:]=55
    assert prep.slice_arrays(x,q,nz,s)['cpu_fp32.bin'].tobytes()==result['cpu_fp32.bin'].tobytes()


def test_bad_nz_and_scale_rejected():
    x,q,nz,s=arrays();nz[0,0,0,0]^=1
    with pytest.raises(ValueError,match='inverse'):prep.slice_arrays(x,q,nz,s)
    x,q,nz,s=arrays();s[0,0]=np.nan
    with pytest.raises(ValueError,match='scale'):prep.slice_arrays(x,q,nz,s)


def test_numerical_disagreement_is_reported_without_tolerance():
    golden=np.ones((16,64),dtype='<f2').tobytes();bad=bytearray(golden);bad[0]^=1
    payload={'cpu_fp32.bin':golden,'cpu_fp64.bin':golden}
    result=run.numerical_report(payload,{'native':[golden,golden],'custom':[bytes(bad),bytes(bad)]})
    assert result['status']=='FAIL_NUMERICAL'
    assert result['comparisons']['custom'][0]['native']['bit_mismatches']==1
    assert result['comparisons']['custom'][0]['native']['max_ulp']==1
    assert result['group_refresh']=='NOT_RUN'
    assert run.numerical_report(payload,{'native':[golden]*2,'custom':[golden]*2})['status']=='K128_BITWISE_MATCH'
    for incomplete in ({},{'native':[golden]*2},{'native':[],'custom':[]}):
        with pytest.raises(ValueError,match='both implementations'):run.numerical_report(payload,incomplete)
    with pytest.raises(ValueError,match='repeat'):
        run.numerical_report(payload,{'native':[golden]*2,'custom':[golden,bytes(bad)]})


def test_untimed_runtime_gates_do_not_accept_false_guards_or_other_shapes():
    ex=dict(status='EXECUTION_COMPLETED_NUMERICS_UNCHECKED',op='DFlashAntiQuantK128',
        soc='Ascend310P3',shape=[16,128,64],guards=True,readonly=True,repeat_equal=True,
        repetitions=2,device=0,cpu_fallback=False,timing={'status':'NOT_RUN','warmup':0,'repetitions':0})
    run.check_execution(ex,'custom',0)
    for key,value in [('guards',False),('readonly',False),('repeat_equal',False),('shape',[16,256,64]),
                      ('cpu_fallback',True),('timing',{'status':'MEASURED','warmup':5,'repetitions':30})]:
        bad=copy.deepcopy(ex);bad[key]=value
        with pytest.raises(ValueError):run.check_execution(bad,'custom',0)


def test_sdk_source_symlink_escape_and_permission_error_fail_closed(tmp_path):
    cann=tmp_path/'sdk';cann.mkdir();outside=tmp_path/'outside.h';outside.write_text('SetAntiQuantVector')
    (cann/'version.info').write_text('9.0.0');(cann/'escape.h').symlink_to(outside)
    r,_=sdk.collect(cann,tmp_path/'out')
    assert r['status']=='BLOCKED_AUDIT' and 'escapes' in r['error']
    def denied(*args,**kwargs):
        kwargs['onerror'](PermissionError('denied source'))
    with patch.object(sdk.os,'walk',side_effect=denied):
        r,_=sdk.collect(cann,tmp_path/'denied')
    assert r['status']=='BLOCKED_AUDIT' and 'PermissionError' in r['error']


def test_identity_and_source_dequant_guard(tmp_path):
    p=tmp_path/'file';p.write_bytes(b'original');item=common.identity(p);p.write_bytes(b'mutated')
    with pytest.raises(ValueError,match='identity'):common.verify(item)
    kernel=ROOT/'op_kernel/d_flash_anti_quant_k128.cpp';assert run.source_no_explicit_dequant(kernel)['status']=='SOURCE_CHECKED'
    p.write_text(kernel.read_text()+'\nCast(dst,src,mode,128);\n')
    with pytest.raises(ValueError,match='explicit'):run.source_no_explicit_dequant(p)


def test_poc_cannot_claim_performance_or_other_shapes():
    assert 'K128' in run.__doc__
    source=(ROOT/'test/runner.cpp').read_text()
    assert 'if(argc!=3)' in source and 'Timing timing;' in source
    assert 'w_fp16_kn.bin' in source and 'ACL_FORMAT_FRACTAL_NZ' in source
    assert 'RunMeasuredCalls' in source and 'CheckNamedGuards' in source
    assert 'r.s.buffer.CopyOut()!=s' in source and 'r.o.buffer.CopyOut()!=o' in source
    host=(ROOT/'op_host/d_flash_anti_quant_k128.cpp').read_text()
    assert 'p.get_baseK()!=128' in host and 'c->SetBlockDim(1)' in host
    kernel=(ROOT/'op_kernel/d_flash_anti_quant_k128.cpp').read_text()
    assert kernel.count('PipeBarrier<PIPE_ALL>();')==2
    assert kernel.index('mm.SetAntiQuantVector')<kernel.index('mm.IterateAll')<kernel.index('mm.End')


def test_native_and_custom_runner_cpp_syntax_only(tmp_path):
    compiler=shutil.which('c++')
    if not compiler:pytest.skip('local compiler unavailable')
    (tmp_path/'acl').mkdir()
    (tmp_path/'acl/acl.h').write_text('''#pragma once
#include <cstddef>
#include <cstdint>
using aclrtStream=void*;using aclDataType=int;using aclFormat=int;
constexpr int ACL_SUCCESS=0,ACL_MEM_MALLOC_HUGE_FIRST=0,ACL_MEMCPY_HOST_TO_DEVICE=1,ACL_MEMCPY_DEVICE_TO_HOST=2,ACL_FLOAT16=1,ACL_INT8=2,ACL_FORMAT_ND=2,ACL_FORMAT_FRACTAL_NZ=29;
const char* aclGetRecentErrMsg();const char* aclrtGetSocName();
int aclrtMalloc(void**,size_t,int);int aclrtFree(void*);int aclrtMemset(void*,size_t,int,size_t);int aclrtMemcpy(void*,size_t,const void*,size_t,int);
int aclInit(const char*);int aclFinalize();int aclrtSetDevice(int);int aclrtResetDevice(int);int aclrtCreateStream(aclrtStream*);int aclrtDestroyStream(aclrtStream);int aclrtSynchronizeStream(aclrtStream);
''')
    decl='''#pragma once
struct aclTensor;struct aclOpExecutor;
aclTensor* aclCreateTensor(const int64_t*,uint64_t,int,const int64_t*,int64_t,int,const int64_t*,uint64_t,void*);
int aclDestroyTensor(const aclTensor*);
'''
    (tmp_path/'mm.h').write_text(decl+'''int aclnnMmGetWorkspaceSize(const aclTensor*,const aclTensor*,aclTensor*,int8_t,uint64_t*,aclOpExecutor**);
int aclnnMm(void*,uint64_t,aclOpExecutor*,aclrtStream);
''')
    (tmp_path/'aclnn_d_flash_anti_quant_k128.h').write_text(decl+'''int aclnnDFlashAntiQuantK128GetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**);
int aclnnDFlashAntiQuantK128(void*,uint64_t,aclOpExecutor*,aclrtStream);
''')
    for flags in ([],['-DANTIQUANT_NATIVE_REFERENCE=1','-DANTIQUANT_MM_HEADER="mm.h"']):
        r=subprocess.run([compiler,'-std=c++17','-fsyntax-only','-I'+str(tmp_path),'-I'+str(run.QUANT/'test'),
            *flags,str(ROOT/'test/runner.cpp')],capture_output=True,text=True)
        assert r.returncode==0,r.stderr
