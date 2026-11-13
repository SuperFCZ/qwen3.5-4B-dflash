"""Actual Runner read/Make expressions + CPU-only ACL allocation fixtures. No NPU."""
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import x_transform as transform
import protocol_x as protocol
from test_down import fixture

QUANT=HERE.parent.parent/'draft_quant/test'


def contracts(text):
    reads={name:int(count) for name,count in re.findall(r'ReadBytes\(folder\+"/([^\"]+)",(\d+)\)',text)}
    output=int(re.search(r'const size_t outputBytes=(\d+);',text)[1])
    tensors={}
    for name,logical,storage,dtype,size,data in re.findall(
            r'r\.([xwsy])\.Make\(\{([^}]+)\},\{([^}]+)\},(ACL_\w+),ACL_FORMAT_ND,(\w+)(?:,&(\w+))?\);',text):
        logical=tuple(map(int,logical.split(',')));storage=tuple(map(int,storage.split(',')))
        size=output if size=='outputBytes' else int(size)
        width={'ACL_INT8':1,'ACL_UINT64':8,'ACL_FLOAT16':2}[dtype]
        assert logical==storage and math.prod(storage)*width==size
        tensors[name]=(logical,dtype,size,data)
    assert set(tensors)=={'x','w','s','y'}
    assert tensors['x']==((16,9728),'ACL_INT8',155648,'x')
    assert tensors['w']==((10,304,16,16,32),'ACL_INT8',24903680,'w')
    assert tensors['s']==((2560,),'ACL_UINT64',20480,'s')
    assert tensors['y']==((16,2560),'ACL_FLOAT16',81920,'')
    for name,file in [('x','a_i8.bin'),('w','b_i8.bin'),('s','quant_u64_doc_candidate.bin')]:
        assert reads[file]==tensors[name][2]
    assert reads['expected-0.bin']==reads['expected-1.bin']==output
    total=sum(tensors[k][2] for k in ('x','w','s'))
    reported=int(re.search(r'initial_input_payload_bytes\\":(\d+)',text)[1])
    assert reported==total==25079808
    return reads,tensors


@pytest.mark.parametrize('kind',transform.NAMES)
def test_all_file_descriptor_buffer_and_payload_sizes(kind):
    text=(HERE/kind/'runner/runner.cpp').read_text()
    contracts(text)
    assert transform.files(kind)['runner/runner.cpp']==text.encode()


@pytest.mark.parametrize('kind',transform.NAMES)
def test_scale_regression_and_activation_cascade_are_rejected(kind):
    text=(HERE/kind/'runner/runner.cpp').read_text()
    for wrong in (
        text.replace('"/quant_u64_doc_candidate.bin",20480','"/quant_u64_doc_candidate.bin",155648'),
        text.replace('ACL_UINT64,ACL_FORMAT_ND,20480,&s','ACL_UINT64,ACL_FORMAT_ND,155648,&s'),
        text.replace('"/a_i8.bin",155648','"/a_i8.bin",20480'),
        text.replace('ACL_INT8,ACL_FORMAT_ND,155648,&x','ACL_INT8,ACL_FORMAT_ND,20480,&x'),
    ):
        with pytest.raises(AssertionError):contracts(wrong)


def test_one_pass_mapping_does_not_rewrite_new_activation_bytes():
    pattern=r'\b(?:'+'|'.join(transform.NUMBERS)+r')\b'
    assert re.sub(pattern,lambda m:transform.NUMBERS[m[0]],'40960 155648')=='155648 20480'


@pytest.mark.parametrize('kind',transform.NAMES)
def test_actual_reader_and_tensor_allocate_with_cpu_acl_mock(tmp_path,kind):
    compiler=shutil.which('c++')
    if not compiler:pytest.skip('host C++ compiler unavailable')
    text=(HERE/kind/'runner/runner.cpp').read_text();contracts(text)
    tensor=text[text.index('struct Tensor {'):text.index('struct Session {')]
    read_expr=re.search(r'ReadBytes\(folder\+"/quant_u64_doc_candidate.bin",\d+\)',text)[0]
    make_expr=re.search(r'r\.s\.Make\([^;]+;',text)[0].replace('r.s.','tensor.')
    # Existing declaration-only CPU header. Implement only the guarded buffer
    # operations used by actual Tensor::Make; no aclInit or operator execution.
    stub=HERE.parent/'stage4k/test/N128/cpu_include'
    source=tmp_path/'reader.cpp'
    source.write_text('''#include "runner_common.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
struct aclTensor{};
size_t allocated=0;
const char* aclGetRecentErrMsg(){return "CPU fixture only";}
int aclrtMalloc(void**p,size_t n,int){allocated=n;*p=std::malloc(n);return *p?0:1;}
int aclrtFree(void*p){std::free(p);return 0;}
int aclrtMemset(void*p,size_t capacity,int value,size_t n){if(n>capacity)return 1;std::memset(p,value,n);return 0;}
int aclrtMemcpy(void*p,size_t capacity,const void*q,size_t n,int){if(n>capacity)return 1;std::memcpy(p,q,n);return 0;}
aclTensor* aclCreateTensor(const int64_t*l,uint64_t rank,int dt,const int64_t*stride,int64_t,int,
 const int64_t*s,uint64_t srank,void*){
 if(rank!=1||srank!=1||l[0]!=2560||s[0]!=2560||stride[0]!=1||dt!=ACL_UINT64)throw std::runtime_error("Scale descriptor mismatch");
 return new aclTensor;
}
int aclDestroyTensor(const aclTensor*p){delete p;return 0;}
using namespace draft_quant_test;
'''+tensor+'''
int main(int argc,char**argv){try{
 std::string folder=argv[1];auto s='''+read_expr+''';Tensor tensor;
 '''+make_expr+'''
 if(s.size()!=2560*sizeof(uint64_t)||tensor.buffer.bytes!=20480||allocated!=20480+2*kGuard||tensor.buffer.CopyOut()!=s)
  throw std::runtime_error("Scale file/buffer/guard size mismatch");
 tensor.Release();return 0;
 }catch(const std::exception&e){std::cerr<<e.what();return 2;}}
''')
    exe=tmp_path/'reader'
    subprocess.run([compiler,'-std=c++17','-I'+str(stub),'-I'+str(QUANT),str(source),'-o',str(exe)],check=True,capture_output=True)
    scale=tmp_path/'quant_u64_doc_candidate.bin'
    words=np.arange(2560,dtype='<u8');words.tofile(scale)
    assert scale.stat().st_size==20480
    ok=subprocess.run([exe,tmp_path],capture_output=True,text=True)
    assert ok.returncode==0,ok.stderr
    for size in (20479,20481,155648):
        scale.write_bytes(bytes(size))
        result=subprocess.run([exe,tmp_path],capture_output=True,text=True)
        assert result.returncode==2 and 'input file has wrong size' in result.stderr


@pytest.mark.parametrize('system,limit',[(2097152,0),(2097152,2),(100000,0)])
def test_workspace_remains_sdk_requested_independent_of_input_payload(system,limit):
    platform,row=fixture();policy=dict(aic=8,aiv=8,all=7,selected=7,arch=2002)
    blocks=limit or 7;arena=(system+511)//512*512;stride=arena+1024;user=blocks*stride;total=system+user
    t1=dict(fields=row['fields'],system_workspace_bytes=system)
    text='STAGE4X_CORE_POLICY aic=8 aiv=8 all=7 selected=7\nSTAGE4X_ARCH_PROBE arch=2002\nSTAGE4X_SCOPE tile=-1\n'
    text+=''.join(f'STAGE4X_T1 {key}={value}\n' for key,value in row['fields'].items())
    text+='STAGE4X_T1_CONTRACT match=1 effectiveSharedUb=0\n'
    schedule=f'STAGE4X_SCHEDULE coreLimit={limit} blocks={blocks} cores=7 system={system} arena={arena} stride={stride} user={user} total={total} mode=0'
    result=protocol.plan(text+schedule,0,limit,-1,t1,policy)
    assert result['total']==total
    ranges=[(system+i*stride,system+(i+1)*stride) for i in range(blocks)]
    assert ranges[-1][1]==total and all(a[1]<=b[0] for a,b in zip(ranges,ranges[1:]))
    with pytest.raises(ValueError):protocol.plan(text+schedule.replace(f'total={total}', 'total=25079808'),0,limit,-1,t1,policy)
