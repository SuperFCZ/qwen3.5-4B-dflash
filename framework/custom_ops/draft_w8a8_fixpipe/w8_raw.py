"""Raw integer-only synthetic cases. No scale encoding, FP16 or frozen inputs."""
import numpy as np
from w8_common import identity,verify,write


def oracle(a,b):
    if a.shape!=(16,128) or b.shape!=(128,64) or a.dtype!=np.dtype('i1') or b.dtype!=np.dtype('i1'):
        raise ValueError('raw requires INT8 ND M16/K128/N64')
    return a.astype('<i8')@b.astype('<i8')


def prepare(root):
    root.mkdir()
    a=((np.arange(2048)*17)%256-128).astype('i1').reshape(16,128)
    b=((np.arange(8192)*71)%256-128).astype('i1').reshape(128,64)
    basis=np.zeros_like(a)
    for row,k in enumerate((0,1,15,16,31,32,63,64,95,96,111,112,124,125,126,127)):basis[row,k]=1
    specs=[('signed_dense',a,b),('zero',np.zeros_like(a),b),('k_boundaries',basis,b),
           ('max_positive',np.full_like(a,-128),np.full_like(b,-128)),
           ('max_negative',np.full_like(a,-128),np.full_like(b,127)),
           ('cancellation',np.tile(np.array([127,-127],dtype='i1'),(16,64)),np.full_like(b,127))]
    result=[]
    for name,x,w in specs:
        d=root/name;d.mkdir();acc=oracle(x,w)
        for file,value in [('a.bin',x),('b_nd.bin',w),('acc_int64.bin',acc)]:value.tofile(d/file)
        case=dict(name=name,shape=[16,128,64],raw_only=True,input_count=2,
            files={file:identity(d/file) for file in ('a.bin','b_nd.bin','acc_int64.bin')},
            oracle='CPU INT64 exact dot; abs bound 2097152, INT32 fits',status='CPU_READY_NPU_NOT_RUN')
        write(d/'case.json',case);result.append(case)
    return result


def compare(case,outputs):
    for f in case['files'].values():verify(f)
    a=np.fromfile(case['files']['a.bin']['path'],'i1').reshape(16,128)
    b=np.fromfile(case['files']['b_nd.bin']['path'],'i1').reshape(128,64)
    expected=oracle(a,b)
    saved=np.fromfile(case['files']['acc_int64.bin']['path'],'<i8').reshape(16,64)
    if not np.array_equal(expected,saved):raise ValueError('saved integer oracle disagrees with inputs')
    if len(outputs)!=2:raise ValueError('two raw repeats required')
    checks=[]
    for data in outputs:
        if len(data)!=4096:raise ValueError('INT32 output requires 4096 bytes')
        actual=np.frombuffer(data,'<i4').reshape(16,64).astype(np.int64)
        unequal=actual!=expected;indices=np.argwhere(unequal);first=None
        if len(indices):
            i,j=map(int,indices[0]);first=dict(index=[i,j],expected=int(expected[i,j]),actual=int(actual[i,j]))
        checks.append(dict(equal=not unequal.any(),mismatches=int(unequal.sum()),
            max_abs_integer=int(np.abs(actual-expected).max()),first_difference=first))
    repeat=outputs[0]==outputs[1]
    return dict(status='RAW_INT32_MATCH' if repeat and all(c['equal'] for c in checks) else 'FAIL_NUMERICAL',
                comparisons=checks,repeat_equal=repeat)
