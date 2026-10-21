"""CPU-only weight preparation and fair-protocol checks for fixed Gate input."""
import csv
import itertools
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'test'))
from a2_common import sha256,write_json,record,validate_execution
from reference import compare

SHAPE=(16,2560,19456)
NAME='layer-0-gate_up'
MODES=('fp16','off','raw')
ORDERS=tuple(itertools.permutations(MODES))
OUTPUTS=('actual-0.bin','actual-1.bin','benchmark-last.bin','postcheck.bin')


def identity(path):
    path=Path(path).resolve();return dict(path=str(path),bytes=path.stat().st_size,sha256=sha256(path))


def verify(item):
    if identity(item['path'])!=item:raise ValueError('file identity changed: '+item['path'])
    return Path(item['path'])


def generate_weights(files,output,n=19456,k=2560):
    """W_eff[n,k] = RN16(INT8(q) * scale[k//128,n]), not a FP16 checkpoint."""
    if n<=0 or n%16 or k<=0 or k%128:raise ValueError('requires NZ N16/K32 and group128')
    for name,size in (('q_nk.bin',n*k),('w_nz.bin',n*k),('s_gn.bin',k//128*n*2)):
        verify(files[name])
        if files[name]['bytes']!=size:raise ValueError('weight size/layout mismatch')
    output.mkdir(parents=True,exist_ok=False)
    q=np.memmap(files['q_nk.bin']['path'],mode='r',dtype='i1',shape=(n,k))
    nz=np.memmap(files['w_nz.bin']['path'],mode='r',dtype='i1',shape=(k//32,n//16,16,32))
    scale=np.memmap(files['s_gn.bin']['path'],mode='r',dtype='<f2',shape=(k//128,n))
    if not np.isfinite(scale).all() or not (scale>0).all():raise ValueError('scale must be positive finite GN FP16')
    nk=np.memmap(output/'w_fp16_nk.bin',mode='w+',dtype='<f2',shape=(n,k))
    kn=np.memmap(output/'w_fp16_kn.bin',mode='w+',dtype='<f2',shape=(k,n))
    subnormal=zeros=negative_zeros=0
    for group in range(k//128):
        codes=np.concatenate([nz[group*4+p].reshape(n,32) for p in range(4)],axis=1)
        start=group*128
        if not np.array_equal(codes,q[:,start:start+128]):raise ValueError('NZ inverse differs from frozen signed INT8 NK')
        # All products fit exactly in FP32 before the only FP16 rounding (8+11
        # significand bits). Also independently compare binary64 -> binary16.
        with np.errstate(over='ignore',invalid='ignore'):
            half=np.multiply(codes.astype('<f2'),scale[group,:,None],dtype=np.float16)
            oracle=(codes.astype(np.float64)*scale[group,:,None].astype(np.float64)).astype('<f2')
        if not np.isfinite(half).all():raise ValueError('nonfinite effective weight; no saturation substitution')
        if not np.array_equal(half.view('<u2'),oracle.view('<u2')):raise ValueError('FP16 dequant RN16 disagreement')
        bits=half.view('<u2');magnitude=bits&0x7fff
        subnormal+=int(((magnitude>0)&(magnitude<0x400)).sum());zeros+=int((magnitude==0).sum());negative_zeros+=int((bits==0x8000).sum())
        nk[:,start:start+128]=half;kn[start:start+128,:]=half.T
    nk.flush();kn.flush()
    for start in range(0,k,128):
        if not np.array_equal(nk[:,start:start+128].view('<u2'),kn[start:start+128,:].T.view('<u2')):raise ValueError('FP16 NK/KN bitwise layout disagreement')
    del nk,kn
    for item in files.values():verify(item)
    result=dict(status='CPU_EFFECTIVE_WEIGHT_VERIFIED',group_size=128,logical_shape=[n,k],
        formula='RN16(signed_INT8_NK[n,k] * FP16_GN[k//128,n])',nz_inverse='BIT_EXACT',
        rounding='FP16 multiply checked against FP64 product -> FP16 nearest-even; no per-group output summation',
        files={name:identity(output/name) for name in ('w_fp16_nk.bin','w_fp16_kn.bin')},source_files=files,
        layout={'w_fp16_nk.bin':'ND contiguous [N,K]','w_fp16_kn.bin':'ND contiguous [K,N]'},
        payload_bytes=dict(w8_codes=n*k,w8_scales=k//128*n*2,fp16=n*k*2),
        fp16_to_w8_storage_ratio_not_bandwidth=(n*k*2)/(n*k+k//128*n*2),
        fp16_values=dict(subnormal=subnormal,zeros=zeros,negative_zeros=negative_zeros,device_denormal_behavior='UNRESOLVED_IF_PRESENT'),
        bandwidth_claim='UNRESOLVED: storage bytes are not measured DDR traffic or bandwidth',npu_validation='NOT_RUN')
    write_json(output/'weights.json',result);return result


def check_execution(ex,mode,device,timed):
    runtime='AscendCL native ACLNN' if mode=='fp16' else 'AscendCL ACLNN'
    if timed:validate_execution(ex,SHAPE,device,runtime,5,30,'continuous-v1',256)
    else:
        expected=dict(status='PASS',runtime=runtime,cpu_fallback=False,input_readonly=True,guards_intact=True,
                      repetitions=2,m=16,k=2560,n=19456,device_id=device)
        if any(type(ex.get(k)) is not type(v) or ex.get(k)!=v for k,v in expected.items()):raise ValueError('untimed runtime/ABI/guards differ')
        timing=ex.get('timing',{})
        if timing.get('status')!='NOT_RUN' or timing.get('warmup')!=0 or timing.get('repetitions')!=0:raise ValueError('pre-gate must be untimed')
    if mode=='fp16':
        if any(ex.get(k)!=v for k,v in dict(op='aclnnMm',cube_math_type=0,dtype='float16',weight_layout='ND_KN',soc_name='Ascend310P3').items()):raise ValueError('native FP16 ABI/cubeMathType differs')
    elif ex.get('op')!='DFlashGroupQuantLinear' or ex.get('tile_k')!=256:raise ValueError('W8 launch ABI differs')
    if timed:
        samples=ex['timing'].get('prepare',{}).get('samples_ms',[])
        if len(samples)!=30 or any(not math.isfinite(v) or v<0 for v in samples):raise ValueError('missing prepare samples')


def numeric(golden,actual,mode,shape=SHAPE):
    result=compare(golden,actual,SimpleNamespace(m=shape[0],n=shape[2]))
    a=np.frombuffer(actual,dtype='<f2').astype(np.float64);g=np.frombuffer(golden,dtype='<f2').astype(np.float64)
    error=np.abs(a-g)
    result.update(rmse=float(np.sqrt(np.mean((a-g)**2))) if result['finite'] else None,
                  p99_abs=float(np.quantile(error,.99)) if result['finite'] else None,
                  mismatch_ratio=result['bit_mismatches']/len(a))
    if not result['finite']:raise ValueError('nonfinite output/golden; stop before timing')
    if mode!='fp16' and (not result['bitwise_equal'] or result['max_ulp']!=0):raise ValueError('W8 Native Golden strict bitwise gate failed')
    result['status']='BITWISE_EQUAL' if result['bitwise_equal'] else 'FP16_DIFFERENCE_REQUIRES_REVIEW'
    result['cause']='UNRESOLVED if differing; no assumption that reduction order explains every mismatch'
    return result


def stats(samples):
    if not samples or any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in samples):raise ValueError('invalid samples')
    s=sorted(samples);n=len(s)
    return dict(median_ms=(s[(n-1)//2]+s[n//2])/2,p95_ms=s[math.ceil(.95*n)-1],samples_ms=list(samples))


def pipe_report(root):
    """Retain profiler units and every task, including layout/copy overhead."""
    files=[];rows=[]
    norm=lambda x:''.join(c.lower() for c in x if c.isalnum())
    for path in sorted(root.rglob('*.csv')):
        files.append(record(path,root))
        with path.open(newline='',encoding='utf-8-sig') as f:
            for number,row in enumerate(csv.DictReader(f),2):
                if None in row:raise ValueError('malformed profiler CSV')
                fields={k:v for k,v in row.items() if k and v and any(p in norm(k) for p in ('mte1','mte2','mte3','cube','mac','vec','scalar','taskduration'))}
                names={k:v for k,v in row.items() if k and norm(k) in ('opname','optype','kernelname','taskname','taskid','streamid')}
                if fields:rows.append(dict(file=str(path.relative_to(root)),line=number,identity=names,metrics=fields))
    groups={}
    for row in rows:
        duration=[v for k,v in row['metrics'].items() if norm(k)=='taskdurationus']
        if len(duration)!=1:continue
        try:value=float(duration[0])
        except ValueError:continue
        if not math.isfinite(value) or value<=0:continue
        label=tuple((k,v) for k,v in row['identity'].items() if norm(k) in ('opname','optype','kernelname','taskname'))
        if not label:continue
        group=groups.setdefault((row['file'],label),dict(file=row['file'],identity=dict(label),samples_us=[]))
        group['samples_us'].append(value)
    for group in groups.values():
        sample=stats(group['samples_us']);group.update(median_task_us=sample['median_ms'],p95_task_us=sample['p95_ms'])
    return dict(status='RAW_COLUMNS_FOUND' if rows else 'UNRESOLVED',files=files,rows=rows,task_durations=list(groups.values()),
        task_duration_scope='Per profiler CSV/task name only, includes untimed/warmup/measured/postcheck launches; no per-call assignment or cross-task sum',
        formal_latency=False,pipe_time_addition='FORBIDDEN: counters may overlap; no sum as kernel duration',
        traffic_bytes='UNRESOLVED',instruction_cost='UNRESOLVED: PipeUtilization is not instruction PC/source evidence')
