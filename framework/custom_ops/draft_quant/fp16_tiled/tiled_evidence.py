"""Read-only reuse of the accepted same-effective-weight experiment."""
import json
from pathlib import Path
import numpy as np
from tiled_contract import identity,verify,SHAPE,OUTPUTS,stats
from fp16_gate_common import check_execution,ORDERS
from a2_common import checked_file


def reuse_prior(path,source):
    r=json.loads(path.read_text());watched={}
    def watch(item):verify(item);watched[item['path']]=item;return Path(item['path'])
    watch(identity(path))
    if r.get('status')!='EXPERIMENT_COMPLETED_DIAGNOSTIC_ONLY' or r.get('phase')!='complete' or r.get('w8_native_bitwise')!='PASS' or r.get('fp16_native_golden_bitwise')!='BITWISE_EQUAL' or r.get('shape')!=list(SHAPE):raise ValueError('requires completed same-weight bitwise experiment')
    if r['w8_builds']!=source['modes']:raise ValueError('historical W8 build/runner differs')
    if r['source_identity']!=source['identity']:raise ValueError('historical Capture/Native Golden identity differs')
    weights=r['weights'];weight_path=path.parent/'weights/weights.json';watch(identity(weight_path))
    if json.loads(weight_path.read_text())!=weights or weights['status']!='CPU_EFFECTIVE_WEIGHT_VERIFIED' or weights['source_files']!=source['files']:raise ValueError('historical FP16 generation binding differs')
    for item in r['watched'].values():watch(item)
    for item in weights['files'].values():watch(item)
    golden=verify(source['golden']).read_bytes();summaries={}
    for mode in ('fp16','off','raw'):
        pre=r['prechecks'][mode];ex=json.loads(watch(pre['execution_file']).read_text());check_execution(ex,mode,r['device']['id'],False)
        if ex!=pre['execution']:raise ValueError('historical precheck changed')
        for item in pre['outputs'].values():
            if watch(item).read_bytes()!=golden:raise ValueError('historical precheck not bitwise Golden')
        samples=[]
        if len(r['rounds'])!=6:raise ValueError('historical crossing incomplete')
        for index,round in enumerate(r['rounds']):
            if round['index']!=index or tuple(round['order'])!=ORDERS[index]:raise ValueError('historical crossing order differs')
            result=round['results'][mode];ex=json.loads(watch(result['execution_file']).read_text());check_execution(ex,mode,r['device']['id'],True)
            if ex!=result['execution'] or not result['formal_latency'] or result['profiled']:raise ValueError('historical formal timing invalid')
            if set(result['outputs'])!=set(OUTPUTS):raise ValueError('historical tail/postcheck missing')
            for item in result['outputs'].values():
                if watch(item).read_bytes()!=golden:raise ValueError('historical timing output differs')
            if result['timing']!=stats(ex['timing']['execute_sync']['samples_ms']):raise ValueError('historical samples differ')
            samples.extend(result['timing']['samples_ms'])
        summaries[mode]=stats(samples)
        if summaries[mode]!=r['timing_summary'][mode]:raise ValueError('historical timing summary differs')
        prof=r['profiles'][mode];ex=json.loads(watch(prof['execution_file']).read_text());check_execution(ex,mode,r['device']['id'],True)
        if ex!=prof['execution'] or not prof['profiled'] or prof['formal_latency'] or set(prof['outputs'])!=set(OUTPUTS):raise ValueError('historical profiling evidence differs')
        for item in prof['outputs'].values():
            if watch(item).read_bytes()!=golden:raise ValueError('historical profiler output differs')
        profile_root=Path(prof['execution_file']['path']).parent.parent/'profiler'
        if not prof['pipe_counters']['files']:raise ValueError('historical profiler CSV inventory missing')
        for item in prof['pipe_counters']['files']:watch(identity(checked_file(profile_root,item)))
    # Recheck the actual effective weights; do not trust a status string alone.
    m,k,n=SHAPE
    nk=np.memmap(verify(weights['files']['w_fp16_nk.bin']),mode='r',dtype='<f2',shape=(n,k))
    q=np.memmap(verify(source['files']['q_nk.bin']),mode='r',dtype='i1',shape=(n,k))
    s=np.memmap(verify(source['files']['s_gn.bin']),mode='r',dtype='<f2',shape=(k//128,n))
    for g in range(k//128):
        expected=(q[:,g*128:(g+1)*128].astype(np.float64)*s[g,:,None].astype(np.float64)).astype('<f2')
        if not np.isfinite(expected).all() or not np.array_equal(expected.view('<u2'),nk[:,g*128:(g+1)*128].view('<u2')):raise ValueError('historical FP16 weight no longer equals RN16(q * GN scale)')
    return weights,dict(report=identity(path),device=r['device'],timing_summary=summaries,profiles=r['profiles'],effective_weight='RECHECKED_RN16_BITWISE',
        epoch='historical, not rerun; compare new Off drift before interpreting residual gaps'),watched
