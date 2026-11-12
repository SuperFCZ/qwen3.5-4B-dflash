import re,struct,math
import numpy as np
from x_common import verify,read,w
from x_transform import NAMES
from reference_x import numeric,expected,comparison

def fault_expected(mode,repeat):
    value=bytearray([255 if repeat else 127]*81920)
    if mode==2:words=(0x53474753,3,384,32,36864,16384,2,1)
    elif mode in (3,4,5):words=(0x53564641,mode-3,0x80000001,384,2,1,63,mode-2)
    elif mode in (6,7):words=(0x53474146,mode-6,0x80000001,256,1,127,mode-6,1)
    else:raise ValueError('fault mode')
    value[:32]=struct.pack('<8I',*words);return bytes(value)


def validate(row,case,evidence,t1,device,host=False,timed=False,anchor=None):
    mode,limit,tile=row['mode'],row['limit'],row['tile'];kind=row['variant']
    ex=read(verify(row['execution']));pl=plan(verify(row['log']).read_text(),mode,limit,tile,t1,evidence['formal_core_policy'])
    if ex['workspace_bytes']!=pl['total']:raise ValueError('actual Workspace differs')
    adm=read(verify(row['admission']))
    if adm!=dict(status='ADMITTED_BEFORE_EXECUTION',workspace_bytes=pl['total'],declared_total=pl['total'],blocks=pl['blocks'],cores=pl['cores'],core_policy=pl['core_policy'],host_proof_checked=not host):raise ValueError('admission mismatch')
    if host:
        if ex!=dict(status='HOST_TILING_ONLY',npu='NOT_RUN',device=device,workspace_bytes=pl['total'],output_mode=0,core_limit=0,tile_id=-1):raise ValueError('Host-only contract')
        return dict(status='FORMAL_DOWN_HOST_ADMITTED',plan=pl)
    required=dict(soc='Ascend310P3',shape=[16,9728,2560],tile_count=10,group_count=1,guards=True,readonly=True,repeat_equal=True,repetitions=2,device=device,cpu_fallback=False,
        output_mode=mode,core_limit=limit,tile_id=tile,guard_coverage='FULL_K_TILE_384_WORD_THREE_BOUNDARIES',prepare_calls=38 if timed else 2,verify_calls=4 if timed else 2,
        poison_verified=True,poison_patterns=[127,255],timed_prepare_io=False,timed_log_stable=True,timed_log_checked=timed,sync_completed=True,resource_admission='BEFORE_EXECUTION',
        ub_guard_protocol='UB_AND_CORE_ARENA_POISON_ON_FAILURE',layout='nd',input_count=3,op=NAMES[kind][0])
    if kind=='fault':
        if timed or mode not in range(2,8) or limit!=1 or tile!=0:raise ValueError('fault cannot benchmark')
        required.update(status='FAULT_PROTOCOL_MATCH',reference_kind='FAULT_WITNESS',fault_witness_match=True)
    else:
        if mode!=0 or limit not in (0,2) or tile not in (0,-1):raise ValueError('normal scope')
        required.update(status='EXPLORATORY_EXECUTION_SAFE',finite_active_output=True,inactive_poison_intact=True,active_coverage_checked=True,reference_kind='CPU_FULL_K_REFERENCE',guard_coverage_unchanged=True,arena_detector='XOR_OR_U32')
    if any(type(ex.get(k))!=type(v) or ex[k]!=v for k,v in required.items()):raise ValueError('execution safety/ABI contract')
    transfers=[ex.get('initial_copyin_wall_ms')]+ex.get('output_copyout_wall_ms',[])
    if ex.get('initial_input_payload_bytes')!=25079808 or len(transfers)!=(5 if timed else 3) or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in transfers):raise ValueError('transfer ledger incomplete')
    timing=w.timing(ex,timed);outputs=[verify(f).read_bytes() for f in row['outputs']]
    if len(outputs)!=(4 if timed else 2):raise ValueError('missing repeats/tail/postcheck')
    if kind=='fault':
        if any(raw!=fault_expected(mode,i) for i,raw in enumerate(outputs)):raise ValueError('fault witness differs')
        return dict(status='FAULT_COVERAGE_MATCH',plan=pl)
    details=[]
    for i,raw in enumerate(outputs):
        result=numeric(raw,expected(case,tile,min(i,1)),tile,min(i,1),outputs[0] if i else None)
        if result!=read(verify(row['numeric'][i])):raise ValueError('numeric JSON differs')
        if anchor and raw!=verify(anchor).read_bytes():raise ValueError('independent Smoke output drift')
        details.append(result)
    match=all(x['mismatch_count']==0 for x in details)
    if ex['numeric_status']!=('BITWISE_REFERENCE_OBSERVATION' if match else 'NUMERIC_MISMATCH_UNRESOLVED'):raise ValueError('numeric state drift')
    if timed and not match:raise ValueError('CPU Golden mismatch blocks formal timing')
    y=np.frombuffer(outputs[0],'<f2').reshape(16,2560);s=slice(0,256) if tile==0 else slice(None)
    refs={'effective_fp64':w.array(case['files']['reference_f64.bin'],'<f8',(16,2560)),
          'w8a16_same_real_input':w.array(case['same_input_w8a16_down'],'<f2',(16,2560)),
          'original_w8a16_mlp':w.array(case['original_w8a16_mlp'],'<f2',(16,2560))}
    return dict(status='DOWN_GOLDEN_AND_SAFETY_MATCH' if match else 'NUMERIC_MISMATCH_UNRESOLVED',quality='UNRESOLVED',
        numeric=details,metrics={k:comparison(v[:,s],y[:,s]) for k,v in refs.items()},plan=pl,timing=timing,
        transfers=dict(initial_copyin_wall_ms=ex['initial_copyin_wall_ms'],output_copyout_wall_ms=ex['output_copyout_wall_ms'],initial_payload_bytes=25079808,output_bytes=81920))


def require_numeric(rows):
    if not rows or any(x['validation']['status']!='DOWN_GOLDEN_AND_SAFETY_MATCH' for x in rows):
        raise ValueError('strict Down Golden mismatch: retain diagnostics, no downstream NPU/timing admission')

def plan(text,mode,limit,tile,t1,policy):
 def unique(pattern):
  rows=re.findall(pattern,text)
  if not rows or any(x!=rows[0] for x in rows):raise ValueError('missing/drifting Host diagnostic '+pattern)
  return rows[0]
 vals=tuple(map(int,unique(r'STAGE4X_CORE_POLICY aic=(\d+) aiv=(\d+) all=(\d+) selected=(\d+)')))
 if vals!=tuple(policy[x] for x in ('aic','aiv','all','selected')) or unique(r'STAGE4X_ARCH_PROBE arch=(\S+)')!='2002':raise ValueError('formal Core policy differs')
 if int(unique(r'STAGE4X_SCOPE tile=(-?\d+)'))!=tile:raise ValueError('tile scope drift')
 for k,v in t1['fields'].items():
  if int(unique(r'STAGE4X_T1 '+k+r'=(-?\d+)'))!=v:raise ValueError('new Full-K Tiling differs '+k)
 if unique(r'STAGE4X_T1_CONTRACT match=(\d+) effectiveSharedUb=0')!='1':raise ValueError('Tiling contract rejected')
 x=list(map(int,unique(r'STAGE4X_SCHEDULE coreLimit=(\d+) blocks=(\d+) cores=(\d+) system=(\d+) arena=(\d+) stride=(\d+) user=(\d+) total=(\d+) mode=(\d+)')))
 l,blocks,cores,system,arena,stride,user,total,m=x
 if l!=limit or m!=mode or cores!=policy['selected'] or blocks!=(limit or min(10,cores)) or blocks>cores or system!=t1['system_workspace_bytes'] or arena!=(system+511)//512*512 or stride!=arena+1024 or user!=blocks*stride or total!=system+user or not 0<total<=512*1024*1024:raise ValueError('Workspace/Core formula mismatch')
 return dict(blocks=blocks,cores=cores,core_policy=policy,total=total,system=system,arena=arena,stride=stride,user=user,owners=[[t for t in range(i,10,blocks) if tile<0 or tile==t] for i in range(blocks)])
