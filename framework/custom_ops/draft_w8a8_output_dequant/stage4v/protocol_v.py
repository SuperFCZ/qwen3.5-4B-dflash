"""Full-K-specific numerical, lifetime, scheduling and artifact admission."""
from pathlib import Path
import re,struct
import numpy as np
from v_common import HERE,q,NAME,COVERAGE,verify,read,contract_header,verify_audit
from reference_v import verify_case

def schedule(stage):
 if stage=='op-host':return [(0,0,-1)]
 if stage=='bounded':return [(0,0,t) for t in (0,37,38,75)]
 if stage=='full':return [(l,m,-1) for l in (0,2) for m in (1,0)]
 if stage=='fault':return [(1,m,0) for m in (2,3,4,5)]
 raise ValueError('invalid phase')

def plan(text,mode,limit,tile,t1,policy):
 def unique(pattern):
  rows=re.findall(pattern,text)
  if not rows or any(x!=rows[0] for x in rows):raise ValueError('missing/drifting Host diagnostic '+pattern)
  return rows[0]
 vals=tuple(map(int,unique(r'STAGE4V_CORE_POLICY aic=(\d+) aiv=(\d+) all=(\d+) selected=(\d+)')))
 if vals!=tuple(policy[x] for x in ('aic','aiv','all','selected')) or unique(r'STAGE4V_ARCH_PROBE arch=(\S+)')!='2002':raise ValueError('formal Core policy differs')
 if int(unique(r'STAGE4V_SCOPE tile=(-?\d+)'))!=tile:raise ValueError('tile scope drift')
 for k,v in t1['fields'].items():
  if int(unique(r'STAGE4V_T1 '+k+r'=(-?\d+)'))!=v:raise ValueError('new Full-K Tiling differs '+k)
 if unique(r'STAGE4V_T1_CONTRACT match=(\d+) effectiveSharedUb=0')!='1':raise ValueError('Tiling contract rejected')
 x=list(map(int,unique(r'STAGE4V_SCHEDULE coreLimit=(\d+) blocks=(\d+) cores=(\d+) system=(\d+) arena=(\d+) stride=(\d+) user=(\d+) total=(\d+) mode=(\d+)')))
 l,blocks,cores,system,arena,stride,user,total,m=x
 if l!=limit or m!=mode or cores!=policy['selected'] or blocks!=(limit or min(76,cores)) or blocks>cores or system!=t1['system_workspace_bytes'] or arena!=(system+511)//512*512 or stride!=arena+1024 or user!=blocks*stride or total!=system+user or not 0<total<=512*1024*1024:raise ValueError('Workspace/Core formula mismatch')
 return dict(blocks=blocks,cores=cores,core_policy=policy,total=total,system=system,arena=arena,stride=stride,user=user,owners=[[t for t in range(i,76,blocks) if tile<0 or tile==t] for i in range(blocks)])

def expected(maps,mode,blocks,tile,repeat):
 y=np.full((16,19456),0x7f7f if repeat==0 else 0xffff,'<u2')
 if mode==2:
  raw=bytearray(y.tobytes());raw[:32]=struct.pack('<8I',0x53565357,3,384,32,36864,0,1,1);return bytes(raw)
 if mode>=3:
  raw=bytearray(y.tobytes());raw[:32]=struct.pack('<8I',0x53564641,mode-3,0x80000001,384,2,1,63,mode-2);return bytes(raw)
 for t in (range(76) if tile<0 else [tile]):
  value=np.full((16,256),t%blocks+1+t/16,'<f2') if mode==1 else maps['golden_fp16.bin'][:,t*256:(t+1)*256]
  y[:,t*256:(t+1)*256]=value.view('<u2')
 return y.tobytes()

def check_execution(ex,device,mode,limit,tile,timed):
 required=dict(status='EXECUTION_COMPLETED_GOLDEN_MATCH',op=NAME,soc='Ascend310P3',shape=[16,2560,19456],tile_count=76,group_count=1,guards=True,readonly=True,repeat_equal=True,repetitions=2,device=device,cpu_fallback=False,output_mode=mode,core_limit=limit,tile_id=tile,guard_coverage=COVERAGE,kernel_guards_unchanged=False,reference_kind='FAULT_WITNESS' if mode>=2 else 'SCHEDULER_WITNESS' if mode==1 else 'CPU_FULL_K_REFERENCE',prepare_calls=38 if timed else 2,verify_calls=4 if timed else 2)
 if any(type(ex.get(k))!=type(v) or ex[k]!=v for k,v in required.items()):raise ValueError('Full-K execution protocol differs')
 # Validate shared continuous-v1 and external checks only after Full-K contract.
 normalized=dict(ex,op=q.bc.full.NAME,tile_count=304,group_count=20,output_mode=0,core_limit=0,kernel_guards_unchanged=True)
 return q.bc.check_execution(normalized,device,0,timed)

def validate_row(row,r,timed=False,host=False):
 mode,limit,tile=row['mode'],row['core_limit'],row['tile_id'];ex=read(verify(row['execution']))
 pl=plan(verify(row['log']).read_text(),mode,limit,tile,r['t1'],r['evidence']['formal_core_policy'])
 adm=read(verify(row['admission']));want=dict(status='ADMITTED_BEFORE_EXECUTION',workspace_bytes=ex['workspace_bytes'],declared_total=pl['total'],blocks=pl['blocks'],cores=pl['cores'],core_policy=pl['core_policy'],host_proof_checked=not host)
 if adm!=want or ex['workspace_bytes']<pl['total']:raise ValueError('Workspace admission differs')
 if host:
  if ex!=dict(status='HOST_TILING_ONLY',npu='NOT_RUN',device=r['device'],workspace_bytes=ex['workspace_bytes'],output_mode=0,core_limit=0,tile_id=-1):raise ValueError('Host-only scope differs')
  return dict(status='HOST_TILING_ONLY',plan=pl)
 timing=check_execution(ex,r['device'],mode,limit,tile,timed);maps=verify_case(read(verify(r['case'])))
 count=4 if timed else 2
 if len(row['outputs'])!=count:raise ValueError('outputs incomplete')
 failures=[]
 for i,f in enumerate(row['outputs']):
  raw=verify(f).read_bytes();target=expected(maps,mode,pl['blocks'],tile,min(i,1))
  if raw!=target:
   a=np.frombuffer(raw,'<u2');e=np.frombuffer(target,'<u2');idx=np.flatnonzero(a!=e)
   first=int(idx[0]);m,n=divmod(first,19456)
   failures.append(dict(output=i,bit_mismatches=int(idx.size),first=dict(row=m,channel=n,tile=n//256,actual_bits=int(a[first]),reference_bits=int(e[first]),dot=int(maps['dot_i32.bin'][m,n]),scale_word=int(maps['quant_u64_doc_candidate.bin'][n//256,n%256]))))
 return dict(status='MATCH' if not failures else 'NUMERICAL_FAIL',failures=failures,plan=pl,timing=timing,workspace_bytes=ex['workspace_bytes'],coverage=COVERAGE,reference_kind=ex['reference_kind'])

def verify_build(b,r):
 from build_v import installed
 if b['status']!='STAGE4V_BUILT_NPU_NOT_RUN' or any(b[k]!=r[k] for k in ('sources','evidence','device','case','t1','prepare_report','host_report')):raise ValueError('Build binding differs')
 for name in ('normal','fault'):
  v=b['variants'][name];verify_audit(v['audit']);item=v['build']
  if v['audit']['fault']!=(name=='fault') or item['status']!='STAGE4V_PRIVATE_BUILT' or installed(Path(item['opp']))!=item['installed']:raise ValueError('wrong installed variant')
  want={Path(x['generated']['path']).parent.name+'/'+Path(x['generated']['path']).name:verify(x['generated']).read_bytes() for x in v['audit']['files'] if Path(x['generated']['path']).suffix in ('.cpp','.h')}
  want['op_host/t1_contract.h']=contract_header(r['t1']).encode()
  got={Path(x['path']).parent.name+'/'+Path(x['path']).name:verify(x).read_bytes() for x in item['generated_sources']}
  if got!=want:raise ValueError('compiled source differs')
  if (Path(item['opp']).parent/'operator.json').read_bytes()!=(HERE/'operator.json').read_bytes():raise ValueError('OPP definition changed')
  if item['linked_libraries']['LIB_custom']!=item['installed']['api']:raise ValueError('Runner links another private API')
  q.verify_tree(item)
 if {x['binary']['sha256'] for x in b['variants']['normal']['build']['installed']['kernels']}=={x['binary']['sha256'] for x in b['variants']['fault']['build']['installed']['kernels']}:raise ValueError('fault binary not independent')
