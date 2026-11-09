"""Conditional resource/dependency contracts; not executable device admission."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4v_vdeq16_probe'))
import vdeq_common as prev
identity,verify,read,write,fresh=prev.identity,prev.verify,prev.read,prev.write,prev.fresh


def sources():return [identity(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]


def decision():
 return dict(status='BLOCKED',reason='FORMAL_CONTEXT_OBSERVATION_AND_EXTERNAL_CO1_RESOURCE_ADMISSION_MISSING',
  source_contract='CONDITIONAL_SOURCE_SUPPORT_WITH_EXPLICIT_EVENTS',formal_50_fields='BLOCKED_NO_VERIFIED_OBSERVATION_INTERFACE',
  external_co1_formal_host='NOT_RUN',npu='NOT_RUN',kernel='NOT_CREATED',opp='NOT_CREATED',device_runner='NOT_CREATED',
  cause='UNRESOLVED',benchmark='PROHIBITED')


def dependencies():
 return [
  dict(producer='Full-K MMAD (PIPE_M)',consumer='INT32 Echo DataCopy (PIPE_V)',buffer='external CO1',required='M_V SetFlag/WaitFlag after final MMAD',evidence='event:572-575,902-909,1043-1048',status='SOURCE_SUPPORTED_NOT_AUTOMATIC'),
  dict(producer='GM Scale load (MTE2)',consumer='VDEQ16 (V)',buffer='256 UINT64 words in UB',required='MTE2_V SetFlag/WaitFlag',evidence='quant:143-158; event:890-891,1030-1032',status='SOURCE_SUPPORTED'),
  dict(producer='INT32 Echo (V)',consumer='VDEQ16 (V)',buffer='same read-only CO1; disjoint UB destinations',required='preserve PIPE_V/PIPE_ALL ordering; no L0C writer/rebind',evidence='external:37-61; datacopy:487-505,542-572',status='CONDITIONAL_ON_OWNERSHIP'),
  dict(producer='Echo/VDEQ16 (V)',consumer='Scalar UB Guard read (S)',buffer='UB payload/redzones',required='V_S SetFlag/WaitFlag',evidence='event:914-916,1117-1119',status='SOURCE_SUPPORTED'),
  dict(producer='Echo/VDEQ16 (V)',consumer='GM output copy (MTE3)',buffer='UB echo/FP16 output',required='V_MTE3 SetFlag/WaitFlag',evidence='event:899-901,1039-1041',status='SOURCE_SUPPORTED'),
  dict(producer='last L0C consumer (V)',consumer='subsequent Cube writer (M)',buffer='external CO1',required='V_M SetFlag/WaitFlag before reuse; caller lifetime retained',evidence='event:908-909,1047-1048; external:59-61',status='SOURCE_SUPPORTED_NO_REUSE_IN_INITIAL_SCOPE'),
  dict(producer='End cleanup',consumer='any result reader',buffer='input queues/L0 pool state',required='End is NOT a substitute for M_V or consumer completion',evidence='scheduler_base:149-167; external:61',status='NO_IMPLICIT_COMPLETION_CLAIM'),
 ]


def conditional_budget(fields,capacity,system_bytes,l0c_start=0):
 """Accept only measured future CO1 fields. Never produces HOST_FEASIBLE."""
 checks={}
 def check(k,v):checks[k]=bool(v)
 keys=('baseM','baseN','baseK','singleCoreM','singleCoreN','singleCoreK','usedCoreNum','depthA1','depthB1','stepM','stepN','stepKa','stepKb','dbL0A','dbL0B','dbL0C','shareL1Size','shareL0CSize','shareUbSize','transLength','isBias','shareMode','Ka','Kb','M','N')
 if any(type(fields.get(k))is not int for k in keys) or any(type(capacity.get(k))is not int or capacity[k]<=0 for k in ('UB','L1','L0A','L0B','L0C')):raise ValueError('complete actual Tiling/capacities required')
 p=fields
 check('shape',[p[k] for k in ('singleCoreM','singleCoreN','singleCoreK','baseM','baseN','baseK','M','N','Ka','Kb')]==[16,256,2560,16,256,128,16,256,2560,2560])
 check('scope',p['usedCoreNum']==1 and p['isBias']==0 and p['shareMode']==0)
 from resources_v import l1_contract
 l1_checks,l1= l1_contract(p,capacity['L1']);checks.update(l1_checks)
 check('db',all(p[k] in (1,2) for k in ('dbL0A','dbL0B','dbL0C')))
 check('l0a',2048*p['dbL0A']<=capacity['L0A'])
 check('l0b',32768*p['dbL0B']<=capacity['L0B'])
 # First/only caller-owned CO1 allocation, before Matmul registration. No normal queue C allocation.
 check('external_l0c',type(l0c_start)is int and l0c_start==0 and l0c_start%1024==0 and l0c_start+16384<=capacity['L0C'])
 check('reported_shared',p['shareL0CSize']>=0 and p['shareUbSize']>=0)
 # Four disjoint UB payloads; original Reserve is not consumed. No assumption that CO1 needs zero scratch.
 payload=dict(scratch=202240,echo=16384,scale=2048,fp16=8192)
 total=sum(payload.values())+2048+8192
 check('ub',capacity['UB']==262144 and total<=capacity['UB'])
 check('scratch',0<=p['transLength']<=202240 and p['transLength']%32==0 and 2*32768<=202240)
 check('workspace',type(system_bytes)is int and system_bytes>0 and system_bytes+(system_bytes+511)//512*512+1024<=512*1024*1024)
 return dict(status='CONDITIONAL_ARITHMETIC_CHECKS_PASS' if all(checks.values()) else 'BLOCKED',checks=checks,
  blockers=[k for k,v in checks.items() if not v],l1=l1,ub=dict(payload=payload,guards=2048,reserve=8192,total=total),
  l0c=dict(active_bytes=16384,start=l0c_start,reported_shared=p['shareL0CSize'],effective_sdk_shared=0,reason='CType physical L0C: SchedulerBase sets sharedL0cSize=0; external allocator owns tensor'),
  shared_ub=dict(reported=p['shareUbSize'],effective=0,reason='dav_2002 getter'),
  formal_host='NOT_PROVEN_BY_THIS_CALCULATION')


def validate_trace(actions):
 """CPU dependency-state model, not an emulator or a Kernel implementation."""
 allocated=False;pending=False;ready=False;echo=False;deq=False;vector_done=False;released=False
 for op in actions:
  if op=='allocate':
   if allocated:raise ValueError('multiple/aliased external allocations')
   allocated=True
  elif op=='mmad':
   if not allocated or pending or released or (ready and not vector_done):raise ValueError('premature Cube write')
   pending=True;ready=False;vector_done=False;echo=False;deq=False
  elif op=='M_V':
   if not pending:raise ValueError('event without producer')
   ready=True;pending=False
  elif op in ('echo','vdeq'):
   if not ready or released:raise ValueError('consumer without completion/lifetime')
   if op=='echo':echo=True
   else:
    if not echo:raise ValueError('diagnostic requires full echo first')
    deq=True
  elif op=='consumer_complete':
   if not echo or not deq:raise ValueError('missing consumer')
   vector_done=True
  elif op=='end':
   # Conservative review constraint: retain End only after both consumers.
   # It is never used to establish producer/consumer completion.
   if pending or not vector_done:raise ValueError('End before explicit producer/consumer completion')
  elif op=='release':
   if pending or not vector_done:raise ValueError('premature release')
   released=True
  else:raise ValueError('unknown operation')
 if not (echo and deq and vector_done):raise ValueError('incomplete observation')
 return 'CPU_DEPENDENCY_MODEL_ONLY'


def resource_design():
 return dict(status='DESIGN_ONLY_NOT_FORMAL_RESOURCE_ADMISSION',ub=dict(scratch_ceiling=202240,echo=16384,scale=2048,fp16=8192,guards=2048,reserve=8192,total=239104),
  l0c=dict(first_and_only_allocation=16384,start_required=0,alignment_required=1024,effective_sdk_shared_if_ctype_co1=0),
  pending=['actual CO1 GetTiling return and all50 fields','actual Formal TilingContext capacities/core policy','L1 depths/steps, L0A/B db factors','SDK transLength bound and any additional consumers','system Workspace and arena request','runtime event ID availability and address checks'],
  historical_vecin_resources='REFERENCE_ONLY_NOT_CO1_PROOF',host_query='NOT_RUN')
