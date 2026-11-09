"""Pure CPU event-ownership and drain specifications, not SDK/device emulation."""
from copy import deepcopy
import phase2_contract as a

MANUAL_B=('M_V','V_M','V_S','MTE2_V','V_MTE3','MTE3_MTE2','MTE2_S','S_MTE3','MTE3_S')


class Pools:
 """Separate software reservations from hardware-in-flight obligations.
 Fetch models bitmap lookup only. Other methods enforce the proposed caller
 contract, including constraints the SDK allocator itself does NOT enforce.
 """
 def __init__(self,events=a.EVENTS,limit=8):
  self.limit=limit;self.reserved={e:{} for e in events};self.active={};self.completed=set()
 def fetch(self,event):
  return next((i for i in range(self.limit) if i not in self.reserved[event]),self.limit)
 def alloc(self,event,owner='app'):
  slot=self.fetch(event)
  if slot>=self.limit:raise ValueError('preflight exhausted; do not call raw SDK Alloc')
  if (event,slot) in self.active:raise ValueError('allocation would collide with unreserved SDK use')
  self.reserved[event][slot]=owner;return slot
 def use(self,event,slot,owner='app'):
  if type(slot)is not int or not 0<=slot<self.limit:raise ValueError('invalid ID')
  key=(event,slot)
  if key in self.active:raise ValueError('same event/ID still active')
  if owner=='sdk':
   if slot in self.reserved[event]:raise ValueError('SDK selected reserved app ID')
  elif self.reserved[event].get(slot)!=owner:raise ValueError('no application reservation')
  self.completed.discard(key);self.active[key]=dict(owner=owner,wait_issued=False)
 def issue_wait(self,event,slot):
  key=(event,slot)
  if key not in self.active:raise ValueError('unmatched wait')
  self.active[key]['wait_issued']=True
 def observe_completed_wait(self,event,slot):
  # External proof token, NOT a consequence of issuing WaitFlag in this model.
  key=(event,slot)
  if key not in self.active or not self.active[key]['wait_issued']:raise ValueError('no issued wait')
  del self.active[key];self.completed.add(key)
 def release(self,event,slot):
  if slot not in self.reserved[event]:raise ValueError('cannot release SDK-only Fetch')
  if (event,slot) in self.active:raise ValueError('premature release; hardware use remains')
  del self.reserved[event][slot]
 def snapshot(self,event):
  return dict(free=[i for i in range(self.limit) if i not in self.reserved[event]],
   allocated=sorted(self.reserved[event]),sdk_fetch_candidate=self.fetch(event),
   in_use=sorted(i for e,i in self.active if e==event),
   wait_issued=sorted(i for (e,i),x in self.active.items() if e==event and x['wait_issued']),
   wait_completed=sorted(i for e,i in self.completed if e==event),
   releasable=sorted(i for i in self.reserved[event] if (event,i) not in self.active))


def sv_trace(plan):
 p=Pools();rows=[]
 def snap(t):rows.append(dict(time=t,hard_event='S_V',**p.snapshot('S_V')))
 snap('initial')
 if plan=='A':
  token=p.alloc('S_V');snap('manual reserved')
  p.use('S_V',token);p.issue_wait('S_V',token);snap('manual wait issued; not yet complete')
  p.observe_completed_wait('S_V',token);snap('V_S join witnessed completion')
  p.release('S_V',token);snap('manual release before SDK window')
 elif plan!='B':raise ValueError('unknown design')
 for j in range(16):
  slot=p.fetch('S_V');snap(f'j{j}: SDK Fetch, mask unchanged')
  p.use('S_V',slot,'sdk');p.issue_wait('S_V',slot);snap(f'j{j}: SDK Set/Wait issued')
  # Proposed V_S scalar completion join after each DataCopy; not inferred from C++ return.
  p.observe_completed_wait('S_V',slot);snap(f'j{j}: V_S join confirms prior V work complete')
 return rows


def manual_pool_trace():
 """One modeled lifetime per independent pool, using external completion tokens."""
 p=Pools();rows=[]
 for event in MANUAL_B:
  slot=p.alloc(event)
  rows.append(dict(time='reserved before user producers',hard_event=event,**p.snapshot(event)))
 for event in MANUAL_B:
  slot=next(iter(p.reserved[event]))
  p.use(event,slot);p.issue_wait(event,slot)
  rows.append(dict(time='wait issued, completion unproved',hard_event=event,**p.snapshot(event)))
  p.observe_completed_wait(event,slot)
  rows.append(dict(time='required completion chain observed',hard_event=event,**p.snapshot(event)))
  p.release(event,slot)
  rows.append(dict(time='released after last completed use',hard_event=event,**p.snapshot(event)))
 return dict(scope='CPU_MODEL_EXTERNAL_COMPLETION_TOKENS_NOT_DEVICE_TRACE',rows=rows,
             sdk_remaining_capacity='UNPROVED_SELECTED_MATMUL_PEAK_DEMAND')


def layout_b():
 r=deepcopy(a.layout());r.update(variant='B_PROPOSED',explicit_ub_bytes=232192,total_budget=240384,unallocated_margin=21760,formal_host='REQUIRED_NEW_BUDGET_NOT_RUN')
 r['buffers']['diagnostic']=dict(raw=230912,payload=231168,bytes=768,end=232192)
 r['guards'] += [[230912,231168],[231936,232192]]
 r['aliases']={};r['diagnostic_views']=dict(arena=[231168,231680],witness=[231680,231936])
 r['additional_allocation']=dict(arena=512,witness=256,guard=512,total=1280)
 return r


class Drain:
 """Safety requirements for ordinary errors; never a hardware progress oracle."""
 def __init__(self,tickets=MANUAL_B,sdk_initialized=True):
  self.tickets=set(tickets);self.pending=set();self.healthy=True;self.m_join=False
  self.sdk_initialized=sdk_initialized;self.sdk_cleanup_proven=not sdk_initialized;self.ended=False;self.released=False;self.success=False
 def issue(self,op):
  if not self.healthy or self.released:raise ValueError('hardware state unavailable/released')
  if self.ended and op=='mmad':raise ValueError('no Matmul restart in error cleanup')
  needed={'mmad':{'M_V','V_S'},'vector':{'V_S'},'load':{'MTE2_S'},'store':{'MTE3_S'}}[op]
  if not needed<=self.tickets:raise ValueError('required drain ticket missing BEFORE issue')
  if op=='mmad':self.m_join=False;self.sdk_cleanup_proven=False
  self.pending.add(op)
 def join(self,event):
  if not self.healthy:raise ValueError('no completion proof after hardware fault/timeout')
  if event not in self.tickets:raise ValueError('late allocation forbidden')
  if event=='M_V':
   if 'mmad' not in self.pending:raise ValueError('do not wait for an unissued producer')
   self.m_join=True # consumer V wait issued; S not yet permitted to clean up
  elif event=='V_S':
   if 'vector' not in self.pending and not self.m_join:raise ValueError('do not wait for an unissued producer')
   self.pending.discard('vector')
   if self.m_join:self.pending.discard('mmad');self.m_join=False
  elif event in ('MTE2_S','MTE3_S'):
   op='load' if event=='MTE2_S' else 'store'
   if op not in self.pending:raise ValueError('do not wait for an unissued producer')
   self.pending.remove(op)
  else:raise ValueError('not a scalar-observed completion join')
 def barrier_all_or_return(self):
  # Deliberately credits no completion to an unproven drain shortcut.
  pass
 def fault(self):self.healthy=False
 def prove_sdk_cleanup(self):
  if not self.healthy or self.pending:raise ValueError('SDK cleanup proof cannot precede drains')
  self.sdk_cleanup_proven=True # external theorem/token; pending in this review
 def end(self):
  if not self.healthy or self.pending or not self.sdk_cleanup_proven:raise ValueError('End cannot drain or prove SDK cleanup')
  self.ended=True
 def release_buffers(self):
  if not self.healthy or self.pending or not self.ended:raise ValueError('premature/unsafe buffer release')
  self.released=True
 def publish(self,success=False):
  if not self.healthy or self.pending or not self.ended or self.released:raise ValueError('cannot publish before safe terminal checkpoint or after release')
  if success:raise ValueError('ordinary-error path must never emit success Witness')
  self.issue('store') # publication itself is asynchronous and needs MTE3_S
  self.success=False


def outcome():
 return dict(status='BLOCKED',revision='PHASE2_1',selected_design='B',
  sv='CONDITIONALLY_NONCONFLICTING_NO_APP_S_V_DURING_DIRECT_VDEQ16',
  blockers=['selected Matmul Init/Iterate/End peak event demand with pre-reserved manual drain IDs not established by the four reviewed files',
            'SDK-owned MTE1/L1/L0A/L0B lifetime at every ordinary-error exit not established; M/V/MTE2/MTE3 joins alone do not prove all SDK cleanup',
            'SchemeB actual TPipe/resource/Formal Host admission and independent review pending'],
  hardware_fault='NO_PROVEN_IN_KERNEL_DRAIN; NO_SUCCESS_WITNESS; HOST_RUNTIME_QUIESCENCE_REQUIRED',
  kernel_implementation='NOT_AUTHORIZED',kernel='NOT_CREATED',opp='NOT_CREATED',aclnn='NOT_CALLED',npu='NOT_RUN',benchmark='NOT_RUN',profile='NOT_RUN')
