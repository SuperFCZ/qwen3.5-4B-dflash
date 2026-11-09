"""Pure CPU design contracts. No device API, Kernel, arithmetic emulation or Golden."""
EVENTS=('M_V','V_M','V_S','MTE2_V','V_MTE3','MTE3_MTE2','MTE2_S','S_MTE3','MTE3_S','S_V')
BUFFERS={
 'scratch':dict(raw=0,payload=256,bytes=202240,end=202752),
 'echo':dict(raw=202752,payload=203008,bytes=16384,end=219648),
 'scale':dict(raw=219648,payload=219904,bytes=2048,end=222208),
 'fp16':dict(raw=222208,payload=222464,bytes=8192,end=230912),
}


def geometry():
 return [dict(j=j,channels=[16*j,16*(j+1)],blockCount=1,blockLen=1,srcStride=0,dstStride=0,
  l0c=[1024*j,1024*(j+1)],raw_echo=[203008+1024*j,203008+1024*(j+1)],
  fp16=[222464+512*j,222464+512*(j+1)],scale=[219904+128*j,219904+128*(j+1)]) for j in range(16)]


def validate_geometry(rows):
 if any(any(type(r.get(k))is not int for k in ('j','blockCount','blockLen','srcStride','dstStride')) or any(type(x)is not int for k in ('channels','l0c','raw_echo','fp16','scale') for x in r[k]) for r in rows):raise ValueError('integer geometry fields required')
 if rows!=geometry():raise ValueError('N16 byte/parameter/Scale mapping differs from reviewed design')
 return dict(source_bytes=16384,echo_bytes=16384,fp16_bytes=8192,scale_bytes=2048,scope='CPU_RANGE_PROOF_ONLY')


def layout():
 guards=[]
 for b in BUFFERS.values():guards.extend([[b['raw'],b['payload']],[b['payload']+b['bytes'],b['end']]])
 return dict(buffers=BUFFERS,guards=guards,explicit_ub_bytes=230912,reserve=8192,total_budget=239104,
  unallocated_margin=23040,reserve_physical_top=[253952,262144],
  l0c=dict(start=0,bytes=16384,alignment=1024,guard='RANGE_ONLY_NO_CANARIES'),
  aliases=dict(arena_A=[222464,222976],witness_A=[222976,223232],arena_B=[203008,203520],witness_B=[203520,203776]),
  workspace_reference_bytes=4195328,formal_host='REQUIRED_NEW_ABI_NOT_RUN')


def scale_offsets(raw):
 if len(raw)!=2048:raise ValueError('frozen256 UINT64 words required')
 # Only slice bytes. No re-encoding, float multiplication or coefficient fitting.
 return [raw[128*j:128*(j+1)] for j in range(16)]


def lifetime(actions):
 """Abstract completion markers; explicitly not a model of hardware timing."""
 done=set();vreads=False;location='FP16';pending={};held=set();released=False
 def need(*names):
  if any(n not in done for n in names):raise ValueError('missing completed dependency '+','.join(names))
 for action in actions:
  if isinstance(action,tuple):
   op,event,slot=action
   if event not in EVENTS or type(slot)is not int or not 0<=slot<8:raise ValueError('event ID/type')
   key=(event,slot)
   if op=='alloc':
    if key in held:raise ValueError('event ID already held')
    held.add(key)
   elif op=='signal':
    if key not in held or key in pending:raise ValueError('signal without reservation or reused before completion')
    pending[key]='signaled'
   elif op=='wait_complete':
    if pending.get(key)!='signaled':raise ValueError('unmatched wait')
    del pending[key];done.add(event)
   elif op=='release':
    if key not in held or key in pending:raise ValueError('premature event release')
    held.remove(key)
   else:raise ValueError('unknown event operation')
   continue
  if action=='arena_init_write':done.discard('MTE3_MTE2');done.add(action)
  elif action=='arena_init_read':need('arena_init_write','MTE3_MTE2');done.discard('MTE2_S');done.add(action)
  elif action=='scale_load':done.discard('MTE2_V');done.add(action)
  elif action=='mmad':
   need('arena_init_read','MTE2_S')
   if vreads or released:raise ValueError('L0C overwritten during consumer lifetime')
   done.discard('M_V');done.add(action)
  elif action=='echo':
   need('mmad','M_V')
   if released:raise ValueError('released CO1')
   vreads=True;done.add(action)
  elif action=='raw_gm_issue':need('echo','V_MTE3');done.discard('MTE3_S');done.add(action)
  elif action=='migrate_witness':
   need('raw_gm_issue','MTE3_S');location='ECHO';done.discard('S_V');done.add(action)
  elif action=='vdeq':
   need('echo','scale_load','MTE2_V','migrate_witness','S_V')
   if released:raise ValueError('released CO1')
   if location!='ECHO':raise ValueError('FP16 still contains live witness')
   done.discard('V_S');done.discard('V_M');done.discard('V_MTE3');done.add(action)
  elif action=='end':
   need('vdeq','V_S','V_M');vreads=False;done.add(action)
  elif action=='arena_post':need('end');done.discard('MTE2_S');done.add(action)
  elif action=='fp16_gm_issue':need('vdeq','end','arena_post','MTE2_S','V_MTE3');done.discard('MTE3_S');done.add(action)
  elif action=='finalize_witness':need('fp16_gm_issue','MTE3_S');done.discard('S_MTE3');done.add(action)
  elif action=='publish':need('finalize_witness','S_MTE3');done.add(action)
  elif action=='release_co1':need('end','publish');released=True
  else:raise ValueError('unknown action')
 if 'publish' not in done or held or pending:raise ValueError('incomplete lifecycle/event release')
 return 'CPU_DESIGN_CONTRACT_ONLY_NOT_HARDWARE_EVIDENCE'


def observation_status(raw_match,repeat,safety,match_actual,match_cpu):
 if any(type(x)is not bool for x in (raw_match,repeat,safety,match_actual,match_cpu)):raise ValueError('boolean evidence flags required')
 if not raw_match or not repeat or not safety:return dict(status='BLOCKED',interpret_fp16=False)
 relation='MATCHES_BOTH' if match_actual and match_cpu else 'MATCHES_FROZEN_ACTUAL' if match_actual else 'MATCHES_CPU_GOLDEN' if match_cpu else 'MATCHES_NEITHER'
 return dict(status='DIAGNOSTIC_OBSERVATION_VALID',interpret_fp16=True,relation=relation,hardware_rounding='UNKNOWN',golden='UNCHANGED')
