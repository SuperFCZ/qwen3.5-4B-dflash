"""Independent Raw INT32 budget; no FP16 Scale/codec admission dependency."""
from evidence import v
from resources_v import l1_contract

def validate(platform,row):
 p=row.get('fields') or {};cap=platform.get('capacity',{});checks={}
 def check(k,x):checks[k]=bool(x)
 check('device',platform.get('soc')=='Ascend310P3' and platform.get('arch')==2002 and platform.get('platform_source')=='NO_ARGUMENT_MANAGER_ON_SELECTED_DEVICE')
 check('request',row.get('name')=='RAW_INT32_N256' and (row.get('requested_n'),row.get('requested_base_n'),row.get('scratch'))==(256,256,202240))
 check('raw_type',row.get('c_dtype')=='INT32' and row.get('c_position')=='VECIN' and row.get('c_format')=='NZ' and row.get('scale_input') is False and 'SetDequantType' not in row.get('setters',{}))
 check('get_tiling',row.get('setup_accepted') is True and type(row.get('get_tiling_rc')) is int and row['get_tiling_rc']==0)
 check('fields',all(type(p.get(k)) is int for k in v.p.FIELDS))
 check('capacity',all(type(cap.get(k)) is int and cap[k]>0 for k in ('UB','L1','L0A','L0B','L0C')))
 if not checks['fields'] or not checks['capacity']:return dict(status='BLOCKED',checks=checks,blockers=[k for k,x in checks.items() if not x])
 check('full_k',p['usedCoreNum']==1 and [p[k] for k in ('singleCoreM','singleCoreN','singleCoreK','baseM','baseN','baseK','M','N','Ka','Kb')]==[16,256,2560,16,256,128,16,256,2560,2560])
 lc,ld=l1_contract(p,cap['L1']);checks.update(lc)
 check('l0_double_buffer',all(p[k] in (1,2) for k in ('dbL0A','dbL0B','dbL0C')))
 check('L0A',0<2048*p['dbL0A']<=cap['L0A'])
 check('L0B',cap['L0B']==65536 and 0<32768*p['dbL0B']<=cap['L0B'])
 check('L0C',p['shareL0CSize']>=0 and 0<16384*p['dbL0C']+p['shareL0CSize']<=cap['L0C'])
 check('UB',cap['UB']==262144 and 202240+16384+512+1536+8192<=cap['UB'])
 check('shared_ub',0<=p['shareUbSize']<=2147483647) # identity-bound2002 getter returns0
 check('scratch',0<=p['transLength']<=202240 and p['transLength']%32==0 and 2*32768+16384<=202240)
 system=platform.get('system_workspace_bytes');check('workspace',type(system) is int and system>0 and system+(system+511)//512*512+1024<=512*1024*1024)
 return dict(status='RAW_RESOURCE_CHECKS_PASS' if all(checks.values()) else 'BLOCKED',checks=checks,blockers=[k for k,x in checks.items() if not x],
  l1=ld,ub=dict(scratch=202240,int32_output=16384,arena_temp=512,guards=1536,reserve=8192,total=228864,scale_buffer=0,effective_shared=0,reported_shared=p['shareUbSize']),
  layout=dict(local='NZ[16,16,16] INT32',local_index='(n//16)*256+m*16+n%16',gm='ND[16,256] INT32',output_bytes=16384),
  proposed_block_dim=1,tiling_context='NOT_RUN_REQUIRES_SEPARATE_FORMAL_OP_HOST',npu='NOT_RUN')
