"""Conservative host capacity checks, not device allocation/lifetime certification."""
CONFIGS={'FULL_K_N256':(256,256,202240,8704)}
def align(v,a):return (v+a-1)//a*a

def l1_contract(p,capacity):
 """Fixed no-bias GM INT8 / CFG_NORM contract, not a general SDK tiler."""
 checks={}
 fields=('baseM','baseN','baseK','singleCoreM','singleCoreN','singleCoreK',
         'depthA1','depthB1','stepM','stepN','stepKa','stepKb','shareL1Size','shareMode','isBias')
 checks['l1_integer_fields']=all(type(p.get(k)) is int and 0<=p[k]<=2147483647 for k in fields)
 if not checks['l1_integer_fields']:
  return checks,dict(allocation_bytes=None)
 checks['l1_fixed_scope']=([p[k] for k in ('baseM','baseN','baseK','singleCoreM','singleCoreN','singleCoreK')]==[16,256,128,16,256,2560]
                           and p['shareMode']==0 and p['isBias']==0)
 blocks=20
 checks['l1_step_mn']=p['stepM']==1 and p['stepN']==1
 # Conservative admitted family: no partial step batches in this fixed20-block K.
 # These are explicit Host admission checks, NOT claims about dav_2002 debug asserts.
 steps=[p['stepKa'],p['stepKb']]
 checks['l1_step_k']=all(0<x<=blocks and blocks%x==0 for x in steps)
 checks['l1_step_compatibility']=all(x>0 for x in steps) and (steps[0]%steps[1]==0 or steps[1]%steps[0]==0)
 counts=[]
 for tag,major,step,depth in [('a',p['stepM'],p['stepKa'],p['depthA1']),('b',p['stepN'],p['stepKb'],p['depthB1'])]:
  batch=major*step
  ok=batch>0 and 0<depth<=blocks and depth%batch==0 and depth//batch in (1,2)
  checks['l1_'+tag+'_depth_step']=ok
  counts.append(depth//batch if ok else None)
 a=2048*p['depthA1'];b=32768*p['depthB1'];total=a+b
 checks['l1_report_matches_depths']=total>0 and p['shareL1Size']==total
 checks['L1_capacity']=type(capacity) is int and 0<total<=capacity and 0<p['shareL1Size']<=capacity
 return checks,dict(a1_basic_block_bytes=2048,b1_basic_block_bytes=32768,a1_bytes=a,b1_bytes=b,
  allocation_bytes=total,reported_shareL1Size=p['shareL1Size'],unexplained_reported_bytes=p['shareL1Size']-total,
  a_step_batches=counts[0],b_step_batches=counts[1],
  normal_allocator='cacheNum is total basic blocks; cache plus queue sums to depth, not depth*step',
  extra_l1_consumers=0,shareL1Size_role='same allocation pool bound, not an additive allocation',
  scope='fixed GM INT8 A/B, no Bias, Scale in UB; no device execution claim')

def validate(platform,row):
 name=row['name'];n,bn,scratch,user=CONFIGS[name];fail=[];checks={};detail={}
 def check(k,ok):
  checks[k]=bool(ok)
  if not ok:fail.append(k)
 check('real_platform',platform.get('platform_source')=='NO_ARGUMENT_MANAGER_ON_SELECTED_DEVICE' and platform.get('soc')=='Ascend310P3' and platform.get('arch')==2002)
 cores=[platform.get(k) for k in ('aic','aiv','all')]
 check('coupled_core_counts',all(type(x) is int and x>0 for x in cores) and cores[0]==cores[1] and cores[2]<=cores[0])
 check('fixed_request',(row.get('requested_n'),row.get('requested_base_n'),row.get('scratch'))==(n,bn,scratch))
 check('sdk_setup',row.get('setup_accepted') is True)
 check('sdk_get_tiling',type(row.get('get_tiling_rc')) is int and row['get_tiling_rc']==0)
 p=row.get('fields') or {};cap=platform.get('capacity',{})
 required=('usedCoreNum','singleCoreM','singleCoreN','singleCoreK','baseM','baseN','baseK','depthA1','depthB1','dbL0A','dbL0B','dbL0C','transLength','stepM','stepN','stepKa','stepKb','shareMode','shareL1Size','shareL0CSize','shareUbSize','isBias')
 check('required_sdk_fields',all(type(p.get(k)) is int for k in required))
 check('platform_capacities',all(type(cap.get(k)) is int and cap[k]>0 for k in ('UB','L1','L0A','L0B','L0C')))
 if not checks['required_sdk_fields'] or not checks['platform_capacities']:
  return dict(name=name,status='BLOCKED',checks=checks,blockers=fail,details=detail)
 check('L0B_310P3_limit',cap['L0B']==65536)
 check('single_core_target',[p[k] for k in ('usedCoreNum','singleCoreM','singleCoreN','singleCoreK')]==[1,16,n,2560])
 check('base_target',[p[k] for k in ('baseM','baseN','baseK')]==[16,bn,128])
 check('full_k_original_shape',all(p.get(k)==v for k,v in [('Ka',2560),('Kb',2560),('M',16),('N',256)]))
 check('no_bias',p['isBias']==0)
 check('positive_depths',all(p[k]>0 for k in ('depthA1','depthB1','dbL0A','dbL0B','dbL0C','stepM','stepN','stepKa','stepKb')))
 check('nonnegative_sharing',all(p[k]>=0 for k in ('shareMode','shareL1Size','shareL0CSize','shareUbSize')))
 # Verified MatmulTiling::GetShareUbSize() returns zero on dav_2002.
 # Keep Host raw value; do not equate it with a TPipe allocation or erase it.
 effective_sdk_ub=0 if platform.get('arch')==2002 else p['shareUbSize']
 check('shared_ub_accounted',platform.get('arch')==2002 and 0<=p['shareUbSize']<=2147483647)
 check('default_share_mode',p['shareMode']==0)
 total=scratch+user+1536+8192+effective_sdk_ub
 check('ub_capacity',total<=cap['UB'] and cap['UB']==262144)
 check('scratch_budget',scratch<=202240 and scratch%256==0)
 scale=n*8;trans=p['transLength']
 check('scale_region',trans>=0 and trans%32==0 and trans+scale<=scratch)
 a=align(p['baseM'],16)*align(p['baseK'],32)
 b=align(p['baseN'],16)*align(p['baseK'],32)
 acc=align(p['baseM'],16)*align(p['baseN'],16)*4
 # Source-proven NORMAL allocation; reject unaccounted shared-pool differences.
 l1_checks,l1_detail=l1_contract(p,cap['L1'])
 for key,ok in l1_checks.items():check(key,ok)
 l1=l1_detail['allocation_bytes']
 check('l0_double_buffer_factors',all(p[k] in (1,2) for k in ('dbL0A','dbL0B','dbL0C')))
 l0a=a*p['dbL0A'];l0b=b*p['dbL0B'];l0c=acc*p['dbL0C']+p['shareL0CSize']
 for k,value in [('L0A',l0a),('L0B',l0b),('L0C',l0c)]:check(k+'_capacity',0<value<=cap[k])
 # Reviewed normal nontransposed INT8 ND input needs two B conversion tiles.
 # Local VECIN/NZ C is user-owned; additionally bound an entire output staging
 # tile inside scratch conservatively, rather than assuming omitted space is free.
 layout=2*b+16*n*2
 check('nd_conversion_and_c_bound',layout<=scratch)
 system=platform.get('system_workspace_bytes')
 check('system_workspace_known',type(system) is int and system>0)
 detail=dict(requested_scratch=scratch,user_ub=user,guards=1536,reserve=8192,total_ub=total,scale_storage_bound=scale,
  transLength=trans,b_two_conversion_tiles=2*b,full_c_staging_bound=16*n*2,
  scratch_peak_bound=max(layout,trans+scale),l1_conservative_bound=l1,l1_allocation=l1_detail,l0a_bound=l0a,l0b_bound=l0b,l0c_bound=l0c,
  reported_shareUbSize=p['shareUbSize'],effective_sdk_reserved_ub=effective_sdk_ub,extra_shared_ub=effective_sdk_ub,
  shared_ub_policy='DAV_2002_GETTER_RETURNS_ZERO_IDENTITY_BOUND',
  coefficient_region_in_scratch=[trans,trans+p['baseN']*8],coefficient_logical_bound=[trans,trans+scale],
  coefficient_bytes_already_inside_scratch=p['baseN']*8,
  field_generation_formula='UNRESOLVED; capacity accounting uses actual returned transLength and verified device consumer',
  shared_l1=p['shareL1Size'],shared_l0c=p['shareL0CSize'],
  padding='M/N rounded to16; INT8 K rounded to32; fixed requests have no tail padding',
  scope='SDK GetTiling acceptance plus conservative capacity checks; not runtime allocation, alias or event proof',
  base_output_contract='one M/N output per Full-K logical Matmul; internal K blocks are not outer partial outputs',
  device_allocation='NOT_RUN')
 return dict(name=name,status='HOST_RESOURCE_CHECKS_PASS' if not fail else 'BLOCKED',checks=checks,blockers=fail,details=detail)
