"""Conservative host capacity checks, not device allocation/lifetime certification."""
CONFIGS={'N64':(64,64,239104,12288),'T1':(128,128,226816,24576),'T2':(128,64,226816,24576)}
def align(v,a):return (v+a-1)//a*a

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
 check('sdk_get_tiling',type(row.get('get_tiling_rc')) is int and row['get_tiling_rc']>=0)
 p=row.get('fields') or {};cap=platform.get('capacity',{})
 required=('usedCoreNum','singleCoreM','singleCoreN','singleCoreK','baseM','baseN','baseK','depthA1','depthB1','dbL0A','dbL0B','dbL0C','transLength','stepM','stepN','stepKa','stepKb','shareMode','shareL1Size','shareL0CSize','shareUbSize','isBias')
 check('required_sdk_fields',all(type(p.get(k)) is int for k in required))
 check('platform_capacities',all(type(cap.get(k)) is int and cap[k]>0 for k in ('UB','L1','L0A','L0B','L0C')))
 if not checks['required_sdk_fields'] or not checks['platform_capacities']:
  return dict(name=name,status='BLOCKED',checks=checks,blockers=fail,details=detail)
 check('single_core_target',[p[k] for k in ('usedCoreNum','singleCoreM','singleCoreN','singleCoreK')]==[1,16,n,128])
 check('base_target',[p[k] for k in ('baseM','baseN','baseK')]==[16,bn,128])
 check('no_bias',p['isBias']==0)
 check('positive_depths',all(p[k]>0 for k in ('depthA1','depthB1','dbL0A','dbL0B','dbL0C','stepM','stepN','stepKa','stepKb')))
 check('nonnegative_sharing',all(p[k]>=0 for k in ('shareMode','shareL1Size','shareL0CSize','shareUbSize')))
 # Verified MatmulTiling::GetShareUbSize() returns zero on dav_2002.
 # Keep Host raw value; do not equate it with a TPipe allocation or erase it.
 effective_sdk_ub=0 if platform.get('arch')==2002 else p['shareUbSize']
 check('shared_ub_accounted',platform.get('arch')==2002 and 0<=p['shareUbSize']<=2147483647)
 check('default_share_mode',p['shareMode']==0)
 total=scratch+user+2560+8192+effective_sdk_ub
 check('ub_capacity',total<=cap['UB'] and cap['UB']==262144)
 check('scratch_budget',scratch<=(239104 if n==64 else 226816) and scratch%256==0)
 scale=n*8;trans=p['transLength']
 check('scale_region',trans>=0 and trans%32==0 and trans+scale<=scratch)
 a=align(p['baseM'],16)*align(p['baseK'],32)
 b=align(p['baseN'],16)*align(p['baseK'],32)
 acc=align(p['baseM'],16)*align(p['baseN'],16)*4
 # This is a conservative upper bound, not a claim about measured live usage.
 l1=a*p['depthA1']*p['stepM']*p['stepKa']+b*p['depthB1']*p['stepN']*p['stepKb']+p['shareL1Size']
 l0a=a*p['dbL0A'];l0b=b*p['dbL0B'];l0c=acc*p['dbL0C']+p['shareL0CSize']
 for k,value in [('L1',l1),('L0A',l0a),('L0B',l0b),('L0C',l0c)]:check(k+'_capacity',0<value<=cap[k])
 # Reviewed normal nontransposed INT8 ND input needs two B conversion tiles.
 # Local VECIN/NZ C is user-owned; additionally bound an entire output staging
 # tile inside scratch conservatively, rather than assuming omitted space is free.
 layout=2*b+16*n*2
 check('nd_conversion_and_c_bound',layout<=scratch)
 system=platform.get('system_workspace_bytes')
 check('system_workspace_known',type(system) is int and system>0)
 detail=dict(requested_scratch=scratch,user_ub=user,guards=2560,reserve=8192,total_ub=total,scale_storage_bound=scale,
  transLength=trans,b_two_conversion_tiles=2*b,full_c_staging_bound=16*n*2,
  scratch_peak_bound=max(layout,trans+scale),l1_conservative_bound=l1,l0a_bound=l0a,l0b_bound=l0b,l0c_bound=l0c,
  reported_shareUbSize=p['shareUbSize'],effective_sdk_reserved_ub=effective_sdk_ub,extra_shared_ub=effective_sdk_ub,
  shared_ub_policy='DAV_2002_GETTER_RETURNS_ZERO_IDENTITY_BOUND',
  coefficient_region_in_scratch=[trans,trans+p['baseN']*8],coefficient_logical_bound=[trans,trans+scale],
  coefficient_bytes_already_inside_scratch=p['baseN']*8,
  field_generation_formula='UNRESOLVED; capacity accounting uses actual returned transLength and verified device consumer',
  shared_l1=p['shareL1Size'],shared_l0c=p['shareL0CSize'],
  padding='M/N rounded to16; INT8 K rounded to32; fixed requests have no tail padding',
  scope='SDK GetTiling acceptance plus conservative capacity checks; not runtime allocation, alias or event proof',
  t2_output_contract='two N64 base outputs per logical N128 group; placement/lifecycle requires a later Kernel review' if name=='T2' else 'one base output per group',
  device_allocation='NOT_RUN')
 return dict(name=name,status='HOST_RESOURCE_CHECKS_PASS' if not fail else 'BLOCKED',checks=checks,blockers=fail,details=detail)


def decide(resources,layout_ok):
 rows={x['name']:x for x in resources}
 valid=[n for n in ('T1','T2') if rows.get(n,{}).get('status')=='HOST_RESOURCE_CHECKS_PASS']
 ok=layout_ok and rows.get('N64',{}).get('status')=='HOST_RESOURCE_CHECKS_PASS' and bool(valid)
 return dict(status='STAGE4I_N128_HOST_TILING_FEASIBLE' if ok else 'STAGE4I_N128_BLOCKED',feasible_candidates=valid if ok else [],
  policy='N64 live control and at least one explicitly requested N128 candidate must pass; candidate failures remain separately reported',
  npu_kernel_execution='NOT_RUN',benchmark='NOT_RUN',profile='NOT_RUN',n128_kernel_compile='NOT_RUN',next_kernel='REQUIRES_EXPLICIT_ADMISSION')
