import hashlib,json
from pathlib import Path
from scheduler_common import c
from platform_contracts import CORE,CAPS,unique_object


def assignments(tiles,blocks):
    if type(tiles) is not int or type(blocks) is not int or not 0<blocks<=tiles:raise ValueError('invalid cyclic partition')
    return [list(range(b,tiles,blocks)) for b in range(blocks)]


def cpu_plan(tiles,generic,aic,aiv):
    if any(type(x) is not int or x<=0 for x in (tiles,generic,aic,aiv)) or aic!=aiv:raise ValueError('coupled resource contract')
    # Correlated source model, NOT a substitute for the installed SDK return value.
    slices=min(tiles,aiv);ratio=aiv//aic;calc=(slices+ratio-1)//ratio
    return dict(baseline=min(tiles,generic),slice_num=slices,expected_calc=calc)


def read_query(platform_path,scheduling_path,opp,route,variant,shape):
    m,k,n=shape;tiles=n//64
    if n%64 or tiles<=0:raise ValueError('partial N tile')
    ps=[json.loads(x,object_pairs_hook=unique_object) for x in Path(platform_path).read_text().splitlines()]
    ss=[json.loads(x,object_pairs_hook=unique_object) for x in Path(scheduling_path).read_text().splitlines()]
    if not ps or len(ps)!=len(ss) or (route=='ACLNN' and len(ps)!=1):raise ValueError('callback evidence count differs')
    p=ps[0];s=ss[0]
    if any(x!=p for x in ps) or any(x!=s for x in ss):raise ValueError('repeated callback drift')
    fields=set(CORE)|{'schema','route','m','k','n','platform_info_present','resource_map_ok','resource_map','capacity','host_dso','soc_enum','arch','workspace'}
    if set(p)!=fields or p['schema']!=1 or p['route']!=route or [p[x] for x in ('m','k','n')]!=list(shape):raise ValueError('platform schema/shape differs')
    if p['platform_info_present'] is not True or p['resource_map_ok'] is not True or p['arch']!=2002 or p['soc_enum']!=2:raise ValueError('not a real dav_2002 310P Context')
    if set(p['capacity'])!=CAPS or any(type(x) is not int or x<=0 for x in p['capacity'].values()):raise ValueError('capacity query failed')
    if any(type(p[x]) is not int or p[x]<0 for x in (*CORE,'workspace')):raise ValueError('platform numeric type')
    if p['coreLimit']!=0 or p['availableCores']!=p['GetCoreNum'] or p['platform_raw_core']!=p['platform_raw_core_locked'] or p['GetCoreNum']!=p['platform_raw_core']:raise ValueError('generic Core view changed')
    model=cpu_plan(tiles,p['GetCoreNum'],p['GetCoreNumAic'],p['GetCoreNumAiv'])
    want=dict(schema=1,variant=variant,route=route,m=m,k=k,n=n,tile_n=64,n_tiles=tiles,core_limit=0,
        generic=p['GetCoreNum'],aic=p['GetCoreNumAic'],aiv=p['GetCoreNumAiv'],slice_num=model['slice_num'],
        calc_tsch_block_dim=model['expected_calc'],baseline_block_dim=model['baseline'],
        selected_block_dim=model['baseline'] if variant=='A' else model['expected_calc'],
        baseline_assignments=assignments(tiles,model['baseline']),candidate_assignments=assignments(tiles,model['expected_calc']),approval='HOST_ONLY_NOT_DEVICE_ADMITTED')
    if set(s)!=set(want)|{'resources','tiling_bytes','tiling_hex','independent_vector'}:raise ValueError('scheduling field inventory differs')
    for key,value in want.items():
        if isinstance(value,int) and type(s[key]) is not int:raise ValueError('scheduling numeric type differs')
    if any(s.get(k)!=v for k,v in want.items()):raise ValueError('SDK scheduling result/assignment differs from coupled contract')
    if p['blockDim']!=s['selected_block_dim']:raise ValueError('actual SetBlockDim differs')
    for k in ('baseline_assignments','candidate_assignments'):
        flat=[t for row in s[k] for t in row]
        if sorted(flat)!=list(range(tiles)):raise ValueError('overlap/missing N tile')
    if not isinstance(p['resource_map'],dict) or any(not isinstance(x,dict) or any(not isinstance(v,str) for v in x.values()) for x in p['resource_map'].values()):raise ValueError('resource map types differ')
    if type(s['independent_vector']) is not int or s['independent_vector']<0:raise ValueError('independent Vector getter type')
    info=p['resource_map'].get('SoCInfo',{})
    if info.get('core_type_list')=='CubeCore,VectorCore':raise ValueError('separated resource model cannot enter coupled candidate')
    if 'vector_core_cnt' not in info or int(info['vector_core_cnt'])!=s['independent_vector']:raise ValueError('GetCoreNumVector/raw resource disagreement')
    r=s['resources']
    if set(r)!={'user_ub','scratch','l1_used','l0a_used','l0b_used','l0c_used','codes_bytes','scale_bytes','weight_bytes','raw_banks','tile_k','workspace'}:raise ValueError('resource field inventory differs')
    if any(type(v) is not int or v<0 for v in r.values()) or r['workspace']!=p['workspace']:raise ValueError('resource scalar contract')
    if r['user_ub']+r['scratch']>p['capacity']['UB']:raise ValueError('UB over capacity')
    for used,cap in [('l1_used','L1'),('l0a_used','L0A'),('l0b_used','L0B'),('l0c_used','L0C')]:
        if r[used]>p['capacity'][cap]:raise ValueError('resource over capacity: '+cap)
    data=bytes.fromhex(s['tiling_hex'])
    if s['tiling_bytes']!=len(data) or not data or not any(data):raise ValueError('parent Tiling serialization missing')
    dso=Path(p['host_dso']).resolve(strict=True);root=Path(opp['install']).resolve()
    known={str(Path(x['path']).resolve()):x['sha256'] for x in opp['preflight']['files']}
    if root not in dso.parents or known.get(str(dso))!=c.sha(dso):raise ValueError('unbound callback Host DSO')
    return dict(platform=p,schedule=s,callbacks=len(ps),platform_file=c.identity(platform_path),scheduling_file=c.identity(scheduling_path),
                host_dso=c.identity(dso),tiling_sha256=hashlib.sha256(data).hexdigest(),scope='HOST_OBSERVATION_ONLY')


def compare_pair(a,b):
    # Across A/B only selected launch policy/Host artifact may differ.
    for key in ('capacity','workspace','GetCoreNum','GetCoreNumAic','GetCoreNumAiv','platform_raw_core','platform_raw_core_locked','resource_map','soc_enum','arch'):
        if a['platform'][key]!=b['platform'][key]:raise ValueError('A/B platform drift: '+key)
    if a['schedule']['resources']!=b['schedule']['resources'] or a['tiling_sha256']!=b['tiling_sha256']:raise ValueError('A/B resources or serialized Tiling changed')
    return dict(A_block_dim=a['schedule']['selected_block_dim'],B_block_dim=b['schedule']['selected_block_dim'],
                slice_num=b['schedule']['slice_num'],calc_tsch_block_dim=b['schedule']['calc_tsch_block_dim'],
                tile_coverage='EXACT_ONCE_BOTH',resources='EXACT_EQUAL',tiling='PARENT_SERIALIZATION_BITWISE_EQUAL',
                npu='NOT_RUN',decision='HOST_CANDIDATE_ONLY_REQUIRES_SEPARATE_DEVICE_SMOKE')
