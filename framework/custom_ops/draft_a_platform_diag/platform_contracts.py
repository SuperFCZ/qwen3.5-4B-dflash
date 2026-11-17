"""Diagnostic acceptance observes resource views; it never grants Core8 PASS."""
import json
from pathlib import Path
from platform_common import c

CORE=('GetCoreNum','GetCoreNumAic','GetCoreNumAiv','platform_raw_core','platform_raw_core_locked','availableCores','blockDim','coreLimit')
CAPS={'UB','L1','L0A','L0B','L0C'}
ENV_KEYS=('ASCEND_CUSTOM_OPP_PATH','ASCEND_OPP_PATH','LD_LIBRARY_PATH','CANN_ROOT','ASCEND_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES','ASCEND_DEVICE_ID','ASCEND_VNPU_SPECS','ASCEND_RUNTIME_OPTIONS')


def environment(env):return {k:env.get(k) for k in ENV_KEYS}


def unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('duplicate JSON key: '+key)
        result[key]=value
    return result


def read_platform(path,route,opp):
    rows=[json.loads(x,object_pairs_hook=unique_object) for x in Path(path).read_text().splitlines()]
    if not rows:raise ValueError('PlatformInfo diagnostic missing')
    fields=set(CORE)|{'schema','route','m','k','n','platform_info_present','resource_map_ok','resource_map','capacity','host_dso','soc_enum','arch','workspace'}
    for r in rows:
        if not isinstance(r,dict) or set(r)!=fields:raise ValueError('PlatformInfo record field set differs')
        if r.get('schema')!=1 or r.get('route')!=route or [r.get(x) for x in ('m','k','n')]!=[16,12800,2560]:raise ValueError('diagnostic shape/route differs')
        if r.get('platform_info_present') is not True or r.get('resource_map_ok') is not True:raise ValueError('PlatformInfo map unavailable')
        for key in (*CORE,'soc_enum','arch','workspace'):
            if type(r.get(key)) is not int or r[key]<0:raise ValueError('invalid platform numeric field')
        if r['coreLimit']!=0 or r['GetCoreNum']!=r['availableCores'] or r['blockDim']!=min(40,r['availableCores']) or r['blockDim']==0:
            raise ValueError('original scheduling policy changed or unstable')
        if set(r['capacity'])!=CAPS or any(type(v) is not int or v<=0 for v in r['capacity'].values()):raise ValueError('capacity query failed')
        if not isinstance(r['resource_map'],dict) or any(not isinstance(v,dict) or any(not isinstance(k,str) or not isinstance(x,str) for k,x in v.items()) for v in r['resource_map'].values()):raise ValueError('resource map schema')
        dso=Path(r['host_dso']).resolve(strict=True);root=Path(opp['install']).resolve()
        known={str(Path(x['path']).resolve()):x['sha256'] for x in opp['preflight']['files']}
        if root not in dso.parents or known.get(str(dso))!=c.sha(dso):raise ValueError('callback loaded an unbound Host DSO')
    if any(r!=rows[0] for r in rows):raise ValueError('PlatformInfo drift across repeated callbacks')
    return dict(observation=rows[0],callbacks=len(rows),source=c.identity(path),host_dso=c.identity(rows[0]['host_dso']),scope='FORMAL_TILING_CONTEXT_OBSERVATION_NOT_NUMERICAL_PASS')


def compare(atc,aclnn):
    if atc['host_dso']!=aclnn['host_dso']:raise ValueError('routes loaded different Host DSO identities')
    a=atc['observation'];b=aclnn['observation']
    changes={k:{'ATC':a[k],'ACLNN':b[k]} for k in (*CORE,'soc_enum','arch','workspace','capacity') if a[k]!=b[k]}
    maps={}
    for group in sorted(set(a['resource_map'])|set(b['resource_map'])):
        left=a['resource_map'].get(group,{});right=b['resource_map'].get(group,{})
        for key in sorted(set(left)|set(right)):
            if left.get(key)!=right.get(key):maps[group+'.'+key]=dict(ATC=left.get(key),ACLNN=right.get(key))
    return dict(status='PLATFORM_VIEWS_DIFFER' if changes or maps else 'PLATFORM_VIEWS_MATCH',field_differences=changes,resource_map_differences=maps,
        host_dso=atc['host_dso'],selected_runtime_cores=b['GetCoreNum'],diagnosis='Different observed values do not establish vNPU or hardware failure',
        experiment_policy='Use runtime-reported cores for unchanged ACLNN scheduling; do not mix this with a differently tiled ATC OM',
        core8_admission='NOT_GRANTED',virtualization='UNRESOLVED_UNTIL_DEVICE_MAPPING_EVIDENCE')


def validate_runtime(r,journal):
    if r.get('status')!='HOST_TILING_ONLY' or r.get('execute_calls')!=0 or (r.get('m'),r.get('k'),r.get('n'))!=(16,12800,2560):raise ValueError('not single-shape Host-only')
    for k in ('acl_init','set_device','context_present','context_stable','stream_created','guards','readonly','poison_unchanged'):
        if r.get(k) is not True:raise ValueError('runtime initialization/safety failed: '+k)
    expected=['aclInit','aclrtSetDevice','aclrtGetDevice','context_after_set_device','aclrtCreateStream','context_after_stream','GetWorkspaceSize','GetWorkspaceSize','aclrtSynchronizeStream']
    if [x.get('event') for x in journal]!=expected or any(type(x.get('returncode')) is not int or x['returncode']!=0 for x in journal):raise ValueError('runtime initialization journal failed')
    if journal[1].get('requested_device')!=0 or journal[2].get('device')!=0 or journal[3].get('present') is not True or journal[4].get('present') is not True or journal[5].get('same_context') is not True:raise ValueError('runtime selection journal differs')
    for i,row in enumerate(journal[6:8]):
        if row.get('call')!=i or row.get('workspace')!=r.get('workspace') or row.get('executor_present') is not True:raise ValueError('workspace/executor journal differs')
    if type(r.get('workspace')) is not int or r['workspace']<=0:raise ValueError('missing ACLNN workspace')
    if r['device']!=0 or r['explicit_context_created'] is not False:raise ValueError('changed device/context policy')


def loaded_libraries(maps,opp,cann):
    """Record actual process mappings; inode/device namespace equivalence is not assumed."""
    paths=set();rows=[];root=Path(opp['install']).resolve();cann=Path(cann).resolve()
    for line in Path(maps).read_text().splitlines():
        fields=line.split(None,5)
        if len(fields)!=6 or not fields[5].startswith('/') or '.so' not in Path(fields[5]).name:continue
        if fields[5].endswith(' (deleted)'):raise ValueError('deleted loaded DSO: '+fields[5])
        paths.add(fields[5])
    if not paths:raise ValueError('empty runtime DSO snapshot')
    expected={str(Path(x['path']).resolve()):x['sha256'] for x in opp['preflight']['files']}
    found_private=False
    for alias in sorted(paths):
        target,links=c.alias_target(alias);identity=c.identity(target)
        if root in target.parents:
            if expected.get(str(target))!=identity['sha256']:raise ValueError('unknown private DSO: '+alias)
            group='PRIVATE_DIAGNOSTIC_OPP';found_private=True
        elif cann in target.parents:group='SELECTED_CANN'
        elif Path('/usr/local/Ascend/driver') in target.parents:group='DRIVER_SEPARATE_FROM_CANN'
        elif '/usr/local/Ascend/' in alias or '/usr/local/Ascend/' in str(target) or 'libcust_op' in target.name:
            raise ValueError('different Ascend installation/private OPP: '+alias)
        else:group='SYSTEM'
        rows.append(dict(alias=alias,links=links,file=identity,category=group,
                         mapping_scope='proc maps pathname plus current file SHA; not a mapped-page content hash'))
    if not found_private:raise ValueError('private API/Host library not present in actual process mappings')
    return rows


def verify_loaded(rows):
    for row in rows:
        target,links=c.alias_target(row['alias'])
        if str(target)!=row['file']['path'] or links!=row['links']:raise ValueError('loaded library alias retargeted')
        c.verify(row['file'])
