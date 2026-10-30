"""Stage4J admission: immutable full Golden, real T1 contract, current SDK identity."""
import json
import re
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4i'))
import run_stage4i as host_i
import layout as layout_i
import resources as resources_i
old=layout_i.old
reference=layout_i.reference
import run_stage4b as run_b
common,identity,verify,verify_tree,write,fresh,metrics=old.common,old.identity,old.verify,old.verify_tree,old.write,old.fresh,old.metrics
NAME='DFlashW8A8N128Gate';SNAKE='d_flash_w8_a8_n128_gate'
BOUNDED_TILES=[0,75,76,151]
FIELDS=('usedCoreNum','singleCoreM','singleCoreN','singleCoreK','baseM','baseN','baseK','depthA1','depthB1','dbL0A','dbL0B','dbL0C','transLength','stepM','stepN','stepKa','stepKb','iterateOrder','shareMode','shareL1Size','shareL0CSize','shareUbSize','isBias','Ka','Kb','M','N')
SETTINGS=dict(op=NAME,A=[20,16,128],B=[152,20,128,128],scale=[152,20,128],output=[16,19456],
    tile_n=128,groups=20,base=[16,128,128],scratch=226816,user_ub=24576,ub_guards=2560,reserve=8192,
    core_policy=old.SETTINGS['core_policy'],workspace=old.SETTINGS['workspace'],workspace_cap=old.SETTINGS['workspace_cap'],
    bounded_tiles=BOUNDED_TILES,performance='NOT_RUN',guard_protocol='FULL_STAGE4B_GUARDS')

def sources():
    paths=[p for folder in (HERE,HERE.parent/'stage4i') for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
    rows={x['path']:x for x in old.sources()}
    rows.update({str(p.resolve()):identity(p) for p in paths})
    return [rows[k] for k in sorted(rows)]

def read(path):
    item=identity(path);r=json.loads(verify(item).read_text());verify_tree(r);verify(item);return r

def evidence(args):
    root=args.stage4i_root
    tr=read(root/'tiling-report.json');sdk=read(root/'sdk-binding.json');lr=read(root/'layout-report.json');rr=read(root/'resource-report.json')
    if tr.get('status')!='STAGE4I_N128_HOST_TILING_FEASIBLE' or tr.get('phase')!='complete' or tr.get('host_compile')!='COMPLETED':raise ValueError('accepted Stage4I.1 required')
    if any(tr['commands'][k].get('returncode')!=0 for k in ('configure','compile','ldd','query')):raise ValueError('Stage4I commands incomplete')
    for name in ('npu_kernel_execution','benchmark','profile','n128_kernel_compile'):
        if tr.get(name)!='NOT_RUN':raise ValueError('unexpected Stage4I scope')
    current=host_i.bind(args.cann_root)
    for k in ('status','files','versions','shared_ub_contract'):
        if sdk.get(k)!=current[k]:raise ValueError('current SDK differs from accepted Stage4I: '+k)
    if current['shared_ub_contract']['host_library_identity']!='EXACT_MATCH_RECORDED_SERVER_LIBRARY':raise ValueError('server Host library identity required')
    host_i.verify_dependency_binding(sdk['dependency_binding'])
    q=tr['actual_host_query']
    raw=read(verify(tr['host_query_file']))
    # Driver adds per-case tiling identities after reading host-query.json.
    normalized=json.loads(json.dumps(q))
    for row in normalized['cases']:row.pop('tiling_binary',None)
    if normalized!=raw or q['device']!=args.device:raise ValueError('Host query binding mismatch')
    t1=[x for x in q['cases'] if x['name']=='T1']
    if len(t1)!=1:raise ValueError('one T1 required')
    t1=t1[0];check=resources_i.validate(q,t1)
    if check['status']!='HOST_RESOURCE_CHECKS_PASS' or [x for x in rr['cases'] if x['name']=='T1']!=[check]:raise ValueError('T1 resource admission differs')
    fields=t1['fields']
    required=dict(singleCoreM=16,singleCoreN=128,singleCoreK=128,baseM=16,baseN=128,baseK=128,depthA1=1,depthB1=1,dbL0A=2,dbL0B=2,dbL0C=1,transLength=16384,shareUbSize=1024,shareMode=0)
    if any(type(fields.get(k)) is not int for k in FIELDS) or any(fields[k]!=v for k,v in required.items()):raise ValueError('not the accepted T1 configuration')
    if lr.get('status')!='CPU_FROZEN_LAYOUT_MATCH' or lr.get('validated_tile_groups')!=3040 or lr.get('source_report')!=identity(args.stage4b_report):raise ValueError('frozen layout provenance mismatch')
    b=read(args.stage4b_report)
    if b.get('status')!='STAGE4B_FULL_GATE_MATCH' or b.get('phase')!='complete' or b.get('settings')!=old.SETTINGS or b.get('sources')!=old.sources() or b.get('device')!=args.device:raise ValueError('unchanged accepted Stage4B required')
    case=read(verify(b['case']));maps=reference.verify_case(case,numerical=False)
    if b['case']!=lr['case']:raise ValueError('Stage4I/4B case identity mismatch')
    policy=run_b.validate_host_probe(verify(b['host_probe_report']),b['multi_build_report'],b)
    limits=[0,2 if min(304,policy['selected'])>2 else 1]
    expected=[(l,m) for l in limits for m in [21,0]+[g+1 for g in b['diagnostic_groups']]]
    if [(x['core_limit'],x['mode']) for x in b['runs']]!=expected:raise ValueError('incomplete historical NPU coverage')
    for row in b['runs']:
        mode,limit=row['mode'],row['core_limit'];ex=read(verify(row['execution']));old.check_execution(ex,args.device,mode,limit)
        plan=old.plan_from_log(verify(b['commands'][f'l{limit}_m{mode}']['log_identity']).read_text(),mode,limit)
        if plan!=row['plan'] or plan['core_policy']!=policy or read(verify(row['resource_admission']))!=run_b.admission_expected(ex,plan,False):raise ValueError('historical resource proof mismatch')
        result=reference.compare(case,maps,mode,plan['blocks'],[verify(x).read_bytes() for x in row['outputs']])
        if result!=row['comparison'] or result['status']!='BITWISE_MATCH' or row['status']!='BITWISE_MATCH':raise ValueError('historical output mismatch')
    return dict(stage4i_reports={n:identity(root/n) for n in ('tiling-report.json','resource-report.json','sdk-binding.json','layout-report.json')},
        stage4b_report=identity(args.stage4b_report),stage4b_case=b['case'],preflight=b['evidence']['preflight'],
        frozen=case['source'],sdk=current,formal_core_policy=policy,t1=dict(fields={k:fields[k] for k in FIELDS},capacity=q['capacity'],system_workspace_bytes=q['system_workspace_bytes'],scratch=t1['scratch']),
        layout=lr,prior_runners_executed=False,native_comparison='SKIPPED_SEMANTICS_UNRESOLVED')

def contract_header(t1):
    """Compile exact server T1 values into the new Op Host, never Core counts."""
    if set(t1['fields'])!=set(FIELDS):raise ValueError('incomplete T1 fields')
    rows=['#pragma once','#include <cstdio>','template<class T> bool S4JAcceptedT1(T& p,uint64_t ub,uint64_t l1,uint64_t l0a,uint64_t l0b,uint64_t l0c,uint64_t system,uint64_t scratch) {',' bool ok=true;']
    for key in FIELDS:
        v=t1['fields'][key]
        if type(v) is not int or v<0 or v>2147483647:raise ValueError('invalid T1 field '+key)
        rows += [f' std::fprintf(stderr,"STAGE4J_T1 {key}=%lld\\n",(long long)p.get_{key}());',f' ok = (p.get_{key}()=={v}) && ok;']
    for var,key in [('ub','UB'),('l1','L1'),('l0a','L0A'),('l0b','L0B'),('l0c','L0C')]:
        v=t1['capacity'][key]
        if type(v) is not int or v<=0 or v>2**32:raise ValueError('invalid capacity')
        rows += [f' ok = ({var}=={v}ULL) && ok;']
    system=t1['system_workspace_bytes']
    if type(system) is not int or system<=0 or system>512*1024*1024 or t1['scratch']!=226816:raise ValueError('invalid workspace/scratch')
    rows += [f' ok = system=={system}ULL && scratch==226816 && ok;',
        ' std::fprintf(stderr,"STAGE4J_T1_CONTRACT match=%d effectiveSharedUb=0\\n",int(ok));',' return ok;','}']
    return '\n'.join(rows)+'\n'

def schedule(cores,limit,tile=-1):
    if type(cores) is not int or cores<1 or type(limit) is not int or limit not in range(153) or type(tile) is not int or tile not in range(-1,152):raise ValueError('invalid schedule')
    blocks=min(152,cores) if limit==0 else limit
    if blocks>cores:raise ValueError('requested blocks exceed scheduling limit')
    return blocks,[[t for t in range(c,152,blocks) if tile<0 or t==tile] for c in range(blocks)]

def plan_from_log(text,mode,limit,tile,t1):
    # Reuse tested Core/Arena parser with only shape-specific constants normalized.
    normal=text.replace('STAGE4J','STAGE4B').replace('scratch=226816','scratch=239104').replace('userUB=24576','userUB=12288').replace('single=16/128/128 base=16/128/128','single=16/64/128 base=16/64/128')
    # The historical parser's automatic maximum is304; supported actual Core counts
    # must not reach this difference (still reject, never silently reinterpret).
    p=old.plan_from_log(normal,mode,limit)
    blocks,owners=schedule(p['cores'],limit,tile)
    if p['blocks']!=blocks:raise ValueError('N128 schedule mismatch')
    scopes=set(re.findall(r'STAGE4J_SCOPE tile=(-?\d+)',text))
    if scopes!={str(tile)}:raise ValueError('scope absent or changed')
    for k,v in t1['fields'].items():
        if set(re.findall(r'STAGE4J_T1 '+k+r'=(-?\d+)',text))!={str(v)}:raise ValueError('T1 differs: '+k)
    if set(re.findall(r'STAGE4J_T1_CONTRACT match=(\d+) effectiveSharedUb=(\d+)',text))!={('1','0')} or p['system']!=t1['system_workspace_bytes']:raise ValueError('T1 resource contract not accepted')
    p.update(owners=owners,tile_id=tile);return p

def check_execution(ex,device,mode,limit,tile):
    if ex.get('op')!=NAME or ex.get('tile_count')!=152 or ex.get('tile_id')!=tile:raise ValueError('Stage4J execution scope differs')
    translated=dict(ex,op=old.NAME,tile_count=304)
    old.check_execution(translated,device,mode,limit)
