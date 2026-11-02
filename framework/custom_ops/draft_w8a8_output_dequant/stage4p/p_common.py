"""N256 admission binds Stage4O Host results and accepted Stage4N XOR sources."""
import sys,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4o'))
import o_layout as layout_i
import o_resources as resource_o
import run_stage4o as host_o
n=layout_i.n;j=n.j;old=j.old;reference=j.reference;jref=n.k.jref
common,identity,verify,verify_tree,write,fresh,metrics,read=j.common,j.identity,j.verify,j.verify_tree,j.write,j.fresh,j.metrics,j.read
NAME='DFlashW8A8N256Gate';SNAKE='d_flash_w8_a8_n256_gate';BOUNDED_TILES=[0,37,38,75]
SETTINGS=dict(j.SETTINGS,op=NAME,B=[76,20,128,256],scale=[76,20,256],tile_n=256,base=[16,256,128],scratch=202240,user_ub=49152,bounded_tiles=BOUNDED_TILES,guard_protocol='STAGE4N_640_WORD_XOR_OR')
FIELDS=j.FIELDS

def sources():
 rows={x['path']:x for x in layout_i.source_files()}
 for p in HERE.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts:rows[str(p.resolve())]=identity(p)
 return [rows[x] for x in sorted(rows)]

def evidence(args):
 root=args.stage4o_root;tr=read(root/'tiling-report.json');lr=read(root/'layout-report.json');rr=read(root/'resource-report.json');sdk=read(root/'sdk-binding.json')
 if tr.get('status')!='STAGE4O_N256_HOST_TILING_FEASIBLE' or tr.get('phase')!='complete' or tr.get('sources')!=layout_i.source_files() or any(tr['commands'][k]['returncode']!=0 for k in ('configure','compile','ldd','query')):raise ValueError('unchanged successful Stage4O required')
 frozen=layout_i.freeze(args.stage4n_root)
 if frozen!=lr['stage4n'] or frozen['device']!=args.device:raise ValueError('Stage4N freeze differs')
 current=host_o.bind(args.cann_root)
 if any(current[k]!=sdk[k] for k in ('status','files','versions','shared_ub_contract')):raise ValueError('SDK differs from Stage4O')
 host_o.verify_dependency_binding(sdk['dependency_binding'])
 q=tr['actual_host_query'];raw=read(verify(tr['host_query_file']));normalized=json.loads(json.dumps(q))
 for x in normalized['cases']:x.pop('tiling_binary',None)
 if normalized!=raw or q['device']!=args.device or [r['name'] for r in q['cases']]!=['N256']:raise ValueError('N256 query binding differs')
 row=q['cases'][0];check=resource_o.validate(q,row)
 if check['status']!='HOST_RESOURCE_CHECKS_PASS' or rr['cases']!=[check] or lr['validated_tile_groups']!=1520:raise ValueError('N256 resources/layout not accepted')
 f=row['fields'];required=dict(singleCoreM=16,singleCoreN=256,singleCoreK=128,baseM=16,baseN=256,baseK=128,depthA1=1,depthB1=1,dbL0A=2,dbL0B=2,dbL0C=1,transLength=34816,shareL1Size=34816,shareL0CSize=16384,shareUbSize=2048)
 if any(f.get(k)!=v for k,v in required.items()) or any(type(f.get(k)) is not int for k in FIELDS):raise ValueError('not the admitted BaseN256 configuration')
 if check['details']['l0b_bound']!=65536 or q['capacity']['L0B']!=65536:raise ValueError('L0B boundary differs')
 if not layout_i.core_consistency(q,frozen['formal_core_policy']):raise ValueError('Core reference differs')
 build_n=read(verify(frozen['build']))
 original=[x for x in build_n['opps']['Xor']['build']['generated_sources'] if Path(x['path']).parent.name=='op_kernel' and Path(x['path']).name=='d_flash_w8_a8_n128_gate.cpp']
 from p_transform import ng,J
 if len(original)!=1 or verify(original[0]).read_bytes()!=ng.transform((J/'op_kernel/d_flash_w8_a8_n128_gate.cpp').read_bytes()):raise ValueError('accepted Stage4N XOR source differs from derivation baseline')
 case=read(verify(frozen['case']))
 return dict(stage4o_reports={f:identity(root/f) for f in ('tiling-report.json','resource-report.json','layout-report.json','sdk-binding.json')},
  stage4n=frozen,accepted_xor_kernel=original[0],n128_case=frozen['case'],stage4b_case=case['stage4b_case'],preflight=frozen['candidate']['preflight'],sdk=current,layout=lr,
  formal_core_policy=frozen['formal_core_policy'],t1=dict(fields={k:f[k] for k in FIELDS},capacity=q['capacity'],system_workspace_bytes=q['system_workspace_bytes'],scratch=202240))

def schedule(cores,limit,tile=-1):
 if type(cores) is not int or cores<1 or type(limit) is not int or limit not in range(77) or type(tile) is not int or tile not in range(-1,76):raise ValueError('invalid schedule')
 blocks=min(76,cores) if limit==0 else limit
 if blocks>cores:raise ValueError('requested blocks exceed scheduling limit')
 return blocks,[[t for t in range(c,76,blocks) if tile<0 or t==tile] for c in range(blocks)]

def plan_from_log(text,mode,limit,tile,t1):
 # Reuse structural Core/Arena validation. Exact N256 fields remain untouched.
 normal=text.replace('STAGE4P','STAGE4J').replace('scratch=202240','scratch=226816').replace('userUB=49152','userUB=24576').replace('single=16/256/128 base=16/256/128','single=16/128/128 base=16/128/128')
 result=j.plan_from_log(normal,mode,limit,tile,t1)
 blocks,owners=schedule(result['cores'],limit,tile)
 if result['blocks']!=blocks:raise ValueError('N256 schedule mismatch')
 result['owners']=owners;return result

def check_execution(ex,device,mode,limit,tile):
 if ex.get('op')!=NAME or ex.get('tile_count')!=76:raise ValueError('N256 ABI mismatch')
 j.check_execution(dict(ex,op=j.NAME,tile_count=152),device,mode,limit,tile)

def contract_header(t1):
    """Compile exact server T1 values into the new Op Host, never Core counts."""
    if set(t1['fields'])!=set(FIELDS):raise ValueError('incomplete T1 fields')
    rows=['#pragma once','#include <cstdio>','template<class T> bool S4PAcceptedT1(T& p,uint64_t ub,uint64_t l1,uint64_t l0a,uint64_t l0b,uint64_t l0c,uint64_t system,uint64_t scratch) {',' bool ok=true;']
    for key in FIELDS:
        v=t1['fields'][key]
        if type(v) is not int or v<0 or v>2147483647:raise ValueError('invalid T1 field '+key)
        rows += [f' std::fprintf(stderr,"STAGE4P_T1 {key}=%lld\\n",(long long)p.get_{key}());',f' ok = (p.get_{key}()=={v}) && ok;']
    for var,key in [('ub','UB'),('l1','L1'),('l0a','L0A'),('l0b','L0B'),('l0c','L0C')]:
        v=t1['capacity'][key]
        if type(v) is not int or v<=0 or v>2**32:raise ValueError('invalid capacity')
        rows += [f' ok = ({var}=={v}ULL) && ok;']
    system=t1['system_workspace_bytes']
    if type(system) is not int or system<=0 or system>512*1024*1024 or t1['scratch']!=202240:raise ValueError('invalid workspace/scratch')
    rows += [f' ok = system=={system}ULL && scratch==202240 && ok;',
        ' std::fprintf(stderr,"STAGE4P_T1_CONTRACT match=%d effectiveSharedUb=0\\n",int(ok));',' return ok;','}']
    return '\n'.join(rows)+'\n'
