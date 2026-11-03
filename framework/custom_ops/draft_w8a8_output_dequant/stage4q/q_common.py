"""Read-only Stage4N/P proofs and new paired samples; no historical Kernel build."""
import sys,copy
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4p'))
import p_common as p
import run_stage4p as rp
import p_reference as pr
import fault_validation_p as pf
sys.path.insert(0,str(HERE.parent/'stage4l'))
k=p.n.k;bc=k.bc;rc=k.rc;common=k.common
identity,verify,verify_tree,read,write,fresh,statistics_ms=k.identity,k.verify,k.verify_tree,k.read,k.write,k.fresh,k.statistics_ms
ORDER=(('N128','N256'),('N256','N128'),('N128','N256'));SCOPE=k.SCOPE

def sources():
 rows={x['path']:x for x in p.sources()}
 for directory in (HERE,HERE.parent/'stage4l'):
  for f in directory.rglob('*'):
   if f.is_file() and '__pycache__' not in f.parts and '.pytest_cache' not in f.parts:rows[str(f.resolve())]=identity(f)
 return [rows[x] for x in sorted(rows)]

def evidence(args):
 pe=p.evidence(args);full=read(args.stage4p_root/'full/report.json');build=read(verify(full['build_report']))
 if full['evidence']!=pe or full['sources']!=p.sources() or full['device']!=args.device:raise ValueError('Stage4P source/evidence/device changed')
 rp.verify_build_sources(build);rp.same(full,build)
 prepared=read(verify(full['prepare_report']))
 if prepared['status']!='STAGE4P_PREPARED_NPU_NOT_RUN' or prepared['case']!=full['case']:raise ValueError('Stage4P Prepare differs')
 case=read(verify(full['case']));maps=pr.verify_case(case,numerical=True)
 for stage,key in [('host-only','host_report'),('bounded','bounded_report')]:rp.validate_proof(verify(full[key]),full,maps,stage)
 rp.validate_proof(args.stage4p_root/'full/report.json',full,maps,'full')
 fault=read(args.stage4p_root/'faults/report.json');expected=dict(full,full_report=identity(args.stage4p_root/'full/report.json'))
 pf.verify_faults(args.stage4p_root/'faults/report.json',expected)
 if fault['coverage'].get('injections_per_sweep')!=409600 or fault['coverage'].get('repeats')!=2:raise ValueError('fault coverage incomplete')
 n128=copy.deepcopy(pe['stage4n']['candidate'])
 n256=dict(opp=build['opp'],installed=build['installed'],files=case['files'],policy=full['core_policy'],preflight=pe['preflight'],t1=pe['t1'],runner=build['runner'],case=full['case'],build_report=full['build_report'])
 if n128['policy']!=n256['policy'] or n256['policy']['selected']!=7:raise ValueError('formal Core policy differs')
 oldcase=read(verify(n128['case']));oldmaps=k.jref.verify_case(oldcase)
 import numpy as np
 for tile in range(76):
  for key in ('b_i8.bin','quant_u64_doc_candidate.bin'):
   if maps[key][tile].tobytes()!=np.concatenate([oldmaps[key][tile*2],oldmaps[key][tile*2+1]],axis=-1).tobytes():raise ValueError('packing relation differs')
 if case['files']['golden_fp16.bin']!=n128['files']['golden_fp16.bin'] or case['files']['a_i8.bin']['sha256']!=n128['files']['a_i8.bin']['sha256']:raise ValueError('Golden/A mismatch')
 return dict(variants=dict(N128=n128,N256=n256),stage4n=pe['stage4n'],stage4p_full=identity(args.stage4p_root/'full/report.json'),stage4p_fault=identity(args.stage4p_root/'faults/report.json'),scope=SCOPE)

def environment(proof,cann):
 e=common.env(cann,Path(proof['opp']))
 for key in ('DFLASH_STAGE4J_EXPECTED_CORE_POLICY','DFLASH_STAGE4P_EXPECTED_CORE_POLICY'):e.pop(key,None)
 prefix='STAGE4P' if proof['t1']['fields']['baseN']==256 else 'STAGE4J'
 e['DFLASH_'+prefix+'_EXPECTED_CORE_POLICY']=','.join(str(proof['policy'][x]) for x in ('aic','aiv','all','selected','arch'));return e

def immutable(proof):
 verify_tree(proof)
 for name,v in proof['variants'].items():
  if (rp.installed if name=='N256' else k.jb.installed)(Path(v['opp']))!=v['installed']:raise ValueError('OPP changed')

def check_execution(ex,device,name,timed):
 if ex.get('guard_coverage')!='EXACT_640_WORD_GROUP_XOR_OR':raise ValueError('guard protocol differs')
 if name=='N256':
  if ex.get('op')!=p.NAME or ex.get('tile_count')!=76:raise ValueError('N256 ABI differs')
  ex=dict(ex,op=k.jc.NAME,tile_count=152)
 return k.check_execution(ex,device,'N128',timed)

def plan(text,name,v):
 out=(p.plan_from_log if name=='N256' else k.jc.plan_from_log)(text,0,0,-1,v['t1'])
 if out['core_policy']!=v['policy'] or out['blocks']!=7:raise ValueError('Core policy drift')
 return out

def validate_run(row,proof,device,timed,profiled=False):
 v=proof['variants'][row['variant']];ex=read(verify(row['execution']));stats=check_execution(ex,device,row['variant'],timed);folder=Path(row['execution']['path']).parent
 actual=plan(verify(row['log']).read_text(),row['variant'],v)
 if row['profiled']!=profiled or row['status']!='FROZEN_GOLDEN_MATCH' or actual!=row['plan'] or stats!=row['timing'] or bc.verify_outputs(v,folder,timed)!=row['outputs']:raise ValueError('run proof differs')
 if row['workspace_bytes']!=ex['workspace_bytes'] or ex['workspace_bytes']<actual['total'] or read(verify(row['admission']))!=bc.stage4b.admission_expected(ex,actual,False):raise ValueError('workspace mismatch')

def comparison(rows):
 if [(r['round'],r['position'],r['variant']) for r in rows]!=[(i,j,n) for i,o in enumerate(ORDER) for j,n in enumerate(o)]:raise ValueError('paired order incomplete')
 stats={}
 for n in ('N128','N256'):
  selected=[r for r in rows if r['variant']==n]
  if any(r['profiled'] or not r['formal_latency_evidence'] or r['status']!='FROZEN_GOLDEN_MATCH' for r in selected):raise ValueError('invalid new timing')
  if len({r.get('workspace_bytes') for r in selected})!=1:raise ValueError('Workspace changed across rounds')
  stats[n]={key:statistics_ms([s for r in selected for s in r['timing'][key]['samples_ms']],90) for key in ('execute_sync','prepare')}
 ratios=[next(r['timing']['execute_sync']['median_ms'] for r in rows if r['round']==i and r['variant']=='N128')/next(r['timing']['execute_sync']['median_ms'] for r in rows if r['round']==i and r['variant']=='N256') for i in range(3)]
 ratio=stats['N128']['execute_sync']['median_ms']/stats['N256']['execute_sync']['median_ms']
 return dict(statistics=stats,round_ratios=ratios,n128_over_n256=ratio,decision='GO_FURTHER_REVIEW' if ratio>=1.03 and all(v>1 for v in ratios) else 'NO_GO_NO_MATERIAL_IMPROVEMENT',scope=SCOPE,
  interpretation='Net implementation effects including reduced group Guard frequency; not pure Matmul throughput',historical_samples_used=False)
