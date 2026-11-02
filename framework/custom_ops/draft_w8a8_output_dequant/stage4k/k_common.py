"""Read-only historical admission and exact continuous-v1 comparison contracts."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4c'))
import bench_common as bc
import run_stage4c as rc
sys.path.insert(0,str(HERE.parent/'stage4j'))
import j_common as jc
import run_stage4j as rj
import j_reference as jref
import build_n128 as jb
from bench_common import common,identity,verify,verify_tree,statistics_ms
write=bc.write;fresh=bc.fresh
ORDER=(('N64','N128'),('N128','N64'),('N64','N128'))
SCOPE='PREPACKED_RESIDENT_W8A8_KERNEL_ONLY'

def sources():
 rows={x['path']:x for x in bc.sources()+jc.sources()}
 for p in HERE.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts:rows[str(p.resolve())]=identity(p)
 return [rows[k] for k in sorted(rows)]

def read(path):return jc.read(path)

def evidence(args):
 n64=bc.evidence(args)
 history=read(args.stage4c_report)
 expected=dict(sources=bc.sources(),evidence=n64,device=args.device,runner_build_report=history['runner_build_report'])
 rc.prior_report(args.stage4c_report,'STAGE4C_BASELINE_MEASURED',expected)
 historical_build=read(verify(history['runner_build_report']))
 if historical_build.get('status')!='STAGE4C_BENCHMARK_PREPARED' or any(historical_build.get(k)!=expected[k] for k in ('sources','evidence','device')):raise ValueError('Stage4C build mismatch')
 args.stage4b_report=args.stage4b_root/'smoke/report.json'
 je=jc.evidence(args);j=read(args.stage4j_root/'full/report.json')
 if j.get('sources')!=jc.sources() or j.get('evidence')!=je or j.get('device')!=args.device:raise ValueError('Stage4J source/SDK/evidence changed')
 case=read(verify(j['case']));maps=jref.verify_case(case,numerical=True)
 for stage,key in [('host-only','host_report'),('bounded','bounded_report')]:
  if rj.validate_proof(verify(j[key]),j,maps,stage)!=n64['policy']:raise ValueError('formal Core policy differs')
 if rj.validate_proof(args.stage4j_root/'full/report.json',j,maps,'full')!=n64['policy']:raise ValueError('full Core policy differs')
 build=read(verify(j['build_report']));rj.same(j,build)
 if build.get('status')!='STAGE4J_BUILT_NPU_NOT_RUN' or jb.installed(Path(build['opp']))!=build['installed']:raise ValueError('N128 private OPP changed')
 if j['evidence']['stage4b_case']!=n64['case'] or case['stage4b_case']!=n64['case']:raise ValueError('different frozen cases')
 if case['files']['golden_fp16.bin']!=n64['files']['golden_fp16.bin'] or case['files']['a_i8.bin']['sha256']!=n64['files']['a_i8.bin']['sha256']:raise ValueError('Golden/Activation differs')
 # Recheck the accepted N64->N128 packing relation, not just tensor sizes.
 oldcase=read(verify(n64['case']));oldmaps=bc.reference.verify_case(oldcase,numerical=False)
 verify_packing(oldmaps,maps)
 n128=dict(opp=build['opp'],installed=build['installed'],files=case['files'],policy=n64['policy'],preflight=n64['preflight'],build_report=j['build_report'],case=j['case'],runner=build['runner'],t1=je['t1'])
 if n64['policy']['selected']!=7:raise ValueError('experiment requires accepted 7-Block policy, not a fallback')
 return dict(variants=dict(N64=n64,N128=n128),stage4c=identity(args.stage4c_report),stage4j=identity(args.stage4j_root/'full/report.json'),stage4i=je['stage4i_reports'],sdk=je['sdk'],scope=SCOPE)

def environment(proof,cann):
 e=common.env(cann,Path(proof['opp']))
 for prefix in ('STAGE4B','STAGE4J'):e.pop('DFLASH_'+prefix+'_EXPECTED_CORE_POLICY',None)
 prefix='STAGE4J' if 't1' in proof else 'STAGE4B'
 e['DFLASH_'+prefix+'_EXPECTED_CORE_POLICY']=','.join(str(proof['policy'][k]) for k in ('aic','aiv','all','selected','arch'))
 return e

def immutable(proof):
 verify_tree(proof)
 for name,p in proof['variants'].items():
  if (bc.installed if name=='N64' else jb.installed)(Path(p['opp']))!=p['installed']:raise ValueError('historical OPP changed '+name)

def check_execution(ex,device,name,timed):
 x=dict(ex)
 if name=='N128':
  if x.get('op')!=jc.NAME or x.get('tile_count')!=152:raise ValueError('wrong N128 ABI')
  x.update(op=bc.full.NAME,tile_count=304)
 return bc.check_execution(x,device,0,timed)

def plan(text,name,p):
 result=bc.full.plan_from_log(text,0,0) if name=='N64' else jc.plan_from_log(text,0,0,-1,p['t1'])
 if result['core_policy']!=p['policy'] or result['blocks']!=7:raise ValueError('actual scheduling drift')
 return result

def validate_run(row,proof,device,timed,profiled=False):
 name=row['variant'];p=proof['variants'][name];ex=read(verify(row['execution']))
 stats=check_execution(ex,device,name,timed)
 folder=Path(row['execution']['path']).parent
 outputs=bc.verify_outputs(p,folder,timed)
 actual=plan(verify(row['log']).read_text(),name,p)
 if ex['workspace_bytes']<actual['total'] or read(verify(row['admission']))!=bc.stage4b.admission_expected(ex,actual,False):raise ValueError('workspace admission mismatch')
 if row.get('profiled') is not profiled:raise ValueError('profile samples cannot enter formal timing')
 if stats!=row['timing'] or outputs!=row['outputs'] or actual!=row['plan'] or row['status']!='FROZEN_GOLDEN_MATCH':raise ValueError('run proof changed')
 return stats

def comparison(runs):
 if [(r['round'],r['position'],r['variant']) for r in runs]!=[(i,j,v) for i,order in enumerate(ORDER) for j,v in enumerate(order)]:raise ValueError('paired order incomplete')
 result=dict(status='STAGE4K_COMPARISON_COMPLETED',scope=SCOPE,variants={},rounds=[],historical_samples_used=False,
  attribution='Full Guard protocol preserved, but group/tile Guard counts also fall with N128; ratio is not pure Matmul speedup')
 for name in ('N64','N128'):
  rows=[r for r in runs if r['variant']==name]
  if any(r['profiled'] or not r['formal_latency_evidence'] or r['status']!='FROZEN_GOLDEN_MATCH' for r in rows):raise ValueError('invalid formal samples')
  result['variants'][name]={key:statistics_ms([s for r in rows for s in r['timing'][key]['samples_ms']],90) for key in ('execute_sync','prepare')}
  if len({r['workspace_bytes'] for r in rows})!=1:raise ValueError('workspace drift')
  result['variants'][name].update(blocks=7,workspace_bytes=rows[0]['workspace_bytes'])
 for i in range(3):
  rows={r['variant']:r for r in runs if r['round']==i}
  m={name:rows[name]['timing']['execute_sync']['median_ms'] for name in rows}
  result['rounds'].append(dict(round=i,order=list(ORDER[i]),median_ms=m,n64_over_n128=m['N64']/m['N128']))
 result['n64_over_n128']=result['variants']['N64']['execute_sync']['median_ms']/result['variants']['N128']['execute_sync']['median_ms']
 return result


def verify_packing(oldmaps,maps):
 import numpy as np
 for t in range(152):
  for key in ('b_i8.bin','quant_u64_doc_candidate.bin'):
   expected=np.concatenate([oldmaps[key][2*t],oldmaps[key][2*t+1]],axis=-1)
   if maps[key][t].tobytes()!=expected.tobytes():raise ValueError('packing relation changed '+key+' tile='+str(t))
