"""Local contract must match server bytes before any candidate build/execution."""
import sys,copy
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4q'))
import q_common as q
import run_stage4q as qr
import source_t as source
materialize=source.materialize;verify_audit=source.verify_audit
p=q.p;rp=q.rp;pr=q.pr
NAME=p.NAME;SNAKE=p.SNAKE;common=q.common
identity,verify,verify_tree,read,write,fresh=q.identity,q.verify,q.verify_tree,q.read,q.write,q.fresh
environment=q.environment;bc=q.bc;rc=q.rc
COVERAGE = {
 'baseline': 'FULL_GROUP_XOR_OR_GUARD',
 'candidate': 'WEAKER_GUARD_DIAGNOSTIC_ONLY',
}
LIMITATION = 'Group-boundary detection removed: transient corruption restored before Tile-end can escape; persistent faults are detected later. Numerical agreement does not establish safety equivalence.'

def sources():
 rows={x['path']:x for x in q.sources()}
 for f in HERE.rglob('*'):
  if f.is_file() and '__pycache__' not in f.parts and '.pytest_cache' not in f.parts:rows[str(f.resolve())]=identity(f)
 return [rows[k] for k in sorted(rows)]

def evidence(args):
 base=q.evidence(args);tr=read(args.stage4q_root/'benchmark/benchmark-report.json')
 current=dict(sources=q.sources(),evidence=base,device=args.device,machine=tr['machine'])
 qr.prior(verify(tr['build_report']),'build',current);current['build_report']=tr['build_report']
 qr.prior(args.stage4q_root/'benchmark/benchmark-report.json','benchmark',current)
 prof=read(args.stage4q_root/'profile/profile-report.json')
 if prof.get('status')!='STAGE4Q_PROFILE_COMPLETED' or prof.get('benchmark_report')!=identity(args.stage4q_root/'benchmark/benchmark-report.json') or prof['evidence']!=base or prof['sources']!=q.sources():raise ValueError('Stage4Q Profile not bound')
 for row in prof['runs']:q.validate_run(row,base,args.device,True,True)
 from profile_rows import review
 profile_review={n:review(raw) for n,raw in prof['raw_profile'].items()}
 baseline=base['variants']['N256'];full=read(verify(base['stage4p_full']))
 return dict(baseline=baseline,preflight=baseline['preflight'],t1=baseline['t1'],case=baseline['case'],formal_core_policy=baseline['policy'],
  stage4p_full=base['stage4p_full'],stage4p_fault=base['stage4p_fault'],stage4p_prepare=full['prepare_report'],
  stage4q=identity(args.stage4q_root/'benchmark/benchmark-report.json'),profile=identity(args.stage4q_root/'profile/profile-report.json'),profile_review=profile_review,
  guard_coverage=COVERAGE,limitation=LIMITATION,scope=q.SCOPE)

def immutable(proof):
 verify_tree(proof)
 if rp.installed(Path(proof['baseline']['opp']))!=proof['baseline']['installed']:raise ValueError('baseline OPP changed')

def checked_build(b,r):
 if b.get('guard_coverage')!=COVERAGE:raise ValueError('guard protocol label missing or changed')
 if b.get('status')!='STAGE4T_BUILT_NPU_NOT_RUN' or any(b.get(k)!=r[k] for k in ('sources','evidence','device')):raise ValueError('build scope changed')
 from p_common import contract_header
 for kind in ('candidate',):
  item=b['variants'][kind];verify_audit(item['audit']);built=item['build'];verify_tree(built)
  if item['audit'].get('coverage')!=COVERAGE['candidate'] or built['status']!='STAGE4T_PRIVATE_BUILT' or rp.installed(Path(built['opp']))!=built['installed']:raise ValueError('variant identity differs')
  expected={Path(x['candidate']['path']).parent.name+'/'+Path(x['candidate']['path']).name:verify(x['candidate']).read_bytes() for x in item['audit']['files'] if Path(x['candidate']['path']).suffix in ('.cpp','.h')}
  expected['op_host/t1_contract.h']=contract_header(r['evidence']['t1']).encode()
  actual={Path(x['path']).parent.name+'/'+Path(x['path']).name:verify(x).read_bytes() for x in built['generated_sources']}
  if expected!=actual:raise ValueError('compiled source differs')
  if (Path(built['opp']).parent/'operator.json').read_bytes()!=(HERE.parent/'stage4p/operator.json').read_bytes():raise ValueError('ABI drift')
 for name in ('baseline','candidate'):
  bench=b['bench_runners'][name]
  proof=copy.deepcopy(r['evidence']['baseline'])
  if name=='candidate':proof.update(opp=b['variants']['candidate']['build']['opp'],installed=b['variants']['candidate']['build']['installed'])
  if bench.get('diagnostic') is not (name=='candidate'):raise ValueError('Runner coverage compile identity mismatch')
  if bench['status']!='STAGE4T_BENCH_RUNNER_BUILT' or bench['evidence']!=proof or bench['linked_libraries']['LIB_custom']!=proof['installed']['api']:raise ValueError('benchmark runner/OPP binding differs')
  verify_tree(bench)
 baseline={x['binary']['sha256'] for x in r['evidence']['baseline']['installed']['kernels']}
 if baseline=={x['binary']['sha256'] for x in b['variants']['candidate']['build']['installed']['kernels']}:raise ValueError('binary contrast unestablished')

ORDER=(('baseline','candidate'),('candidate','baseline'),('baseline','candidate'))
def comparison(rows):
 if [(r['round'],r['position'],r['variant']) for r in rows]!=[(i,j,n) for i,o in enumerate(ORDER) for j,n in enumerate(o)]:raise ValueError('paired order incomplete')
 if any(r['profiled'] or not r['formal_latency_evidence'] or r['status']!='FROZEN_GOLDEN_MATCH' for r in rows):raise ValueError('invalid measured evidence')
 if any(r['plan']!=rows[0]['plan'] or r['workspace_bytes']!=rows[0]['workspace_bytes'] for r in rows):raise ValueError('resource drift')
 stats={n:{key:q.statistics_ms([s for r in rows if r['variant']==n for s in r['timing'][key]['samples_ms']],90) for key in ('execute_sync','prepare')} for n in ('baseline','candidate')}
 ratios=[next(r['timing']['execute_sync']['median_ms'] for r in rows if r['round']==i and r['variant']=='baseline')/next(r['timing']['execute_sync']['median_ms'] for r in rows if r['round']==i and r['variant']=='candidate') for i in range(3)]
 ratio=stats['baseline']['execute_sync']['median_ms']/stats['candidate']['execute_sync']['median_ms']
 return dict(statistics=stats,round_ratios=ratios,baseline_over_candidate=ratio,net_implementation_delta_ms=stats['baseline']['execute_sync']['median_ms']-stats['candidate']['execute_sync']['median_ms'],decision='DIAGNOSTIC_MEASUREMENT_ONLY_NOT_PRODUCTION',guard_coverage=COVERAGE,
  scope=q.SCOPE,interpretation='Net removal effect including compiler changes; not pure Guard instruction time and not safety-equivalent; no production promotion')


def check_execution(ex,device,name,timed):
 if name not in COVERAGE or ex.get('guard_coverage')!=COVERAGE[name] or ex.get('kernel_guards_unchanged') is not (name=='baseline'):
  raise ValueError('Stage4T guard coverage mislabeled')
 if ex.get('op')!=p.NAME or ex.get('tile_count')!=76:raise ValueError('N256 ABI mismatch')
 # Common external safety/timing validator expects its original shape/unchanged flag.
 # Normalize ONLY after the independent Stage4T coverage contract above, as Stage4L/N do.
 # Persisted execution and reports retain actual diagnostic labels, never FULL coverage.
 return bc.check_execution(dict(ex,op=bc.full.NAME,tile_count=304,kernel_guards_unchanged=True),device,0,timed)

def validate_run(row,proof,device,timed,profiled=False):
 name=row['variant'];v=proof['variants'][name];ex=read(verify(row['execution']))
 stats=check_execution(ex,device,name,timed);folder=Path(row['execution']['path']).parent
 actual=q.plan(verify(row['log']).read_text(),'N256',v)
 if row.get('guard_coverage')!=COVERAGE[name] or row['profiled']!=profiled or row['status']!='FROZEN_GOLDEN_MATCH' or actual!=row['plan'] or stats!=row['timing'] or bc.verify_outputs(v,folder,timed)!=row['outputs']:
  raise ValueError('run proof differs')
 if row['workspace_bytes']!=ex['workspace_bytes'] or ex['workspace_bytes']<actual['total'] or read(verify(row['admission']))!=bc.stage4b.admission_expected(ex,actual,False):raise ValueError('workspace mismatch')
