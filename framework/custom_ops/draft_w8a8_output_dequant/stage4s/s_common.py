"""Local contract must match server bytes before any candidate build/execution."""
import sys,copy
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4q'))
import q_common as q
import run_stage4q as qr
import source_s as source
materialize=source.materialize;verify_audit=source.verify_audit
p=q.p;rp=q.rp;pr=q.pr
NAME=p.NAME;SNAKE=p.SNAKE;common=q.common
identity,verify,verify_tree,read,write,fresh=q.identity,q.verify,q.verify_tree,q.read,q.write,q.fresh
environment=q.environment;bc=q.bc;rc=q.rc
SDK={
 'impl/basic_api/kernel_operator_data_copy_intf_impl.h':'0a432937742d2625565d3eb1ef3caa930a190271faadb0344345d2382235737b',
 'impl/basic_api/dav_m200/kernel_operator_data_copy_impl.h':'8a890aa80c39380519bcba461bfd19a67060e650253622d8b49b4b2c01ea3c8f',
 'include/basic_api/kernel_struct_data_copy.h':'8402ed99bc1be3c05f688fb92e9262531ebce6199ba953c827dcef2a47bf5343',
 'impl/basic_api/utils/kernel_check_data_copy_util.h':'63b0874089739c09f37c390608bd0a66d99bb7c928e10c94f46b7bf823051c05',
 'impl/basic_api/utils/kernel_utils_mode_cpu.h':'794ec64d4e7e871ca79704c87770d920946544bcefe155f22b4dd7607f4d1f40',
 'impl/basic_api/utils/kernel_utils_constants.h':'3ce93de8c25d198386e87656481f6bc9dfbeae86fe66a248f7e3d36c86cd32da'}

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
 ids=[]
 for rel,sha in SDK.items():
  item=identity(args.cann_root/'aarch64-linux/asc'/rel)
  if args.cann_root not in Path(item['path']).parents or item['sha256']!=sha:raise ValueError('DataCopy SDK mismatch '+rel)
  ids.append(item)
 baseline=base['variants']['N256'];full=read(verify(base['stage4p_full']))
 return dict(baseline=baseline,preflight=baseline['preflight'],t1=baseline['t1'],case=baseline['case'],formal_core_policy=baseline['policy'],
  stage4p_full=base['stage4p_full'],stage4p_fault=base['stage4p_fault'],stage4p_prepare=full['prepare_report'],
  stage4q=identity(args.stage4q_root/'benchmark/benchmark-report.json'),profile=identity(args.stage4q_root/'profile/profile-report.json'),profile_review=profile_review,
  sdk=dict(status='REVIEWED_DATACOPY_IDENTITIES_MATCHED',files=ids),scope=q.SCOPE)

def immutable(proof):
 verify_tree(proof)
 if rp.installed(Path(proof['baseline']['opp']))!=proof['baseline']['installed']:raise ValueError('baseline OPP changed')

def checked_build(b,r):
 if b.get('status')!='STAGE4S_BUILT_NPU_NOT_RUN' or any(b.get(k)!=r[k] for k in ('sources','evidence','device')):raise ValueError('build scope changed')
 from p_common import contract_header
 for kind in ('candidate','fault'):
  item=b['variants'][kind];verify_audit(item['audit']);built=item['build'];verify_tree(built)
  if item['audit']['diagnostic']!=(kind=='fault') or built['status']!='STAGE4S_PRIVATE_BUILT' or rp.installed(Path(built['opp']))!=built['installed']:raise ValueError('variant identity differs')
  expected={Path(x['candidate']['path']).parent.name+'/'+Path(x['candidate']['path']).name:verify(x['candidate']).read_bytes() for x in item['audit']['files'] if Path(x['candidate']['path']).suffix in ('.cpp','.h')}
  expected['op_host/t1_contract.h']=contract_header(r['evidence']['t1']).encode()
  actual={Path(x['path']).parent.name+'/'+Path(x['path']).name:verify(x).read_bytes() for x in built['generated_sources']}
  if expected!=actual:raise ValueError('compiled source differs')
  if (Path(built['opp']).parent/'operator.json').read_bytes()!=(HERE.parent/'stage4p/operator.json').read_bytes():raise ValueError('ABI drift')
 for name in ('baseline','candidate'):
  bench=b['bench_runners'][name]
  proof=copy.deepcopy(r['evidence']['baseline'])
  if name=='candidate':proof.update(opp=b['variants']['candidate']['build']['opp'],installed=b['variants']['candidate']['build']['installed'])
  if bench['status']!='STAGE4S_BENCH_RUNNER_BUILT' or bench['evidence']!=proof or bench['linked_libraries']['LIB_custom']!=proof['installed']['api']:raise ValueError('benchmark runner/OPP binding differs')
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
 return dict(statistics=stats,round_ratios=ratios,baseline_over_candidate=ratio,decision='GO_FURTHER_REVIEW' if ratio>=1.03 and all(x>1 for x in ratios) else 'NO_GO_NO_MATERIAL_IMPROVEMENT',
  scope=q.SCOPE,interpretation='Net final-store change including compiler effects; source calls are not machine instructions or isolated wall time')
