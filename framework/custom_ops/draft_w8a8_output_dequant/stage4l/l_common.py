"""2x2 attribution, scoped safety, and immutable upstream admission."""
import sys,copy,difflib
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4k'))
import k_common as k
import run_stage4k as rk
sys.path.insert(0,str(HERE.parent/'stage4d'))
import pair_common as d
import run_stage4d as rd
import guard_variant as gv
from k_common import identity,verify,verify_tree,read,statistics_ms
write=k.write;fresh=k.fresh;environment=k.environment
NAMES=('N64Full','N128Full','N64Weak','N128Weak')
# Williams-balanced order: each position once; each ordered adjacent pair once.
ORDER=tuple(tuple(NAMES[i] for i in order) for order in ((0,1,3,2),(1,2,0,3),(2,3,1,0),(3,0,2,1)))
POLICY={n:('DIAGNOSTIC_ONLY_WEAKER_GUARD_COVERAGE' if n.endswith('Weak') else 'FULL_GROUP_GUARDS') for n in NAMES}
SCOPE=k.SCOPE
bc=k.bc;rc=k.rc;common=k.common

def sources():
 rows={x['path']:x for x in k.sources()+d.sources()}
 for p in HERE.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts:rows[str(p.resolve())]=identity(p)
 return [rows[x] for x in sorted(rows)]

def transform(raw):
 # Stage4D's exact group-tail transformation with only N128 buffer lengths.
 scan=gv.SCAN.replace('(),2048','(),4096').replace('(),4096)||!CheckGuard(sumRaw','(),8192)||!CheckGuard(sumRaw').replace('sumRaw.Get<uint32_t>(),4096','sumRaw.Get<uint32_t>(),8192')
 text=raw.decode();start=text.find(scan)
 if text.count(scan)!=1 or not text[:start].endswith('SetFlag<HardEvent::V_S>(groupReady);WaitFlag<HardEvent::V_S>(groupReady);\n') or not text[start+len(scan):].startswith('    }\n    // All modes'):raise ValueError('N128 group guard anchor drift')
 return (text[:start]+text[start+len(scan):]).encode()

def materialize(root):
 base=HERE.parent/'stage4j';root.mkdir();rows=[]
 for p in [base/'operator.json']+[p for sub in ('op_host','op_kernel') for p in (base/sub).iterdir() if p.is_file()]:
  dst=root/p.relative_to(base);dst.parent.mkdir(parents=True,exist_ok=True);raw=p.read_bytes();data=transform(raw) if p.parent.name=='op_kernel' else raw
  dst.write_bytes(data);rows.append(dict(original=identity(p),generated=identity(dst)))
 original=(base/'op_kernel/d_flash_w8_a8_n128_gate.cpp').read_text();candidate=(root/'op_kernel/d_flash_w8_a8_n128_gate.cpp').read_text()
 diff=''.join(difflib.unified_diff(original.splitlines(True),candidate.splitlines(True),fromfile='N128Full',tofile='N128Weak'))
 (root/'group-guard-only.diff').write_text(diff)
 audit=dict(status='EXACT_GROUP_SCAN_ONLY',files=rows,diff=identity(root/'group-guard-only.diff'),safety_protocol=POLICY['N128Weak'],removed_group_checks=5)
 write(root/'source-audit.json',audit);return audit

def verify_audit(audit):
 if audit['status']!='EXACT_GROUP_SCAN_ONLY' or audit['safety_protocol']!=POLICY['N128Weak']:raise ValueError('audit policy changed')
 base=HERE.parent/'stage4j'
 wanted={str(p.resolve()) for p in [base/'operator.json']+[p for sub in ('op_host','op_kernel') for p in (base/sub).iterdir() if p.is_file()]}
 if len(audit['files'])!=len(wanted) or {x['original']['path'] for x in audit['files']}!=wanted:raise ValueError('audit inventory differs')
 for row in audit['files']:
  src=verify(row['original']);dst=verify(row['generated']);raw=src.read_bytes()
  if dst.read_bytes()!=(transform(raw) if src.parent.name=='op_kernel' else raw):raise ValueError('unapproved source delta')
 original=(base/'op_kernel/d_flash_w8_a8_n128_gate.cpp').read_text()
 expected=''.join(difflib.unified_diff(original.splitlines(True),transform(original.encode()).decode().splitlines(True),fromfile='N128Full',tofile='N128Weak'))
 if verify(audit['diff']).read_text()!=expected:raise ValueError('audit diff changed')

def evidence(args):
 base=k.evidence(args);p=read(args.stage4k_root/'benchmark/benchmark-report.json')
 current=dict(sources=k.sources(),evidence=base,device=args.device,machine=p['machine'])
 rk.prior(verify(p['build_report']),'build',current);current['build_report']=p['build_report']
 rk.prior(args.stage4k_root/'benchmark/benchmark-report.json','benchmark',current)
 prof=read(args.stage4k_root/'profile/profile-report.json')
 if prof.get('status')!='STAGE4K_PROFILE_COMPLETED' or prof.get('evidence')!=base or prof.get('sources')!=k.sources() or prof.get('benchmark_report')!=identity(args.stage4k_root/'benchmark/benchmark-report.json'):raise ValueError('accepted Stage4K Profile required')
 if [x['variant'] for x in prof['runs']]!=['N64','N128']:raise ValueError('Stage4K Profile incomplete')
 for row in prof['runs']:k.validate_run(row,base,args.device,True,True)
 args.stage4c_root=args.stage4c_report.parent.parent
 de=d.evidence(args);dr=read(args.stage4d_root/'benchmark/benchmark-report.json');db=read(verify(dr['pair_build_report']))
 if dr.get('sources')!=d.sources() or dr.get('evidence')!=de or dr.get('device')!=args.device or de['stage4b']!=base['variants']['N64']:raise ValueError('Stage4D B provenance differs')
 rd.accepted_report(args.stage4d_root/'benchmark/benchmark-report.json','STAGE4D_AB_MEASURED_DIAGNOSTIC_ONLY',dr,db)
 verify_compiled_b(db)
 variants={n:copy.deepcopy(base['variants']['N128' if n.startswith('N128') else 'N64']) for n in NAMES}
 variants['N64Weak'].update(opp=db['variants']['B']['opp'],installed=db['variants']['B']['installed'],historical_runner=db['variants']['B']['runner'])
 # N128 Weak is built later; never authorize Full OPP under a Weak label.
 variants['N128Weak'].update(opp=None,installed=None)
 for n,v in variants.items():v['safety_protocol']=POLICY[n]
 return dict(variants=variants,stage4k=identity(args.stage4k_root/'benchmark/benchmark-report.json'),stage4k_profile=identity(args.stage4k_root/'profile/profile-report.json'),
  stage4d=identity(args.stage4d_root/'benchmark/benchmark-report.json'),stage4d_build=dr['pair_build_report'],stage4d_audit=db['source_audit'],upstream=base,scope=SCOPE)

def immutable(proof):
 verify_tree(proof);gv.verify_audit(proof['stage4d_audit'],verify)
 for n,p in proof['variants'].items():
  if p['opp'] and (k.jb.installed if n.startswith('N128') else k.bc.installed)(Path(p['opp']))!=p['installed']:raise ValueError('OPP drift '+n)

def check_execution(ex,device,name,timed):
 if ex.get('stage4l_variant')!=name or ex.get('guard_coverage')!=POLICY[name] or ex.get('kernel_guards_unchanged') is not name.endswith('Full'):raise ValueError('guard coverage mislabeled')
 x=dict(ex,kernel_guards_unchanged=True) # normalize only after explicit scoped coverage validation
 return k.check_execution(x,device,'N128' if name.startswith('N128') else 'N64',timed)

def plan(text,name,p):return k.plan(text,'N128' if name.startswith('N128') else 'N64',p)

def validate_run(row,proof,device,timed,profiled=False):
 n=row['variant'];p=proof['variants'][n];ex=read(verify(row['execution']));stats=check_execution(ex,device,n,timed)
 folder=Path(row['execution']['path']).parent;actual=plan(verify(row['log']).read_text(),n,p)
 if row.get('safety_protocol')!=POLICY[n] or row.get('profiled') is not profiled or row['status']!='FROZEN_GOLDEN_MATCH' or row['plan']!=actual or row['timing']!=stats:raise ValueError('scoped run proof differs')
 if ex['workspace_bytes']<actual['total'] or read(verify(row['admission']))!=k.bc.stage4b.admission_expected(ex,actual,False) or k.bc.verify_outputs(p,folder,timed)!=row['outputs']:raise ValueError('resource/Golden proof differs')
 return stats

def comparison(runs):
 if [(r['round'],r['position'],r['variant']) for r in runs]!=[(i,j,n) for i,o in enumerate(ORDER) for j,n in enumerate(o)]:raise ValueError('four balanced rotations required')
 stats={}
 for width in ('N64','N128'):
  rows=[r for r in runs if r['variant'].startswith(width)]
  for key in ('plan','workspace_bytes'):
   if any(r.get(key)!=rows[0].get(key) for r in rows):raise ValueError('Full/Weak resource drift '+width)
 for n in NAMES:
  rows=[r for r in runs if r['variant']==n]
  if any(r['profiled'] or not r['formal_latency_evidence'] or r['safety_protocol']!=POLICY[n] or r['status']!='FROZEN_GOLDEN_MATCH' for r in rows):raise ValueError('unaccepted samples')
  stats[n]={key:statistics_ms([s for r in rows for s in r['timing'][key]['samples_ms']],120) for key in ('execute_sync','prepare')}
 def contrasts(m):
  a=m['N64Full']-m['N64Weak'];b=m['N128Full']-m['N128Weak']
  return dict(n64_guard_net_ms=a,n128_guard_net_ms=b,full_tile_ratio=m['N64Full']/m['N128Full'],weak_tile_ratio=m['N64Weak']/m['N128Weak'],interaction_ms=a-b)
 med={n:stats[n]['execute_sync']['median_ms'] for n in NAMES}
 rounds=[dict(round=i,**contrasts({r['variant']:r['timing']['execute_sync']['median_ms'] for r in runs if r['round']==i})) for i in range(4)]
 return dict(status='STAGE4L_MEASURED_DIAGNOSTIC_ONLY',statistics=stats,rounds=rounds,**contrasts(med),scope=SCOPE,
  interpretation='Net compiled-code effects; not pure Guard instruction time or additive Pipe/critical-path attribution',safety_protocols=POLICY,adoption='NOT_ELIGIBLE_WEAK_VARIANTS')


def verify_compiled_b(db):
 for sub in ('op_host','op_kernel'):
  for file in (Path(db['source_audit']['variants']['B']['source_dir'])/sub).iterdir():
   if file.is_file():
    generated=[x for x in db['variants']['B']['generated_sources'] if Path(x['path']).name==file.name and Path(x['path']).parent.name==sub]
    if len(generated)!=1 or verify(generated[0]).read_bytes()!=file.read_bytes():raise ValueError('Stage4D compiled source differs from audit')
