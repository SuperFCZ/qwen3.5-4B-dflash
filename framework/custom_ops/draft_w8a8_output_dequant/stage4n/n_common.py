"""Immutable N128 Full evidence; new XOR and fault-only private artifacts."""
import sys,copy
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4k'))
import k_common as k
import guard_transform as gt
materialize=gt.materialize;verify_audit=gt.verify_audit;j=gt.j
identity=k.identity;verify=k.verify;read=k.read;write=k.write;fresh=k.fresh;common=k.common;environment=k.environment
NAMES=('Full','Xor');ORDER=(('Full','Xor'),('Xor','Full'),('Full','Xor'))
POLICY=dict(Full='FULL_GROUP_GUARDS',Xor='EXACT_640_WORD_GROUP_XOR_OR')
SCOPE=k.SCOPE
bc=k.bc;rc=k.rc

def sources():
 rows={x['path']:x for x in k.sources()}
 for p in HERE.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts:rows[str(p.resolve())]=identity(p)
 return [rows[x] for x in sorted(rows)]

def evidence(args):
 old=k.evidence(args)
 bindings=[]
 for rel,sha in [('kernel_tensor_impl.h','1e5d6669d91e5687db76fcace1cb1b2a9af0dfde92c4b5ee65dd2b23b99b9e0a'),('kernel_macros.h','500ea245ecddc927f644ae8cb2a75abe914164b402d372c7293297135ef0d3f9'),('kernel_event.h','d974ecee61e39402dad6fa368e036b538bc1f69b05989b6b3a9fc9c946bd609c'),('kernel_operator_block_sync_intf_impl.h','7f0fa56ad683b86f2ded10b984645e698d05a04e541c04d25cedb69c4be70c6f')]:
  item=identity(args.cann_root/'aarch64-linux/asc/impl/basic_api'/rel)
  if args.cann_root not in Path(item['path']).parents or item['sha256']!=sha:raise ValueError('Scalar SDK identity mismatch '+rel)
  bindings.append(item)
 old['scalar_sdk']=bindings
 full=old['variants']['N128'];return dict(upstream=old,variants={n:copy.deepcopy(full) for n in ('Full','Xor','Fault')},scope=SCOPE)

def check_build(b):
 for name in ('Xor','Fault'):
  item=b['opps'][name]
  if item['audit']['kind']!=('xor' if name=='Xor' else 'fault'):raise ValueError('timing/fault artifact swapped')
  verify_audit(item['audit']);w=item['build'];k.verify_tree(w)
  if w['status']!='STAGE4N_PRIVATE_OPP_BUILT' or k.jb.installed(Path(w['opp']))!=w['installed']:raise ValueError('private OPP differs')
  expected={Path(x['generated']['path']).parent.name+'/'+Path(x['generated']['path']).name:verify(x['generated']).read_bytes() for x in item['audit']['files'] if Path(x['generated']['path']).suffix in ('.cpp','.h')}
  expected['op_host/t1_contract.h']=j.contract_header(w['evidence']['t1']).encode()
  actual={Path(x['path']).parent.name+'/'+Path(x['path']).name:verify(x).read_bytes() for x in w['generated_sources']}
  if expected!=actual:raise ValueError('compiled source delta differs')
  if (Path(w['opp']).parent/'operator.json').read_bytes()!=(HERE.parent/'stage4j/operator.json').read_bytes():raise ValueError('ABI changed')
 hashes=[{x['binary']['sha256'] for x in p['installed']['kernels']} for p in [b['evidence']['variants']['Full'],b['opps']['Xor']['build'],b['opps']['Fault']['build']]]
 if any(hashes[i]==hashes[t] for i in range(3) for t in range(i)):raise ValueError('identical binaries cannot establish experiment identities')

def active(proof,b):
 check_build(b);r=copy.deepcopy(proof)
 for name in ('Xor','Fault'):
  w=b['opps'][name]['build'];r['variants'][name].update(opp=w['opp'],installed=w['installed'])
 return r

def immutable(proof):
 k.verify_tree(proof)
 for p in proof['variants'].values():
  if k.jb.installed(Path(p['opp']))!=p['installed']:raise ValueError('OPP identity drift')

def check_execution(ex,device,name,timed):
 if ex.get('stage4n_variant')!=name or ex.get('guard_coverage')!=POLICY[name] or ex.get('kernel_guards_unchanged') is not (name=='Full'):raise ValueError('coverage label differs')
 return k.check_execution(dict(ex,kernel_guards_unchanged=True),device,'N128',timed)

def plan(text,name,p):return k.plan(text,'N128',p)

def validate_run(row,proof,device,timed,profiled=False):
 p=proof['variants'][row['variant']];ex=read(verify(row['execution']));stats=check_execution(ex,device,row['variant'],timed)
 folder=Path(row['execution']['path']).parent;actual=plan(verify(row['log']).read_text(),row['variant'],p)
 if row['safety_protocol']!=POLICY[row['variant']] or row['profiled']!=profiled or row['status']!='FROZEN_GOLDEN_MATCH' or actual!=row['plan'] or stats!=row['timing']:raise ValueError('scoped run proof differs')
 if ex['workspace_bytes']<actual['total'] or read(verify(row['admission']))!=k.bc.stage4b.admission_expected(ex,actual,False) or k.bc.verify_outputs(p,folder,timed)!=row['outputs']:raise ValueError('resource/output proof differs')

def comparison(rows):
 if [(r['round'],r['position'],r['variant']) for r in rows]!=[(i,t,n) for i,o in enumerate(ORDER) for t,n in enumerate(o)]:raise ValueError('three paired rounds required')
 if any(r['profiled'] or not r['formal_latency_evidence'] or r['status']!='FROZEN_GOLDEN_MATCH' or r['safety_protocol']!=POLICY[r['variant']] for r in rows):raise ValueError('invalid formal evidence')
 if any(r['plan']!=rows[0]['plan'] or r['workspace_bytes']!=rows[0]['workspace_bytes'] for r in rows):raise ValueError('paired resource drift')
 stats={n:{key:k.statistics_ms([v for r in rows if r['variant']==n for v in r['timing'][key]['samples_ms']],90) for key in ('execute_sync','prepare')} for n in NAMES}
 ratios=[next(r['timing']['execute_sync']['median_ms'] for r in rows if r['round']==i and r['variant']=='Full')/next(r['timing']['execute_sync']['median_ms'] for r in rows if r['round']==i and r['variant']=='Xor') for i in range(3)]
 ratio=stats['Full']['execute_sync']['median_ms']/stats['Xor']['execute_sync']['median_ms']
 # Predeclared materiality gate; not a statistical confidence or production claim.
 go=ratio>=1.03 and all(x>1 for x in ratios)
 return dict(statistics=stats,round_ratios=ratios,full_over_xor=ratio,decision='GO_FURTHER_REVIEW' if go else 'NO_GO_NO_MATERIAL_IMPROVEMENT',
  criterion='aggregate median ratio>=1.03 and each round ratio>1; otherwise stop this implementation',scope=SCOPE,
  interpretation='net compiled guard implementation effect, not pure branch cost; 640 uint32 loads per group retained',weak='HISTORICAL_DIAGNOSTIC_ONLY_NOT_PAIRED')
