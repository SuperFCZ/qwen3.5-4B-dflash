"""Read-only history and new experiment identities; no old module state mutation."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
sys.path.insert(0,str(BASE/'stage4v_fullk_profile'))
import profile_evidence as pe
from explore_protocol import OUTPUTS
__all__=["OUTPUTS","NAMES"]
c=pe.c;v=c.v;common=v.common
identity,verify,read,write,fresh,verify_tree=c.identity,c.verify,c.read,c.write,c.fresh,c.verify_tree
from sg_transform import NAMES,verify_audit


def sources():return [identity(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]


def bind(a,root,r):
 proof=pe.bind(a.experiment_root,a.cann_root,a.device,root,r)
 r['machine']=proof.pop('machine')
 from explore_numeric import observe
 anchor=verify(proof['full_anchor']).read_bytes()
 tile=read(verify(proof['reports']['tile0']))
 for i,out in enumerate(tile['runs'][0]['outputs']):
  if observe(verify(out).read_bytes(),anchor,0,i)['mismatch_count']:raise ValueError('historical Tile0 and Full anchors disagree')
 profile=read(a.profile_report);verify_tree(profile)
 if profile.get('status') not in ('FULLK_PROFILE_COMPLETED','FULLK_PROFILE_COMPLETED_METRICS_PARTIAL') or profile.get('capture_status')!='TWO_SAFE_FULL_EXECUTIONS_COMPLETED' or profile.get('error') or profile.get('integrity_error'):raise ValueError('safe complete Profile required')
 if profile['sources']!=pe.sources() or profile['evidence']['reports']!=proof['reports']:raise ValueError('Profile identity chain differs')
 if pe.validate(profile['run'],proof['proof'],a.device)!=profile['run']['validation']:raise ValueError('Profile safety changed')
 if any(verify(x).read_bytes()!=verify(proof['full_anchor']).read_bytes() for x in profile['run']['outputs']):raise ValueError('Profile repeat changed')
 sdk=[]
 for spec in read(HERE/'sdk-review.json')['sources']:
  f=identity(a.cann_root/spec['relative_path'])
  if a.cann_root not in Path(f['path']).parents or f['sha256']!=spec['sha256']:raise ValueError('Guard SDK identity differs')
  count=len(verify(f).read_text().splitlines())
  if any(not 1<=lo<=hi<=count for lo,hi in spec['lines']):raise ValueError('SDK line range invalid')
  sdk.append(dict(file=f,lines=spec['lines']))
 return dict(history=proof,profile_report=identity(a.profile_report),sdk=sdk,
  baseline_runner=proof['build']['artifacts']['runner'],baseline=proof['proof'],anchor=proof['full_anchor'])


def env(evidence,cann,opp=None):
 p=evidence['baseline'];modified=dict(p)
 if opp is not None:modified['normal']=dict(p['normal'],opp=str(opp))
 return c.environment(modified,cann)


def verify_bound(evidence):
 verify_tree(evidence);pe.verify_bound(evidence['history']);c.unchanged(evidence['baseline'])


def inspect_installed(item,kind):
 from sg_build import installed
 if installed(Path(item['opp']),kind)!=item['installed']:raise ValueError('private candidate OPP drift')
 verify_tree(item)


def verify_build(b):
 for kind in ('candidate','fault'):
  x=b['variants'][kind]
  item=x['build']
  if item.get('status')!='GUARD_AB_PRIVATE_BUILT' or item.get('cann_compile')!='COMPLETED' or item.get('installation')!='ISOLATED_OPP_VERIFIED' or any(z.get('returncode')!=0 for z in item['commands'].values()):raise ValueError('incomplete private build')
  verify_audit(x['audit'],verify);inspect_installed(x['build'],kind)
  if x['build']['linked_libraries']['LIB_custom']!=x['build']['installed']['api']:raise ValueError('wrong API')
  if x['build']['generated_sources']!=x['compiled_sources']:raise ValueError('compiled source inventory changed')
  want={Path(f['path']).parent.name+'/'+Path(f['path']).name:verify(f).read_bytes() for f in x['audit']['files'] if Path(f['path']).suffix in ('.h','.cpp')}
  want['op_host/t1_contract.h']=v.contract_header(b['evidence']['baseline']['t1']).encode()
  got={Path(f['path']).parent.name+'/'+Path(f['path']).name:verify(f).read_bytes() for f in x['compiled_sources']}
  if want!=got:raise ValueError('compiled source differs from exact transform')
  if read(Path(x['build']['opp']).parent/'operator.json')!=read(verify(next(f for f in x['audit']['files'] if Path(f['path']).name=='operator.json'))):raise ValueError('operator definition changed')
  for dep in x['build']['dependencies'].values():
   c.verify_runtime(dep)
   for private in dep['private_opp_libraries']:
    if c.trace(private['requested_path'])!={k:private[k] for k in ('requested_path','real_path','symlink_chain')}:raise ValueError('private alias drift')
 # Fault executable/ELF cannot be selected for performance.
 if b['variants']['candidate']['build']['runner']==b['variants']['fault']['build']['runner']:raise ValueError('fault Runner aliased to candidate')
