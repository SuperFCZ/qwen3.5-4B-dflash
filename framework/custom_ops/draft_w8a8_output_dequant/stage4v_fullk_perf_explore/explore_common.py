"""Rebind frozen normal OPP and failures; never mutate a historical experiment."""
import importlib.util
from pathlib import Path
import sys
import re
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE/'stage4v'))
import v_common as v
import protocol_v as old_protocol
import run_stage4v as old_run
spec=importlib.util.spec_from_file_location('_fullk_explore_frozen_evidence',BASE/'stage4v_diagnostics/evidence.py')
frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
sys.path.insert(0,str(BASE/'stage4v_raw_int32_probe'))
from raw_probe_runtime_identity import collect,verify_runtime,dependencies,confined,trace
from run_stage2b import command
from explore_numeric import observe,known_six
identity,verify,verify_tree,read,write,fresh=v.identity,v.verify,v.verify_tree,v.read,v.write,v.fresh


def sources():
 files=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
 files.extend([BASE/'stage4v_diagnostics/evidence.py',BASE/'stage4v_raw_int32_probe/raw_probe_runtime_identity.py',v.common.QUANT/'test/runner_common.h'])
 return [identity(p) for p in sorted(files)]


def bind(root,device):
 data=frozen.load(root);r=data['frozen'];b=read(verify(r['build_report']))
 if device!=r['device']:raise ValueError('device differs from frozen Build')
 for phase in ('prepare','host','build'):old_run.prior(verify(data['identities'][phase]),phase,r)
 if r.get('integrity_error'):raise ValueError('historical integrity failure')
 text=verify(data['identities']['log']).read_text()
 ex=read(verify(data['identities']['execution']))
 actual=verify(data['identities']['actual']).read_bytes();expected=verify(data['identities']['expected']).read_bytes()
 comparison=validate_historical_failure(text,ex,actual,expected)
 normal=b['variants']['normal']['build'];policy=r['evidence']['formal_core_policy']
 if policy!=dict(aic=8,aiv=8,all=7,selected=7,arch=2002):raise ValueError('frozen formal policy changed; no fallback')
 adm=read(verify(data['identities']['admission']))
 if adm['workspace_bytes']!=16784384:raise ValueError('frozen Workspace identity differs')
 return dict(identities=data['identities'],original_sources=r['sources'],normal=normal,
  normal_audit=b['variants']['normal']['audit'],t1=r['t1'],case=r['case'],policy=policy,workspace=adm['workspace_bytes'],device=device,
  machine=r['machine'],original_evidence=r['evidence'],historical_comparison=comparison,
  historical_failure='STRICT_GOLDEN_ONLY_BY_BOUND_RUNNER_CONTROL_FLOW',historical_ub_arena='INDIRECT_COMPLETE_FINITE_TILE_WITH_POISON_ON_FAILURE; NOT_DIRECT_UB_READBACK')



def validate_historical_failure(text,ex,actual,expected):
 failure='FAIL: strict Stage4V Golden mismatch; no timing after failed precheck'
 if [x for x in text.splitlines() if x.startswith('FAIL:')]!=[failure]:raise ValueError('historical failure not exclusively strict Golden mismatch')
 if re.search(r'status=(?:CORRUPT|UNALLOCATED)|acl\w* failed|synchronize failed',text):raise ValueError('historical safety failure')
 for name in ('a','b','scale','o','workspace'):
  for side in ('prefix','suffix'):
   if not re.search(r'guard buffer='+name+r' side='+side+r' .*status=INTACT changed_bytes=0(?:\n|$)',text):raise ValueError('historical GM guard log missing')
 if ex.get('status')!='FAIL' or ex.get('timing_complete') is not False or ex['timing']['execute_sync']['samples_ms']:raise ValueError('unexpected historical failure execution')
 comparison=observe(actual,expected,0,0);known_six(comparison)
 return comparison

def environment(proof,cann):
 env=v.environment(dict(opp=proof['normal']['opp'],policy=proof['policy']),cann)
 env['FULLK_EXPLORE_WORKSPACE_BYTES']=str(proof['workspace'])
 return env


def unchanged(proof):
 old_protocol.verify_build(read(verify(proof['identities']['build'])),read(verify(proof['identities']['bounded'])))
 verify_tree(proof)
 v.immutable(proof['original_evidence'])


def machine(root,proof,cann,r):
 result=v.q.rc.machine(root,proof['original_evidence']['baseline'],cann,r)
 for key in ('os','version_files'):
  if result[key]!=proof['machine'][key]:raise ValueError('machine differs from frozen Stage4V '+key)
 return result


def build(root,proof,cann,r):
 env=environment(proof,cann);normal=proof['normal'];opp=Path(normal['opp'])
 for tag,path in [('historical_runner',verify(normal['runner'])),('normal_host',verify(normal['installed']['host']))]:
  command(['ldd',path],root/(tag+'-ldd.log'),env,r,tag+'_ldd')
  dep=runtime(root/(tag+'-ldd.log'),cann,proof)
  r.setdefault('dependencies',{})[tag]=dep
 command(['cmake','-S',HERE/'runner','-B',root/'runner-build','-DCANN_ROOT='+str(cann),'-DOPP_ROOT='+str(opp),'-DCOMMON_INCLUDE='+str(v.common.QUANT/'test'),'-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],root/'cmake.log',env,r,'configure')
 command(['cmake','--build',root/'runner-build','--parallel'],root/'compile.log',env,r,'compile')
 runner=root/'runner-build/fullk_explore_runner';command(['ldd',runner],root/'runner-ldd.log',env,r,'runner_ldd')
 r['dependencies']['exploration_runner']=runtime(root/'runner-ldd.log',cann,proof)
 links={}
 for line in (root/'runner-build/CMakeCache.txt').read_text().splitlines():
  if line.startswith('LIB_') and ':FILEPATH=' in line:
   key,path=line.split(':FILEPATH=',1);item=identity(path);boundary=opp if key=='LIB_custom' else cann
   if boundary.resolve() not in Path(item['path']).parents:raise ValueError('linked library escaped bound root')
   links[key]=item
 if set(links)!={'LIB_custom','LIB_ascendcl','LIB_nnopbase','LIB_acl_op_compiler'} or links['LIB_custom']!=normal['installed']['api']:raise ValueError('wrong normal API linkage')
 command(['nm','-u',runner],root/'symbols.log',env,r,'symbols')
 text=(root/'symbols.log').read_text()
 for sym in ('aclnnDFlashW8A8FullKChannelGetWorkspaceSize','aclnnDFlashW8A8FullKChannel'):
  if not re.search(r'\b'+sym+r'\b',text):raise ValueError('wrong ACLNN ABI')
 r['artifacts']=dict(runner=identity(runner),linked_libraries=links,compile_commands=identity(root/'runner-build/compile_commands.json'),symbols=identity(root/'symbols.log'))
 r['kernel_build']='NOT_RUN_REUSED_FROZEN_NORMAL_OPP';r['opp_install']='NOT_RUN'


def verify_artifacts(b):
 verify_tree(b['artifacts'])
 if b['artifacts']['linked_libraries']['LIB_custom']!=b['evidence']['normal']['installed']['api']:raise ValueError('artifact normal API differs')
 for d in b['dependencies'].values():
  verify_runtime(d)
  for private in d['private_opp_libraries']:
   if trace(private['requested_path'])!={k:private[k] for k in ('requested_path','real_path','symlink_chain')}:raise ValueError('private OPP alias changed')


def runtime(log,cann,proof):
 original=identity(log);opp=Path(proof['normal']['opp']).resolve();private=[];other=[]
 allowed=[proof['normal']['installed'][k] for k in ('api','host')]
 for needed,path in dependencies(log.read_text()):
  if needed=='libcust_opapi.so' and opp not in path.parents:raise ValueError('custom API resolved outside normal OPP')
  if opp in path.parents:
   entry=confined(path,(opp,))
   if entry['file'] not in allowed:raise ValueError('unknown private OPP dependency')
   private.append(dict(entry,needed=needed))
  else:other.append(f'{needed} => {path} (0x0)')
 filtered=log.with_name(log.stem+'-sdk-driver-system.log');filtered.write_text('\n'.join(other)+'\n')
 history=[dict(label='preflight',report=proof['original_evidence']['preflight'],content=read(verify(proof['original_evidence']['preflight']))),
  dict(label='frozen_bounded',report=proof['identities']['bounded'],content=read(verify(proof['identities']['bounded'])))]
 r=collect(filtered,cann,history);r.update(original_ldd=original,private_opp_libraries=private)
 verify(original);return r
