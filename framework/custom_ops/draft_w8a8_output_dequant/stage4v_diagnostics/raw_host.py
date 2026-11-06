#!/usr/bin/env python3
"""Raw INT32 Host-only query. No OPP, Runner or compute Kernel is created."""
import argparse,os,platform,shutil,hashlib
from pathlib import Path
from evidence import HERE,v,load,identity,verify,verify_tree,read,write,fresh,sources
from sdk_review import bind
from raw_resources import validate
import raw_host_dependency_identity as dependency_validator
from raw_host_dependency_identity import compiled_dependencies,verify_dependency_binding


def dependency_validator_identity():
 """Reject foreign/cached validators before reading or generating build evidence."""
 expected=(Path(__file__).resolve().parent/'raw_host_dependency_identity.py').resolve(strict=True)
 if Path(dependency_validator.__file__).resolve(strict=True)!=expected:
  raise ValueError('Raw Host dependency validator module is outside Diagnostics')
 if dependency_validator.__spec__ is None or Path(dependency_validator.__spec__.origin).resolve(strict=True)!=expected:
  raise ValueError('Raw Host dependency validator import origin differs')
 functions={}
 for name,fn in (('compiled_dependencies',compiled_dependencies),('verify_dependency_binding',verify_dependency_binding)):
  if fn is not getattr(dependency_validator,name) or fn.__module__!='raw_host_dependency_identity' or Path(fn.__code__.co_filename).resolve(strict=True)!=expected:
   raise ValueError('Raw Host dependency validator function origin differs: '+name)
  functions[name]=dict(module=fn.__module__,source_path=str(expected))
 return dict(status='DIAGNOSTICS_VALIDATOR_IDENTITY_VERIFIED',module='raw_host_dependency_identity',file=identity(expected),functions=functions)


# Also protect direct imports and --help; do not alter sys.modules or search paths.
VALIDATOR_AT_IMPORT=dependency_validator_identity()


def bound_validator(source_records):
 current=dependency_validator_identity()
 if current!=VALIDATOR_AT_IMPORT or [x for x in source_records if x['path']==current['file']['path']]!=[current['file']]:
  raise ValueError('Raw Host dependency validator source identity changed')
 return current


def main(args):
 root=fresh(args.output_dir,[args.frozen_root,args.cann_root]);r=dict(status='NOT_RUN',source_review='NOT_RUN',host_feasible='NOT_RUN',build_pass='NOT_RUN',npu='NOT_RUN',opp='NOT_CREATED',runner='NOT_CREATED',sources=sources(),cause='UNRESOLVED',benchmark='PROHIBITED')
 try:
  r['dependency_validator']=bound_validator(r['sources'])
  data=load(args.frozen_root);r['evidence']=data['identities'];frozen=data['frozen']
  if args.device!=frozen['device']:raise ValueError('device differs from frozen run')
  if platform.system()!='Linux' or platform.machine() not in ('aarch64','arm64'):raise ValueError('real Linux ARM64 Host receiver required')
  sdk=bind(args.cann_root);old=v.p.host_o;release=old.bind(args.cann_root)
  prior=read(verify(frozen['host_report']))['sdk']
  if any(release[k]!=prior[k] for k in ('status','files','versions','shared_ub_contract')):raise ValueError('SDK differs from frozen Stage4V')
  r['sdk']=sdk;r['release']=release;r['source_review']='SOURCE_REVIEWED'
  case=data['case'];bf=verify(case['files']['b_i8.bin'])
  with bf.open('rb') as f:tile=f.read(2560*256)
  r['input_binding']=dict(a=case['files']['a_i8.bin'],b_source=case['files']['b_i8.bin'],b_offset_bytes=0,b_bytes=len(tile),b_slice_sha256=hashlib.sha256(tile).hexdigest(),tile_id=0,requantization=False,scale_input='NONE')
  env=v.common.env(args.cann_root);r['commands']={};tools={n:shutil.which(n) for n in ('cmake','c++','ldd')}
  if not all(tools.values()):raise ValueError('Host tool missing')
  r['tools']={k:identity(x) for k,x in tools.items()}
  for name,argv in [('configure',[tools['cmake'],'-S',HERE/'raw_host','-B',root/'host-build','-DCANN_ROOT='+str(args.cann_root),'-DCMAKE_CXX_COMPILER='+tools['c++'],'-DCMAKE_EXPORT_COMPILE_COMMANDS=ON']),('compile',[tools['cmake'],'--build',root/'host-build','--parallel'])]:
   r['commands'][name]=old.command(argv,root/(name+'.log'),env,300);write(root/'raw-host-report.json',r)
   if r['commands'][name]['returncode']!=0:raise ValueError('Host '+name+' failed')
  exe=root/'host-build/raw_int32_host';r['host_binary']=identity(exe);r['host_compile']='COMPLETED'
  r['linked_libraries']=old.linked_libraries(root/'host-build/CMakeCache.txt',args.cann_root)
  r['dependency_binding']=compiled_dependencies(root/'host-build',args.cann_root,HERE/'raw_host/probe.cpp',release['files'][0]['file'])
  if bound_validator(r['sources'])!=r['dependency_validator']:raise ValueError('validator changed before Host query')
  for name,argv in [('ldd',[tools['ldd'],exe]),('query',[exe,args.device,root])]:
   r['commands'][name]=old.command(argv,root/(name+'.log'),env,300)
   if r['commands'][name]['returncode']!=0 or name=='ldd' and 'not found' in (root/'ldd.log').read_text():raise ValueError('Host '+name+' failed')
  r['host_query']=identity(root/'host-query.json');raw=read(verify(r['host_query']))
  if raw['device']!=args.device or len(raw['cases'])!=1:raise ValueError('query scope differs')
  row=raw['cases'][0];r['resources']=validate(raw,row)
  r['formal_policy_reference']=frozen['evidence']['formal_core_policy']
  if not v.p.layout_i.core_consistency(raw,r['formal_policy_reference']):raise ValueError('manager outside accepted formal bounds')
  if r['resources']['status']!='RAW_RESOURCE_CHECKS_PASS':raise ValueError('Raw blockers: '+','.join(r['resources']['blockers']))
  r['tiling_binary']=identity(root/'RAW_INT32_N256-tiling.bin')
  keys=('singleCoreM','singleCoreN','singleCoreK','baseM','baseN','baseK','depthA1','depthB1','stepM','stepN','stepKa','stepKb','dbL0A','dbL0B','dbL0C','shareL1Size','shareL0CSize')
  r['versus_frozen_fp16']={k:dict(raw=row['fields'][k],fp16=frozen['t1']['fields'][k],same=row['fields'][k]==frozen['t1']['fields'][k]) for k in keys}
  r.update(status='RAW_HOST_CANDIDATE_FEASIBLE_CONTEXT_PENDING',host_feasible='STANDALONE_HOST_FEASIBLE',formal_tiling_context='NOT_RUN',
    next_gate='Review new Raw Tiling/ABI and enforce exact formal TilingContext Core/resource identity before any compute execution. No OPP/Runner generated by this tool.')
 except Exception as e:r.update(status='BLOCKED',error=f'{type(e).__name__}: {e}')
 finally:
  try:
   for f in r['sources']:verify(f)
   if 'dependency_validator' in r and bound_validator(r['sources'])!=r['dependency_validator']:raise ValueError('validator binding changed')
   for key in ('evidence','sdk','release','linked_libraries'):verify_tree(r.get(key,{}))
   if 'dependency_binding' in r:verify_dependency_binding(r['dependency_binding'])
  except Exception as e:r.update(status='BLOCKED',integrity_error=str(e))
  write(root/'raw-host-report.json',r);write(root/'resource-report.json',r.get('resources',dict(status='NOT_RUN')))
 print(r['status']+': '+str(root/'raw-host-report.json'));return int(r['status']=='BLOCKED')

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--frozen-root',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--device',type=int,default=0);ap.add_argument('--cann-root',type=Path,default=Path(os.environ.get('CANN_ROOT','/usr/local/Ascend/cann-9.0.0')))
 args=ap.parse_args();args.cann_root=args.cann_root.resolve()
 if args.device<0:ap.error('negative device')
 raise SystemExit(main(args))
