"""Actual selected-device Host query, independently built before the compute OPP."""
from pathlib import Path
import platform,re,shutil
from nz_common import HERE,p,q,identity,verify,write,read
from nz_resources import validate
from dependency_identity_v import compiled_dependencies,verify_dependency_binding

def query(args,root,r):
 if platform.system()!='Linux' or platform.machine() not in ('aarch64','arm64'):raise ValueError('Linux ARM64 Host receiver required')
 old=p.host_o;sdk=old.bind(args.cann_root);
 contract=read(HERE.parent/'stage4v/sdk-contract.json');sdk['full_k_contract']=identity(HERE.parent/'stage4v/sdk-contract.json');sdk['full_k_files']=[]
 for row in contract['sources']:
  f=identity(args.cann_root/'aarch64-linux/asc'/row['relative_path'])
  if args.cann_root not in Path(f['path']).parents or f['sha256']!=row['sha256']:raise ValueError('Full-K SDK contract identity differs '+row['id'])
  sdk['full_k_files'].append(f)
 sdk['full_k_source_status']='SOURCE_VERIFIED_FOR_SELECTED_SDK_IDENTITIES_ONLY'
 r['sdk']=sdk;write(root/'sdk-binding.json',sdk)
 env=q.common.env(args.cann_root);env.pop('ASCEND_CUSTOM_OPP_PATH',None)
 tools={n:shutil.which(n) for n in ('cmake','c++','ldd')}
 if not all(tools.values()):raise ValueError('Host tools absent')
 sdk['tools']={n:identity(f) for n,f in tools.items()}
 r['commands']={};r['host_compile']='RUNNING'
 for name,argv in [('configure',[tools['cmake'],'-S',HERE/'host','-B',root/'host-build','-DCANN_ROOT='+str(args.cann_root),'-DCMAKE_CXX_COMPILER='+tools['c++'],'-DCMAKE_EXPORT_COMPILE_COMMANDS=ON']),('compile',[tools['cmake'],'--build',root/'host-build','--parallel'])]:
  r['commands'][name]=old.command(argv,root/(name+'.log'),env,300);write(root/'report.json',r)
  if r['commands'][name]['returncode']!=0:raise ValueError('Host '+name+' failed')
 exe=root/'host-build/stage4v_host';r['host_compile']='COMPLETED';sdk['executable']=identity(exe)
 sdk['linked_libraries']=old.linked_libraries(root/'host-build/CMakeCache.txt',args.cann_root)
 sdk['dependency_binding']=compiled_dependencies(root/'host-build',args.cann_root,HERE/'host/probe.cpp',sdk['files'][0]['file'])
 for name,argv in [('ldd',[tools['ldd'],exe]),('query',[exe,args.device,root])]:
  r['commands'][name]=old.command(argv,root/(name+'.log'),env,300)
  if r['commands'][name]['returncode']!=0 or name=='ldd' and 'not found' in (root/'ldd.log').read_text():raise ValueError('Host '+name+' failed')
 sdk['loaded_libraries']=[identity(x) for x in re.findall(r'=>\s+(/\S+)\s+\(', (root/'ldd.log').read_text()) if args.cann_root in Path(x).resolve().parents]
 r['host_query']=identity(root/'host-query.json');raw=read(verify(r['host_query']))
 if raw['device']!=args.device or len(raw['cases'])!=1 or raw['cases'][0]['name']!='FULL_K_N256':raise ValueError('Host query scope mismatch')
 resource=validate(raw,raw['cases'][0]);r['resources']=resource
 write(root/'resource-report.json',resource);write(root/'sdk-binding.json',sdk)
 if resource['status']!='HOST_RESOURCE_CHECKS_PASS':raise ValueError('Full-K resource blockers: '+','.join(resource['blockers']))
 policy=r['evidence']['formal_core_policy']
 if not p.layout_i.core_consistency(raw,policy):raise ValueError('Manager bounds inconsistent with formal policy')
 f=raw['cases'][0]['fields']
 t1=dict(fields={k:f[k] for k in p.FIELDS},capacity=raw['capacity'],system_workspace_bytes=raw['system_workspace_bytes'],scratch=202240)
 system=t1['system_workspace_bytes'];total=system+policy['selected']*((system+511)//512*512+1024)
 if not 0<total<=512*1024*1024:raise ValueError('formal Workspace upper bound')
 r['tiling_binary']=identity(root/'FULL_K_N256-tiling.bin');r['t1']=t1
 r['core_policy_note']='Manager only; formal TilingContext must be rechecked by new Op before launch'
 verify_dependency_binding(sdk['dependency_binding']);write(root/'tiling-report.json',r)
 return t1
