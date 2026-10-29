#!/usr/bin/env python3
"""Host-only N64/N128 Tiling and CPU layout experiment. Never builds/launches a compute Kernel."""
import argparse,json,os,platform,re,shutil,subprocess
from pathlib import Path
from layout import HERE,old,identity,verify,verify_tree,real
from resources import CONFIGS,validate,decide
from dependency_identity import compiled_dependencies,verify_dependency_binding
write,fresh=old.write,old.fresh

SDK={
 'include/tiling/platform/platform_ascendc.h':('44cf3f35292f15a8735b0d43b7e88198ca15d7b4a3bf44719eb3f099697f7f2a',[[140,141],[149,161]]),
 'aarch64-linux/asc/include/adv_api/matmul/matmul_tiling_base.h':('44dea3b20ddeea8987471f8f226cfeadbdc65140f746ee1f3bab33b8b8e3350b',[[253,266],[317,350]]),
 'aarch64-linux/asc/impl/adv_api/detail/matmul/resource/local_workspace/matmul_local_workspace.h':('dbffdce3b74c3e18414c1c5e89e4a4260ec64654eeccc58f172ed89f7ae6c08b',[[155,183],[248,317]]),
 'aarch64-linux/asc/impl/adv_api/detail/matmul/stage/copy_cube_out/quant/quant_processor_datacopy.h':('10438170dde8cb131bf3ba9443955d101d5e7936aa27dc38127677cb1b898c7c',[[113,158]]),
}

# Fixed paths verified from this server's installation metadata and Raw Preflight.
VERSION_PATHS=('aarch64-linux/ascend_toolkit_install.info','ascend_toolkit_install.info',
 'version.info','version.cfg','aarch64-linux/asc/version.info','aarch64-linux/asc/version.cfg',
 'share/info/asc-devkit/version.info','opp/version.info')

def version_fields(text,relative):
 # A dependency constraint, directory name, compatible_version or innerversion
 # is never evidence of the installed release. Only the observed key=value form.
 found=[];packages=[]
 for number,line in enumerate(text.splitlines(),1):
  match=re.fullmatch(r'\s*(version|package_name)\s*=\s*(.*?)\s*',line,re.I)
  if not match:continue
  record=dict(key=match[1],value=match[2],line=number)
  (found if match[1].lower()=='version' else packages).append(record)
 if len(found)!=1:raise ValueError('CANN 9.0.0 version unverified: missing/duplicate version field in '+relative)
 result=dict(version_field=found[0],reported_version=found[0]['value'],cann_9_0_0=found[0]['value']=='9.0.0')
 if Path(relative).name=='ascend_toolkit_install.info':
  if len(packages)!=1 or packages[0]['value']!='Ascend-cann-toolkit':raise ValueError('CANN 9.0.0 version unverified: wrong/ambiguous toolkit package in '+relative)
  result['package_field']=packages[0]
 return result


def bind(cann):
 rows=[];versions=[];missing=[]
 for rel,(sha,lines) in SDK.items():
  item=identity(cann/rel)
  if cann not in Path(item['path']).parents or item['sha256']!=sha:raise ValueError('reviewed SDK identity mismatch: '+rel)
  rows.append(dict(file=item,lines=lines))
 for rel in VERSION_PATHS:
  p=cann/rel
  if not p.is_file():
   if p.is_symlink():raise ValueError('broken version metadata link: '+rel)
   missing.append(rel);continue
  item=identity(p)
  if cann not in Path(item['path']).parents:raise ValueError('version file escapes selected CANN root: '+rel)
  if item['bytes']>65536:raise ValueError('version metadata exceeds 64 KiB: '+rel)
  text=verify(item).read_text(encoding='utf-8-sig');verify(item)
  versions.append(dict(file=item,relative=rel,**version_fields(text,rel)))
 if not versions:raise ValueError('CANN 9.0.0 version unverified: no recognized version metadata files')
 wrong=[x['relative']+'='+x['reported_version'] for x in versions if not x['cann_9_0_0']]
 if wrong:raise ValueError('CANN 9.0.0 version unverified: conflicting/unsupported release: '+', '.join(wrong))
 return dict(status='FIXED_SDK_IDENTITIES_MATCHED',files=rows,versions=versions,version_missing=missing,
  version_policy='Exact installed release field; all discovered declarations must agree with 9.0.0; no directory/tool-error fallback',
  source_snapshots_saved=False,platform_route='GetInstance() only; selected real device; no customSocVersion')


def command(argv,path,env,timeout=180):
 argv=list(map(str,argv));write(str(path)+'.command.json',dict(argv=argv))
 with path.open('wb') as out:
  try:
   result=subprocess.run(argv,stdout=out,stderr=subprocess.STDOUT,env=env,timeout=timeout)
   rc=result.returncode;status='COMPLETED' if rc==0 else 'FAILED'
  except subprocess.TimeoutExpired:
   rc=None;status='TIMEOUT'
 record=dict(argv=argv,returncode=rc,status=status,log=identity(path))
 return record


def table(rows,resource):
 out=['| Metric | N64 | N128 T1 | N128 T2 |','|---|---|---|---|']
 indexed={x['name']:x for x in rows};checks={x['name']:x for x in resource}
 def cell(name,key):
  r=indexed.get(name,{});p=r.get('fields') or {};d=checks.get(name,{}).get('details',{})
  if key=='GetTiling':return str(r.get('get_tiling_rc','NOT_RUN'))
  if key=='singleCore M/N/K':return '/'.join(str(p.get('singleCore'+k,'UNKNOWN')) for k in 'MNK')
  if key=='base M/N/K':return '/'.join(str(p.get('base'+k,'UNKNOWN')) for k in 'MNK')
  if key=='L1 depth':return str([p.get('depthA1'),p.get('depthB1')])
  if key=='L0 db':return str([p.get('dbL0A'),p.get('dbL0B'),p.get('dbL0C')])
  if key=='Resource status':return checks.get(name,{}).get('status','NOT_RUN')
  return str({'Scratch requested':r.get('scratch','NOT_RUN'),'transLength':p.get('transLength','UNKNOWN'),
   'Scale storage bound':d.get('scale_storage_bound','NOT_EVALUATED'),'UB budget total':d.get('total_ub','NOT_EVALUATED')}.get(key))
 for key in ('GetTiling','singleCore M/N/K','base M/N/K','Scratch requested','transLength','Scale storage bound','L1 depth','L0 db','UB budget total','Resource status'):
  out.append('| '+key+' | '+' | '.join(cell(n,key) for n in CONFIGS)+' |')
 return '\n'.join(out)+'\n\nRequested budgets and calculated bounds are not measurements of live resource usage. No Kernel was executed.\n'


def linked_libraries(cache,cann):
 linked=[]
 for line in cache.read_text().splitlines():
  if line.startswith('LIB_') and ':FILEPATH=' in line:
   key,value=line.split(':FILEPATH=',1);item=identity(value)
   if cann not in Path(item['path']).parents:raise ValueError('linked library outside selected CANN: '+key)
   linked.append(dict(name=key,file=item))
 if {x['name'] for x in linked}!={'LIB_tiling_api','LIB_register','LIB_c_sec','LIB_platform','LIB_ascendcl'}:raise ValueError('unexpected Host libraries')
 return linked


def main(args):
 historical_root=args.stage4b_report.parent.parent if args.stage4b_report.parent.name=='smoke' else args.stage4b_report.parent
 root=fresh(args.output_dir,[args.cann_root,historical_root])
 sources=[identity(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
 base=dict(npu_kernel_execution='NOT_RUN',benchmark='NOT_RUN',profile='NOT_RUN',n128_kernel_compile='NOT_RUN',opp_installation='NOT_RUN')
 tr=dict(status='NOT_RUN',phase='layout',sources=sources,host_compile='NOT_RUN',commands={},**base)
 lr=dict(status='NOT_RUN',**base);rr=dict(status='NOT_RUN',cases=[],**base);sdk=dict(status='NOT_RUN')
 rows=[]
 try:
  lr=real(args.stage4b_report);lr.update(base);write(root/'layout-report.json',lr)
  if platform.system()!='Linux' or platform.machine() not in ('aarch64','arm64'):raise ValueError('real Linux aarch64 CANN Host-only receiver required')
  sdk=bind(args.cann_root);write(root/'sdk-binding.json',sdk)
  tr['phase']='host_compile';tr['host_compile']='RUNNING';tools={n:shutil.which(n) for n in ('cmake','c++','ldd')}
  if not all(tools.values()):raise ValueError('Host build tools missing')
  sdk['tools']={k:identity(v) for k,v in tools.items()}
  env=dict(os.environ);env.pop('ASCEND_CUSTOM_OPP_PATH',None)
  env['ASCEND_HOME_PATH']=str(args.cann_root);env['ASCEND_INSTALL_PATH']=str(args.cann_root)
  libs=[str(args.cann_root/'lib64'),str(args.cann_root/'aarch64-linux/lib64')]
  inherited=[p for p in env.get('LD_LIBRARY_PATH','').split(':') if p and '/vendors/' not in p and '/opp/' not in p]
  env['LD_LIBRARY_PATH']=':'.join(libs+inherited)
  for key,argv in [('configure',[tools['cmake'],'-S',HERE,'-B',root/'host-build','-DCANN_ROOT='+str(args.cann_root),'-DCMAKE_CXX_COMPILER='+tools['c++'],'-DCMAKE_EXPORT_COMPILE_COMMANDS=ON']),
                   ('compile',[tools['cmake'],'--build',root/'host-build','--parallel'])]:
   tr['commands'][key]=command(argv,root/(key+'.log'),env,300);write(root/'tiling-report.json',tr)
   if tr['commands'][key]['returncode']!=0:raise ValueError('Host-only '+key+' failed; no fallback')
  tr['host_compile']='COMPLETED';exe=root/'host-build/stage4i_host';sdk['executable']=identity(exe)
  sdk['compile_commands']=identity(root/'host-build/compile_commands.json')
  sdk['linked_libraries']=linked_libraries(root/'host-build/CMakeCache.txt',args.cann_root)
  sdk['dependency_binding']=compiled_dependencies(root/'host-build',args.cann_root,HERE/'probe.cpp',sdk['files'][0]['file'])
  sdk['compiled_sdk_headers']=sdk['dependency_binding']['headers']
  tr['commands']['ldd']=command([tools['ldd'],exe],root/'ldd.log',env)
  if tr['commands']['ldd']['returncode']!=0 or 'not found' in (root/'ldd.log').read_text():raise ValueError('Host dependency unresolved')
  sdk['loaded_sdk_libraries']=[identity(p) for p in re.findall(r'=>\s+(/\S+)\s+\(', (root/'ldd.log').read_text()) if args.cann_root in Path(p).resolve().parents]
  tr['phase']='host_query';tr['commands']['query']=command([exe,args.device,root],root/'host-query.log',env)
  if tr['commands']['query']['returncode']!=0:raise ValueError('Host query failed; see host-query.log')
  tr['host_query_file']=identity(root/'host-query.json')
  result=json.loads(verify(tr['host_query_file']).read_text());rows=result['cases']
  if result['device']!=args.device or [x['name'] for x in rows]!=list(CONFIGS):raise ValueError('Host query scope differs')
  for row in rows:
   if row.get('fields') is not None:
    row['tiling_binary']=identity(root/(row['name']+'-tiling.bin'))
    if row['tiling_binary']['bytes']!=row['serialized_bytes']:raise ValueError('serialized Tiling size differs')
  tr['actual_host_query']=result
  rr=dict(status='RESOURCE_CHECKS_COMPLETED',cases=[validate(result,x) for x in rows],**base)
  tr.update(decide(rr['cases'],lr['status']=='CPU_FROZEN_LAYOUT_MATCH'),phase='complete')
 except Exception as error:
  tr.update(status='STAGE4I_N128_BLOCKED',error=f'{type(error).__name__}: {error}')
  if tr['host_compile']=='RUNNING':tr['host_compile']='FAILED_OR_INCOMPLETE'
 finally:
  try:
   for s in sources:verify(s)
   verify_tree(lr);verify_tree(sdk);verify_tree(tr)
   if 'dependency_binding' in sdk:verify_dependency_binding(sdk['dependency_binding'])
  except Exception as error:tr.update(status='STAGE4I_N128_BLOCKED',integrity_error=str(error))
  write(root/'layout-report.json',lr);write(root/'resource-report.json',rr);write(root/'sdk-binding.json',sdk);write(root/'tiling-report.json',tr)
  (root/'comparison.md').write_text(table(rows,rr['cases']))
 print(tr['status']+': '+str(root/'tiling-report.json'))
 return int(tr['status']!='STAGE4I_N128_HOST_TILING_FEASIBLE')

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--device',type=int,default=0)
 p.add_argument('--cann-root',type=Path,default=Path('/usr/local/Ascend/cann-9.0.0'))
 p.add_argument('--stage4b-report',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-full-gate-20261027-093617/smoke/report.json'))
 args=p.parse_args();args.cann_root=args.cann_root.resolve()
 if args.device<0:p.error('negative device')
 raise SystemExit(main(args))
