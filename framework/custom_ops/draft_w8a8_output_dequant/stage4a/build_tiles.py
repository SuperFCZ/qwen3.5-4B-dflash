"""Private Local NZ OPP build only; no earlier-stage builds or installs."""
import json,platform,re,shutil,sys
from pathlib import Path
from tile_common import HERE,NAME,SNAKE,common,identity,verify,write
from run_stage2b import command
QUANT=common.QUANT
env=common.env

def core_header_binding(project,cann,root):
    # Compiler-produced Host dependencies only; never scan the installed SDK tree.
    headers={};deps=[];notes=[];paths=sorted((project/'build_out/op_host').rglob('*.o.d'))
    for dep in paths[:32]:
        if dep.stat().st_size>4*1024*1024:notes.append('oversized dependency file: '+str(dep));continue
        deps.append(identity(dep))
        for name in re.findall(r'(/[^\s\\]*platform_ascendc\.h)',dep.read_text(errors='replace').replace('\\\n',' ')):
            p=Path(name).resolve()
            if cann.resolve() not in p.parents or p.stat().st_size>1024*1024:notes.append('unaccepted platform header: '+name);continue
            item=identity(p);lines=p.read_text().splitlines();verify(item)
            matches=[dict(line=i,text=line) for i,line in enumerate(lines,1) if any(s in line for s in ('GetCoreNum','GetCurNpuArch','GetSocVersion'))]
            headers[str(p)]=dict(source=item,api_lines=matches)
    if len(paths)>32:notes.append('dependency file count capped at 32')
    result=dict(status='INSTALLED_HEADER_BOUND' if headers else 'HEADER_NOT_RESOLVED_FROM_DEPENDENCIES',
        headers=list(headers.values()),dependencies=deps,notes=notes,
        limitation='Declarations and runtime getter results do not explain the 8/8/7 difference. Out-of-line implementation semantics are not fabricated.')
    write(root/'core-sdk-binding.json',result);return result

def registration_evidence(project,primary):
    # Successful compiler diagnostics may be kept in nested build logs instead
    # of build.sh stdout. Search only this generated project's logs, bounded.
    marker='STAGE4A_REGIST_EXPANSION='
    candidates=[primary]+sorted((project/'build_out').rglob('*.log'))
    scanned=[];skipped=[];expanded=[];contexts=[];seen=set();remaining=32*1024*1024
    for path in candidates:
        real=path.resolve()
        if real in seen:continue
        seen.add(real)
        if real!=primary.resolve() and project.resolve() not in real.parents:
            skipped.append(dict(path=str(path),reason='OUTSIDE_GENERATED_PROJECT'));continue
        if len(scanned)>=64:
            skipped.append(dict(path=str(path),reason='FILE_COUNT_LIMIT'));continue
        try:
            size=real.stat().st_size
            if size>8*1024*1024 or size>remaining:
                skipped.append(dict(path=str(path),bytes=size,reason='BYTE_LIMIT'));continue
            item=identity(real);text=real.read_text(errors='replace');verify(item)
            scanned.append(item);remaining-=size
            for i,line in enumerate(text.splitlines(),1):
                if marker not in line:continue
                payload=line.split(marker,1)[1].strip()
                row=dict(log=item,line=i,text=line)
                if not payload or re.search(r'#\s*pragma\s+message\b',line) or 'S4A_TEXT' in payload or 'REGIST_MATMUL_OBJ(' in payload:contexts.append(row)
                else:expanded.append(row)
        except OSError as error:skipped.append(dict(path=str(path),reason=f'{type(error).__name__}: {error}'))
    return dict(status='COMPILER_EXCERPTS_COLLECTED' if expanded else ('EXPANSION_SEARCH_INCOMPLETE' if skipped else 'EXPANSION_NOT_EMITTED'),
        evidence=expanded,source_context=contexts,log=identity(primary),scanned_logs=scanned,skipped_logs=skipped,
        interpretation='Bounded log evidence only. A pragma source echo is not an expansion. Absent diagnostics do not prove a wrong Workspace pointer.',
        limitation='Compiler warnings may be suppressed or captured outside these logs; installed macro behavior remains unresolved without emitted evidence.',
        runtime_reservation='unchanged; full platform arena per logical core plus framework system space')

def installed(opp):
    sys.path.insert(0,str(QUANT/'test'))
    from opp_preflight import check_root,check_binary_path
    from compile_raw import elf_info
    check_root(opp)
    directory=opp/'vendors/customize/op_impl/ai_core/tbe/kernel/ascend310p'/SNAKE
    paths=sorted(directory.glob('*.o'))
    if not paths:raise ValueError('missing Stage4A installed kernel')
    records=[]
    for p in paths:
        meta=check_binary_path(p)
        if opp.resolve() not in p.resolve().parents or opp.resolve() not in meta.resolve().parents:raise ValueError('installed kernel escaped private OPP')
        binary,elf=elf_info(p,False)
        if not isinstance(json.loads(meta.read_text()),dict):raise ValueError('invalid installed kernel JSON')
        records.append(dict(binary=binary,metadata=identity(meta),elf=elf))
    header=opp/'vendors/customize/op_api/include'/('aclnn_'+SNAKE+'.h')
    library=opp/'vendors/customize/op_api/lib/libcust_opapi.so'
    alias=opp/'vendors/customize/op_impl/ai_core/tbe/op_tiling/liboptiling.so'
    for p in (header,library,alias):
        if not p.is_file() or not p.stat().st_size or opp.resolve() not in p.resolve().parents:raise ValueError('missing/escaped installed API or runtime Host library: '+str(p))
    host,info=elf_info(alias,True)
    # Binding the actual installed package, not equating it to an uninstalled build .so.
    return dict(kernels=records,header=identity(header),api=identity(library),host=host,host_elf=info,host_alias=str(alias),
                all_files=[identity(p) for p in sorted(opp.rglob('*')) if p.is_file()])


def build(args,root,r):
    if sys.platform!='linux' or platform.machine() not in ('aarch64','arm64'):raise ValueError('CANN Linux aarch64 receiver required')
    sys.path.insert(0,str(QUANT/'test'))
    from opp_preflight import check_root
    check_root(root)
    pf=json.loads(verify(r['evidence']['preflight']).read_text())
    for name in ('msopgen','cmake','c++'):
        tool=shutil.which(name)
        if name not in pf['tools'] or not tool or identity(tool)!=pf['tools'][name]['file']:raise ValueError('tool differs from Stage1 environment: '+name)
    r['tools']=pf['tools'];e=env(args.cann_root);project=root/'CustomOp'
    r.update(phase='msopgen',cann_compile='RUNNING');write(root/'report.json',r)
    definition=root/'operator.json';shutil.copyfile(HERE/'operator.json',definition)
    command(['msopgen','gen','-i',definition,'-c','ai_core-Ascend310P3','-lan','cpp','-out',project,'-op',NAME],root/'msopgen.log',e,r,'msopgen')
    bound=[]
    for sub in ('op_host','op_kernel'):
        wanted=set()
        for p in (HERE/sub).iterdir():
            if p.is_file():shutil.copyfile(p,project/sub/p.name);wanted.add(p.name);bound.append(identity(project/sub/p.name))
        actual={p.name for p in (project/sub).iterdir() if p.suffix in ('.cpp','.h','.hpp','.cc')}
        if actual!=wanted:raise ValueError('unexpected generated source inventory')
    r['generated_sources']=bound;r['phase']='cann_compile';write(root/'report.json',r)
    command(['bash','-c','cd "$1" && bash build.sh','stage4a',project],root/'build.log',e,r,'build',3600)
    r['cann_compile']='BUILD_RETURNED_ZERO'
    r['core_sdk_binding']=core_header_binding(project,args.cann_root,root)
    r['workspace_registration']=registration_evidence(project,root/'build.log')
    for f in bound:verify(f)
    packages=list((project/'build_out').glob('custom_opp_*.run'))
    if len(packages)!=1:raise ValueError('missing/ambiguous Stage4A package')
    r['package']=identity(packages[0]);opp=root/'opp';r['phase']='isolated_install';r['installation']='RUNNING';write(root/'report.json',r)
    command(['bash',packages[0],'--install-path='+str(opp)],root/'install.log',e,r,'install')
    r['phase']='installed_opp_preflight';r['installed']=installed(opp);r['installation']='ISOLATED_OPP_VERIFIED'
    command(['ldd',r['installed']['host']['path']],root/'host-ldd.log',env(args.cann_root,opp),r,'host_ldd')
    r['core_sdk_binding']['host_library']=r['installed']['host']
    r['core_sdk_binding']['host_dependencies_log']=identity(root/'host-ldd.log')
    dependencies=[]
    for name in re.findall(r'=>\s+(/\S+)\s+\(', (root/'host-ldd.log').read_text()):
        if args.cann_root.resolve() in Path(name).resolve().parents:dependencies.append(identity(name))
    r['core_sdk_binding']['sdk_libraries']=dependencies
    write(root/'core-sdk-binding.json',r['core_sdk_binding'])
    r['phase']='runner_build';write(root/'report.json',r)
    command(['cmake','-S',HERE/'test','-B',root/'runner-build','-DCANN_ROOT='+str(args.cann_root),
        '-DOPP_ROOT='+str(opp),'-DCOMMON_INCLUDE='+str(QUANT/'test'),'-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],root/'cmake.log',e,r,'runner_configure')
    command(['cmake','--build',root/'runner-build','--parallel'],root/'runner-build.log',e,r,'runner_build')
    runner=root/'runner-build/stage4a_runner'
    command(['ldd',runner],root/'ldd.log',env(args.cann_root,opp),r,'ldd')
    if 'not found' in (root/'ldd.log').read_text():raise ValueError('unresolved runtime library')
    links={}
    for line in (root/'runner-build/CMakeCache.txt').read_text().splitlines():
        if line.startswith('LIB_') and ':FILEPATH=' in line:
            key,value=line.split(':FILEPATH=',1);item=identity(value)
            expected=opp if key=='LIB_custom' else args.cann_root
            if expected.resolve() not in Path(item['path']).parents:raise ValueError('Runner linked wrong library: '+key)
            links[key]=item
    if set(links)!={'LIB_custom','LIB_ascendcl','LIB_nnopbase','LIB_acl_op_compiler'}:raise ValueError('missing Runner library identity')
    r.update(status='STAGE4A_BUILT_NPU_NOT_RUN',phase='complete',cann_compile='COMPLETED',opp=str(opp),
             runner=identity(runner),linked_libraries=links,compile_commands=identity(root/'runner-build/compile_commands.json'))
