"""Build byte-audited private A/B projects; no historical OPP is installed or edited."""
import json,platform,re,shutil,sys
from pathlib import Path
from pair_common import HERE,NAME,SNAKE,common,identity,verify,write
from build_full import core_header_binding,registration_evidence
from run_stage2b import command
QUANT=common.QUANT
env=common.env

def installed(opp):
    sys.path.insert(0,str(QUANT/'test'))
    from opp_preflight import check_root,check_binary_path
    from compile_raw import elf_info
    check_root(opp)
    directory=opp/'vendors/customize/op_impl/ai_core/tbe/kernel/ascend310p'/SNAKE
    paths=sorted(directory.glob('*.o'))
    if not paths:raise ValueError('missing Stage4B installed kernel')
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


def build_variant(args,root,r,source_dir):
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
    definition=root/'operator.json';shutil.copyfile(source_dir/'operator.json',definition)
    command(['msopgen','gen','-i',definition,'-c','ai_core-Ascend310P3','-lan','cpp','-out',project,'-op',NAME],root/'msopgen.log',e,r,'msopgen')
    bound=[]
    for sub in ('op_host','op_kernel'):
        wanted=set()
        for p in (source_dir/sub).iterdir():
            if p.is_file():shutil.copyfile(p,project/sub/p.name);wanted.add(p.name);bound.append(identity(project/sub/p.name))
        actual={p.name for p in (project/sub).iterdir() if p.suffix in ('.cpp','.h','.hpp','.cc')}
        if actual!=wanted:raise ValueError('unexpected generated source inventory')
    r['generated_sources']=bound;r['phase']='cann_compile';write(root/'report.json',r)
    command(['bash','-c','cd "$1" && bash build.sh','stage4d',project],root/'build.log',e,r,'build',3600)
    r['cann_compile']='BUILD_RETURNED_ZERO'
    r['core_sdk_binding']=core_header_binding(project,args.cann_root,root)
    r['workspace_registration']=registration_evidence(project,root/'build.log')
    for f in bound:verify(f)
    packages=list((project/'build_out').glob('custom_opp_*.run'))
    if len(packages)!=1:raise ValueError('missing/ambiguous Stage4B package')
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
    runner=root/'runner-build/stage4d_runner'
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
    r.update(status='PRIVATE_VARIANT_BUILT_NPU_NOT_RUN',phase='complete',cann_compile='COMPLETED',opp=str(opp),
             runner=identity(runner),linked_libraries=links,compile_commands=identity(root/'runner-build/compile_commands.json'))
