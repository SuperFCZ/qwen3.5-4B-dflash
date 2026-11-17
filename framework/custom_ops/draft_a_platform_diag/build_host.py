"""Clone a frozen private OPP; rebuild CPU Host DSOs only; preserve all Kernel bytes."""
from pathlib import Path
import os
import shutil
from platform_common import c
from derive import audit_host

HOST_LIBS=('libcust_opmaster_rt2.0.so','libcust_opsproto_rt2.0.so')


def tree(root):
    root=Path(root).resolve();result={}
    for p in sorted(root.rglob('*')):
        if not p.is_file() and not p.is_symlink():continue
        target=p.resolve(strict=True)
        if root not in target.parents:raise ValueError('private tree symlink escapes: '+str(p))
        if p.is_symlink() and Path(os.readlink(p)).is_absolute():raise ValueError('absolute private alias cannot be cloned: '+str(p))
        if not target.is_file():raise ValueError('unsupported private directory symlink: '+str(p))
        result[str(p.relative_to(root))]=dict(file=c.identity(target),alias=os.readlink(p) if p.is_symlink() else None,
            resolved_relative=str(target.relative_to(root)))
    return result


def compare_install(old,new,allowed):
    if set(old)!=set(new):raise ValueError('private OPP file inventory changed')
    changed=[]
    for rel,a in old.items():
        b=new[rel]
        if a['alias']!=b['alias'] or a['resolved_relative']!=b['resolved_relative']:raise ValueError('private OPP alias changed: '+rel)
        equal=all(a['file'][k]==b['file'][k] for k in ('bytes','sha256'))
        if not equal:
            if a['resolved_relative'] not in allowed:raise ValueError('unexpected OPP content change: '+rel)
            changed.append(rel)
    kernels=[p for p in old if '/kernel/ascend310p/' in p and p.endswith('.o')]
    if not kernels:raise ValueError('no original Kernel binaries')
    if not changed:raise ValueError('Host logging change absent from built libraries')
    return dict(changed_files=changed,kernels=[dict(relative_path=p,bytes=old[p]['file']['bytes'],sha256=old[p]['file']['sha256'],exact_equal=True) for p in kernels],
        all_kernel_bytes_equal=True,all_non_host_files_equal=True)


def preset_options(project,cann):
    p=c.read(project/'CMakePresets.json');matches=[x for x in p['configurePresets'] if x['name']=='default']
    if len(matches)!=1 or matches[0].get('inherits'):raise ValueError('unsupported generated preset')
    preset=matches[0];values={}
    for k,v in preset['cacheVariables'].items():
        raw=v['value'] if isinstance(v,dict) else v
        value=str(raw).replace('${sourceDir}',str(project))
        if '${' in value:raise ValueError('unresolved generated CMake variable: '+k)
        values[k]=value
    if (Path(values['ASCEND_CANN_PACKAGE_PATH']).resolve()!=cann or values['ASCEND_COMPUTE_UNIT']!='ascend310p' or
        values['ENABLE_CROSS_COMPILE'].lower() not in ('false','off','0') or values['ASCEND_PACK_SHARED_LIBRARY'].lower() not in ('false','off','0')):
        raise ValueError('Host-only private package configuration differs')
    return ['-G',preset['generator']]+['-D'+k+'='+v for k,v in values.items()]+['-DCMAKE_EXPORT_COMPILE_COMMANDS=ON']


def build(root,prior,variant='B'):
    root=Path(root);base=prior['reports']['build']['variants'][variant];old_opp=base['opp']
    c.verify_tree(base);c.verify_opp(old_opp);audit=audit_host()
    old_project=c.verify(base['build_report']).parent/'CustomOp'
    if c.sha(old_project/'op_host/d_flash_group_quant_linear.cpp')!=audit['original']['sha256']:raise ValueError('accepted generated Host source differs')
    project=root/'CustomOp'
    shutil.copytree(old_project,project,symlinks=True,ignore=shutil.ignore_patterns('build_out','host_native_tiling','__pycache__','.git','.pytest_cache'))
    before=tree(project)
    for row in before.values():
        original=c.identity(old_project/row['resolved_relative'])
        if any(original[k]!=row['file'][k] for k in ('bytes','sha256')):raise ValueError('generated source changed during copy')
        row['file']=original  # Old identities must continue to point at immutable old files.
    (project/'op_host/d_flash_group_quant_linear.cpp').write_bytes(c.verify(audit['candidate']).read_bytes())
    (project/'op_host/platform_diag.h').write_bytes(c.verify(audit['writer']).read_bytes())
    cmake=project/'CMakeLists.txt';text=cmake.read_text()
    remove='if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel)\n    add_subdirectory(op_kernel)\nendif()'
    if text.count(remove)!=1:raise ValueError('generated Kernel target block differs; refusing a Kernel build')
    cmake.write_text(text.replace(remove,'# Host-only evidence rebuild: device Kernel targets deliberately absent.'))
    host_cmake=project/'op_host/CMakeLists.txt'
    with host_cmake.open('a') as f:f.write('\ntarget_link_libraries(cust_optiling PRIVATE ${CMAKE_DL_LIBS})\ntarget_link_libraries(cust_op_proto PRIVATE ${CMAKE_DL_LIBS})\n')
    after=tree(project)
    changed={p for p in before if before[p]['file']['sha256']!=after[p]['file']['sha256']}
    if changed!={'CMakeLists.txt','op_host/CMakeLists.txt','op_host/d_flash_group_quant_linear.cpp'} or set(after)-set(before)!={'op_host/platform_diag.h'}:
        raise ValueError('generated project changed beyond approved Host logging/build scope')
    out=root/'host-build';install=root/'opp';cann=Path(prior['current']['cann_root'])
    env=c.selected_env(prior['current']['evidence'],install);env.pop('DFLASH_GROUP_QUANT_EVIDENCE_PATH',None)
    c.command(['cmake','-S',project,'-B',out,*preset_options(project,cann)],root/'configure.log',env)
    c.command(['cmake','--build',out,'--target','cust_optiling','cust_op_proto','--parallel','2'],root/'host-build.log',env)
    commands=c.read(out/'compile_commands.json')
    if any('/op_kernel/' in x['file'] or 'ccec' in x.get('command','') for x in commands):raise ValueError('device compilation entered Host-only build')
    old_root=Path(old_opp['install']);old_tree=tree(old_root)
    shutil.copytree(old_root,install,symlinks=True)
    allowed=[];libraries=[]
    for name in HOST_LIBS:
        generated={p.resolve() for p in out.rglob(name) if p.is_file()}
        targets={old_root/row['resolved_relative'] for row in old_tree.values() if Path(row['resolved_relative']).name==name}
        if len(generated)!=1 or len(targets)!=1:raise ValueError('ambiguous Host DSO artifact: '+name)
        source=next(iter(generated));original=next(iter(targets));relative=str(original.relative_to(old_root));target=install/relative
        target.unlink();shutil.copy2(source,target);allowed.append(relative)
        old_deps=c.dependencies(c.identity(original),root/(name+'-old-ldd.log'),env)
        new_deps=c.dependencies(c.identity(target),root/(name+'-new-ldd.log'),env)
        if old_deps!=new_deps:raise ValueError('Host runtime dependency identities changed: '+name)
        libraries.append(dict(old=c.identity(original),new=c.identity(target),dependencies=new_deps))
    installed=tree(install);comparison=compare_install(old_tree,installed,set(allowed))
    from opp import preflight
    new_opp=dict(install=str(install),preflight=preflight(install),kind='HOST_EVIDENCE_ONLY_DERIVED_OPP')
    c.verify_opp(new_opp);c.verify_opp(old_opp);c.verify_tree(base)
    return dict(status='PLATFORM_DIAG_HOST_DSOS_BUILT_NPU_NOT_RUN',variant=variant,opp=new_opp,base_opp=old_opp,
        project_audit=audit,project_before=before,project_after=after,library_transition=libraries,opp_comparison=comparison,
        compile_commands=c.identity(out/'compile_commands.json'),cmake_cache=c.identity(out/'CMakeCache.txt'),
        kernel_compile='NOT_RUN_FROZEN_BINARY_COPY',old_probes='OLD_OPP_ONLY_NOT_TRANSFERRED',new_probes='NOT_RUN',
        required_before_new_npu_acceptance='This package is Host-diagnostic-only. No new numerical acceptance is granted; no 104-probe execution is scheduled.')
