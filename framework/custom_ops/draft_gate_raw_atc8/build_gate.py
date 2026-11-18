"""Build only the Gate device target per variant. Preserve the accepted control byte-for-byte."""
import argparse
import os
from pathlib import Path
import shutil
import sys
from gate_common import (HERE,HEADER,STEM,GATE,OFF_ELF,OFF_HEADER,RAW_HEADER,KERNEL_SOURCE,API,c,command,
                         derive_header,gate_kernel,host,one,preset_options,require,tree,check_root)
from flow_build import HOST_LIBS
from opp import preflight
from gate_plan import elf_sections,sdk_layout

DEVICE_CMAKE='if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel)\n    add_subdirectory(op_kernel)\nendif()'
DISABLED='# Host-only evidence rebuild: device Kernel targets deliberately absent.'


def copy_snapshot(snapshot,original,dest):
    """Materialize only the admitted generated files, never copy a stale build cache."""
    dest.mkdir()
    for rel,row in snapshot.items():
        name=Path(rel);resolved=Path(row['resolved_relative'])
        require(not name.is_absolute() and not resolved.is_absolute() and '..' not in name.parts+resolved.parts,'unconfined generated project')
        p=original/name
        require(p.resolve(strict=True)==original/resolved==c.verify(row['file']),'source alias retargeted: '+rel)
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
        if p.is_symlink():
            require(os.readlink(p)==row['alias'] and not Path(row['alias']).is_absolute(),'invalid project alias')
            target.symlink_to(row['alias'])
        else:
            require(row['alias'] is None,'project alias removed')
            shutil.copy2(p,target)
    compare_tree(snapshot,tree(dest),set())


def compare_tree(before,after,allowed):
    require(before.keys()==after.keys(),'OPP/source inventory changed')
    changed=set()
    for rel,old in before.items():
        new=after[rel]
        require((old['alias'],old['resolved_relative'])==(new['alias'],new['resolved_relative']),'alias changed: '+rel)
        if old['file']['sha256']!=new['file']['sha256']:
            require(old['resolved_relative'] in allowed,'unexpected file change: '+rel)
            changed.add(old['resolved_relative'])
    return changed


def compile_device(project,output,cann,env,mode):
    require(not output.exists(),'nonempty compiler output')
    require((project/'op_host/scheduler_variant.h').read_text()=='#pragma once\n#define DFLASH_SCHEDULER_POLICY 0\n','not scheduler A')
    require(c.sha(project/'op_kernel'/(STEM+'.cpp'))==KERNEL_SOURCE,'device source drift')
    for side in ('op_host','op_kernel'):
        require(c.sha(project/side/HEADER)=={'off':OFF_HEADER,'raw':RAW_HEADER}[mode],'not the reviewed generated header')
    require(DEVICE_CMAKE in (project/'CMakeLists.txt').read_text(),'no genuine device target')
    command(['cmake','-S',project,'-B',output,*preset_options(project,cann),'-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],
            output.parent/(mode+'-configure.log'),env)
    command(['cmake','--build',output,'--target','cust_optiling','cust_op_proto',GATE+'_ascend310p','--','-j2','VERBOSE=1'],
            output.parent/(mode+'-compile.log'),env)
    base=output/'op_kernel/binary/ascend310p'
    executable=one(base.rglob('*.o'),'need exactly one newly compiled device ELF')
    require(executable.parent.name==STEM and executable.read_bytes().startswith(b'\x7fELF'),'not a newly generated Gate ELF')
    # Same two SDK postprocessors as the generated aggregate binary target, but scoped to Gate.
    util=project/'cmake/util'
    command([sys.executable,util/'insert_simplified_keys.py','-p',base],output.parent/(mode+'-keys.log'),env)
    command([sys.executable,util/'ascendc_ops_config.py','-p',base,'-s','ascend310p'],output.parent/(mode+'-index.log'),env)
    metadata=executable.with_suffix('.json');info=c.read(metadata)
    require(info['sha256']==c.sha(executable) and info['binFileName']==executable.stem,'generated SDK object seal differs')
    recipes=[c.identity(x) for x in sorted(output.rglob('*')) if x.is_file() and (
             x.name=='compile_commands.json' or x.suffix=='.sh' or (x.name=='build.make' and GATE in str(x.parent)))]
    require(any(x['path'].endswith('build.make') for x in recipes),'device compile recipe missing')
    tools={}
    for tool in ('cmake','c++','make','python3','opc','bisheng'):
        selected=shutil.which(tool,path=env['PATH'])
        require(selected,'missing compiler tool: '+tool)
        if tool in ('opc','bisheng'):require(cann.resolve() in Path(selected).resolve().parents,'compiler outside selected SDK: '+tool)
        tools[tool]=dict(alias=selected,file=c.identity(selected))
    return dict(binary=c.identity(executable),metadata=c.identity(metadata),recipes=recipes,command_path_tools=tools,
                config=c.identity(base/(STEM+'.json')),index=c.identity(base/'binary_info_config.json'))


def install_raw_outputs(old,install,build,before,kernel,compiler,folder,env):
    kernel_rel=Path(kernel['elf']['path']).relative_to(old)
    metadata_rel=Path(kernel['metadata']['path']).relative_to(old)
    require(kernel_rel.suffix=='.o' and metadata_rel==kernel_rel.with_suffix('.json'),'unexpected Gate object/metadata targets')
    outputs={kernel_rel:compiler['binary'],metadata_rel:compiler['metadata']}
    host_paths={}
    for lib in HOST_LIBS:
        source=one({p.resolve() for p in build.rglob(lib) if p.is_file()},'ambiguous Host DSO')
        host_rel=Path(one({v['resolved_relative'] for v in before.values() if Path(v['resolved_relative']).name==lib},'unknown installed Host DSO'))
        outputs[host_rel]=c.identity(source)
        host_paths[lib]=host_rel
    require(len(outputs)==2+len(HOST_LIBS),'overlapping Raw install targets')
    for target,file in outputs.items():
        dest=install/target;source=c.verify(file)
        require(dest.resolve(strict=True)==dest and install in dest.parents and old not in dest.parents,'unsafe package target')
        require(build in source.parents and source.name==target.name,'wrong Raw artifact for '+str(target))
    for target,file in outputs.items():
        dest=install/target
        shutil.copyfile(c.verify(file),dest)
        require(dest.stat().st_size==file['bytes'] and c.sha(dest)==file['sha256'],'installed artifact differs: '+str(target))
    for lib,host_rel in host_paths.items():
        require(c.dependencies(c.identity(install/host_rel),folder/(lib+'-ldd.log'),env)==c.dependencies(c.identity(old/host_rel),folder/(lib+'-base-ldd.log'),env),'Host link dependencies changed')
    return {str(target) for target in outputs}


def private_build(folder,src,variant,prior):
    anchor=src['off_package'];pack=c.read(c.verify(anchor));opp=pack['opp'];c.verify_opp(opp)
    old=Path(opp['install']).resolve();install=folder/'opp';check_root(install)
    before=tree(old);shutil.copytree(old,install,symlinks=True);compare_tree(before,tree(install),set())
    snapshot=pack['project_after'];original=Path(snapshot['CMakeLists.txt']['file']['path']).parent
    project=folder/'CustomOp';copy_snapshot(snapshot,original,project)
    cfg=project/'op_host'/HEADER;raw=derive_header(cfg.read_bytes())
    require((project/'op_kernel'/HEADER).read_bytes()==cfg.read_bytes(),'mismatched generated Header')
    kernel=gate_kernel(opp);require(kernel['elf']['sha256']==OFF_ELF,'control ELF changed')
    selected=c.read(c.verify(src['flow_build']))['variants']['A']
    env,env_binding=host.envs.for_prior(prior,install)
    # Recheck original trusted tool set; no inherited LD/preload/custom OPP path.
    allowed=set()
    cmake=project/'CMakeLists.txt';parent=c.verify(pack['project_before']['CMakeLists.txt']['file']).read_text()
    require(parent.count(DEVICE_CMAKE)==1 and parent.replace(DEVICE_CMAKE,DISABLED)==cmake.read_text(),'device target restore not exact')
    cmake.write_text(parent)
    if variant=='raw':
        for side in ('op_host','op_kernel'):(project/side/HEADER).write_bytes(raw)
    compare_tree(snapshot,tree(project),{'CMakeLists.txt','op_host/'+HEADER,'op_kernel/'+HEADER} if variant=='raw' else {'CMakeLists.txt'})
    build=folder/'device-build';cache=folder/'build-cache';cache.mkdir()
    build_env,_=host.envs.for_prior(prior,None)
    require(not build_env['ASCEND_CUSTOM_OPP_PATH'],'private OPP in device compiler environment')
    compiler=compile_device(project,build,Path(src['cann']),dict(build_env,ASCEND_CACHE_PATH=str(cache)),variant)
    kernel_rel=Path(kernel['elf']['path']).relative_to(old)
    require(Path(compiler['binary']['path']).name==kernel_rel.name,'unexpected compiler binary name/dispatch key')
    configs=install/'vendors/customize/op_impl/ai_core/tbe/kernel/config/ascend310p'
    require(c.read(configs/(STEM+'.json'))==c.read(c.verify(compiler['config'])) and
            c.read(configs/'binary_info_config.json')[GATE]==c.read(c.verify(compiler['index']))[GATE],'Gate registration drift')
    if variant=='off':
        if compiler['binary']['sha256']!=OFF_ELF:
            old_code=elf_sections(c.verify(kernel['elf']).read_bytes());new_code=elf_sections(c.verify(compiler['binary']).read_bytes())
            c.write(folder/'off-elf-drift.json',dict(expected=kernel['elf'],actual=compiler['binary'],source_header=c.identity(cfg),
                    old_code=old_code,new_code=new_code,executable_sections_equal=old_code==new_code,
                    reason=('Non-executable ELF differences' if old_code==new_code else 'Changed executable sections')+
                           '; underlying compiler/build cause unresolved. Historical numerical acceptance NOT inherited.'))
            raise ValueError('fresh Off ELF differs from accepted Off; see off-elf-drift.json')
        require((install/kernel_rel).resolve()==install/kernel_rel,'unsafe control binary target')
        shutil.copyfile(c.verify(compiler['binary']),install/kernel_rel)
    else:
        allowed.update(install_raw_outputs(old,install,build,before,kernel,compiler,folder,env))
        # Source packages can contain the same device configuration in the dynamic implementation directory.
        for dest in sorted({p.resolve(strict=True) for p in install.rglob(HEADER)}):
            require(c.sha(dest)==OFF_HEADER and install in dest.parents,'unbound installed configuration')
            dest.write_bytes(raw);allowed.add(str(dest.relative_to(install)))
    delta=sorted(compare_tree(before,tree(install),allowed))
    frozen=dict(install=str(install),preflight=preflight(install),kind='GATE8_PRIVATE_'+variant.upper())
    c.verify_opp(frozen);k=gate_kernel(frozen)
    if variant=='off':require(not delta and k['elf']['sha256']==OFF_ELF,'Off differs: no inherited acceptance')
    else:
        require(k['elf']['sha256']!=OFF_ELF and all(any(Path(rel).name==lib for rel in delta) for lib in HOST_LIBS),'Raw device/Host rebuild missing')
        require(elf_sections(c.verify(k['elf']).read_bytes())!=elf_sections(c.verify(kernel['elf']).read_bytes()),'Raw OPP code sections are identical to control')
    dynamic=[];tbe=install/'vendors/customize/op_impl/ai_core/tbe'
    wrapper=one((p for p in tbe.rglob(STEM+'.py') if p.parent.name=='dynamic'),'unbound/ambiguous packaged ATC implementation')
    for cpp in sorted(tbe.rglob(STEM+'.cpp')):
        require(install in cpp.resolve().parents and c.sha(cpp)==KERNEL_SOURCE,'unexpected packaged Gate implementation')
        header=cpp.parent/HEADER
        require(install in header.resolve(strict=True).parents and c.sha(header)=={'off':OFF_HEADER,'raw':RAW_HEADER}[variant],'packaged ATC dynamic header drift')
        dynamic.append(dict(wrapper=c.identity(wrapper),source=c.identity(cpp),header=c.identity(header)))
    require(dynamic and 'BUILD_KERNEL_SRC' not in env,'no bound dynamic Gate source or unexpected source override')
    api=c.identity(install/'vendors/customize/op_api/lib/libcust_opapi.so')
    require(api['sha256']==selected['api']['sha256']==API,'unexpected API drift')
    # Rebind the standard-manifest plus this exact private package; never reuse Off records under Raw paths.
    standard=src['standard_manifest'];manifest=host.dsos.with_private(c.read(c.verify(standard)),frozen,standard)
    c.write(folder/'manifest.json',manifest)
    env,env_binding=host.envs.for_prior(prior,install)
    host.envs.verify_binding(env_binding)
    runner=folder/'runner-build'
    command(['cmake','-S',HERE/'runner','-B',runner,'-DCANN_ROOT='+src['cann'],'-DCOMMON_INCLUDE='+str(c.QUANT/'test')],folder/'runner-configure.log',env)
    command(['cmake','--build',runner,'--target','gate_om_runner'],folder/'runner-compile.log',env)
    binary=c.identity(runner/'gate_om_runner');deps=c.dependencies(binary,folder/'runner-ldd.log',env)
    require(deps==selected['om']['deps'],'Gate OM Runner standard library identities differ from accepted full Runner')
    # Keep full Draft Runner and source math untouched.
    codec=sdk_layout(folder/'layout',project,Path(src['cann']),env,c.verify(prior['current']['evidence']['tools']['c++']))
    commands=[c.identity(p) for parent in (folder,folder/'layout') for p in sorted(parent.iterdir()) if p.is_file() and
              (p.suffix=='.log' or p.name.endswith('.command.json'))]
    return dict(opp=frozen,env=env_binding,manifest=c.identity(folder/'manifest.json'),api=api,kernel=k,opp_executable_sections=elf_sections(c.verify(k['elf']).read_bytes()),
                dynamic=dynamic,codec=codec,
                generated_sources=tree(project),package_files=tree(install),changed_files=delta,compiler=compiler,commands=commands,
                compile_policy='FRESH_GATE_ONLY_DEVICE_TARGET' if variant=='raw' else 'FRESH_GATE_ONLY_DEVICE_TARGET_BYTE_EXACT_CONTROL',
                off_build_provenance=anchor,runner=dict(runner=binary,deps=deps),om=selected['om'])


if __name__=='__main__':
    import traceback
    parser=argparse.ArgumentParser();parser.add_argument('request',type=Path);args=parser.parse_args()
    request=c.read(args.request);dest=Path(request['output']);dest.mkdir(exist_ok=False)
    report=dict(status='BLOCKED',variant=request['variant'])
    try:
        from gate_common import sources
        require(request['sources']==sources(),'build worker source drift')
        src=c.read(c.verify(request['source_report']))['binding'];prior=c.read(c.verify(src['host_source']))['binding']['prior']
        report['result']=private_build(dest,src,request['variant'],prior)
        c.verify_tree(report['result']);report['status']='PRIVATE_GATE_PACKAGE_BUILT'
    except Exception as err:report['error']=repr(err);(dest/'traceback.txt').write_text(traceback.format_exc())
    c.write(dest/'report.json',report);sys.exit(int(report['status']=='BLOCKED'))
