"""Offline Linux ARM64 compile/link check; does not run ACLNN, ATC, or a Kernel."""
import argparse,json,os,shutil,subprocess,tempfile
from pathlib import Path
from scheduler_common import c,HERE,PD
from scheduler_build import preset_options
from scheduler_derive import verify


def main(a):
    verify();root=Path(tempfile.mkdtemp(prefix='core-scheduler-',dir=a.output_parent))
    r=dict(status='BLOCKED',scope='LOCAL_ONLY_NOT_SERVER_VERIFIED',npu='NOT_RUN',atc='NOT_RUN',kernel_compile='NOT_RUN',output=str(root));commands=[]
    def run(argv,name):
        with (root/(name+'.log')).open('wb') as log:rc=subprocess.run(list(map(str,argv)),stdout=log,stderr=subprocess.STDOUT).returncode
        commands.append(dict(argv=list(map(str,argv)),returncode=rc,log=c.identity(root/(name+'.log'))))
        if rc:raise RuntimeError(name+' failed')
    try:
        template=a.template.resolve(strict=True);project=root/'CustomOp'
        shutil.copytree(template,project,symlinks=True,ignore=shutil.ignore_patterns('build_out','__pycache__'))
        for name in ('d_flash_group_quant_linear.cpp','scheduler_evidence.h'):
            shutil.copyfile(HERE/'op_host'/name,project/'op_host'/name)
        shutil.copyfile(PD/'op_host/platform_diag.h',project/'op_host/platform_diag.h')
        (project/'op_host/scheduler_variant.h').write_text('#pragma once\n#define DFLASH_SCHEDULER_POLICY '+str(int(a.variant=='B'))+'\n')
        cm=project/'CMakeLists.txt';text=cm.read_text()
        text=text.replace('if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel)\n    add_subdirectory(op_kernel)\nendif()','# No device targets in offline Host check.')
        if 'add_subdirectory(op_kernel)' in text:raise ValueError('device target remains')
        cm.write_text(text)
        with (project/'op_host/CMakeLists.txt').open('a') as f:f.write('\ntarget_link_libraries(cust_optiling PRIVATE ${CMAKE_DL_LIBS})\ntarget_link_libraries(cust_op_proto PRIVATE ${CMAKE_DL_LIBS})\n')
        build=root/'host-build';run(['cmake','-S',project,'-B',build,*preset_options(project,a.cann_root)],'configure')
        run(['cmake','--build',build,'--target','cust_optiling','cust_op_proto','cust_opapi','--parallel','2'],'host-build')
        api=root/'opp/vendors/customize/op_api';(api/'lib').mkdir(parents=True);(api/'include').mkdir()
        for p in build.rglob('aclnn_*.h'):shutil.copy2(p,api/'include'/p.name)
        libs={p.resolve() for p in build.rglob('libcust_opapi.so')}
        if len(libs)!=1:raise ValueError('ambiguous local API')
        shutil.copy2(next(iter(libs)),api/'lib/libcust_opapi.so')
        runner=root/'runner';run(['cmake','-S',HERE/'runner','-B',runner,'-DCANN_ROOT='+str(a.cann_root),'-DPRIVATE_OPP='+str(root/'opp'),'-DCOMMON_INCLUDE='+str(c.QUANT/'test')],'runner-configure')
        run(['cmake','--build',runner,'--parallel','2'],'runner-build')
        run(['ldd',runner/'platform_diag_runner'],'runner-ldd')
        if 'not found' in (root/'runner-ldd.log').read_text():raise ValueError('unresolved local Runner dependency')
        for name in ('libcust_opmaster_rt2.0.so','libcust_opsproto_rt2.0.so'):
            found={p.resolve() for p in build.rglob(name)}
            if len(found)!=1:raise ValueError('Host library count')
            path=next(iter(found));run(['ldd','-r',path],name)
            if any(x in (root/(name+'.log')).read_text() for x in ('not found','undefined symbol')):raise ValueError('Host unresolved symbol')
        r.update(status='LOCAL_HOST_AND_RUNNER_COMPILED_LINKED',runner=c.identity(runner/'platform_diag_runner'),
            sdk=[dict(file=c.identity(a.cann_root/x['relative']),review_sha256=x['sha256']) for x in c.read(HERE/'sdk-review.json')['installed_headers']],
            compile_commands=c.identity(build/'compile_commands.json'))
        if any(x['file']['sha256']!=x['review_sha256'] for x in r['sdk']):raise ValueError('local SDK changed')
    except Exception as e:r.update(status='BLOCKED',error=str(e))
    r['commands']=commands;c.write(root/'report.json',r);print(json.dumps(r,indent=2));return int(r['status']=='BLOCKED')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--variant',choices=('A','B'),required=True);p.add_argument('--template',type=Path,required=True);p.add_argument('--output-parent',type=Path,required=True);p.add_argument('--cann-root',type=Path,default=Path(os.environ.get('CANN_ROOT','/usr/local/Ascend/cann-9.0.0')))
    raise SystemExit(main(p.parse_args()))
