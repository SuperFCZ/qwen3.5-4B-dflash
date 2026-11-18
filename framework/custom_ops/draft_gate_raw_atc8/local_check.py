"""ARM64 offline check. Never invokes ATC, loads a model, or starts an NPU application."""
import argparse
import os
from pathlib import Path
import shutil
import tempfile
import traceback
from build_gate import compile_device, DEVICE_CMAKE
from gate_common import c,command,HERE,HEADER,STEM,derive_header,require
from gate_plan import elf_sections,sdk_layout


def main(args):
    out=Path(tempfile.mkdtemp(prefix='gate8-check-',dir=args.output_parent));project=out/'CustomOp'
    report=dict(status='BLOCKED',scope='LOCAL_ONLY_NOT_SERVER_VERIFIED',npu='NOT_RUN',atc='NOT_RUN')
    try:
        shutil.copytree(args.template,project,symlinks=True)
        for name in ('platform_diag.h','scheduler_evidence.h',STEM+'.cpp'):
            require(c.sha(project/'op_host'/name)==c.sha(HERE.parent/'draft_a_core_scheduler_flow/op_host'/name),'need validated general-shape Scheduler A fixture')
        cm=project/'CMakeLists.txt';src=cm.read_text()
        require(src.count('# Host-only compile check')==1 and 'add_subdirectory(op_kernel)' not in src,'unknown local template')
        cm.write_text(src.replace('# Host-only compile check',DEVICE_CMAKE))
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
        command(['cmake','-S',HERE/'runner','-B',out/'runner','-DCANN_ROOT='+str(args.cann),'-DCOMMON_INCLUDE='+str(c.QUANT/'test')],out/'runner-configure.log',env)
        command(['cmake','--build',out/'runner','--parallel','2'],out/'runner-build.log',env)
        binary=c.identity(out/'runner/gate_om_runner');report.update(runner=binary,runner_deps=c.dependencies(binary,out/'runner-ldd.log',env))
        report['compilers']={};report['sections']={}
        for mode in ('off','raw'):
            # Separate source trees/caches/uploads; the only device source edit is the audited generated header.
            work=out/mode;work.mkdir();target=work/'CustomOp';shutil.copytree(project,target,symlinks=True)
            if mode=='raw':
                for sub in ('op_host','op_kernel'):
                    h=target/sub/HEADER;h.write_bytes(derive_header(h.read_bytes()))
            cache=work/'cache';cache.mkdir()
            result=compile_device(target,work/'device-build',args.cann,dict(env,ASCEND_CACHE_PATH=str(cache)),mode)
            report['compilers'][mode]=result;report['sections'][mode]=elf_sections(c.verify(result['binary']).read_bytes())
        require(report['sections']['off']!=report['sections']['raw'],'local switch produced identical executable sections')
        a,b=[report['compilers'][mode] for mode in ('off','raw')]
        require(all(c.read(c.verify(a[k]))==c.read(c.verify(b[k])) for k in ('config','index')),'local registry changed')
        report['layout']=sdk_layout(out/'codec',project,args.cann,env,shutil.which('c++'))
        report['status']='LOCAL_GATE8_RUNNER_BOTH_HOST_DEVICE_BUILDS_AND_SERIALIZER_MATCH'
    except Exception as exc:report['error']=repr(exc);(out/'traceback.txt').write_text(traceback.format_exc())
    c.write(out/'report.json',report);print(report['status'],report.get('error'),out,flush=True)
    return int(report['status']=='BLOCKED')


if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--template',type=Path,required=True)
    cli.add_argument('--output-parent',type=Path,required=True);cli.add_argument('--cann',type=Path,default=Path('/usr/local/Ascend/cann-9.0.0'))
    raise SystemExit(main(cli.parse_args()))
