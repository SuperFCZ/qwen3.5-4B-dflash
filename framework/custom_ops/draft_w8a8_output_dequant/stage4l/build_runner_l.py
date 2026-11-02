import json,platform,shutil
from pathlib import Path
from l_common import HERE,verify,identity,environment,common
from run_stage2b import command

def build(args,root,r):
    if platform.system()!='Linux' or platform.machine() not in ('aarch64','arm64'):raise ValueError('Linux aarch64 CANN receiver required')
    proof=r['evidence'];pf=json.loads(verify(proof['preflight']).read_text())
    for name in ('cmake','c++'):
        tool=shutil.which(name)
        if not tool or identity(tool)!=pf['tools'][name]['file']:raise ValueError('compiler differs from accepted environment')
    env=environment(proof,args.cann_root);r['phase']='runner_build'
    command(['cmake','-S',HERE/args.variant,'-B',root/'runner-build','-DCANN_ROOT='+str(args.cann_root),'-DOPP_ROOT='+proof['opp'],
        '-DCOMMON_INCLUDE='+str(common.QUANT/'test'),'-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],root/'cmake.log',env,r,'configure')
    command(['cmake','--build',root/'runner-build','--parallel'],root/'runner-build.log',env,r,'runner_build')
    runner=root/'runner-build/stage4l_runner';command(['ldd',runner],root/'ldd.log',env,r,'ldd')
    if 'not found' in (root/'ldd.log').read_text():raise ValueError('missing Runner dependency')
    links={}
    for line in (root/'runner-build/CMakeCache.txt').read_text().splitlines():
        if line.startswith('LIB_') and ':FILEPATH=' in line:
            key,value=line.split(':FILEPATH=',1);item=identity(value)
            expected=Path(proof['opp']) if key=='LIB_custom' else args.cann_root
            if expected.resolve() not in Path(item['path']).parents:raise ValueError('wrong linked library')
            links[key]=item
    if set(links)!={'LIB_custom','LIB_ascendcl','LIB_nnopbase','LIB_acl_op_compiler'}:raise ValueError('missing link identities')
    r.update(status='STAGE4L_RUNNER_BUILT',phase='complete',runner_compile='COMPLETED',runner=identity(runner),
        linked_libraries=links,compile_commands=identity(root/'runner-build/compile_commands.json'),
        kernel_build='NOT_RUN_REUSED_ACCEPTED_OPP',opp_installation='NOT_RUN_REUSED_ACCEPTED_OPP')
