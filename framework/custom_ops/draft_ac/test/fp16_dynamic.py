"""Independent C++ adapter build/reuse; no OPP, AIR or OM compilation."""
from pathlib import Path
import subprocess
from common import ROOT, sha256, write_json
from fp16_evidence import load


def sources():
    paths=['framework/custom_ops/draft_ac/test/fp16_dynamic_runner.cpp',
           'framework/custom_ops/draft_ac/test/fp16_dynamic/CMakeLists.txt',
           'framework/custom_ops/draft_quant/test/runner_common.h']
    return {p:sha256(ROOT/p) for p in paths}


def runner(output, env, reuse=None):
    if reuse:
        r=load(reuse);p=Path(r['runner']['path']).resolve()
        if r.get('source_sha256')!=sources() or not p.is_file() or sha256(p)!=r['runner']['sha256']:
            raise ValueError('dynamic adapter build identity differs')
        return p,r
    output.mkdir()
    cann=env.get('CANN_ROOT',env.get('ASCEND_HOME_PATH','/usr/local/Ascend/ascend-toolkit/latest'))
    commands=[['cmake','-S',str(ROOT/'framework/custom_ops/draft_ac/test/fp16_dynamic'),'-B',str(output),'-DCANN_ROOT='+cann],
              ['cmake','--build',str(output),'--parallel']]
    write_json(output/'commands.json',commands)
    with (output/'build.log').open('w') as log:
        for command in commands:subprocess.run(command,env=env,cwd=output,stdout=log,stderr=subprocess.STDOUT,check=True)
    p=output/'fp16_dynamic_runner'
    r=dict(source_sha256=sources(),runner=dict(path=str(p),sha256=sha256(p)),cann_root=cann,
           status='BUILT_NOT_NPU_VALIDATED',scope='C++ dynamic replay adapter; original FP16 OM unchanged')
    write_json(output/'build.json',r)
    return p,r
