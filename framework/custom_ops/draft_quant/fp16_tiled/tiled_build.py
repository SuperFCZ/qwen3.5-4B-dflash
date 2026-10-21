"""New standalone OPP, with original W8 recipe sources read-only."""
import json
from pathlib import Path
import platform
import shutil
import sys
from tiled_contract import HERE,identity,write_json
from tiled_sources import generate,SNAKE,NEW
from a3_launch import configure_build
from opp_preflight import check_root,check_binary_path
from gate_copy_sdk import inspect as inspect_sdk
from experiment import command,child_environment


def installed(root):
    check_root(root);directory=root/'vendors/customize/op_impl/ai_core/tbe/kernel/ascend310p'/SNAKE
    binaries=sorted(directory.glob('*.o'))
    if not binaries:raise ValueError('no installed custom FP16 Kernel')
    result=[]
    for binary in binaries:
        metadata=check_binary_path(binary)
        if binary.stat().st_size==0 or not isinstance(json.loads(metadata.read_text()),dict):raise ValueError('invalid custom FP16 OPP object/JSON')
        result.append(dict(binary=identity(binary),metadata=identity(metadata)))
    api=root/'vendors/customize/op_api/include'/f'aclnn_{SNAKE}.h'
    if not api.is_file() or api.stat().st_size==0:raise ValueError('custom FP16 ACLNN header missing')
    return dict(kernels=result,header=identity(api),status='FILESYSTEM_VERIFIED_NPU_NOT_RUN')


def build(args,root,config):
    if platform.machine() not in ('arm64','aarch64') or sys.platform!='linux':raise ValueError('requires real Linux aarch64 CANN receiver')
    if config['gate_prefetch_mode']!='off' or config['gate_copy_mode']!='loop' or config['k_tile']!=256 or config['scale_mode']!='broadcast' or config['dequant_mode']!='batched':raise ValueError('derive only accepted W8 Off Loop/K256/Broadcast')
    root.mkdir();check_root(root);project=root/'CustomOp';opp=root/'opp';env=child_environment(args.cann_root)
    write_json(root/'sdk.json',inspect_sdk(args.cann_root,'prefetch'))
    command(['msopgen','gen','-i',HERE/'DFlashGateFp16Tiled.json','-c','ai_core-Ascend310P3','-lan','cpp','-out',project,'-op',NEW],root/'msopgen.log',env)
    audit=generate(HERE.parent,project,root/'derived-source')
    shutil.copyfile(HERE.parent/'op_host/d_flash_group_quant_linear_contract.h',project/'op_host/d_flash_group_quant_linear_contract.h')
    configure_build(project/'op_host/d_flash_group_quant_linear_build_config.h',root/'baseline-build-config.json',config['core_limit'],config['dequant_mode'],
        project/'op_kernel/d_flash_group_quant_linear_build_config.h',config['pipeline_mode'],config['kv_m80_mode'],config['scale_mode'],256,'loop','off')
    write_json(root/'source-adaptation.json',audit)
    command(['bash','-c','cd "$1" && bash build.sh','build',str(project)],root/'build.log',env,3600)
    packages=list((project/'build_out').glob('custom_opp_*.run'))
    if len(packages)!=1:raise ValueError('expected exactly one new OPP package')
    command(['bash',packages[0],'--install-path='+str(opp)],root/'install.log',env)
    opp_audit=installed(opp);write_json(root/'opp-preflight.json',opp_audit)
    command(['cmake','-S',HERE,'-B',root/'runner-build','-DCANN_ROOT='+str(args.cann_root),'-DOPP_ROOT='+str(opp),'-DRUNNER_SOURCE='+audit['runner']['path'],'-DCOMMON_INCLUDE='+str(HERE.parent/'test')],root/'runner-cmake.log',env)
    command(['cmake','--build',root/'runner-build','--parallel'],root/'runner-build.log',env)
    runner=root/'runner-build/dflash_gate_fp16_tiled_test';env=child_environment(args.cann_root,opp/'vendors/customize')
    command(['ldd',runner],root/'runner-ldd.log',env)
    files=[identity(p) for p in opp.rglob('*') if p.is_file()]+[identity(runner),identity(packages[0])]
    for item in audit.values():files.extend([identity(item['path']),identity(item['diff'])])
    for path in (root/'baseline-build-config.json',project/'op_host/d_flash_group_quant_linear_build_config.h',project/'op_kernel/d_flash_group_quant_linear_build_config.h',project/'op_host/d_flash_group_quant_linear_contract.h',HERE.parent/'test/runner_common.h'):
        files.append(identity(path))
    report=dict(status='BUILT_NOT_NPU_VERIFIED',op=NEW,baseline_config=config,source_adaptation=audit,installed=opp_audit,runner=identity(runner),files=files,
        scope='isolated custom FP16 tile input; baseline allocation labels do not imply dequant execution')
    write_json(root/'build-report.json',report)
    return runner,env,report
