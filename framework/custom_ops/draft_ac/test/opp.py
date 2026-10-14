"""Build one NEW vendor package from frozen A/C sources/configs; no DLL overlay."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from common import ROOT,HERE,A_ROOT,C_ROOT,sha256,write_json
from a3_launch import load_build_config
from opp_preflight import check_root,check_binary_path

OPS={'DFlashGroupQuantLinear':('d_flash_group_quant_linear',A_ROOT),
     'DFlashDraftLmHeadTop1':('d_flash_draft_lm_head_top1',C_ROOT),
     'DFlashDraftLmHeadTop1Audit':('d_flash_draft_lm_head_top1_audit',C_ROOT)}


def validate_sources(a_config,c_release):
    a=load_build_config(a_config)
    if (a.get('dequant_mode'),a.get('scale_mode'),a.get('k_tile'),a.get('kv_m80_mode'))!=('batched','broadcast',256,'a-ub'):
        raise ValueError('provide the accepted A5.1 broadcast/K256/A-UB build identity; never use tracked K128 defaults')
    for name,digest in a['source_sha256'].items():
        if name.startswith(('op_host/','op_kernel/')) and sha256(A_ROOT/name)!=digest: raise ValueError('frozen A source differs: '+name)
    c=json.loads(Path(c_release).read_text())
    if c.get('status')!='PASS' or c.get('new_release_acceptance')!='PASS' or c.get('correctness')!='PASS': raise ValueError('need real C Release PASS evidence')
    config=c['build']
    if config.get('build_kind')!='release' or config.get('operators')!=['d_flash_draft_lm_head_top1','d_flash_draft_lm_head_top1_audit']:
        raise ValueError('C must be the accepted two-operator release without probes')
    if (config.get('conversion'),config.get('top1'),config.get('profile'))!=('compare_select','vector','none'): raise ValueError('not frozen C Vector release')
    for name,digest in c['contract']['source_sha256'].items():
        if sha256(C_ROOT/name)!=digest: raise ValueError('frozen C source differs: '+name)
    if sha256(config['header'])!=config['header_sha256']: raise ValueError('C build header changed')
    return a,config


def preflight(install,ambient=''):
    vendor=Path(install)/'vendors/customize'; kernels=vendor/'op_impl/ai_core/tbe/kernel/ascend310p'
    records=[]
    for ge,(kernel,_) in OPS.items():
        objects=sorted((kernels/kernel).glob('*.o'))
        if not objects: raise ValueError(f'missing {ge} kernel objects')
        for obj in objects:
            meta=check_binary_path(obj)
            if not obj.stat().st_size or not isinstance(json.loads(meta.read_text()),dict): raise ValueError('invalid OPP pair')
            records.append(dict(op=ge,binary=str(obj),bytes=obj.stat().st_size,sha256=sha256(obj),json=str(meta),json_bytes=meta.stat().st_size,json_sha256=sha256(meta)))
        header=vendor/f'op_api/include/aclnn_{kernel}.h'
        if not header.is_file() or not header.stat().st_size: raise ValueError(f'missing ACLNN header {header}')
    proto_libs=list((vendor/'op_proto').rglob('*.so'))
    if not proto_libs or any(not p.stat().st_size for p in proto_libs): raise ValueError('missing GE prototype shared library')
    # Some msopgen packages expose only the registration .so, no readable REG_OP header.
    # ATC, not a header grep, is the decisive registration gate.
    # Do not stack two customize/libcust_opapi.so or ambiguous GE registrations.
    for item in filter(None,(ambient+os.pathsep+os.pathsep.join(str(Path(p).parent.parent) for p in os.environ.get('LD_LIBRARY_PATH','').split(os.pathsep) if Path(p).parent.name=='op_api')).split(os.pathsep)):
        other=Path(item).resolve()
        if other==vendor.resolve(): continue
        headers=list((other/'op_proto/inc').glob('*.h'))+list(other.glob('*/op_proto/inc/*.h'))
        text='\n'.join(p.read_text(errors='replace') for p in headers)
        if any(re.search(r'REG_OP\s*\(\s*'+name+r'\s*\)',text) for name in OPS) or any((other/'op_impl/ai_core/tbe/kernel/ascend310p'/name).exists() for name,_ in OPS.values()):
            raise ValueError(f'ambiguous old A/C OPP active: {other}; remove only old A/C vendor paths, preserve receiver ADN/GDR')
    return dict(status='PASS',scope='registration/filesystem only; GE/ATC execution NOT_RUN',vendor=str(vendor),kernels=records,
                files=[dict(path=str(p),sha256=sha256(p)) for p in sorted(vendor.rglob('*')) if p.is_file() and (p.suffix in ('.h','.so','.json') or '.so.' in p.name)])


def build(args):
    root=args.output_dir.resolve(); check_root(root); root.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',ge_atc='NOT_RUN',npu='NOT_RUN',full_draft='NOT_RUN'); write_json(root/'opp.json',report)
    try:
        a,c=validate_sources(args.a_build_config,args.c_release)
        report.update(a_build_sha256=sha256(args.a_build_config),c_release_sha256=sha256(args.c_release))
        project=root/'CustomOp'; install=root/'opp'
        for index,(op,(_,source)) in enumerate(OPS.items()):
            cmd=['msopgen','gen','-i',str(source/('DFlashGroupQuantLinear.json' if source==A_ROOT else 'DFlashDraftLmHeadTop1.json')),
                 '-c','ai_core-Ascend310P3','-lan','cpp','-out',str(project),'-op',op]
            if index: cmd+=['-m','1']
            subprocess.run(cmd,check=True)
        for source in (A_ROOT,C_ROOT):
            for part in ('op_host','op_kernel'):
                for file in (source/part).iterdir():
                    if file.suffix in ('.h','.cpp'): shutil.copy2(file,project/part/file.name)
        for header in ('head_contract.h',): shutil.copy2(C_ROOT/'op_host'/header,project/'op_kernel'/header)
        for part in ('op_host','op_kernel'):
            shutil.copyfile(a['header'],project/part/'d_flash_group_quant_linear_build_config.h')
            shutil.copyfile(c['header'],project/part/'head_config.h')
        # Enforce exact generated target sources without changing any frozen source.
        with (project/'op_host/CMakeLists.txt').open('a') as out:
            out.write('\nget_target_property(ac_sources cust_optiling SOURCES)\nset(ac_names)\nforeach(s IN LISTS ac_sources)\n get_filename_component(n "${s}" NAME)\n list(APPEND ac_names "${n}")\nendforeach()\n')
            for name,_ in OPS.values(): out.write(f'if(NOT "{name}.cpp" IN_LIST ac_names)\n message(FATAL_ERROR "missing {name}")\nendif()\n')
        subprocess.run(['bash','build.sh'],cwd=project,check=True)
        packages=list((project/'build_out').glob('custom_opp_*.run'))
        if len(packages)!=1: raise ValueError('expected one joint OPP installer')
        env=os.environ.copy(); env.pop('ASCEND_CUSTOM_OPP_PATH',None)
        subprocess.run(['bash',str(packages[0]),'--install-path='+str(install)],env=env,check=True)
        report['preflight']=preflight(install,os.environ.get('ASCEND_CUSTOM_OPP_PATH',''))
        report.update(status='BUILT_NOT_NPU_VALIDATED',install=str(install),a_config=a,c_config=c,
                      installer=dict(path=str(packages[0]),sha256=sha256(packages[0])))
    except Exception as e: report.update(status='FAIL',error=f'{type(e).__name__}: {e}')
    write_json(root/'opp.json',report)
    if report['status']=='FAIL': raise RuntimeError(report['error'])
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--a-build-config',type=Path,required=True); p.add_argument('--c-release',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True); build(p.parse_args())
