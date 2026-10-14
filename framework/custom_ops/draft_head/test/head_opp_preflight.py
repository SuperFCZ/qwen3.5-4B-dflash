#!/usr/bin/env python3
"""Check BOTH installed Head object/JSON sets. Filesystem evidence, not NPU validation."""
import argparse
import hashlib
import json
from pathlib import Path

from common import QUANT  # Adds the existing shared test helpers to sys.path.
from opp_preflight import check_root, check_binary_path

OP_KERNELS=('d_flash_draft_lm_head_top1','d_flash_draft_lm_head_top1_audit')


def check_install(root,diagnostics=False,vector_top1=False,release=False):
    if release and (diagnostics or vector_top1): raise ValueError("release cannot contain probes")
    root=Path(root).absolute()
    check_root(root)
    operators=[]
    for name in OP_KERNELS+(('d_flash_head_probe','d_flash_cast_probe') if diagnostics else ())+(('d_flash_top1_probe',) if vector_top1 else ()):
        directory=root/'vendors/customize/op_impl/ai_core/tbe/kernel/ascend310p'/name
        if not directory.is_dir(): raise ValueError(f'missing installed Head operator directory: {directory}')
        binaries=sorted(directory.glob('*.o'))
        if not binaries: raise ValueError(f'no installed {name} kernel objects under {directory}')
        records=[]
        for binary in binaries:
            # Reuse the CANN 9.0 FIRST '.o'/'.json' derivation and lexical /
            # resolved-path checks; never mutate the A checker's OP_KERNEL.
            metadata=check_binary_path(binary)
            data=binary.read_bytes()
            if not data: raise ValueError(f'empty kernel object: {binary}')
            raw_json=metadata.read_bytes()
            try: info=json.loads(raw_json)
            except (ValueError,UnicodeError) as error:
                raise ValueError(f'invalid kernel JSON {metadata}: {error}') from error
            if not isinstance(info,dict): raise ValueError(f'kernel JSON must be an object: {metadata}')
            records.append(dict(binary=str(binary),json=str(metadata),binary_bytes=len(data),json_bytes=len(raw_json),
                binary_sha256=hashlib.sha256(data).hexdigest(),json_sha256=hashlib.sha256(raw_json).hexdigest()))
        operators.append(dict(name=name,directory=str(directory),kernels=records))
    headers=[]; libraries=[]
    if release:
        kernel_root=root/'vendors/customize/op_impl/ai_core/tbe/kernel/ascend310p'
        expected={kernel_root/name for name in OP_KERNELS}
        extra_objects=[str(p) for p in kernel_root.rglob('*.o') if p.parent not in expected]
        extra_dirs=[str(p) for p in kernel_root.iterdir() if p.is_dir() and 'probe' in p.name]
        if extra_objects or extra_dirs: raise ValueError(f'release has extra operators: {extra_objects+extra_dirs}')
        include=root/'vendors/customize/op_api/include'
        wanted={'aclnn_'+name+'.h' for name in OP_KERNELS}
        actual={p.name for p in include.glob('aclnn_d_flash*.h')}
        if actual!=wanted: raise ValueError(f'release ACLNN header inventory differs: {sorted(actual)}')
        for name in sorted(wanted):
            path=include/name; raw=path.read_bytes()
            if not raw: raise ValueError(f'empty release header: {path}')
            headers.append(dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        for path in sorted((root/'vendors/customize').rglob('*.so*')):
            if not path.is_file(): continue
            raw=path.read_bytes()
            if not raw: raise ValueError(f'empty installed library: {path}')
            libraries.append(dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        if not any('/op_api/lib/' in item['path'] for item in libraries): raise ValueError('missing release ACLNN shared library')
    return dict(status='PASS',scope='installed Head/probe object/JSON paths and JSON parsing only' if diagnostics else 'both installed Head object/JSON paths and JSON parsing only',
        install_root=str(root),operators=operators,release_only=release,headers=headers,libraries=libraries,npu_status='NOT_RUN')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install-root',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--diagnostics',action='store_true')
    parser.add_argument('--vector-top1',action='store_true')
    parser.add_argument('--release',action='store_true')
    args=parser.parse_args()
    try:
        report=check_install(args.install_root,args.diagnostics,args.vector_top1,args.release)
        print(f"PASS: {len(report['operators'])} Head/probe OPP object/JSON sets; NPU NOT_RUN; {args.install_root}",flush=True)
    except (OSError,ValueError) as error:
        report=dict(status='FAIL',scope='Head OPP filesystem preflight',install_root=str(args.install_root.absolute()),
                    npu_status='NOT_RUN',error=f'{type(error).__name__}: {error}')
        print(f"FAIL: {report['error']}",flush=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__': raise SystemExit(main())
