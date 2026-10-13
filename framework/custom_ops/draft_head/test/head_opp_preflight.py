#!/usr/bin/env python3
"""Check BOTH installed Head object/JSON sets. Filesystem evidence, not NPU validation."""
import argparse
import hashlib
import json
from pathlib import Path

from common import QUANT  # Adds the existing shared test helpers to sys.path.
from opp_preflight import check_root, check_binary_path

OP_KERNELS=('d_flash_draft_lm_head_top1','d_flash_draft_lm_head_top1_audit')


def check_install(root):
    root=Path(root).absolute()
    check_root(root)
    operators=[]
    for name in OP_KERNELS:
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
    return dict(status='PASS',scope='both installed Head object/JSON paths and JSON parsing only',
        install_root=str(root),operators=operators,npu_status='NOT_RUN')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install-root',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    try:
        report=check_install(args.install_root)
        print(f'PASS: both Head OPP object/JSON sets; NPU NOT_RUN; {args.install_root}',flush=True)
    except (OSError,ValueError) as error:
        report=dict(status='FAIL',scope='Head OPP filesystem preflight',install_root=str(args.install_root.absolute()),
                    npu_status='NOT_RUN',error=f'{type(error).__name__}: {error}')
        print(f"FAIL: {report['error']}",flush=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__': raise SystemExit(main())
