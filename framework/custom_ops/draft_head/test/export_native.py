#!/usr/bin/env python3
"""Static native full-vocabulary Head and Head+Top1 OMs, M=1..15."""
import argparse
import gc
import os
from pathlib import Path
import sys
from common import ABI,K,N,REPO,load_capture,checked_file,record,sha256,write_json
sys.path[:0]=[str(REPO/'framework/python'),str(REPO)]


def export(args):
    if os.environ.get('ASCEND310P_SIMULATION_ONLY')=='1': raise ValueError('native CANN export required')
    import numpy as np
    import torch
    import torch_npu
    import torchair
    from qwen35_dflash.ascend310p.contracts import AirGraphSpec
    from qwen35_dflash.ascend310p.exporter import export_air_bundle
    from qwen35_dflash.ascend310p.runtime_input_export import canonical_runtime_input_abi
    from qwen35_dflash.ascend310p.compiler import compile_air_bundle,resolve_atc_executable
    from qwen35_dflash.ascend310p.utils import require_run_output
    cap_path=args.capture.resolve(); cap=load_capture(cap_path)
    root=require_run_output(args.output_dir); root.mkdir(parents=True,exist_ok=False)
    device=f'npu:{args.device_id}'; torch.npu.set_device(device)
    if '310P' not in torch.npu.get_device_name(args.device_id).upper(): raise ValueError('requires Ascend310P')
    atc=resolve_atc_executable(args.atc)
    # Two explicit read-only ND inputs match C's standalone ABI. This is NOT
    # the production constant-head graph or an end-to-end latency baseline.
    wfile=checked_file(cap_path.parent,cap['weight']['file'])
    memory=np.memmap(wfile,dtype='<f2',mode='c',shape=(N,K))
    weight=torch.from_numpy(memory).to(device)
    hidden=torch.from_numpy(np.fromfile(checked_file(cap_path.parent,cap['gears']['c64']['hidden']),dtype='<f2').reshape(15,K)).to(device)
    report=dict(abi=ABI,status='RUNNING',capture_sha256=sha256(cap_path),models=[],
                weight_binding='runtime-readonly-ND',production_const_head_parity='NOT_RUN',
                native_execution='NOT_RUN',custom_head_validation='NOT_RUN',full_draft_validation='NOT_RUN',end_to_end='NOT_RUN')
    path=root/'native.json'; write_json(path,report)
    for m in range(1,16):
        for kind in ('logits','top1'):
            item=dict(m=m,kind=kind,status='RUNNING'); report['models'].append(item)
            try:
                class Native(torch.nn.Module):
                    def forward(self,hidden,weight):
                        value=torch.nn.functional.linear(hidden,weight)
                        return value if kind=='logits' else torch.argmax(value,dim=-1)
                model=Native().eval()
                spec=AirGraphSpec(name=f'head_{kind}_m{m}',role='diagnostic',model=model,
                    example_args=(hidden[:m].contiguous(),weight),input_names=('hidden','weight'),output_names=('y',),dynamic=False,
                    metadata=dict(head_contract=ABI,weight_binding='runtime-readonly-ND',
                        tensor_abi=dict(inputs=[dict(name='hidden',dtype='float16',shape=[m,K]),dict(name='weight',dtype='float16',shape=[N,K])],
                                        outputs=[dict(name='y',dtype='float16' if kind=='logits' else 'int64',shape=[m,N] if kind=='logits' else [m])])) )
                with canonical_runtime_input_abi(torchair,public_inputs=spec.example_args,public_names=spec.input_names,
                                                  require_static_shapes=True,public_output_names=spec.output_names) as io_audit:
                    air=export_air_bundle(lambda _:(spec,),{},root/f'm{m:02d}-{kind}'/'artifacts')
                if io_audit['status']!='PASS': raise ValueError('native public ABI normalization failed')
                item['runtime_input_abi']=io_audit
                compiled=compile_air_bundle(air['manifest_path'],atc_bin=atc,soc_version='Ascend310P3',
                    extra_args=['--precision_mode=must_keep_origin_dtype','--deterministic=0'])
                graph=compiled['graphs'][0]
                if graph.get('input_names')!=['hidden','weight'] or graph.get('output_names')!=['y']:
                    raise ValueError('native input/output ABI changed')
                om=checked_file(Path(compiled['manifest_path']).parent,graph['om'])
                item.update(status='PASS',om=record(om,root),deployment=record(Path(compiled['manifest_path']),root),
                            air=record(Path(air['manifest_path']),root),atc_command=graph['atc_command'])
            except Exception as error:
                item.update(status='FAIL',error=f'{type(error).__name__}: {error}')
            finally:
                print(f"native export M{m} {kind}: {item['status']}",flush=True)
                torch._dynamo.reset(); gc.collect(); write_json(path,report)
    report['status']='PASS' if all(x['status']=='PASS' for x in report['models']) and sha256(cap_path)==report['capture_sha256'] else 'FAIL'
    write_json(path,report)
    return 0 if report['status']=='PASS' else 1

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device-id',type=int,default=0); p.add_argument('--atc',default='atc')
    return export(p.parse_args())
if __name__=='__main__': raise SystemExit(main())
