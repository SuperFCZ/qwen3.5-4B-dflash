#!/usr/bin/env python3
"""Raw FP16 tile semantics on NPU, separate from the 41-case Head acceptance."""
import argparse
from pathlib import Path
import subprocess
import numpy as np
from common import write_json,record,sha256


def winner(row):
    # Integer oracle mirrors Better's mathematics without FP operations/NaN conversion.
    def key(b):
        b=int(b); nan=(b&0x7fff)>0x7c00
        return (nan,0 if nan else (-(b&0x7fff) if b&0x8000 else b))
    return max(range(64),key=lambda i:key(row[i]))


def rows():
    cases=[]
    for v in (0,0x8000,1,0x8001,0x7c00,0xfc00,0x7e01,0xfe21,0x7c01,0xfc01):
        cases.append(np.full(64,v,dtype='<u2'))
    for a,b in ((0,0x8000),(1,2),(0x8002,0x8001),(0x7c00,0x7e12),(0xfe13,0x7e01),(0x7c01,0xfc01)):
        for i in (0,1,31,32,62,63):
            row=np.full(64,a,dtype='<u2'); row[i]=b; cases.append(row)
    # Every FP16 bit pattern appears, with shuffled lane order and mixed boundaries.
    rng=np.random.default_rng(20261013)
    all_bits=np.arange(65536,dtype='<u2'); rng.shuffle(all_bits)
    cases.extend(all_bits.reshape(-1,64))
    # Also test ordered NON-NaN blocks: a NaN must not hide finite/subnormal
    # ordering failures in the exhaustive shuffled corpus.
    finite=np.arange(65536,dtype='<u2'); finite=finite[(finite&0x7fff)<=0x7c00]
    finite=np.pad(finite,(0,(-len(finite))%64),mode='edge')
    cases.extend(finite.reshape(-1,64))
    cases.extend(rng.integers(0,65536,size=(128,64),dtype=np.uint16))
    return np.array(cases,dtype='<u2')


def run(args):
    root=args.output_dir.resolve(); root.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',scope='Vector Top1 raw-FP16 primitive only',full_41_case_acceptance='NOT_RUN',
                runner_sha256=sha256(args.runner),batches=[])
    samples=rows(); report['rows']=len(samples)
    try:
        for start in range(0,len(samples),15):
            batch=np.zeros((15,64),dtype='<u2'); batch[:len(samples[start:start+15])]=samples[start:start+15]
            expected=np.array([winner(row) for row in batch],dtype='<i8')
            folder=root/f'batch-{start:04d}'; folder.mkdir(); batch.tofile(folder/'logits.bin')
            with (folder/'runner.log').open('w') as log:
                subprocess.run([str(args.runner.resolve()),str(args.device_id),str(folder)],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=120)
            for repeat in range(3):
                raw=(folder/f'ids-{repeat}.bin').read_bytes()
                if raw!=expected.tobytes():
                    actual=np.frombuffer(raw,dtype='<i8'); row=int(np.flatnonzero(actual!=expected)[0])
                    raise ValueError(dict(batch=start,row=row,expected=int(expected[row]),actual=int(actual[row]),
                                          fp16_bits=[f'0x{int(x):04x}' for x in batch[row]]))
            report['batches'].append(dict(start=start,input=record(folder/'logits.bin',root),output=record(folder/'ids-0.bin',root)))
            write_json(root/'suite.json',report)
        if sha256(args.runner)!=report['runner_sha256']: raise ValueError('runner changed')
        report['status']='PASS'
    except Exception as e: report.update(status='FAIL',error=str(e))
    write_json(root/'suite.json',report)
    return 0 if report['status']=='PASS' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--runner',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device-id',type=int,default=0); raise SystemExit(run(p.parse_args()))
