#!/usr/bin/env python3
"""Independent Gate raw-prefetch layer-0 experiment; 2D DMA always disabled."""
import argparse
from pathlib import Path
from run_gate_copy_pair import run

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a51-evidence',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device-id',type=int,default=0)
    p.add_argument('--order',choices=('baseline-first','candidate-first'),default='baseline-first')
    a=p.parse_args()
    if a.device_id<0:p.error('invalid device')
    a.layers=[0];a.experiment='prefetch'
    raise SystemExit(run(a))
