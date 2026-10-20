#!/usr/bin/env python3
"""Independent Gate raw-prefetch selected-layer experiment; 2D DMA always disabled."""
import argparse
from pathlib import Path
from run_gate_copy_pair import run
from run_gate_copy import selected_layers

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a51-evidence',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device-id',type=int,default=0)
    p.add_argument('--order',choices=('baseline-first','candidate-first'),default='baseline-first')
    p.add_argument('--layers',type=int,nargs='+',choices=range(5),default=[0],help='explicit Gate layers; default: layer-0 only')
    a=p.parse_args(argv)
    if a.device_id<0:p.error('invalid device')
    try:a.layers=list(selected_layers(a.layers))
    except ValueError as error:p.error(str(error))
    a.experiment='prefetch'
    return a


if __name__=='__main__':
    raise SystemExit(run(parse_args()))
