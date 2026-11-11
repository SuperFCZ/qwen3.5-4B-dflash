#!/usr/bin/env python3
"""Run the existing real native Draft replay capture in an isolated process."""
import argparse
from pathlib import Path
import subprocess
import sys
import real_mlp_common as c


def main(a):
    root=c.fresh(a.output_dir)
    r=dict(status='NOT_RUN',sources=c.sources(),context_rows=a.context_rows,npu='NOT_RUN',synthetic=False)
    try:
        ck=c.checkpoint(a.draft_dir);r['checkpoint']=ck
        argv=[sys.executable,'-B',str(c.REPO/'framework/custom_ops/draft_quant/test/a2_capture.py'),
              '--scope','a2','--draft-dir',str(a.draft_dir),'--target-dir',str(a.target_dir),
              '--replay-report',str(a.replay_report),'--output-dir',str(root/'capture'),
              '--device-id',str(a.device)]
        if a.feature_layers:argv+=['--feature-layers',a.feature_layers]
        r['command']=argv;r['npu']='ATTEMPTED'
        with (root/'capture.log').open('xb') as log:
            completed=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,check=False)
        r['returncode']=completed.returncode
        if completed.returncode:raise ValueError('native capture failed; inspect capture.log and capture/manifest.json')
        proof,_,_=c.capture_binding(root/'capture/manifest.json',ck)
        if proof['context_rows']!=a.context_rows:raise ValueError('actual frozen gear differs; no padding or generated replacement')
        r.update(status='REAL_DRAFT_CAPTURE_BOUND',npu='COMPLETED',capture=proof,
                 source_runtime='native eager DraftGraph replay; full OM internal parity NOT_RUN')
    except Exception as exc:r.update(status='BLOCKED',error=f'{type(exc).__name__}: {exc}')
    finally:
        if (root/'capture.log').is_file():r['log']=c.identity(root/'capture.log')
        try:
            c.verify_tree(r)
            if r['sources']!=c.sources():raise ValueError('capture source changed')
        except Exception as exc:r.update(status='BLOCKED',integrity_error=str(exc))
        c.write(root/'report.json',r)
    print(r['status']);return int(r['status']=='BLOCKED')


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for k in ('draft-dir','target-dir','replay-report','output-dir'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--context-rows',type=int,choices=(16,64),required=True)
    p.add_argument('--device',type=int,default=0);p.add_argument('--feature-layers')
    a=p.parse_args(argv)
    if a.device<0:p.error('negative device')
    return a


if __name__=='__main__':raise SystemExit(main(parse()))
