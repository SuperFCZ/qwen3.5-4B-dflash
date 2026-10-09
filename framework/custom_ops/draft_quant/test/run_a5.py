#!/usr/bin/env python3
"""Build isolated A5 controls/candidates, run full A4 twice, compare, then profile."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from a2_common import record,sha256,write_json
from compare_a5 import run as compare_runs

HERE=Path(__file__).resolve().parents[1]


def run(args):
    root=args.output_dir.resolve()
    if root.exists(): raise FileExistsError('use a new A5 evidence directory')
    for name in ('A4_C16_BUNDLE','A4_C64_BUNDLE','A4_C16_NATIVE_OM_MANIFEST','A4_C64_NATIVE_OM_MANIFEST'):
        if not Path(os.environ.get(name,'')).is_file(): raise ValueError(f'{name} must point to an existing A4 artifact')
    if not args.skip_profiling and not shutil.which(args.msprof):
        raise ValueError('load the CANN environment so msprof is available, or explicitly --skip-profiling (NOT_RUN)')
    root.mkdir(parents=True)
    report=dict(abi='dflash-a5-pair-v1',status='RUNNING',order=args.order,builds={},profiles=[],
        profiling='NOT_RUN',full_a4_regression='NOT_RUN',paired_correctness='NOT_RUN',
        adoption='NOT_DECIDED',full_draft_validation='NOT_RUN',decode_performance='NOT_RUN')
    summary=root/'suite.json'; write_json(summary,report)
    try:
        modes=('scalar','broadcast') if args.order=='baseline-first' else ('broadcast','scalar')
        for mode in modes:
            pointer=root/f'{mode}-summary.txt'
            env=dict(os.environ,MODEL_PYTHON=sys.executable,DFLASH_SUITE='a4',DFLASH_DEQUANT_MODE='batched',
                DFLASH_SCALE_MODE=mode,DFLASH_KV_M80_MODE='a-ub',DFLASH_PIPELINE_MODE=args.pipeline,
                A2_WARMUP=str(args.warmup),A2_REPETITIONS=str(args.repetitions),DFLASH_SUMMARY_POINTER=str(pointer))
            command=['bash',str(HERE/'run_server.sh')]
            report['phase']=mode; write_json(summary,report)
            print(f'A5 {mode}: full A4; log {root / (mode+"-server.log")}',flush=True)
            with (root/f'{mode}-server.log').open('w') as log:
                subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            path=Path(pointer.read_text().strip())
            report['builds'][mode]=dict(path=str(path),sha256=sha256(path)); write_json(summary,report)
        baseline=Path((root/'scalar-summary.txt').read_text().strip())
        candidate=Path((root/'broadcast-summary.txt').read_text().strip())
        report['full_a4_regression']='PASS'
        report['phase']='paired_comparison'; write_json(summary,report)
        if compare_runs(baseline,candidate,root/'comparison.json'):
            raise ValueError('paired bitwise comparison failed; do not adopt candidate')
        report['comparison']=record(root/'comparison.json',root)
        report['paired_correctness']='PASS'
        if not args.skip_profiling:
            report['profiling']='RUNNING'
            report['phase']='profiling'; write_json(summary,report)
            # Profile EVERY accepted projection for BOTH gears/variants. Runs
            # are separate from continuous-v1 latency samples and use the exact
            # original OPP/runner/input binding checked by profile_a31.py.
            for mode,path in (('scalar',baseline),('broadcast',candidate)):
                for gear in ('c16','c64'):
                    real=json.loads((path.parent/gear/'suite.json').read_text())
                    for case in real['cases']:
                        source=path.parent/gear/case['name']/'custom'
                        output=root/'profiles'/mode/gear/case['name']
                        command=[sys.executable,str(HERE/'test/profile_a31.py'),'--case-dir',str(source),
                                 '--output-dir',str(output),'--msprof',args.msprof]
                        subprocess.run(command,check=True)
                        profile=json.loads((output/'profile.json').read_text())
                        if profile.get('status')!='COMMAND_COMPLETED' or not profile.get('numerical_match'):
                            raise ValueError('profiled correctness failed')
                        report['profiles'].append(dict(mode=mode,gear=gear,name=case['name'],
                            evidence=record(output/'profile.json',root),pipe_counters=profile['pipe_counters']))
                        write_json(summary,report)
            report['profiling']='COMMANDS_COMPLETED'
            # Side-by-side raw columns, retaining original units and all rows.
            indexed={(p['mode'],p['gear'],p['name']):p for p in report['profiles']}
            report['profile_pairs']=[dict(gear=p['gear'],name=p['name'],baseline=p,
                candidate=indexed['broadcast',p['gear'],p['name']]) for p in report['profiles'] if p['mode']=='scalar']
            if len(report['profiles'])!=104 or len(report['profile_pairs'])!=52:
                raise ValueError('profiling requires all 26 projections per gear and variant')
        report.update(status='PASS',phase='complete',full_a4_regression='PASS')
    except Exception as error:
        if report.get('phase')=='profiling': report['profiling']='FAIL'
        if report.get('phase')=='paired_comparison': report['paired_correctness']='FAIL'
        report.update(status='FAIL',error=f'{type(error).__name__}: {error}')
        raise
    finally: write_json(summary,report)
    print(f'A5 evidence: {summary}; PASS is correctness/workflow only, adoption NOT_DECIDED; full Draft/Decode NOT_RUN')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--pipeline',choices=('serial','prefetch'),default='prefetch')
    parser.add_argument('--order',choices=('baseline-first','candidate-first'),default='baseline-first')
    parser.add_argument('--warmup',type=int,default=5); parser.add_argument('--repetitions',type=int,default=30)
    parser.add_argument('--msprof',default='msprof'); parser.add_argument('--skip-profiling',action='store_true')
    args=parser.parse_args()
    if not 3<=args.warmup<=100 or not 10<=args.repetitions<=1000: parser.error('invalid continuous-v1 warmup/repetitions')
    run(args)

if __name__=='__main__': main()
