#!/usr/bin/env python3
"""Independent K-only experiment; no capture/export, no production integration."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from a2_common import record,sha256,write_json
from compare_a51 import run as compare

HERE=Path(__file__).resolve().parents[1]

def run(args):
    root=args.output_dir.resolve(); root.mkdir(parents=True,exist_ok=False)
    report=dict(abi='dflash-a51-pair-v1',status='RUNNING',runs={},profiling='NOT_RUN',
        representative_correctness='NOT_RUN',full_a4_regression='NOT_RUN',paired_correctness='NOT_RUN',
        full_draft_validation='NOT_RUN',decode_performance='NOT_RUN',adoption='NOT_DECIDED')
    summary=root/'suite.json'; write_json(summary,report)
    try:
        base_env=dict(os.environ)
        if args.a5_evidence is not None:
            source=args.a5_evidence.resolve()
            if json.loads((source/'suite.json').read_text()).get('status')!='PASS':
                raise ValueError('A5 evidence must be a passing suite')
            a4_path=Path((source/'scalar-summary.txt').read_text().strip())
            a4=json.loads(a4_path.read_text())
            if a4.get('status')!='PASS' or a4.get('abi')!='dflash-group-quant-linear-a4-suite-v1':
                raise ValueError('A5 pointer must identify full passing A4')
            for gear in ('c16','c64'):
                for field,suffix in (('bundle','BUNDLE'),('native_om_manifest','NATIVE_OM_MANIFEST')):
                    item=a4['inputs'][gear][field]
                    if sha256(item['path'])!=item['sha256']: raise ValueError('A5 capture/native manifest changed')
                    base_env[f'A4_{gear.upper()}_{suffix}']=item['path']
            report['source_a5']=dict(path=str(source),suite_sha256=sha256(source/'suite.json'),
                scalar_a4=dict(path=str(a4_path),sha256=sha256(a4_path)))
        for tile in ((128,256) if args.order=='baseline-first' else (256,128)):
            report['phase']=f'k{tile}'; write_json(summary,report)
            pointer=root/f'k{tile}-summary.txt'
            env=dict(base_env,MODEL_PYTHON=sys.executable,DFLASH_SUITE='a51',DFLASH_K_TILE=str(tile),
                DFLASH_DEQUANT_MODE='batched',DFLASH_SCALE_MODE='broadcast',DFLASH_KV_M80_MODE='a-ub',
                DFLASH_PIPELINE_MODE=args.pipeline,A2_WARMUP='5',A2_REPETITIONS='30',DFLASH_SUMMARY_POINTER=str(pointer))
            print(f'A5.1 K{tile}: six-shape fallback gate, representative gates, then full A4; log {root / ("k"+str(tile)+".log")}',flush=True)
            with (root/f'k{tile}.log').open('w') as log:
                subprocess.run(['bash',str(HERE/'run_server.sh')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            path=Path(pointer.read_text().strip()); report['runs'][str(tile)]=dict(path=str(path),sha256=sha256(path))
        report.update(phase='comparison',representative_correctness='PASS',full_a4_regression='PASS')
        write_json(summary,report)
        code=compare(Path(report['runs']['128']['path']),Path(report['runs']['256']['path']),root/'comparison.json')
        report['comparison']=record(root/'comparison.json',root)
        report.update(status='PASS' if code==0 else 'FAIL',paired_correctness='PASS' if code==0 else 'FAIL')
    except Exception as error:
        if report.get('phase')=='comparison': report['paired_correctness']='FAIL'
        report.update(status='FAIL',error=f'{type(error).__name__}: {error}')
    write_json(summary,report)
    print(f"{report['status']}: {summary}; adoption NOT_DECIDED; full Draft/Decode NOT_RUN",flush=True)
    return 0 if report['status']=='PASS' else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--a5-evidence',type=Path,help='reuse the accepted A5 C16/C64 captures and native OMs')
    parser.add_argument('--pipeline',choices=('serial','prefetch'),default='prefetch')
    parser.add_argument('--order',choices=('baseline-first','candidate-first'),default='baseline-first')
    return run(parser.parse_args())
if __name__=='__main__': raise SystemExit(main())
