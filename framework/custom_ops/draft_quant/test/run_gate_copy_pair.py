#!/usr/bin/env python3
"""Shared isolated Gate A/B driver; one copy OR prefetch variable per run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from a2_common import checked_file,record,sha256,write_json
from a3_launch import load_build_config
from compare_a51 import unwrap
from compare_a5 import load as load_full,projection
from run_gate_copy import SHAPE,copy_evidence

HERE=Path(__file__).resolve().parents[1]


def source_evidence(path):
    pair=json.loads(path.read_text())
    if pair.get('status')!='PASS' or pair.get('paired_correctness')!='PASS':raise ValueError('need accepted A5.1 paired PASS')
    item=pair['runs']['256'];single=Path(item['path'])
    if sha256(single)!=item['sha256']:raise ValueError('accepted K256 suite changed')
    stage,full,_=unwrap(single,256)
    config=load_build_config(Path(stage['build_config_path']))
    if config.get('gate_copy_mode','loop')!='loop' or config.get('gate_prefetch_mode','off')!='off':raise ValueError('accepted baseline must be original copy loop')
    a4,scopes=load_full(full,'broadcast',256,Path(stage['build_config_path']))
    child,child_root,_=scopes['c16']
    return dict(bundle=child['bundle'],native=child['native_om_manifest'],golden=str(child_root/'suite.json'),config=config,
                accepted_pair=dict(path=str(path),sha256=sha256(path)),accepted_stage=dict(path=str(single),sha256=sha256(single)))


def compare_runs(left,right,output,experiment="copy"):
    if experiment not in ("copy","prefetch"):raise ValueError("unknown Gate experiment")
    rows=[];reports=[];roots=[];configs=[]
    modes=('loop','2d') if experiment=='copy' else ('off','raw')
    field='gate_copy_mode' if experiment=='copy' else 'gate_prefetch_mode'
    for path,mode in zip((left,right),modes):
        r=json.loads(path.read_text());cfg=load_build_config(Path(r['build_config_path']))
        if r.get('status')!='PASS' or r['build']!=cfg or cfg.get(field)!=mode or sha256(r['build_config_path'])!=r['build_config_sha256']:
            raise ValueError('failed or mismatched Gate copy build')
        if experiment=='prefetch' and cfg.get('gate_copy_mode')!='loop':raise ValueError('2D copy must be off in raw-prefetch experiment')
        if experiment=='copy' and cfg.get('gate_prefetch_mode','off')!='off':raise ValueError('raw-prefetch must be off in 2D experiment')
        if r.get('experiment','copy')!=experiment:raise ValueError('wrong Gate experiment evidence')
        if not r.get('checks') or tuple(r['checks'])!=tuple(r['selected_cases']) or any(c['status']!='PASS' for c in r['checks'].values()):raise ValueError('untimed correctness gate missing')
        timing_path=checked_file(path.parent,r['timing']);timing=json.loads(timing_path.read_text())
        if timing.get('status')!='PASS' or timing['selected_cases']!=list(r['selected_cases']) or tuple(c['name'] for c in timing['cases'])!=tuple(r['selected_cases']):raise ValueError('incomplete selected timing')
        reports.append((r,timing));roots.append(timing_path.parent);configs.append(cfg)
    a,b=configs
    for key in ('source_sha256','core_limit','dequant_mode','pipeline_mode','kv_m80_mode','scale_mode','k_tile','tiling_abi','k_tile_policy','launch_version'):
        if a.get(key)!=b.get(key):raise ValueError('non-copy build setting differs: '+key)
    for key in ('device_id','selected_cases','bundle_sha256','native_om_manifest_sha256','timing_protocol'):
        if reports[0][1].get(key)!=reports[1][1].get(key):raise ValueError('paired timing identity differs: '+key)
    if reports[0][0]['selected_cases']!=reports[1][0]['selected_cases']:raise ValueError('paired selections differ')
    for old,new in zip(reports[0][1]['cases'],reports[1][1]['cases']):
        for row in (old,new):
            for execution in row['executions'].values():
                if execution['timing']['warmup']!=5 or execution['timing']['repetitions']!=30:raise ValueError('not 5/30')
        row=projection(old,new,*roots,SHAPE,reports[0][0],reports[1][0],False,gate_prefetch=experiment=='prefetch')
        if experiment=='copy' and row['baseline_launch']!=row['candidate_launch']:raise ValueError('Matmul tiling/resources/schedule changed')
        row['copy_plans']=[copy_evidence(root/old['name']/'custom/runner.log',mode) for root,mode in zip(roots,('loop','2d') if experiment=='copy' else ('loop','loop'))]
        rows.append(row)
    result=dict(experiment=experiment,status='PASS' if rows and all(r['status']=='PASS' for r in rows) else 'FAIL',scope='Gate-only same-Capture Native W8 bitwise; isolated runtime-weight ACLNN versus immutable Native OM',
                cases=rows,builds=configs,adoption='NOT_DECIDED',full_draft_token_kv='NOT_RUN',full_a4_regression='NOT_RUN',decode_performance='NOT_RUN',
                interpretation='Raw samples/median/p95 are isolated diagnostic timings. No full-Draft or cross-model speedup inference.')
    write_json(output,result);return result


def run(args):
    experiment=getattr(args,'experiment','copy')
    if experiment not in ('copy','prefetch'):raise ValueError('unknown Gate experiment')
    modes=('loop','2d') if experiment=='copy' else ('off','raw')
    if experiment=='prefetch' and args.layers!=[0]:raise ValueError('Gate prefetch stage is layer-0 only')
    root=args.output_dir.resolve();root.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',abi=f'dflash-gate-{experiment}-pair-v1',experiment=experiment,runs={},full_draft_token_kv='NOT_RUN',decode_performance='NOT_RUN',adoption='NOT_DECIDED')
    summary=root/'suite.json';write_json(summary,report)
    try:
        source=source_evidence(args.a51_evidence.resolve());report['source']=source;cfg=source['config']
        watched={Path(p):sha256(p) for p in (source['bundle'],source['native'],source['golden'],args.a51_evidence)}
        for mode in (modes if args.order=='baseline-first' else tuple(reversed(modes))):
            pointer=root/f'{mode}-summary.txt'
            env=dict(os.environ,MODEL_PYTHON=sys.executable,DEVICE_ID=str(args.device_id),DFLASH_SUITE='gate-copy' if experiment=='copy' else 'gate-prefetch',
                DFLASH_GATE_PREFETCH_MODE=mode if experiment=='prefetch' else 'off',
                DFLASH_GATE_COPY_MODE=mode if experiment=='copy' else 'loop',DFLASH_K_TILE='256',DFLASH_SCALE_MODE='broadcast',DFLASH_DEQUANT_MODE='batched',
                DFLASH_CORE_LIMIT=str(cfg['core_limit']),DFLASH_PIPELINE_MODE=cfg['pipeline_mode'],DFLASH_KV_M80_MODE=cfg['kv_m80_mode'],
                A4_C16_BUNDLE=source['bundle'],A4_C16_NATIVE_OM_MANIFEST=source['native'],GATE_GOLDEN_SUITE=source['golden'],
                GATE_LAYERS=','.join(map(str,args.layers)),DFLASH_SUMMARY_POINTER=str(pointer))
            report['phase']=mode;write_json(summary,report)
            with (root/f'{mode}.log').open('w') as log:subprocess.run(['bash',str(HERE/'run_server.sh')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            path=Path(pointer.read_text().strip())
            report['runs'][mode]=dict(path=str(path),sha256=sha256(path),sdk=record(path.parent.parent/'sdk-copy.json',path.parent.parent),
                installed_opp=record(path.parent.parent/'opp-preflight.json',path.parent.parent),
                source_commit=record(path.parent.parent/'source-commit.txt',path.parent.parent))
        if any(sha256(p)!=h for p,h in watched.items()):raise ValueError('accepted evidence changed')
        if experiment=='copy':compared=compare_runs(*(Path(report['runs'][mode]['path']) for mode in modes),root/'comparison.json')
        else:compared=compare_runs(*(Path(report['runs'][mode]['path']) for mode in modes),root/'comparison.json',experiment='prefetch')
        report.update(status=compared['status'],comparison=record(root/'comparison.json',root),phase='complete')
    except Exception as error:report.update(status='FAIL',error=f'{type(error).__name__}: {error}')
    write_json(summary,report);print(f"{report['status']}: {summary}; full Draft Token/KV NOT_RUN; adoption NOT_DECIDED",flush=True)
    return int(report['status']!='PASS')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--a51-evidence',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--layers',type=int,nargs='+',choices=range(5),default=[0]);p.add_argument('--device-id',type=int,default=0)
    p.add_argument('--order',choices=('baseline-first','candidate-first'),default='baseline-first')
    a=p.parse_args()
    if len(set(a.layers))!=len(a.layers) or a.device_id<0:p.error('invalid layers/device')
    raise SystemExit(run(a))
