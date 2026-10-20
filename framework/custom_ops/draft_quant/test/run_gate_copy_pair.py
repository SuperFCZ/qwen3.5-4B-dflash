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
from run_gate_copy import SHAPE,copy_evidence,selected_layers,selected_cases

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


def summarize(rows):
    if not rows or len({r['name'] for r in rows})!=len(rows):raise ValueError('empty/duplicate Gate summary')
    result=dict(projection_count=len(rows),selected_cases=[r['name'] for r in rows],
        interpretation='Sum of selected isolated projection medians/p95s; NOT full Draft latency or the median/p95 of a sequential execution.')
    for kind in ('baseline','candidate','native_before','native_after'):
        result[kind]=dict(sum_projection_median_ms=sum(r[kind]['median_ms'] for r in rows),
                          sum_projection_p95_ms=sum(r[kind]['p95_ms'] for r in rows))
    result['median_sum_speedup']=result['baseline']['sum_projection_median_ms']/result['candidate']['sum_projection_median_ms']
    result['p95_sum_ratio']=result['baseline']['sum_projection_p95_ms']/result['candidate']['sum_projection_p95_ms']
    return result


def verify_layer_identities(before,after,names):
    if tuple(before)!=tuple(names) or tuple(after)!=tuple(names) or before!=after:raise ValueError('paired per-layer capture/Native/Golden identity differs')
    for name in names:
        row=before[name]
        if row['name']!=name:raise ValueError('per-layer identity name differs')
        if set(row['capture_files'])!={'x.bin','q_nk.bin','w_nz.bin','s_gn.bin','eager-0.bin','eager-1.bin'} or set(row['native'])!={'om','deployment','air_manifest'} or len(row['golden'])!=4:raise ValueError('incomplete per-layer identity inventory')
        for item in [*row['capture_files'].values(),*row['native'].values(),*row['golden']]:
            path=Path(item['path'])
            if not path.is_file() or path.stat().st_size!=item['bytes'] or sha256(path)!=item['sha256']:raise ValueError('per-layer artifact changed: '+str(path))


def compare_runs(left,right,output,experiment="copy"):
    if experiment not in ("copy","prefetch"):raise ValueError("unknown Gate experiment")
    if output.exists() or output.with_suffix('.md').exists():raise FileExistsError('use fresh comparison artifacts')
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
    names=tuple(reports[0][0]['selected_cases'])
    if not names or len(set(names))!=len(names) or any(n not in selected_cases(list(range(5))) for n in names):raise ValueError('invalid selected Gate inventory')
    if names!=tuple(n for n in selected_cases(list(range(5))) if n in names):raise ValueError('noncanonical Gate layer order')
    if experiment=='prefetch':verify_layer_identities(reports[0][0].get('input_identities',{}),reports[1][0].get('input_identities',{}),names)
    for old,new in zip(reports[0][1]['cases'],reports[1][1]['cases']):
        for row in (old,new):
            for execution in row['executions'].values():
                if execution['timing']['warmup']!=5 or execution['timing']['repetitions']!=30:raise ValueError('not 5/30')
        row=projection(old,new,*roots,SHAPE,reports[0][0],reports[1][0],False,gate_prefetch=experiment=='prefetch')
        if experiment=='copy' and row['baseline_launch']!=row['candidate_launch']:raise ValueError('Matmul tiling/resources/schedule changed')
        row['copy_plans']=[copy_evidence(root/old['name']/'custom/runner.log',mode) for root,mode in zip(roots,('loop','2d') if experiment=='copy' else ('loop','loop'))]
        row['p95_speedup']=row['baseline']['p95_ms']/row['candidate']['p95_ms']
        if experiment=='prefetch':row['input_identity']=reports[0][0]['input_identities'][row['name']]
        rows.append(row)
        print(f"{row['name']}: {row['status']} median {row['baseline']['median_ms']:.6f}->{row['candidate']['median_ms']:.6f} ms, p95 {row['baseline']['p95_ms']:.6f}->{row['candidate']['p95_ms']:.6f} ms, speedup {row['median_speedup']:.4f}x",flush=True)
    result=dict(experiment=experiment,status='PASS' if rows and all(r['status']=='PASS' for r in rows) else 'FAIL',scope='Gate-only same-Capture Native W8 bitwise; isolated runtime-weight ACLNN versus immutable Native OM',
                cases=rows,summary=summarize(rows),
                regressions=[dict(name=r['name'],median=r['candidate_slower_median'],p95=r['candidate_slower_p95']) for r in rows if r['candidate_slower_median'] or r['candidate_slower_p95']],
                builds=configs,adoption='NOT_DECIDED',full_draft_token_kv='NOT_RUN',full_a4_regression='NOT_RUN',decode_performance='NOT_RUN',
                interpretation='Raw samples/median/p95 are isolated diagnostic timings. No full-Draft or cross-model speedup inference.')
    write_json(output,result)
    lines=['# Gate independent projection comparison','',
           'Status '+result['status']+' is numerical only; adoption NOT_DECIDED; full_draft_token_kv NOT_RUN.',
           'Each cell is median / p95 (ms). Raw samples and exact identities are in comparison.json.','',
           '| Layer | Off/baseline | Raw/candidate | Native-before | Native-after | Median speedup | Regression vs baseline |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        cells=[f"{row[k]['median_ms']:.6f} / {row[k]['p95_ms']:.6f}" for k in ('baseline','candidate','native_before','native_after')]
        flags=','.join(k for k,flag in (('median',row['candidate_slower_median']),('p95',row['candidate_slower_p95'])) if flag) or 'none'
        lines.append('| '+row['name']+' | '+' | '.join(cells)+f" | {row['median_speedup']:.5f}x | {flags} |")
    lines+=['',result['summary']['interpretation'],'',json.dumps(result['summary'],indent=2)]
    output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    return result


def run(args):
    experiment=getattr(args,'experiment','copy')
    if experiment not in ('copy','prefetch'):raise ValueError('unknown Gate experiment')
    modes=('loop','2d') if experiment=='copy' else ('off','raw')
    layers=selected_layers(args.layers);names=selected_cases(layers)
    root=args.output_dir.resolve();root.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',abi=f'dflash-gate-{experiment}-pair-v1',experiment=experiment,selected_layers=list(layers),selected_cases=names,runs={},full_draft_token_kv='NOT_RUN',decode_performance='NOT_RUN',adoption='NOT_DECIDED')
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
                GATE_LAYERS=','.join(map(str,layers)),DFLASH_SUMMARY_POINTER=str(pointer))
            report['phase']=mode;write_json(summary,report)
            with (root/f'{mode}.log').open('w') as log:subprocess.run(['bash',str(HERE/'run_server.sh')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            path=Path(pointer.read_text().strip())
            completed=json.loads(path.read_text())
            if completed.get('status')!='PASS' or tuple(completed.get('selected_cases',[]))!=names:raise ValueError('run did not validate exactly the requested Gate layers')
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
