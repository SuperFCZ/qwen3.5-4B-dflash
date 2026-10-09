#!/usr/bin/env python3
"""A5.0 paired 26-projection diagnostics per gear; never a full-Draft latency."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import a4_contract
import reference
from a2_common import checked_file, sha256, validate_execution, write_json
from a3_launch import launch_evidence, load_build_config
from compare_a3 import timings


def load(path, mode):
    report = json.loads(path.read_text())
    if (report.get('abi') != 'dflash-group-quant-linear-a4-suite-v1' or report.get('status') != 'PASS' or
        report.get('timing_protocol') != 'continuous-v1' or
        set(report.get('checks', {})) != {'paired_gears','synthetic','c16','c64','launches'} or
        any(c.get('status') != 'PASS' for c in report['checks'].values())):
        raise ValueError('A5 requires complete passing A4 regression in BOTH builds')
    config_path = path.parent.parent / 'build-config.json'
    config = load_build_config(config_path)
    if (report.get('build') != config or sha256(config_path) != report.get('build_config_sha256') or
        config.get('scale_mode') != mode or config.get('dequant_mode') != 'batched' or
        config.get('kv_m80_mode') != 'a-ub' or config.get('launch_version') != 5):
        raise ValueError('need explicit scalar/broadcast paired builds retaining KV M80 a-ub')
    synthetic = json.loads(checked_file(path.parent,report['checks']['synthetic']['summary']).read_text())
    if (synthetic.get('status') != 'PASS' or
        tuple(w['name'] for w in synthetic.get('workloads',[])) != tuple(s.name for s in reference.A4_WORKLOADS) or
        any(w.get('status') != 'PASS' for w in synthetic['workloads'])):
        raise ValueError('missing complete 36-shape A4 synthetic regression')
    scopes = {}
    for gear in (16,64):
        label=f'c{gear}'
        child_path=checked_file(path.parent,report['checks'][label]['summary'])
        child=json.loads(child_path.read_text())
        if (child.get('status') != 'PASS' or child.get('abi') != a4_contract.ABI or child.get('context_rows') != gear or
            child.get('timing_protocol') != 'continuous-v1' or
            tuple(c['name'] for c in child.get('cases',[])) != a4_contract.CASE_NAMES):
            raise ValueError('need all 26 real projections per C16/C64 gear')
        for key in ('bundle','native_om_manifest'):
            if sha256(child[key]) != child[key+'_sha256']:
                raise ValueError('capture/native manifest changed after run')
            identity=report['inputs'][label][key]
            if identity['sha256'] != child[key+'_sha256'] or Path(identity['path']).resolve()!=Path(child[key]).resolve():
                raise ValueError('aggregate/real input identity differs')
        for runner in child['runners'].values():
            if sha256(runner['path'])!=runner['sha256']: raise ValueError('accepted runner binary changed')
        scopes[label]=(child,child_path.parent,gear)
    return report,scopes


def compatible(a,b):
    for key in ('device_id','timing_protocol','inputs'):
        if a.get(key) is None or a[key]!=b.get(key): raise ValueError(f'paired inputs differ: {key}')
    for key in ('source_sha256','core_limit','dequant_mode','pipeline_mode','kv_m80_mode','tiling_abi','launch_version'):
        if a['build'].get(key) is None or a['build'][key]!=b['build'].get(key):
            raise ValueError(f'paired build differs: {key}')


def match_plans(a,b,shape):
    for key in ('m','k','n','tile_m','tile_n','tile_k','weight_reuse_rows','block_dim','available_cores','raw_banks','selected_pipeline'):
        if a[key]!=b[key]: raise ValueError(f'non-scale launch policy changed: {key}')
    if a['user_ub_bytes']+a['matmul_ub_bytes'] != b['user_ub_bytes']+b['matmul_ub_bytes']:
        raise ValueError('total UB budget changed')
    if tuple(shape)==(80,2560,2048):
        for key in ('user_ub_bytes','matmul_ub_bytes','system_workspace_bytes','cube'):
            if a[key]!=b[key]: raise ValueError('KV M80 must retain the exact A-UB resource plan')
        if b['scale']['selected_scale']!='scalar-muls-v1': raise ValueError('KV M80 scale path changed')


def projection(old,new,aroot,broot,shape,a,b):
    if old.get('status')!='PASS' or new.get('status')!='PASS': raise ValueError('failed projection')
    protocol=old['executions']['custom']['timing']
    expected=checked_file(aroot,old['outputs']['native_om'][0]).read_bytes()
    comparisons,plans=[],[]
    for report,case,root in ((a,old,aroot),(b,new,broot)):
        for kind in ('custom','native_om'):
            execution=case['executions'][kind]
            validate_execution(execution,shape,report['device_id'],
                'AscendCL ACLNN' if kind=='custom' else 'AscendCL native OM',
                protocol['warmup'],protocol['repetitions'],'continuous-v1')
            outputs,bracket=case['outputs'][kind],case.get('bracket_outputs',{}).get(kind,[])
            if len(outputs)!=2 or len(bracket)!=2: raise ValueError('missing repeat/tail/postcheck evidence')
            for item in outputs+bracket:
                comparisons.append(reference.compare(expected,checked_file(root,item).read_bytes(),SimpleNamespace(m=shape[0],n=shape[2])))
        cfg=report['build']
        plans.append(launch_evidence(root/case['name']/'custom/runner.log',shape,cfg['core_limit'],
            case['executions']['custom']['workspace_bytes'],cfg['dequant_mode'],cfg['pipeline_mode'],5,cfg['kv_m80_mode'],cfg['scale_mode']))
    match_plans(*plans,shape)
    before,after=timings(old['executions']['custom']),timings(new['executions']['custom'])
    native_before,native_after=timings(old['executions']['native_om']),timings(new['executions']['native_om'])
    return dict(name=old['name'],shape=list(shape),status='PASS' if all(c['finite'] and c['bitwise_equal'] and c['max_ulp']==0 for c in comparisons) else 'FAIL',
        comparisons=comparisons,baseline=before,candidate=after,native_before=native_before,native_after=native_after,
        baseline_launch=plans[0],candidate_launch=plans[1],median_speedup=before['median_ms']/after['median_ms'],
        candidate_native_ratio=after['median_ms']/native_after['median_ms'],
        candidate_slower_median=after['median_ms']>before['median_ms'],candidate_slower_p95=after['p95_ms']>before['p95_ms'])


def totals(rows):
    if tuple(row['name'] for row in rows)!=a4_contract.CASE_NAMES:
        raise ValueError('totals require exactly 26 projections in contract order')
    def summarize(items):
        return dict(projection_count=len(items),**{kind:dict(
            sum_projection_median_ms=sum(r[kind]['median_ms'] for r in items),
            sum_projection_p95_ms=sum(r[kind]['p95_ms'] for r in items))
            for kind in ('baseline','candidate','native_before','native_after')})
    by_shape={}
    for row in rows: by_shape.setdefault(tuple(row['shape']),[]).append(row)
    total=summarize(rows)
    total['median_sum_speedup']=total['baseline']['sum_projection_median_ms']/total['candidate']['sum_projection_median_ms']
    total['candidate_native_median_sum_ratio']=total['candidate']['sum_projection_median_ms']/total['native_after']['sum_projection_median_ms']
    return dict(**total,by_shape=[dict(shape=list(shape),**summarize(items)) for shape,items in by_shape.items()],
        interpretation='sum of 26 isolated projection medians/p95s; NOT median/p95 of a sequential Draft call or end-to-end timing')


def run(baseline,candidate,output):
    if output.exists(): raise FileExistsError('use a new comparison file')
    a,left=load(baseline,'scalar'); b,right=load(candidate,'broadcast'); compatible(a,b)
    result=dict(abi='dflash-a5-comparison-v1',status='RUNNING',full_a4_regression='PASS',
        baseline=dict(path=str(baseline),sha256=sha256(baseline)),candidate=dict(path=str(candidate),sha256=sha256(candidate)),
        scope='PASS is numerical only; isolated continuous-v1',adoption='NOT_DECIDED',
        full_draft_validation='NOT_RUN',decode_performance='NOT_RUN',cases=[],totals={})
    for label in ('c16','c64'):
        old,aroot,gear=left[label]; new,broot,_=right[label]
        for key in ('bundle_sha256','native_om_manifest_sha256','device_id'):
            if old.get(key) is None or old[key]!=new.get(key): raise ValueError('paired projection inputs differ')
        rows=[]
        for expected,x,y in zip(a4_contract.cases(gear),old['cases'],new['cases']):
            row=projection(x,y,aroot,broot,a4_contract.case_shape(expected),a,b)
            rows.append(row); result['cases'].append(dict(gear=label,**row))
            print(f"{label}/{row['name']}: {row['status']} median {row['baseline']['median_ms']:.4f}->{row['candidate']['median_ms']:.4f} ms; "
                  f"p95 {row['baseline']['p95_ms']:.4f}->{row['candidate']['p95_ms']:.4f}; native {row['native_after']['median_ms']:.4f}",flush=True)
        result['totals'][label]=totals(rows)
        print(f"{label} 26 projection totals: {result['totals'][label]}",flush=True)
    result['regressions']=[dict(gear=r['gear'],name=r['name'],median=r['candidate_slower_median'],p95=r['candidate_slower_p95'])
                           for r in result['cases'] if r['candidate_slower_median'] or r['candidate_slower_p95']]
    result['status']='PASS' if all(r['status']=='PASS' for r in result['cases']) else 'FAIL'
    write_json(output,result)
    return 0 if result['status']=='PASS' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','candidate','output'): parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    return run(args.baseline.resolve(),args.candidate.resolve(),args.output.resolve())

if __name__=='__main__': raise SystemExit(main())
