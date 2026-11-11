#!/usr/bin/env python3
"""Assemble separate numerical and timing evidence; never synthesize MLP latency."""
import argparse
from pathlib import Path
import real_mlp_common as c
import run_real_gate as gate


def load(path, status, prepare, sources):
    r=c.read(path);c.verify_tree(r)
    if r.get('status')!=status or r.get('error') or r.get('integrity_error') or r['prepare_report']!=prepare or r['sources']!=sources:
        raise ValueError('incomplete or unrelated report: '+str(path))
    return r


def main(a):
    root=c.fresh(a.output_dir)
    r=dict(status='NOT_RUN',sources=c.sources(),reports={},quality='UNRESOLVED',production='NOT_APPROVED')
    try:
        prep=c.prepared(a.prepare_report);pid=c.identity(a.prepare_report)
        g=load(a.gate_smoke,gate.STATES['smoke'],pid,r['sources'])
        m=load(a.mlp_smoke,'REAL_MLP_BOUNDARIES_RECORDED_W8_DOWN_NOT_RUN',pid,r['sources'])
        if m['gate_smoke_report']!=c.identity(a.gate_smoke):raise ValueError('MLP/Gate smoke chain differs')
        if [x['case'] for x in g['runs']]!=[x['id'] for x in prep['cases']]:raise ValueError('Gate sample set differs')
        for case,row in zip(prep['cases'],g['runs']):
            if row['validation']!=gate.validate(row,case,g['evidence'],g['device']):raise ValueError('gate validation changed')
        numeric=dict(status='REAL_MLP_BOUNDARY_DIAGNOSTICS_QUALITY_UNRESOLVED',checkpoint=prep['checkpoint'],
            CPU=[dict(case=x['id'],context_rows=x['capture']['context_rows'],metrics=x['diagnostics']) for x in prep['cases']],
            gate_npu=[dict(case=x['case'],validation=x['validation']) for x in g['runs']],
            native_and_mixed_mlp=m['cases'],full_w8a8_mlp_npu='NOT_RUN_DOWN_KERNEL_UNSUPPORTED',
            capture_scope='real native eager DraftGraph replay, not full OM internal activation parity',
            historical_201='HISTORICAL_UNRESOLVED_NOT_NEW_INPUT_ACCEPTANCE_RULE')
        perf=dict(status='PARTIAL_METRICS_ONLY',full_w8a8_mlp='NOT_RUN_DOWN_KERNEL_UNSUPPORTED',
            integrated_mixed_mlp='NOT_RUN_SEPARATE_PROCESS_BRIDGE',online_npu_quantization='NOT_RUN',
            device_kernel_time='UNAVAILABLE_NO_DEVICE_TIMESTAMPS',H2D_D2H='EXCLUDED_NOT_MEASURED',
            frozen_0_472195_ms='HISTORICAL_REFERENCE_ONLY_NOT_EXTRAPOLATED',quantization_semantics_differ=True)
        if a.gate_benchmark:
            b=load(a.gate_benchmark,gate.STATES['benchmark'],pid,r['sources'])
            if b['smoke_report']!=c.identity(a.gate_smoke):raise ValueError('gate timing from another Smoke')
            rows=[]
            for row in b['runs']:
                case=next(x for x in prep['cases'] if x['id']==row['case'])
                anchor=next(x['outputs'][0] for x in g['runs'] if x['case']==row['case'])
                if row['validation']!=gate.validate(row,case,b['evidence'],b['device'],timed=True,anchor=anchor):raise ValueError('timing safety drift')
                rows.append(row)
            for case in prep['cases']:
                selected=[x for x in rows if x['case']==case['id']]
                if [x['round'] for x in selected]!=[0,1,2]:raise ValueError('round coverage differs')
                stat={key:c.stats([v for row in selected for v in row['validation']['timing'][key]['samples_ms']],90) for key in ('prepare','execute_sync')}
                if stat!=b['statistics'][case['id']]:raise ValueError('timing summary differs')
            perf['w8a8_gate_resident']=b['statistics']
        if a.mlp_benchmark:
            b=load(a.mlp_benchmark,'REAL_NATIVE_MLP_TIMED_W8_DOWN_NOT_RUN',pid,r['sources'])
            if b['smoke_report']!=c.identity(a.mlp_smoke):raise ValueError('native timing from another Smoke')
            perf['native_eager_mlp_and_mixed_suffix']=b['cases']
        if a.preprocessing:
            b=load(a.preprocessing,'CPU_ONLINE_PREPROCESSING_MEASURED',pid,r['sources'])
            perf['cpu_online_preprocessing']=b['cases']
        for key,value in vars(a).items():
            if isinstance(value,Path) and key!='output_dir':r['reports'][key]=c.identity(value)
        c.write(root/'single-layer-numerics.json',numeric);c.write(root/'single-layer-performance.json',perf)
        r.update(status='REAL_MLP_REPORT_COMPILED_W8_DOWN_NOT_RUN',numerics=c.identity(root/'single-layer-numerics.json'),performance=c.identity(root/'single-layer-performance.json'))
    except Exception as exc:r.update(status='BLOCKED',error=f'{type(exc).__name__}: {exc}')
    finally:
        try:
            c.verify_tree(r)
            if r['sources']!=c.sources():raise ValueError('source changed')
        except Exception as exc:r.update(status='BLOCKED',integrity_error=str(exc))
        c.write(root/'report.json',r)
    print(r['status']);return int(r['status']=='BLOCKED')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for key in ('prepare-report','gate-smoke','mlp-smoke','output-dir'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('gate-benchmark','mlp-benchmark','preprocessing'):p.add_argument('--'+key,type=Path)
    raise SystemExit(main(p.parse_args()))
