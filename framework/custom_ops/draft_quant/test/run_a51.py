#!/usr/bin/env python3
"""A5.1: six fallback shapes, representative real projections, then full A4."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import a4_contract
import run_a2
import run_a4
import run_suite
import a51_fallback
from a2_common import load_bundle, record, sha256, write_json
from a3_launch import load_build_config, launch_evidence

REPRESENTATIVES=('fc',)+tuple(f'layer-0-{kind}' for kind in a4_contract.KINDS)


def collect(root,gear,config):
    real=json.loads((root/'suite.json').read_text())
    if (real.get('status')!='PASS' or real.get('context_rows')!=gear or real.get('timing_protocol')!='continuous-v1' or
        tuple(c['name'] for c in real.get('cases',[]))!=REPRESENTATIVES):
        raise ValueError('representative inventory/gear/strict correctness failed')
    plans=[]
    for case in real['cases']:
        shape=a4_contract.case_shape(next(c for c in a4_contract.cases(gear) if c['name']==case['name']))
        plans.append(launch_evidence(root/case['name']/'custom/runner.log',shape,config['core_limit'],
            case['executions']['custom']['workspace_bytes'],config['dequant_mode'],config['pipeline_mode'],5,
            config['kv_m80_mode'],config['scale_mode'],config['k_tile'],config.get('k_tile_policy')))
    return plans


def run(args):
    config=load_build_config(args.build_config)
    if (config.get('k_tile_policy')!=a51_fallback.K_TILE_POLICY or
        config.get('k_tile') not in (128,256) or config.get('dequant_mode')!='batched' or
        config.get('kv_m80_mode')!='a-ub' or config.get('scale_mode') not in ('scalar','broadcast') or
        args.warmup!=5 or args.repetitions!=30):
        raise ValueError('A5.1 requires current fallback policy, batched, a-ub, explicit K tile, warmup=5/repetitions=30')
    root=args.output_dir.resolve(); root.mkdir(parents=True,exist_ok=False)
    report=dict(abi='dflash-a51-stage-v1',status='RUNNING',phase='preflight',build=config,
        build_config_path=str(args.build_config.resolve()),build_config_sha256=sha256(args.build_config),
        device_id=args.device_id,checks={},launches={},full_a4_regression='NOT_RUN',
        full_draft_validation='NOT_RUN',decode_performance='NOT_RUN')
    summary=root/'suite.json'; write_json(summary,report)
    try:
        a4_contract.paired_gears(load_bundle(args.c16_bundle,'a4'),load_bundle(args.c64_bundle,'a4'))
        report['phase']='fallback-synthetic'; write_json(summary,report)
        target=root/'fallback-synthetic'
        passed=run_suite.run(target,args.runner.resolve(),args.device_id,'a4',config['k_tile'],
                             shape_filter=a51_fallback.TARGET_NAMES,continuous=True)
        report['checks']['fallback-synthetic']=dict(status='PASS' if passed else 'FAIL',summary=record(target/'suite.json',root))
        if not passed: raise ValueError('six-shape fallback gate failed; representative/full A4 NOT_RUN')
        report['launches']['fallback-synthetic']=a51_fallback.collect(target,config,args.device_id)
        write_json(summary,report)
        for gear in (16,64):
            label=f'representative-c{gear}'; report['phase']=label
            child=SimpleNamespace(**vars(args)); child.output_dir=root/label
            child.scope='a4'; child.timing_protocol='continuous-v1'; child.case_filter=REPRESENTATIVES
            child.k_tile=config['k_tile']; child.bundle=getattr(args,f'c{gear}_bundle')
            child.native_om_manifest=getattr(args,f'c{gear}_native_om_manifest')
            write_json(summary,report)
            passed=run_a2.run(child)==0
            report['checks'][label]=dict(status='PASS' if passed else 'FAIL',summary=record(child.output_dir/'suite.json',root))
            if not passed: raise ValueError('representative failed; full A4 NOT_RUN')
            report['launches'][label]=collect(child.output_dir,gear,config); write_json(summary,report)
        report['phase']='full'; write_json(summary,report)
        child=SimpleNamespace(**vars(args)); child.output_dir=root/'full'
        passed=run_a4.run(child)==0
        report['full_a4_regression']='PASS' if passed else 'FAIL'
        report['checks']['full']=dict(status=report['full_a4_regression'],summary=record(child.output_dir/'suite.json',root))
        if not passed: raise ValueError('full A4 failed')
        if load_build_config(args.build_config)!=config or sha256(args.build_config)!=report['build_config_sha256']:
            raise ValueError('build identity changed')
        report.update(status='PASS',phase='complete')
    except Exception as error: report.update(status='FAIL',error=f'{type(error).__name__}: {error}')
    write_json(summary,report)
    print(f"{report['status']}: A5.1 K{config['k_tile']} phase={report['phase']}; {summary}; full Draft/Decode NOT_RUN",flush=True)
    return 0 if report['status']=='PASS' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('build-config','c16-bundle','c64-bundle','c16-native-om-manifest','c64-native-om-manifest','runner','om-runner','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device-id',type=int,default=0); parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--repetitions',type=int,default=30); parser.add_argument('--timeout',type=int,default=1800)
    return run(parser.parse_args())

if __name__=='__main__': raise SystemExit(main())
