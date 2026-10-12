#!/usr/bin/env python3
"""Strict K128/K256 pairing after representative and full A4 gates."""
import argparse
import json
from pathlib import Path
import a4_contract
import a51_fallback
from a2_common import checked_file,sha256,write_json
from a3_launch import load_build_config
from compare_a5 import run as compare_full,projection
from run_a51 import REPRESENTATIVES


def check_protocol(report):
    for case in report['cases']:
        for kind in ('custom','native_om'):
            timing=case['executions'][kind]['timing']
            if timing.get('warmup')!=5 or timing.get('repetitions')!=30 or timing.get('protocol')!='continuous-v1':
                raise ValueError('A5.1 requires warmup=5, repetitions=30, continuous-v1')


def unwrap(path,k_tile):
    report=json.loads(path.read_text()); config=load_build_config(Path(report['build_config_path']))
    if (report.get('abi')!='dflash-a51-stage-v1' or report.get('status')!='PASS' or
        report.get('full_a4_regression')!='PASS' or report.get('build')!=config or config.get('k_tile')!=k_tile or
        config.get('scale_mode')!='broadcast' or config.get('kv_m80_mode')!='a-ub' or
        sha256(report['build_config_path'])!=report['build_config_sha256'] or
        set(report.get('checks',{}))!={'fallback-synthetic','representative-c16','representative-c64','full'} or
        any(c.get('status')!='PASS' for c in report['checks'].values())):
        raise ValueError('need current A5.1 six-shape + representative + full A4 PASS with Broadcast/A-UB')
    fallback=checked_file(path.parent,report['checks']['fallback-synthetic']['summary'])
    plans=a51_fallback.collect(fallback.parent,config,report['device_id'])
    if report.get('launches',{}).get('fallback-synthetic')!=plans:
        raise ValueError('stored fallback launch evidence differs from raw logs')
    full=checked_file(path.parent,report['checks']['full']['summary'])
    full_report=json.loads(full.read_text())
    if full_report['build']!=config: raise ValueError('full build differs from representative build')
    reps={}
    for gear in (16,64):
        child=checked_file(path.parent,report['checks'][f'representative-c{gear}']['summary'])
        value=json.loads(child.read_text())
        check_protocol(value)
        full_child=checked_file(full.parent,full_report['checks'][f'c{gear}']['summary'])
        check_protocol(json.loads(full_child.read_text()))
        if value.get('status')!='PASS' or value.get('context_rows')!=gear or tuple(c['name'] for c in value['cases'])!=REPRESENTATIVES:
            raise ValueError('incomplete representative evidence')
        for key in ('bundle','native_om_manifest'):
            if sha256(value[key])!=value[key+'_sha256'] or value[key+'_sha256']!=full_report['inputs'][f'c{gear}'][key]['sha256']:
                raise ValueError('representative/full capture identity differs')
        reps[gear]=(value,child.parent)
    return report,full,reps


def run(left,right,output):
    if output.exists(): raise FileExistsError('use a fresh paired result')
    a,af,ar=unwrap(left,128); b,bf,br=unwrap(right,256)
    # Representative statistics are separate and are never added to the 26-call total.
    rows=[]
    for gear in (16,64):
        x,xroot=ar[gear]; y,yroot=br[gear]
        for key in ('bundle_sha256','native_om_manifest_sha256','device_id'):
            if x[key]!=y[key]: raise ValueError('representative inputs differ')
        for old,new in zip(x['cases'],y['cases']):
            shape=a4_contract.case_shape(next(c for c in a4_contract.cases(gear) if c['name']==old['name']))
            row=projection(old,new,xroot,yroot,shape,a,b,True)
            rows.append(dict(gear=f'c{gear}',**row))
    if any(row['status']!='PASS' for row in rows):
        write_json(output,dict(status='FAIL',representatives=rows,full_pair='NOT_RUN')); return 1
    status=compare_full(af,bf,output,'k-tile',(Path(a['build_config_path']),Path(b['build_config_path'])))
    report=json.loads(output.read_text()); report['representatives']=rows
    report['fallback_synthetic']='PASS'
    report['fallback_launches']=[a['launches']['fallback-synthetic'],b['launches']['fallback-synthetic']]
    report['staged_runs']=[dict(path=str(p),sha256=sha256(p)) for p in (left,right)]
    write_json(output,report); return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','candidate','output'): parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(); return run(args.baseline.resolve(),args.candidate.resolve(),args.output.resolve())
if __name__=='__main__': raise SystemExit(main())
