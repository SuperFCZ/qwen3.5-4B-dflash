#!/usr/bin/env python3
"""Fresh two-operator release acceptance. Historical PASS never replaces a new NPU run."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from common import HERE,ABI,checked_file,load_capture,sha256,write_json
from conversion_probe import environment
from run_suite import validate_execution,validate_native
from head_opp_preflight import check_install,OP_KERNELS
import build_config

CASES=[f'real-c{g}-m{m:02d}' for g in (16,64) for m in (1,8,15)]
SYNTHETIC=('zero','tile_tie','partition_tie','last_vocab','rounded_tie','rounded_up','row_identity',
           'all_negative','overflow_inf','nan_first','signed_zero_underflow')
ALL_CASES={f'real-c{g}-m{m:02d}' for g in (16,64) for m in range(1,16)}|{'synthetic-'+n for n in SYNTHETIC}


def frozen_contract():
    contract=json.loads((HERE/'release_contract.json').read_text())
    if contract.get('abi')!='dflash-head-release-v1': raise ValueError('unknown release contract')
    for name,digest in contract['source_sha256'].items():
        if sha256(HERE/name)!=digest: raise ValueError(f'frozen algorithm/ABI source changed: {name}')
    return contract


def require_full_suite(data,contract):
    if data.get('abi')!=ABI: raise ValueError('Head suite ABI differs')
    if data.get('status')!='PASS' or data.get('full_41_case_acceptance')!='PASS' or data.get('custom_correctness')!='PASS':
        raise ValueError('fresh/full custom 41-case PASS required')
    cases=data.get('cases',[])
    if len(cases)!=41 or {c['name'] for c in cases}!=ALL_CASES or any(c.get('status')!='PASS' for c in cases):
        raise ValueError('incomplete or failed 41 cases')
    config=data['build']
    if (config.get('conversion'),config.get('top1'),config.get('profile'))!=('compare_select','vector','none'):
        raise ValueError('expected accepted Vector algorithm, not an observer/fallback')
    for name,digest in contract['source_sha256'].items():
        if config.get('source_sha256',{}).get(name)!=digest: raise ValueError(f'evidence algorithm differs: {name}')
    for case in cases:
        if set(case.get('executions',{}))!={'native-logits','native-top1','audit','fused'}: raise ValueError('missing execution mode')
        for mode,value in case['executions'].items():
            validate_execution(value,mode,case['m'],value['device_id'],*( (0,0) if mode=='audit' else (5,30) ))


def load_full_suite(path,contract):
    path=Path(path).resolve(); data=json.loads(path.read_text()); require_full_suite(data,contract)
    for case in data['cases']:
        comparison=json.loads(checked_file(path.parent,case['comparison']).read_text())
        checks=comparison.get('comparisons',{})
        if comparison.get('status')!='PASS' or not {'full_fp16_logits','custom_ids','audit_ids'}<=checks.keys() or not all(c.get('equal') is True for c in checks.values()):
            raise ValueError(f"comparison is not strictly bitwise PASS: {case['name']}")
    return data


def stats(execution):
    values=execution['timing']['execute_sync']['samples_ms']
    if len(values)!=30 or any(not math.isfinite(x) or x<=0 for x in values): raise ValueError('invalid continuous-v1 samples')
    return dict(median_ms=statistics.median(values),p95_ms=sorted(values)[28],samples_ms=values)


def compare_performance(current,baseline,contract,max_ratio=1.10):
    if not math.isfinite(max_ratio) or max_ratio<1: raise ValueError('invalid regression threshold')
    require_full_suite(current,contract); require_full_suite(baseline,contract)
    for field in ('capture_sha256','native_manifest_sha256'):
        if current.get(field)!=baseline.get(field) or not current.get(field): raise ValueError('capture/Native golden identity differs')
    rows=[]; concerns=[]
    for name in CASES:
        new=next(c for c in current['cases'] if c['name']==name); old=next(c for c in baseline['cases'] if c['name']==name)
        for tensor in ('hidden','weight'):
            if new['inputs'][tensor]['sha256']!=old['inputs'][tensor]['sha256']: raise ValueError(f'{name}: input changed')
        for key in ('base_k','single_k','partitions','core_limit','user_ub_bytes','matmul_ub_bytes','user_workspace_bytes','system_workspace_bytes'):
            if new['launches']['fused'][key]!=old['launches']['fused'][key]: raise ValueError(f'{name}: tiling/workspace changed: {key}')
        if new['executions']['fused']['device_id']!=old['executions']['fused']['device_id']: raise ValueError('different device IDs')
        measured={label:stats(case['executions'][mode]) for label,case,mode in (
            ('custom',new,'fused'),('native_top1',new,'native-top1'),('native_head',new,'native-logits'),
            ('baseline_vector',old,'fused'),('baseline_native_top1',old,'native-top1'))}
        ratios={metric:measured['custom'][metric]/measured['baseline_vector'][metric] for metric in ('median_ms','p95_ms')}
        row=dict(name=name,timing=measured,current_to_baseline=ratios,
                 native_to_custom_median=measured['native_top1']['median_ms']/measured['custom']['median_ms'])
        rows.append(row)
        for metric,ratio in ratios.items():
            if ratio>max_ratio: concerns.append(dict(name=name,reason='vector regression',metric=metric,ratio=ratio))
        if row['native_to_custom_median']<1: concerns.append(dict(name=name,reason='custom median slower than Native'))
        drift=measured['native_top1']['median_ms']/measured['baseline_native_top1']['median_ms']
        if drift>max_ratio or drift<1/max_ratio: concerns.append(dict(name=name,reason='Native control drift; review device conditions',ratio=drift))
    return dict(status='REVIEW_REQUIRED' if concerns else 'PASS',threshold=max_ratio,
                scope='performance tripwire only; never changes bitwise correctness',cases=rows,concerns=concerns)


def call(cmd,log,env):
    write_json(log.with_suffix('.command.json'),cmd)
    with log.open('w') as output: subprocess.run(cmd,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)


def build_environment(args,state_path):
    env=os.environ.copy()
    # Exercise the actual defaults, independent of stale experiment environment.
    for key in ('HEAD_CONVERSION','HEAD_TOP1','HEAD_PROFILE_STAGE','HEAD_DIAGNOSTICS','HEAD_TOP1_PROBE','DFLASH_HEAD_CORE_LIMIT'):
        env.pop(key,None)
    env.update(HEAD_CAPTURE=str(args.capture),HEAD_NATIVE_MANIFEST=str(args.native_manifest),HEAD_BUILD_ONLY='1',
               HEAD_NATIVE_ONLY='0',HEAD_BUILD_STATE=str(state_path),MODEL_PYTHON=sys.executable,DEVICE_ID=str(args.device_id))
    return env


def acceptance_markdown(report):
    text=['# C release acceptance', '',f"Status: **{report['status']}**", '',
          f"New build correctness: {report['correctness']}; performance: {report['performance']['status']}.",
          'Full Draft/Decode and A/C integration: NOT_RUN.', '',
          '| Case | Custom median/p95 ms | Native Top1 median/p95 ms | Previous Vector median/p95 ms |',
          '|---|---:|---:|---:|']
    for row in report['performance']['cases']:
        t=row['timing']; fmt=lambda key:f"{t[key]['median_ms']:.4f} / {t[key]['p95_ms']:.4f}"
        text.append(f"| {row['name']} | {fmt('custom')} | {fmt('native_top1')} | {fmt('baseline_vector')} |")
    text+=['', 'Configuration: compare_select + Vector Top1, profile=none; Fused/Audit only.',
           f"Build state: {report['build_state']}",f"Fresh 41-case suite: {report['suite']['path']}",
           f"Historical comparison: {report['baseline']['path']}",
           f"Frozen algorithm commit: {report['contract']['algorithm_commit']}",
           f"Source commit: {report['source_commit']}", '',
           'See release.json for exact inputs, binaries, launch configuration, raw samples and review flags.']
    return '\n'.join(text)+'\n'


def run(args):
    root=args.output_dir.resolve(); root.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',correctness='NOT_RUN',performance=dict(status='NOT_RUN'),
                new_release_acceptance='NOT_RUN',full_draft_validation='NOT_RUN',ac_integration='NOT_RUN')
    summary=root/'release.json'
    write_json(summary,report)
    try:
        contract=frozen_contract(); report['contract']=contract
        load_capture(args.capture); validate_native(args.native_manifest,sha256(args.capture))
        baseline_path=args.baseline_dir/'vector-full/suite.json'
        baseline=load_full_suite(baseline_path,contract)
        report['baseline']=dict(path=str(baseline_path),sha256=sha256(baseline_path),scope='historical comparison only')
        report['environment']=environment(os.environ.get('CANN_ROOT',os.environ.get('ASCEND_HOME_PATH','/usr/local/Ascend/ascend-toolkit/latest')),args.device_id)
        if report['environment']['status']!='PASS': raise ValueError(report['environment']['errors'])
        state_path=root/'build-state.json'; report['build_state']=str(state_path)
        call(['bash',str(HERE/'run_server.sh')],root/'build.log',build_environment(args,state_path))
        state=json.loads(state_path.read_text()); config=build_config.load(state['config'])
        if config['build_kind']!='release' or config['top1']!='vector' or config['core_limit']!=0 or config['operators']!=list(OP_KERNELS):
            raise ValueError('not a default release build')
        if 'top1_runner' in state: raise ValueError('release unexpectedly depends on a probe runner')
        project=Path(state['build_root'])/'CustomOp'
        if sha256(project/'op_kernel/head_config.h')!=config['header_sha256']: raise ValueError('Host/Kernel configuration differs')
        packages=list((project/'build_out').glob('custom_opp_*.run'))
        if len(packages)!=1: raise ValueError('expected one release installer')
        paths=packages+[project/p for p in ('CMakeLists.txt','op_host/CMakeLists.txt','op_kernel/CMakeLists.txt',
            'op_host/head_config.h','op_kernel/head_config.h')]+[HERE/'test/CMakeLists.txt']
        report['build_artifacts']=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha256(p)) for p in paths]
        report['build']=config; report['source_commit']=(Path(state['run_root'])/'source-commit.txt').read_text().strip()
        report['installed_opp']=check_install(state['install'],release=True)
        write_json(summary,report)
        env={**os.environ,**state['environment']}
        cmd=[sys.executable,str(HERE/'test/run_suite.py'),'--capture',str(args.capture),'--native-manifest',str(args.native_manifest),
             '--native-runner',state['native_runner'],'--runner',state['runner'],'--build-config',state['config'],
             '--output-dir',str(root/'data'),'--device-id',str(args.device_id),'--warmup','5','--repetitions','30','--timeout',str(args.timeout)]
        report['correctness']='RUNNING'; write_json(summary,report)
        call(cmd,root/'server.log',env)
        path=root/'data/suite.json'; current=load_full_suite(path,contract)
        if current['build']!=config: raise ValueError('suite build identity changed')
        report['correctness']='PASS'; report['suite']=dict(path=str(path),sha256=sha256(path))
        report['performance']=compare_performance(current,baseline,contract,args.max_regression_ratio)
        if build_config.load(state['config'])!=config or frozen_contract()!=contract: raise ValueError('source/config drift')
        if any(sha256(p['path'])!=p['sha256'] for p in report['build_artifacts']): raise ValueError('build artifacts changed')
        if sha256(baseline_path)!=report['baseline']['sha256']: raise ValueError('historical suite changed')
        if check_install(state['install'],release=True)!=report['installed_opp']: raise ValueError('installed OPP changed')
        report['status']=report['performance']['status']; report['new_release_acceptance']=report['status']
        (root/'acceptance.md').write_text(acceptance_markdown(report))
    except Exception as e:
        report.update(status='FAIL',error=f'{type(e).__name__}: {e}')
        if report['correctness']=='RUNNING': report['correctness']='FAIL'
        report['new_release_acceptance']='FAIL'
    write_json(summary,report)
    print(f"{report['status']}: {summary}; fresh correctness={report['correctness']}; full Draft NOT_RUN",flush=True)
    return 0 if report['status']=='PASS' else 2 if report['status']=='REVIEW_REQUIRED' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('capture','native-manifest','baseline-dir','output-dir'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device-id',type=int,default=0); p.add_argument('--timeout',type=int,default=1800)
    p.add_argument('--max-regression-ratio',type=float,default=1.10)
    a=p.parse_args()
    for name in ('capture','native_manifest','baseline_dir'): setattr(a,name,getattr(a,name).resolve())
    raise SystemExit(run(a))
