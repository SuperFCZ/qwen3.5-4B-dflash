#!/usr/bin/env python3
"""Recover only missing W8 msprof; never execute FP16 or unprofiled benchmarks."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shlex
import subprocess

from common import sha256,write_json
from c16_diagnostic import STATUS,fp16_case,controls,diagnostic_execution,build_runner,attribute,save
from single_graph import load_reuse,verify_reuse
from c_profile import prepare_case,check_outputs,load
from fp16_w8_profile import replay_command
from c_profile_trace import collect
from profile_metadata import inspect
from profile_recovery_checks import snapshot,profiler_inventory,timing_check,output_metrics

FLAGS=['--ascendcl=on','--runtime-api=on','--task-time=on','--aicpu=on','--ai-core=on',
       '--aic-mode=task-based','--aic-metrics=PipeUtilization']
SUFFIXES=('actual-0.bin','actual-1.bin','benchmark-last.bin','postcheck.bin')


def saved_phase(case,folder,runner,device,phase,profile=False):
    command=[str(runner),str(device),str(case['om']),str(folder),str(folder/'inputs')]
    if load(folder/'command.json')!=command:raise ValueError('saved Runner/OM/phase command differs: '+str(folder))
    launch=load(folder/'launch.json')
    if profile:
        if not isinstance(launch,list) or len(launch)!=10 or launch[1:8]!=FLAGS or launch[8]!='--output='+str(folder/'profiler') or launch[9]!='--application='+shlex.join(command):
            raise ValueError('saved msprof command differs')
    elif launch!=command:raise ValueError('saved direct command differs')
    for side in ('inputs','outputs'):
        expected=''.join(t['name']+' '+t['dtype']+' '+' '.join(map(str,t['shape']))+'\n' for t in case['abi'][side])
        if (folder/(side+'.tsv')).read_text()!=expected:raise ValueError('saved ABI TSV differs')
    inputs=load(folder/'input-sha256.json')
    if set(inputs)!=set(case['inputs']):raise ValueError('saved input inventory differs')
    for name,source in case['inputs'].items():
        p=folder/'inputs'/(name+'.bin')
        if not p.is_file() or p.stat().st_size!=source.stat().st_size or sha256(p)!=inputs[name] or inputs[name]!=sha256(source):
            raise ValueError('saved phase input differs: '+str(p))
    if case['model']=='fp16':
        if (folder/'gear.txt').read_text().strip()!='16' or (folder/'diagnostic-phase.txt').read_text()!=phase:
            raise ValueError('saved diagnostic gear/phase differs')
        for name,reference in case['golden'].items():
            p=folder/'reference'/(name+'.bin')
            if not p.is_file() or p.stat().st_size!=reference.stat().st_size or sha256(p)!=sha256(reference):raise ValueError('saved Native reference differs')
        checked=diagnostic_execution(case,folder,phase)
        if phase!='numerics':
            timing_check(checked['execution'])
            for observation in checked['observations']:output_metrics(case,folder,observation)
            repeat=all(sha256(folder/(t['name']+'-'+s))==sha256(folder/(t['name']+'-'+SUFFIXES[0])) for t in case['abi']['outputs'] for s in SUFFIXES[1:])
            if checked['execution']['repeat_equal']!=repeat:raise ValueError('saved FP16 repeat flag differs from boundary bytes')
    else:
        execution=check_outputs(case,folder);timing_check(execution)
        checked=dict(scope=STATUS,raw_runner_execution=execution,own_reference_bitwise_equal=True)
    return checked


def validate(source,expected_files):
    path=source/'c16-diagnostic-report.json';report=load(path)
    if (report.get('status')!=STATUS or report.get('formal_acceptance')!='NOT_RUN_NOT_ELIGIBLE' or
        report.get('historical_fp16_status')!='FAIL_REPLAY_CHECKS' or report.get('gear')!=16 or
        report.get('end_to_end')!='NOT_RUN' or type(report.get('device_id')) is not int or report['device_id']<0):
        raise ValueError('not the original C16 diagnostic-only report')
    rows={c['model']:c for c in report['cases']}
    if len(report['cases'])!=2 or set(rows)!={'fp16','w8_a'}:raise ValueError('original model inventory differs')
    if rows['w8_a'].get('profile_status')!='NOT_RUN' or 'profile' in rows['w8_a'] or (source/'w8_a/profile').exists():
        raise ValueError('W8 profile is not wholly missing; refusing a duplicate or partial rerun')
    if not report.get('source_watch'):raise ValueError('original SHA256 watch inventory missing')
    for p,h in report['source_watch'].items():
        if not Path(p).is_file() or sha256(p)!=h:raise ValueError('original immutable artifact changed: '+p)
    old_args,reuse=load_reuse(Path(report['reuse']['source']))
    if reuse!=report['reuse']:raise ValueError('original OPP/Runner/environment provenance differs')
    for key,value in reuse['environment'].items():
        if report['environment'].get(key)!=value:raise ValueError('original execution environment differs from accepted OPP')
    for key in ('CANN_ROOT','ASCEND_HOME_PATH'):
        if report['environment'].get(key) is not None and os.environ.get(key)!=report['environment'][key]:
            raise ValueError('activate original CANN environment: '+key)
    build=load(source/'runner/build.json')
    if build!=report['build']:raise ValueError('original diagnostic build report differs')
    diagnostic,_=build_runner(source/'UNUSED',{},source/'runner/build.json')  # Mandatory reuse: never builds.
    if diagnostic!=source/'runner/c16_diagnostic_runner':raise ValueError('diagnostic runner not in original build directory')
    identity=rows['fp16']['model_identity']
    fp=fp16_case(Path(identity['replay']),Path(identity['deployment']).parent,Path(identity['deployment']))
    draft=load(Path(reuse['source'])/'draft-suite.json')
    candidates=[r for r in draft['cases'] if r['mode']=='a' and r['gear']==16 and r['status']=='PASS']
    if draft['status']!='PASS' or len(candidates)!=1:raise ValueError('accepted W8 C16 row missing')
    w=prepare_case(Path(reuse['source']),candidates[0],Path(reuse['runner']));w.update(model='w8_a',gear=16,dynamic=False);w['controls']=controls(w)
    for case in (fp,w):
        row=rows[case['model']]
        if row['gear']!=16 or row['abi']!=case['abi'] or row['controls']!=case['controls'] or row['om']!={'path':str(case['om']),'sha256':sha256(case['om'])}:
            raise ValueError('saved model ABI/OM/control identity differs')
        if row['input_sha256']!={n:sha256(p) for n,p in case['inputs'].items()}:raise ValueError('saved model input SHA differs')
        for phase in (('numerics','timing','profile') if case['model']=='fp16' else ('timing',)):
            checked=saved_phase(case,source/case['model']/phase,diagnostic if case['model']=='fp16' else Path(reuse['runner']),report['device_id'],phase,phase=='profile')
            if checked!=row[phase]:raise ValueError('saved phase evidence differs from original embedded report: '+case['model']+'/'+phase)
        execution=row['timing']['execution' if case['model']=='fp16' else 'raw_runner_execution']
        if row['unprofiled_latency']!=execution['timing']['execute_sync']:raise ValueError('original unprofiled latency was changed')
    files=profiler_inventory(source/'fp16/profile/profiler',expected_files)
    return report,fp,w,reuse,old_args,files


def run(args):
    source=args.source_evidence.resolve();root=args.output_dir.resolve()
    if root.is_relative_to(source) or source.is_relative_to(root):raise ValueError('recovery must use a separate new directory')
    root.mkdir(parents=True,exist_ok=False)
    report=dict(status=STATUS,execution='PREPARING_RECOVERY',device_execution='NOT_RUN',cases=[],
                recovery=dict(source=str(source),scope='offline FP16 parse + missing W8 msprof only',fp16_execution='NOT_RERUN',unprofiled_execution='NOT_RERUN'))
    save(report,root)
    before=None;reuse=None
    try:
        if os.environ.get('ASCEND310P_SIMULATION_ONLY')=='1' or os.environ.get('PROFILING_MODE') not in (None,'','false'):
            raise ValueError('real unwrapped profiling environment required')
        print('Recovery preflight: checking saved reports, bytes, timing and profiler inventory; no device execution.',flush=True)
        before=snapshot(source)
        original,fp,w,reuse,old_args,files=validate(source,args.expected_fp16_profiler_files)
        if root.is_relative_to(Path(old_args.native_base).resolve()) or root.is_relative_to(Path(reuse['source'])):
            raise ValueError('output overlaps frozen Native/accepted evidence')
        recovery=report['recovery'];report=copy.deepcopy(original)
        report.pop('error',None);report.pop('integrity_error',None)
        report.update(execution='RECOVERING',device_execution='NOT_RUN',recovery=recovery)
        report['recovery'].update(source_report_sha256=sha256(source/'c16-diagnostic-report.json'),
            original_execution=original.get('execution'),original_error=original.get('error'),
            frozen_inventory=before,fp16_profiler_integrity=files,
            validation='Existing reports/embedded observations, Runner sources+binary, all input/OM hashes, dynamic gears, saved boundary bytes/metrics and 5/30 statistics checked.',
            historical_hash_limit='Original run did not persist checksums for profiler and diagnostic output files. Recovery-time hashes do not establish retroactive authenticity. 30 numerical observations have no retained raw KV; embedded report equality is checked.',
            rmse_recheck='CPU float64 recomputation versus original long-double summation: relative 1e-12 only for saved RMSE consistency; tensor bit counts/max/p99 and W8 golden hashes remain exact.')
        rows={c['model']:c for c in report['cases']}
        for c in report['cases']:c['unprofiled_latency_origin']=dict(source=str(source/c['model']/'timing'),reused=True)
        fdir=root/'fp16';fdir.mkdir()
        print('Parsing existing FP16 graph/profile offline; FP16 and unprofiled timings will not be rerun.',flush=True)
        graph,heads,linear=inspect(fp,rows['fp16'],fdir,existing=source/'fp16/profile')
        rows['fp16']['profile_trace']=collect(source/'fp16/profile/profiler',graph)
        rows['fp16']['attribution']=attribute(rows['fp16']['profile_trace'],heads,linear)
        rows['fp16']['profile_status']='REUSED_OFFLINE';rows['fp16']['profile_origin']=str(source/'fp16/profile')
        save(report,root)
        if args.validate_only:report['execution']='VALIDATED_NO_DEVICE_EXECUTION';return
        env={**os.environ,**reuse['environment']};out=rows['w8_a'];folder=root/'w8_a'/'profile';folder.parent.mkdir()
        command=replay_command(w,folder,Path(reuse['runner']),report['device_id'])
        old_launch=load(source/'fp16/profile/launch.json')
        launch=[args.msprof or old_launch[0],*FLAGS,'--output='+str(folder/'profiler'),'--application='+shlex.join(command)]
        write_json(folder/'launch.json',launch)
        out['profile_status']='STARTED';report['device_execution']='W8_PROFILE_STARTED';save(report,root)
        print('Executing missing W8 A-only C16 msprof only: '+str(folder),flush=True)
        with (folder/'runner.log').open('w') as log:subprocess.run(launch,env=env,cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True)
        out['profile']=dict(scope=STATUS,raw_runner_execution=check_outputs(w,folder),own_reference_bitwise_equal=True)
        out['profile_origin']=str(folder)
        graph,heads,linear=inspect(w,out,folder,existing=source/'w8_a/profile')
        out['profile_trace']=collect(folder/'profiler',graph);out['attribution']=attribute(out['profile_trace'],heads,linear)
        count=len(snapshot(folder/'profiler',profiler=True))
        if not count:raise ValueError('new W8 profiler produced no files')
        out['profiler_inventory']=profiler_inventory(folder/'profiler',count)
        out['profiler_inventory']['integrity_scope']='New W8 capture SHA256 inventory recorded immediately after completion.'
        out['profile_status']='COLLECTED'
        for c in report['cases']:
            if c['unprofiled_latency']!=next(x for x in original['cases'] if x['model']==c['model'])['unprofiled_latency']:
                raise ValueError('recovery overwrote unprofiled timing')
        report.update(execution='RECOVERY_COMPLETED',device_execution='ONLY_W8_PROFILE_EXECUTED')
    except Exception as error:
        report['execution']='RECOVERY_FAILED';report['error']=f'{type(error).__name__}: {error}'
        for c in report['cases']:
            if c.get('profile_status')=='STARTED':c['profile_status']='FAILED'
        raise
    finally:
        try:
            if before is not None and snapshot(source)!=before:raise ValueError('original diagnostic directory changed during recovery')
            if reuse:verify_reuse(reuse)
            for p,h in report.get('source_watch',{}).items():
                if sha256(p)!=h:raise ValueError('original frozen artifact changed: '+p)
        except Exception as error:report['execution']='RECOVERY_FAILED';report['integrity_error']=str(error);raise
        finally:save(report,root)
    print('Recovery completed: '+str(root/'c16-diagnostic-report.json')+'; DIAGNOSTIC_ONLY.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-evidence',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--expected-fp16-profiler-files',type=int,required=True)
    p.add_argument('--msprof');p.add_argument('--validate-only',action='store_true')
    args=p.parse_args()
    if args.expected_fp16_profiler_files<1:p.error('expected profiler inventory must be positive')
    run(args)
