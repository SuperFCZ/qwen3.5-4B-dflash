#!/usr/bin/env python3
"""Same-Cube-recipe custom FP16 versus frozen W8 Off; no W8 optimization."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
from types import SimpleNamespace
from tiled_contract import HERE,SHAPE,OUTPUTS,identity,verify,write_json,pack_tiles,accounting,custom_plan,require_same_plan,verify_execution,stats,compare
from fp16_gate_common import pipe_report
from fp16_gate_evidence import accepted
from experiment import command,child_environment,invoke as invoke_off
from tiled_evidence import reuse_prior
from tiled_build import build

ORDERS=(('off','custom_fp16'),('custom_fp16','off'))


def invoke_custom(folder,source,packed,runner,env,config,device,timed,profiler=None):
    folder.mkdir(parents=True);case=folder/'case';case.mkdir()
    files={'x.bin':source['files']['x.bin'],'w_fp16_tiles.bin':packed['file']}
    for name,item in files.items():(case/name).symlink_to(verify(item))
    argv=[str(runner),str(device),str(case)]+(['5','30','--continuous'] if timed else [])
    write_json(case/'command.json',argv)
    if profiler:
        launcher=folder/'replay.sh';launcher.write_text('#!/usr/bin/env bash\nset -eo pipefail\nexec '+shlex.join(argv)+' > '+shlex.quote(str(case/'runner.log'))+' 2>&1\n')
        command([profiler,'--ai-core=on','--aic-mode=task-based','--aic-metrics=PipeUtilization',f'--output={folder/"profiler"}','--application='+shlex.join(['bash',str(launcher)])],folder/'profile.log',env,3600)
    else:command(argv,case/'runner.log',env)
    ex=json.loads((case/'execution.json').read_text());verify_execution(ex,device,timed)
    plan=custom_plan(case/'runner.log',config,ex['workspace_bytes'])
    names=OUTPUTS if timed else OUTPUTS[:2];actual=[(case/n).read_bytes() for n in names]
    golden=verify(source['golden']).read_bytes()
    comparisons=[compare(golden,a,SimpleNamespace(m=16,n=19456)) for a in actual]
    match=all(d['finite'] and d['bitwise_equal'] and d['max_ulp']==0 for d in comparisons)
    result=dict(status='BITWISE_EQUAL' if match else 'FAIL',comparisons=comparisons,execution=ex,plan=plan,
        outputs={n:identity(case/n) for n in names},profiled=bool(profiler),formal_latency=timed and not profiler)
    if timed and not profiler:result['timing']=stats(ex['timing']['execute_sync']['samples_ms'])
    if profiler:result['pipe_counters']=pipe_report(folder/'profiler')
    write_json(folder/'result.json',result)
    if not match or any(a!=actual[0] for a in actual):raise ValueError('Custom FP16 strict Golden/repeat mismatch; comparison saved, no timing promotion')
    if profiler and not result['pipe_counters']['files']:raise ValueError('missing msprof CSV artifacts')
    for item in files.values():verify(item)
    return result,actual[0]


def run(args):
    root=args.output_dir.absolute();forbidden=[HERE.parent.resolve(),args.prior_report.resolve().parent,args.a51_evidence.resolve().parent,args.prefetch_evidence.resolve().parent]
    if root.resolve()!=root or any(root==p or p in root.parents for p in forbidden):raise ValueError('use new independent output outside frozen artifacts/repo')
    root.mkdir(parents=True,exist_ok=False);root.chmod(0o700)
    report=dict(status='RUNNING',phase='reuse_evidence',shape=SHAPE,checks={},rounds=[],profiles={},
        cpu_preparation='NOT_RUN',cann_build='NOT_RUN',npu_correctness='NOT_RUN',performance='NOT_RUN',
        dequant_only='NOT_RUN_SECOND_STAGE',simulator='NOT_RUN',full_draft_token_kv='NOT_RUN',
        path_accounting=accounting(),interpretation='W8 Off - Custom FP16 is net data-preparation path plus related memory/synchronization difference, NOT pure Dequant time')
    summary=root/'comparison.json';watched={};write_json(summary,report)
    try:
        source=accepted(args.a51_evidence,args.prefetch_evidence);watched.update(source['watched'])
        weights,history,oldfiles=reuse_prior(args.prior_report,source);watched.update(oldfiles);report['historical']=history
        if history['device']['id']!=args.device_id:raise ValueError('current device id differs from historical comparison')
        for p in HERE.iterdir():
            if p.is_file():item=identity(p);watched[item['path']]=item
        report['phase']='cpu_tile_pack';write_json(summary,report)
        packed=pack_tiles(weights['files']['w_fp16_nk.bin'],root/'w_fp16_tiles.bin');write_json(root/'packed-weights.json',packed)
        watched[packed['file']['path']]=packed['file'];report.update(cpu_preparation='BITWISE_PACK_VERIFIED',packed_weights=packed)
        if args.prepare_only:report.update(status='PREPARED_NOT_RUN',phase='complete')
        else:
            report.update(phase='build_custom_fp16',cann_build='RUNNING');write_json(summary,report)
            cfg=source['modes']['off']['config'];runner,env,built=build(args,root/'custom',cfg);report['build']=built;report['cann_build']='BUILD_INSTALL_COMPLETED'
            for item in built['files']:watched[item['path']]=item
            off_runner=Path(source['modes']['off']['runner']);off_env=child_environment(args.cann_root,Path(source['modes']['off']['opp'])/'vendors/customize')
            report['phase']='untimed_correctness';write_json(summary,report)
            off,off_output=invoke_off('off',root/'checks/off',source,weights,off_runner,off_env,args.device_id,False)
            report['checks']['off']=off;write_json(summary,report)
            custom,custom_output=invoke_custom(root/'checks/custom_fp16',source,packed,runner,env,cfg,args.device_id,False)
            report['checks']['custom_fp16']=custom;write_json(summary,report)
            require_same_plan(off['launch'],custom['plan']['allocation_plan'])
            if off_output!=custom_output:raise ValueError('current Off/Custom outputs differ')
            report.update(npu_correctness='UNTIMED_BITWISE_PLAN_GATE_PASSED',phase='crossed_unprofiled_timing');write_json(summary,report)
            for index,order in enumerate(ORDERS):
                round=dict(index=index,order=order,results={});report['rounds'].append(round)
                for mode in order:
                    folder=root/'timing'/str(index)/mode
                    if mode=='off':
                        result,output=invoke_off(mode,folder,source,weights,off_runner,off_env,args.device_id,True)
                        require_same_plan(off['launch'],result['launch'])
                    else:
                        result,output=invoke_custom(folder,source,packed,runner,env,cfg,args.device_id,True)
                        require_same_plan(off['launch'],result['plan']['allocation_plan'])
                    if output!=off_output:raise ValueError('timed-tail/cross-order Golden drift')
                    round['results'][mode]=result;write_json(summary,report)
            report['timing_summary']={m:stats([v for r in report['rounds'] for v in r['results'][m]['timing']['samples_ms']]) for m in ('off','custom_fp16')}
            a,b=report['timing_summary']['off'],report['timing_summary']['custom_fp16']
            report['net_data_preparation_delta']=dict(median_ms=a['median_ms']-b['median_ms'],
                paired_round_median_ms=[r['results']['off']['timing']['median_ms']-r['results']['custom_fp16']['timing']['median_ms'] for r in report['rounds']],
                definition='W8 Off - Custom FP16: net preparation + related memory/sync; includes 10->1 GM copies and 16640->32768 weight bytes per tile')
            old=history['timing_summary'];report['historical_context']=dict(current_off_over_old_off=a['median_ms']/old['off']['median_ms'],
                custom_minus_old_native_ms=b['median_ms']-old['fp16']['median_ms'],causal_attribution='UNRESOLVED: Native uses another tiling/runtime and another measurement epoch; not pure custom Cube or synchronization cost')
            report.update(performance='MEASURED_UNPROFILED',phase='separate_msprof');write_json(summary,report)
            profiler=shutil.which(args.msprof)
            if not profiler:raise ValueError('msprof unavailable')
            item=identity(profiler);watched[item['path']]=item
            for mode in ('off','custom_fp16'):
                if mode=='off':
                    result,output=invoke_off(mode,root/'profiling'/mode,source,weights,off_runner,off_env,args.device_id,True,profiler)
                    require_same_plan(off['launch'],result['launch'])
                else:
                    result,output=invoke_custom(root/'profiling'/mode,source,packed,runner,env,cfg,args.device_id,True,profiler)
                    require_same_plan(off['launch'],result['plan']['allocation_plan'])
                if output!=off_output:raise ValueError('profiling output changed')
                report['profiles'][mode]=result;write_json(summary,report)
            report.update(status='EXPERIMENT_COMPLETED_DIAGNOSTIC_ONLY',phase='complete',npu_correctness='ALL_REQUESTED_CHECKS_PASSED')
    except Exception as error:
        if report['cann_build']=='RUNNING':report['cann_build']='FAILED_OR_INCOMPLETE'
        if report['phase']=='untimed_correctness':report['npu_correctness']='GATE_FAILED_OR_INCOMPLETE'
        report.update(status='BLOCKED',error=f'{type(error).__name__}: {error}')
    finally:
        try:
            for item in watched.values():verify(item)
        except Exception as error:report.update(status='BLOCKED',integrity_error=str(error))
        report['watched']=watched;write_json(summary,report)
        lines=['# Custom FP16 same Cube recipe / W8 Off','', 'Status: '+report['status']+'; phase: '+report['phase'],
            'CPU preparation: '+report['cpu_preparation']+'; CANN build: '+report['cann_build']+'; NPU correctness: '+report['npu_correctness'],'',
            '| Current implementation | median ms | p95 ms | samples |','| --- | ---: | ---: | ---: |']
        for name,t in report.get('timing_summary',{}).items():lines += [f"| {name} | {t['median_ms']:.6f} | {t['p95_ms']:.6f} | {len(t['samples_ms'])} |"]
        if report.get('net_data_preparation_delta'):lines += ['',json.dumps(report['net_data_preparation_delta'],indent=2)]
        lines += ['',report['interpretation'],'Cube recipe, allocation budget and all recorded tiling/launch fields must match current Off. Frozen Off has no full serialized TCubeTiling-byte dump; this is not a claim of an independently compared opaque tiling blob.',
            'FP16 reads more bytes with fewer GM instructions; dequant vector work and six PIPE_V barriers per K disappear. Same MTE2_S/V pairs, three PIPE_ALL per K, post-output fence, startup Duplicate/PIPE_V, UB reservations and SDK call/accumulator structure remain.',
            'Native FP16 and Raw are historical references only. Their timing/profile evidence is reused; neither is executed here. Different epochs cannot prove pure Cube/synchronization cost.',
            'Profiled Task Duration/pipe counters are separate from 5/30 continuous-v1 timing. Do not sum overlapping pipe times.',
            'Dequant-only second stage / Simulator / full Draft: NOT_RUN.',report.get('error','')]
        (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(report['status']+': '+str(summary));return int(report['status']=='BLOCKED')


def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prior-report',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/gate-same-weight-fp16-w8-20261021-014911/comparison.json'))
    p.add_argument('--a51-evidence',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/a51-fallback-20261012-063342/suite.json'))
    p.add_argument('--prefetch-evidence',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/gate-raw-prefetch-reverse-20261020-025853/suite.json'))
    p.add_argument('--cann-root',type=Path,default=Path(os.environ.get('CANN_ROOT','/usr/local/Ascend/cann-9.0.0')))
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--device-id',type=int,default=0);p.add_argument('--msprof',default='msprof');p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();a.cann_root=a.cann_root.resolve()
    if a.device_id<0:p.error('invalid device')
    return a

if __name__=='__main__':raise SystemExit(run(parse()))
