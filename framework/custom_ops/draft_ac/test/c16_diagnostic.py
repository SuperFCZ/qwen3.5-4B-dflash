#!/usr/bin/env python3
"""C16 ONLY: observe known-failing FP16 KV without granting numerical acceptance."""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess

from common import ROOT, sha256, write_json
from fp16_evidence import deployment, inventory, load, replay_payload
from single_graph import load_reuse, verify_reuse
from c_profile import prepare_case, check_outputs
from fp16_w8_profile import replay_command, graph_regions
from c_profile_graph import read_pbtxt, read_om_json, summarize
from c_profile_trace import collect, stats, union


STATUS='DIAGNOSTIC_ONLY'


def controls(case):
    result={}
    for name in ('start_position','valid_rows','proposal_count'):
        spec=next(t for t in case['abi']['inputs'] if t['name']==name)
        dtype=spec['dtype'];data=case['inputs'][name].read_bytes()
        if dtype not in ('int16','int64') or len(data)!=(2 if dtype=='int16' else 8):
            raise ValueError('scalar control ABI: '+name)
        result[name]=struct.unpack('<h' if dtype=='int16' else '<q',data)[0]
    if not 1<=result['valid_rows']<=16 or not 1<=result['proposal_count']<=15 or result['start_position']<0:
        raise ValueError('diagnostic requires legal C16 controls')
    for t in case['abi']['outputs']:
        if t['name']=='draft_top1': continue
        if t['dtype']!='float16' or len(t['shape'])!=4 or result['start_position']+16>t['shape'][2]:
            raise ValueError('C16 KV physical row bounds differ')
    return result


def fp16_case(report_path, search_root, selected=None):
    path=Path(report_path).resolve();r=load(path);request=path.parent/'request.json';binding=None
    if selected:
        candidates=[Path(selected).resolve()]
    elif request.is_file():
        binding=load(request)['deployment_manifest'];p=Path(binding['path'])
        if not p.is_absolute(): raise ValueError('original request Deployment path must be absolute')
        p=p.resolve()
        if sha256(p)!=binding['sha256']: raise ValueError('original FP16 request Deployment hash differs')
        candidates=[p]
    else:
        candidates=[p for p in inventory(search_root)[0] if any(g.get('name')=='draft' and g.get('om',{}).get('sha256')==r.get('draft_om_sha256') for g in load(p)['graphs'])]
    if len(candidates)!=1: raise ValueError('need one FP16 Deployment; use --fp16-deployment to resolve Chunk/MTP ambiguity')
    base=deployment(candidates[0]);g=base['graph']
    if (r.get('status')!='FAIL_REPLAY_CHECKS' or r.get('fake_acl') is not False or
        not base['dynamic'] or r.get('draft_om_sha256')!=g['om']['sha256'] or
        r.get('draft_static64_om_sha256') is not None or r.get('kv_output_audit',{}).get('context_rows')!=16):
        raise ValueError('requires the real known-failing FP16 dynamic C16 replay; no status rewriting')
    phases=r.get('phases',[])
    if len(phases)!=2 or {p.get('phase') for p in phases}!={'isolated','interleaved_prefill'}:
        raise ValueError('missing original replay phases')
    for p in phases:
        if p.get('iterations',0)<2 or any(p.get(k)!=0 for k in ('token_mismatch_iterations','input_restore_mismatch_iterations','input_mutation_iterations')):
            raise ValueError('historical Token/restore/readonly contract is not intact')
        for k in ('full_output_hash_mismatch_iterations','valid_kv_mismatch_iterations'):
            if type(p.get(k)) is not int or not 0<=p[k]<=p['iterations']: raise ValueError('invalid historical mismatch counts')
    case=replay_payload(base,path,16,r)
    case.update(source_status=r['status'],source_phases=phases,reference_role='historical_native_reference_NOT_accepted_golden')
    case['controls']=controls(case)
    if binding: case['watch'][str(request)]=sha256(request)
    return case


def build_runner(output,env,reuse=None):
    paths=['framework/custom_ops/draft_ac/test/fp16_dynamic_runner.cpp','framework/custom_ops/draft_ac/test/c16_diagnostic_metrics.h',
           'framework/custom_ops/draft_ac/test/c16_diagnostic/CMakeLists.txt','framework/custom_ops/draft_quant/test/runner_common.h']
    source={p:sha256(ROOT/p) for p in paths}
    if reuse:
        r=load(reuse);binary=Path(r['runner']['path']).resolve()
        if r.get('build_kind')!='c16_diagnostic_only' or r.get('sources')!=source or sha256(binary)!=r['runner']['sha256']:
            raise ValueError('not the exact diagnostic runner build; formal runner cannot be relabelled')
        return binary,r
    output.mkdir();cann=env.get('CANN_ROOT',env.get('ASCEND_HOME_PATH','/usr/local/Ascend/ascend-toolkit/latest'))
    commands=[['cmake','-S',str(ROOT/'framework/custom_ops/draft_ac/test/c16_diagnostic'),'-B',str(output),'-DCANN_ROOT='+cann],
              ['cmake','--build',str(output),'--parallel']]
    write_json(output/'commands.json',commands)
    with (output/'build.log').open('w') as log:
        for command in commands:subprocess.run(command,cwd=output,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    binary=output/'c16_diagnostic_runner'
    r=dict(status=STATUS,build_kind='c16_diagnostic_only',sources=source,runner=dict(path=str(binary),sha256=sha256(binary)),cann_root=cann)
    write_json(output/'build.json',r);return binary,r


def gear_check(case,folder):
    g=load(folder/'dynamic-gear.json')
    expected=[[dim if t['name']!='features' or axis!=1 else rows for t in case['abi']['inputs'] for axis,dim in enumerate(t['shape'])] for rows in (16,64)]
    if g.get('status')!='PASS' or g.get('selected_rows')!=16 or sorted(g.get('actual_gears',[]))!=sorted(expected):
        raise ValueError('actual NPU dynamic gear differs from complete ABI')
    return g


def diagnostic_execution(case,folder,phase):
    r=load(folder/'execution.json');gear=gear_check(case,folder)
    if r.get('status')!=STATUS or r.get('cpu_fallback') is not False or r.get('guards') is not True or r.get('readonly') is not True:
        raise ValueError('diagnostic execution status/guard/readonly contract differs')
    if type(r.get('repeat_equal')) is not bool: raise ValueError('repeat observation missing')
    count=30 if phase=='numerics' else 4
    observations=[json.loads(line) for line in (folder/'observations.jsonl').read_text().splitlines()]
    expected_names=[f'numerics-{i}' for i in range(30)] if phase=='numerics' else ['actual-0.bin','actual-1.bin','benchmark-last.bin','postcheck.bin']
    if r.get('observed_calls')!=count or [o.get('observation') for o in observations]!=expected_names:
        raise ValueError('diagnostic output observation inventory differs')
    names={t['name'] for t in case['abi']['outputs']}
    for o in observations:
        if o.get('status')!=STATUS or set(o.get('outputs',{}))!=names: raise ValueError('missing Token or KV observation')
        for t in case['abi']['outputs']:
            v=o['outputs'][t['name']]
            if t['name']=='draft_top1':
                if type(v.get('bitwise_equal')) is not bool or len(v.get('actual_ids',[]))!=math.prod(t['shape']): raise ValueError('invalid Token observation')
            else:
                if v.get('all',{}).get('elements')!=math.prod(t['shape']): raise ValueError('incomplete full KV observation')
                for key in ('all','valid_prefix','written_valid','written_padding','untouched_prefix_vs_input','untouched_tail_vs_input'):
                    m=v[key]
                    if m['elements']!=m['finite_pairs']+m['nonfinite_pairs'] or not 0<=m['bit_mismatches']<=m['elements']:
                        raise ValueError('invalid KV metric denominator')
    if phase=='numerics':
        if r.get('phase')!='numerics' or 'timing' in r: raise ValueError('numerics must not supply latency samples')
    else:
        t=r['timing'];samples=t['execute_sync']['samples_ms']
        if (t.get('protocol')!='continuous-v1' or t.get('warmup')!=5 or t.get('repetitions')!=30 or
            t.get('timed_tail_checked') is not True or len(samples)!=30 or any(not math.isfinite(x) or x<=0 for x in samples)):
            raise ValueError('diagnostic timing contract differs')
        if any(not math.isfinite(t['execute_sync'][k]) or t['execute_sync'][k]<=0 for k in ('median_ms','p95_ms')):
            raise ValueError('invalid diagnostic timing statistics')
    return dict(execution=r,actual_dynamic_gear=gear,observations=observations,
        observation_coverage='all 30 separate untimed numerical calls' if phase=='numerics' else
        '2 before + last timed output + 1 postcheck only; intermediate timed outputs NOT_OBSERVED')


def region(name,kind,heads,linear):
    k=kind.lower()
    if name in heads:return 'LM_HEAD'
    if any(x in k for x in ('transdata','transpose','cast','reshape')):return 'TRANSDATA_LAYOUT'
    if 'norm' in k:return 'NORM'
    if any(x in k for x in ('scatter','cacheupdate','indexcopy')):return 'KV_CACHE'
    if any(x in k for x in ('attention','softmax')):return 'ATTENTION'
    if 'weightquant' in k or 'dflashgroupquantlinear' in k:return 'W8_LINEAR'
    if name in linear and 'matmul' in k:return 'FP16_LINEAR'
    if any(x in k for x in ('memcpy','memset','atomicaddr','add','mul','gelu','silu')) and 'matmul' not in k:return 'OTHER_KNOWN'
    return 'UNKNOWN'


def attribute(profile,heads,linear):
    timeline=profile.get('selected_timeline')
    if not timeline:return dict(status='UNKNOWN',reason='no unique measured device timeline; do not divide CSV totals')
    calls=[]
    for call in timeline['iterations']:
        groups={}
        for task in call['tasks']:
            label=region(task['name'],task['op_type'],heads,linear);task['diagnostic_region']=label
            groups.setdefault(label,[]).append((task['start_us'],task['start_us']+task['duration_us']))
        calls.append(groups)
    labels=sorted({k for c in calls for k in c})
    return dict(status=STATUS,region_union_us={k:stats([union(c.get(k,[])) for c in calls]) for k in labels},
                host_and_device_metrics=timeline['metrics'],note='Region unions may overlap; never sum them as wall time. Host sync overlaps device work.')


def save(report,root):
    report['status']=STATUS;report['formal_acceptance']='NOT_RUN_NOT_ELIGIBLE';report['end_to_end']='NOT_RUN'
    latencies={c['model']:c['unprofiled_latency'] for c in report.get('cases',[]) if 'unprofiled_latency' in c}
    if set(latencies)=={'fp16','w8_a'} and latencies['fp16']['median_ms']>0:
        f,w=latencies['fp16'],latencies['w8_a']
        report['latency_comparison']=dict(status=STATUS,w8_minus_fp16_median_ms=w['median_ms']-f['median_ms'],
            w8_over_fp16_median_ratio=w['median_ms']/f['median_ms'],
            limitation='Different layer count/checkpoints; unresolved FP16 correctness. Not formal model speedup.')
    write_json(root/'c16-diagnostic-report.json',report)
    lines=['# C16 FP16 vs W8 A-only — DIAGNOSTIC_ONLY','',
           'FP16 原 FAIL_REPLAY_CHECKS 保留；本报告不授予 bitwise、正式性能或端到端验收。不同 checkpoint/层数，不做跨模型 Golden 比较。',
           '数值观察和延迟分开：30 次单独数值调用；5/30 continuous-v1 计时只检查四个边界输出，中间计时输出未逐次观察；msprof 独立运行。','',
           f"执行：{report['execution']}；NPU：{report.get('device_execution','NOT_RUN')}。"]
    if report.get('error'):lines+=['','错误：'+report['error']]
    if report.get('latency_comparison'):lines+=['','诊断性延迟对照：'+json.dumps(report['latency_comparison'],ensure_ascii=False)]
    for c in report.get('cases',[]):
        lines+=['',f"## {c['model']} / C16",'']
        t=c.get('unprofiled_latency')
        lines+=['- 诊断性无插桩 median/p95 ms：'+(f"{t['median_ms']:.4f} / {t['p95_ms']:.4f}" if t else 'NOT_RUN'),
                '- msprof：'+c.get('profile_status','NOT_RUN'),'- 输入控制：'+json.dumps(c['controls'])]
        for phase in ('numerics','timing','profile'):
            observations=c.get(phase,{}).get('observations',[])
            if observations:
                tokens=sum(not o['outputs']['draft_top1']['bitwise_equal'] for o in observations)
                kv=sum(v['all']['bit_mismatches']>0 for o in observations for n,v in o['outputs'].items() if n!='draft_top1')
                nonfinite=sum(v['all']['actual_nonfinite'] for o in observations for n,v in o['outputs'].items() if n!='draft_top1')
                untouched=sum(v[k]['bit_mismatches'] for o in observations for n,v in o['outputs'].items() if n!='draft_top1' for k in ('untouched_prefix_vs_input','untouched_tail_vs_input'))
                lines+=[f'- {phase}：{len(observations)} 次观测；Token 不一致 {tokens} 次；KV 张量不一致 {kv} 次；非有限元素累计 {nonfinite}；未修改区域 bit mismatch 累计 {untouched}。']
    lines+=['','全部 KV 的 ratio/max_abs/RMSE/p99_abs、非有限计数与有效区/未修改区明细见 JSON；空区域指标为 null。',
        '设备区域为重叠去重后的独立区间统计，区域之间仍可重叠，不能相加解释整图差值。UNKNOWN 不强行归类；尚未采集的设备数据为 NOT_RUN。']
    (root/'c16-diagnostic-analysis.md').write_text('\n'.join(lines)+'\n')


def run(args):
    if not args.diagnostic_only:raise ValueError('explicit --diagnostic-only required')
    source=args.w8_evidence.resolve();root=args.output_dir.resolve();rp=args.fp16_replay.resolve()
    if root.is_relative_to(source) or root.is_relative_to(rp.parent) or source.is_relative_to(root):raise ValueError('new output overlaps frozen evidence')
    root.mkdir(parents=True,exist_ok=False)
    report=dict(status=STATUS,execution='PREPARING',device_execution='NOT_RUN',cases=[],
                historical_fp16_status='NOT_VERIFIED',gear=16,device_id=args.device_id,
                perf_scope='diagnostic latency only: FP16 numerical correctness unresolved')
    watch={};reuse=None
    try:
        if os.environ.get('ASCEND310P_SIMULATION_ONLY')=='1' or os.environ.get('PROFILING_MODE') not in (None,'','false'):
            raise ValueError('real device outside inherited profiling required')
        fp=fp16_case(rp,args.search_root,args.fp16_deployment);watch.update(fp['watch'])
        report['historical_fp16_status']=fp['source_status']
        old,reuse=load_reuse(source)
        if root.is_relative_to(Path(old.native_base).resolve()):raise ValueError('output overlaps Native Base')
        suite=load(source/'suite.json');draft=load(source/'draft-suite.json')
        if suite.get('status')!='PASS' or draft.get('status')!='PASS':raise ValueError('W8 accepted full Draft evidence missing')
        rows=[r for r in draft['cases'] if r['mode']=='a' and r['gear']==16 and r['status']=='PASS']
        if len(rows)!=1:raise ValueError('need one accepted W8 A-only C16 case')
        runner=Path(reuse['runner']);w=prepare_case(source,rows[0],runner)
        w.update(model='w8_a',gear=16,dynamic=False);w['controls']=controls(w)
        watch.update(w['watch']);watch.update({str(source/n):sha256(source/n) for n in ('suite.json','draft-suite.json')})
        cases=[fp,w];env={**os.environ,**reuse['environment']};report['reuse']=reuse
        report['environment']={k:env.get(k) for k in ('ASCEND_CUSTOM_OPP_PATH','LD_LIBRARY_PATH','CANN_ROOT','ASCEND_HOME_PATH')}
        report['source_watch']=watch;report['fp16_source_phases']=fp['source_phases']
        for c in cases:
            report['cases'].append(dict(model=c['model'],gear=16,controls=c['controls'],abi=c['abi'],
                om=dict(path=str(c['om']),sha256=sha256(c['om'])),input_sha256={n:sha256(p) for n,p in c['inputs'].items()},
                model_identity=c.get('identity',dict(evidence=str(source),mode='a',artifacts=rows[0]['artifacts'])),
                historical_latency=c.get('historical'),profile_status='NOT_RUN'))
        save(report,root)
        if args.prepare_only:report['execution']='PREPARED_NOT_EXECUTED';return
        for name,cmd in (('npu-smi',['npu-smi','info']),('msprof-version',[args.msprof,'--version'])):
            with (root/(name+'.log')).open('w') as log:
                result=subprocess.run(cmd,env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=name=='npu-smi')
            report.setdefault('environment_commands',{})[name]=dict(command=cmd,returncode=result.returncode)
        diagnostic,b=build_runner(root/'runner',env,args.diagnostic_build);report['build']=b;watch[str(diagnostic)]=sha256(diagnostic)
        for c in cases:(root/c['model']).mkdir()
        def execute(c,out,phase,profile=False):
            folder=root/c['model']/phase;binary=diagnostic if c['model']=='fp16' else runner
            command=replay_command(c,folder,binary,args.device_id)
            if c['model']=='fp16':
                (folder/'diagnostic-phase.txt').write_text(phase)
                refs=folder/'reference';refs.mkdir()
                for name,p in c['golden'].items():(refs/(name+'.bin')).symlink_to(p)
            launch=command
            if profile:
                out['profile_status']='STARTED'
                launch=[args.msprof,'--ascendcl=on','--runtime-api=on','--task-time=on','--aicpu=on','--ai-core=on',
                        '--aic-mode=task-based','--aic-metrics=PipeUtilization','--output='+str(folder/'profiler'),'--application='+shlex.join(command)]
            write_json(folder/'launch.json',launch);report['execution']='RUNNING';report['device_execution']='STARTED'
            out[phase]=dict(scope=STATUS,execution='RUNNING',directory=str(folder));save(report,root)
            try:
                with (folder/'runner.log').open('w') as log:subprocess.run(launch,env=env,cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True)
            except Exception as error:
                out[phase].update(execution='FAILED',error=str(error),log=str(folder/'runner.log'))
                partial=folder/'observations.jsonl'
                if partial.is_file():
                    out[phase]['partial_observations_path']=str(partial)
                raise
            if c['model']=='fp16':out[phase]=diagnostic_execution(c,folder,phase)
            else:out[phase]=dict(scope=STATUS,raw_runner_execution=check_outputs(c,folder),own_reference_bitwise_equal=True)
            if phase=='timing':
                r=out[phase]['execution'] if c['model']=='fp16' else out[phase]['raw_runner_execution']
                out['unprofiled_latency']=r['timing']['execute_sync']
            save(report,root);return folder
        execute(fp,report['cases'][0],'numerics')
        for c,out in zip(cases,report['cases']):execute(c,out,'timing')
        for c,out in zip(cases,report['cases']):
            folder=execute(c,out,'profile',True);nodes=[]
            pb=[p for p in c['payloads'] if p.name=='dynamo.pbtxt']
            if len(pb)==1:
                try:nodes=read_pbtxt(pb[0]);out['pre_atc_graph']=summarize(nodes,'pre_atc')
                except (ValueError,OSError) as e:out['graph_error']=str(e)
            atc=shutil.which(args.atc)
            if atc:
                target=folder/'om.json';cmd=[atc,'--mode=1','--om='+str(c['om']),'--json='+str(target)]
                write_json(folder/'inspect-command.json',cmd)
                with (folder/'inspect.log').open('w') as log:
                    done=subprocess.run(cmd,env={**env,'ASCEND_WORK_PATH':str(folder)},cwd=folder,stdout=log,stderr=subprocess.STDOUT)
                try:
                    if done.returncode:raise ValueError('OM inspection unavailable')
                    nodes=read_om_json(target);out['compiled_graph']=summarize(nodes,'compiled_om')
                except (ValueError,OSError) as e:out['inspection_error']=str(e)
            graph,heads,linear=graph_regions(nodes) if nodes else ({},set(),set())
            out['profile_trace']=collect(folder/'profiler',graph);out['attribution']=attribute(out['profile_trace'],heads,linear)
            out['profile_status']='COLLECTED';save(report,root)
        report['execution']='COMPLETED';report['device_execution']='EXECUTED_DIAGNOSTIC_ONLY'
    except Exception as e:
        report['execution']='BLOCKED' if report['device_execution']=='NOT_RUN' else 'FAILED';report['error']=f'{type(e).__name__}: {e}'
        for case in report['cases']:
            if case['profile_status']=='STARTED':case['profile_status']='FAILED'
            for phase in ('numerics','timing','profile'):
                if case.get(phase,{}).get('execution')=='RUNNING':case[phase]['execution']='FAILED'
        raise
    finally:
        try:
            for p,h in watch.items():
                if sha256(p)!=h:raise ValueError('frozen source changed: '+p)
            if reuse:verify_reuse(reuse)
        except Exception as e:report['execution']='FAILED';report['integrity_error']=str(e);raise
        finally:save(report,root)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostic-only',action='store_true',required=True)
    p.add_argument('--w8-evidence',type=Path,required=True);p.add_argument('--fp16-replay',type=Path,required=True)
    p.add_argument('--search-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--fp16-deployment',type=Path);p.add_argument('--diagnostic-build',type=Path)
    p.add_argument('--device-id',type=int,default=0);p.add_argument('--msprof',default='msprof');p.add_argument('--atc',default='atc')
    p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    if args.device_id<0:p.error('device ID must be nonnegative')
    run(args)
