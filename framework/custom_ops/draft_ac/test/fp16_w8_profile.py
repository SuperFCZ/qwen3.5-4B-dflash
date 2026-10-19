#!/usr/bin/env python3
"""Frozen six-layer FP16 versus five-layer W8 A-only: no export/build/Target."""
import argparse
import collections
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from datetime import datetime, timezone

from common import ROOT, sha256, write_json
from single_graph import load_reuse, verify_reuse
from c_profile import load, prepare_case, timing
from c_profile_graph import read_pbtxt, read_om_json, summarize, edge_name, category
from c_profile_trace import collect, stats, union
from fp16_evidence import resolve, deployment
from fp16_capture import inspect as inspect_fp16, capture as capture_fp16

WIDTHS={'float16':2,'int16':2,'int64':8,'float32':4}


def region(name, kind, head_names, linear_names):
    """Conservative labels: unnamed MatMul may be Attention, never assume Linear."""
    key=re.sub('[^a-z0-9]','',(name+' '+kind).lower())
    if name in head_names: return 'LM Head'
    if 'transdata' in key: return 'TransData'
    if 'memcpy' in key or 'cast' in key or 'transpose' in key: return 'Other conversion'
    if any(s in key for s in ('scatter','cacheupdate','indexcopy')): return 'KV Cache update'
    if 'norm' in key: return 'Norm'
    if any(s in kind.lower() for s in ('attention','softmax','flashattn')): return 'Attention'
    if name in linear_names or 'dflashgroupquantlinear' in key or 'weightquantbatchmatmul' in key:
        return 'Linear'
    if any(s in key for s in ('attention','softmax','flashattn')): return 'Attention'
    if 'matmul' in key: return 'Unmapped MatMul (Linear/Attention unknown)'
    return 'Other/unmapped'


def graph_regions(nodes):
    summary=summarize(nodes,'attribution');table={n['name']:n for n in nodes}
    heads={n['name'] for n in summary['head_related_nodes']}; linear=set()
    for n in nodes:
        if category(n['type']) not in ('matmul','custom_a') or n['name'] in heads: continue
        producers=[table.get(edge_name(x),{}) for x in n['inputs']]
        if category(n['type'])=='custom_a' or any(p.get('type') in ('Const','FileConstant') for p in producers):
            linear.add(n['name'])
    return summary,heads,linear


def attribution(profile, heads, linear):
    t=profile.get('selected_timeline')
    if not t: return dict(status='UNRESOLVED',reason='no unique 38-call Execute/Sync timeline; CSV totals are not per-call latency')
    sums=collections.defaultdict(list);occupancy=collections.defaultdict(list);calls=[]
    for call in t['iterations']:
        groups=collections.defaultdict(list)
        for task in call['tasks']:
            label=region(task['name'],task['op_type'],heads,linear)
            task['draft_region']=label; groups[label].append(task)
        calls.append(groups)
    labels=sorted({label for groups in calls for label in groups})
    for groups in calls:
        for label in labels:
            tasks=groups.get(label,[])
            sums[label].append(sum(x['duration_us'] for x in tasks))
            occupancy[label].append(union((x['start_us'],x['start_us']+x['duration_us']) for x in tasks))
    return dict(status='MAPPED_WITH_EXPLICIT_UNKNOWNS',region_task_sum_us={k:stats(v) for k,v in sums.items()},
                region_occupancy_us={k:stats(v) for k,v in occupancy.items()},
                note='30 timed calls only. Region occupancies may overlap; task sums and Host sync are not additive wall time.')


def validate_outputs(case, output):
    report=load(output/'execution.json');t=timing(report)
    if case.get('dynamic'):
        gear=load(output/'dynamic-gear.json')
        expected=[[dim if t['name']!='features' or j!=1 else rows for t in case['abi']['inputs'] for j,dim in enumerate(t['shape'])] for rows in (16,64)]
        if gear.get('status')!='PASS' or gear.get('selected_rows')!=case['gear'] or sorted(gear.get('actual_gears',[]))!=sorted(expected):
            raise ValueError('actual NPU dynamic gear evidence differs')
        report['dynamic_gear']=gear
    if any(not isinstance(x,(float,int)) or not math.isfinite(x) or x<=0 for x in t['samples_ms']):
        raise ValueError('nonfinite/nonpositive timing samples')
    if report.get('row_permutation_checked') is not False: raise ValueError('full Draft runner scope differs')
    for spec in case['abi']['outputs']:
        name=spec['name'];expected=case['golden'][name];size=math.prod(spec['shape'])*WIDTHS[spec['dtype']]
        for suffix in ('actual-0.bin','actual-1.bin','benchmark-last.bin','postcheck.bin'):
            p=output/(name+'-'+suffix)
            if p.stat().st_size!=size or sha256(p)!=sha256(expected):
                raise ValueError(case['model']+' output differs from its own frozen Golden: '+name)
    return report


def replay_command(case, folder, runner, device):
    folder.mkdir();inputs=folder/'inputs';inputs.mkdir()
    if case.get('dynamic'): (folder/'gear.txt').write_text(str(case['gear'])+'\n')
    for side in ('inputs','outputs'):
        (folder/(side+'.tsv')).write_text(''.join(t['name']+' '+t['dtype']+' '+' '.join(map(str,t['shape']))+'\n' for t in case['abi'][side]))
    for t in case['abi']['inputs']:
        path=case['inputs'][t['name']];size=math.prod(t['shape'])*WIDTHS[t['dtype']]
        if path.stat().st_size==size: (inputs/(t['name']+'.bin')).symlink_to(path)
        elif case['model']=='fp16' and t['name']=='features' and path.stat().st_size>size:
            # Historical snapshots store the maximum gear; only the bound C16
            # prefix is consumed by the C16 static OM. Preserve original bytes.
            with path.open('rb') as stream: payload=stream.read(size)
            (inputs/'features.bin').write_bytes(payload)
        else: raise ValueError('input allocation differs from selected gear: '+str(path))
    command=[str(runner),str(device),str(case['om']),str(folder),str(inputs)]
    write_json(folder/'command.json',command)
    write_json(folder/'input-sha256.json',{t['name']:sha256(inputs/(t['name']+'.bin')) for t in case['abi']['inputs']})
    return command


def finish(report, root):
    report['comparisons']=[]
    for gear in (16,64):
        items={c['model']:c for c in report['cases'] if c['gear']==gear}
        if set(items)=={'fp16','w8_a'} and all('formal_latency' in c for c in items.values()):
            f,w=(items[k]['formal_latency'] for k in ('fp16','w8_a'))
            report['comparisons'].append(dict(gear=gear,fp16_median_ms=f['median_ms'],w8_a_median_ms=w['median_ms'],
                fp16_p95_ms=f['p95_ms'],w8_a_p95_ms=w['p95_ms'],fp16_over_w8_a=f['median_ms']/w['median_ms'],
                interpretation='different models/inputs: descriptive system latency ratio, NOT quantization-only speedup'))
            fa=items['fp16'].get('attribution',{}).get('region_occupancy_us',{})
            wa=items['w8_a'].get('attribution',{}).get('region_occupancy_us',{})
            if fa and wa:
                report['comparisons'][-1]['profile_w8_minus_fp16_region_mean_us']={
                    k:wa.get(k,{}).get('mean',0)-fa.get(k,{}).get('mean',0) for k in sorted(set(fa)|set(wa))}
                report['comparisons'][-1]['causal_explanation']='UNPROVEN: overlapping regions and different models; not an additive latency decomposition'
    priorities=[]
    for c in report['cases']:
        data=c.get('attribution',{}).get('region_occupancy_us',{})
        for label,value in sorted(data.items(),key=lambda item:-item[1]['mean'])[:3]:
            priorities.append(dict(model=c['model'],gear=c['gear'],region=label,profile_occupancy_us=value['mean'],
                action=('resolve missing graph/task mapping first' if 'unmapped' in label.lower() or 'unknown' in label.lower()
                        else 'inspect compiled layout and critical-path overlap before selecting optimization'),
                predicted_speedup='UNPROVEN'))
    report['optimization_priorities']=priorities or [{'action':'NOT_RUN: collect same-device formal timings and mapped device timelines first','predicted_speedup':'UNPROVEN'}]
    write_json(root/'fp16-w8-report.json',report)
    lines=['# FP16 vs W8 A-only 完整 Draft 对比','',f"状态：{report['status']}；设备验收：{report['device_status']}；端到端：NOT_RUN。",
        'FP16 六层、W8 五层，checkpoint、特征宽度和输入不同；只对各自 Golden 做 bitwise 验证。比值不代表纯 W8 量化收益。',
        '正式延迟：无 msprof，continuous-v1，warmup=5、repetitions=30，Execute+Synchronize；与插桩后的设备归因分开。','',
        '| Model | Gear | Formal median/p95 ms | Profiling |','|---|---|---|---|']
    for c in report['cases']:
        t=c.get('formal_latency');value=f"{t['median_ms']:.4f} / {t['p95_ms']:.4f}" if t else 'NOT_RUN'
        lines.append(f"| {c['model']} | C{c['gear']} | {value} | {c.get('profiling_status','NOT_RUN')} |")
    if report.get('error'): lines+=['','阻塞/错误：`'+report['error']+'`']
    if report.get('fp16_preflight'):
        lines+=['','## 轻量 FP16 兼容检查（设备未执行）','']
        for c in report['fp16_preflight']['candidates']:
            lines+=['- '+c['path']+': '+c['status']+'；Manifest '+c.get('manifest_status','BLOCKED')+'；shape '+c.get('shape_mode','UNKNOWN')]
            if c.get('error'): lines+=['  错误：'+c['error']]
            lines+=['  Replay：'+json.dumps(c.get('replays',{}),ensure_ascii=False),
                    '  采集来源：'+json.dumps(c.get('capture_requests',{}),ensure_ascii=False),
                    '  待验证历史请求候选：'+json.dumps(c.get('possible_original_requests',[]),ensure_ascii=False)]
    for comparison in report['comparisons']:
        lines+=['', '- '+json.dumps(comparison,ensure_ascii=False)]
    lines+=['','## 设备归因与优化优先级','',
        '逐 Kernel、描述符、布局、CSV 原始字段、30 次窗口及 Host launch/sync 明细见 JSON。未映射 MatMul 不直接归类为 Linear；Attention/Norm/KV 按可见算子标签识别，融合内部不可分解。',
        '区域 occupancy 与 task sum 可能重叠，不能相加解释整图差值。FP16 与 W8 层数不同，差值同时包含模型、checkpoint、形状、编译优化与精度的影响。',
        '可选方向：Linear 需核查常量权重布局与搬运；LM Head 应比较 immutable-weight 整图，不能套用 runtime-weight 独立结果；Attention/Norm/KV/TransData 先看设备占比和关键路径。没有设备证据不承诺收益。']
    lines += ['- '+json.dumps(p,ensure_ascii=False) for p in report['optimization_priorities']]
    for c in report['cases']:
        a=c.get('attribution',{}).get('region_occupancy_us',{})
        if a:
            lines+=['',f"### {c['model']} C{c['gear']} 插桩区间 occupancy 均值（μs）",'']
            lines+=['- '+k+': '+f"{v['mean']:.3f}" for k,v in a.items()]
    lines+=['','`tools/profile_om.py` 使用生成路径/首次 stage 窗口。此入口沿用 msprof CLI 采集方式和工具 provenance，但不调用该生成入口，避免重新运行 Target/Prefill 或将采集开销当正式延迟。',
        '动态 FP16 使用独立 C++ gear adapter 查询并验证实际 C16/C64 档位，原 OM 不重编译。新输入/Golden 仅可来自原 FP16 Runner 的真实 Prefill/DebugDraftReplay；缺失来源时 BLOCKED。']
    (root/'fp16-w8-analysis.md').write_text('\n'.join(lines)+'\n')


def run(args):
    root=args.output_dir.resolve();source=args.w8_evidence.resolve();search=args.search_root.resolve()
    if root.is_relative_to(source) or source.is_relative_to(root) or root==search:
        raise ValueError('new output must not overlap frozen W8 evidence/search root')
    root.mkdir(parents=True,exist_ok=False)
    report=dict(status='PREPARING',device_status='NOT_RUN',end_to_end='NOT_RUN',cases=[],
        protocol=dict(name='continuous-v1',warmup=5,repetitions=30,device_id=args.device_id),
        cross_model_bitwise='NOT_APPLICABLE',target_verify='NOT_RUN',
        provenance=dict(profile_om=dict(path=str(ROOT/'tools/profile_om.py'),sha256=sha256(ROOT/'tools/profile_om.py'))))
    watch={};reuse=None
    try:
        if os.environ.get('ASCEND310P_SIMULATION_ONLY')=='1' or os.environ.get('PROFILING_MODE') not in (None,'','false'):
            raise ValueError('formal latency requires real NPU and no inherited profiling wrapper')
        requests={g:p for g,p in ((16,getattr(args,'fp16_c16_request',None)),(64,getattr(args,'fp16_c64_request',None))) if p}
        if args.prepare_only:
            report['fp16_preflight']=inspect_fp16(search,args.fp16_deployment,requests)
            candidates=report['fp16_preflight']['candidates']
            ready=len(candidates)==1 and candidates[0]['status']=='READY_FOR_REPLAY'
            report['status']='PREFLIGHT_ONLY' if ready else 'BLOCKED'
            report['w8_preflight']='NOT_RUN: lightweight FP16 compatibility inspection only'
            print('FP16 preflight: '+report['status']+'; see '+str(root/'fp16-w8-report.json'),flush=True)
            return ready
        old,reuse=load_reuse(source)
        if root.is_relative_to(Path(old.native_base).resolve()): raise ValueError('output overlaps Native Base')
        report['reuse']=reuse
        suite=load(source/'suite.json');draft=load(source/'draft-suite.json')
        if (suite.get('status')!='PASS' or suite.get('single_op_graph')!='PASS'
            or not (suite.get('draft_only')=='PASS' or suite.get('full_draft')=='PASS')
            or draft.get('status')!='PASS' or len(draft['cases'])!=8
            or sorted((r['gear'],r['mode']) for r in draft['cases'])!=sorted((g,m) for g in (16,64) for m in ('off','a','c','ac'))
            or any(r['status']!='PASS' for r in draft['cases'])):
            raise ValueError('need frozen accepted eight-mode W8 evidence')
        watch.update({str(source/n):sha256(source/n) for n in ('suite.json','draft-suite.json')})
        runner=Path(reuse['runner']);cases=[]
        replay_paths={g:p for g,p in ((16,args.fp16_c16_replay),(64,args.fp16_c64_replay)) if p}
        if getattr(args,'capture_fp16',False):
            if not args.fp16_deployment or set(requests)!={16,64}:
                raise ValueError('capture requires explicit FP16 Deployment and original C16/C64 request.json/profile-request.json')
            if replay_paths: raise ValueError('capture and reuse replay inputs are mutually exclusive')
            base=deployment(args.fp16_deployment)
            report['fp16_capture']='RUNNING';finish(report,root)
            replay_paths=capture_fp16(base,requests,root/'fp16-capture',args.device_id,{**os.environ,**reuse['environment']})
            report['fp16_capture']='PASS';report['target_prefill_for_capture']='EXECUTED_ORIGINAL_OM; not Target/Verify acceptance'
            for req in load(root/'fp16-capture/requests.json').values(): watch.update(req['watch'])
        fp16,discovery=resolve(search,args.fp16_deployment,replay_paths)
        for c in fp16:
            if any(root.is_relative_to(Path(c['identity'][k]).parent) for k in ('deployment','replay')):
                raise ValueError('output must be outside FP16 deployment/replay directories')
        report['fp16_discovery']=discovery;cases+=fp16
        for row in draft['cases']:
            if row['mode']!='a': continue
            c=prepare_case(source,row,runner)
            c.update(model='w8_a',gear=row['gear'],layers=5,
                golden={t['name']:c['folder']/'native'/(t['name']+'-actual-0.bin') for t in c['abi']['outputs']},
                identity=dict(evidence=str(source),mode='a',artifacts=row['artifacts']))
            cases.append(c)
        for c in cases:
            for p in c['watch']:
                if Path(p).is_relative_to(root) and not Path(p).is_relative_to(root/'fp16-capture'): raise ValueError('output overlaps source artifact')
            watch.update(c['watch'])
        report['source_watch']=watch
        env=os.environ.copy();env.update(reuse['environment'])
        report['environment']={k:env.get(k) for k in ('ASCEND_CUSTOM_OPP_PATH','LD_LIBRARY_PATH','ASCEND_HOME_PATH','ASCEND_TOOLKIT_HOME')}
        report['environment']['same_process_environment_for_all_cases']=True
        # Alternate model order by gear, with all unprofiled calls before msprof.
        cases.sort(key=lambda c:(c['gear'], c['model']!='fp16' if c['gear']==16 else c['model']=='fp16'))
        for c in cases:
            report['cases'].append(dict(model=c['model'],gear=c['gear'],layers=c['layers'],identity=c['identity'],
                abi=c['abi'],om=dict(path=str(c['om']),sha256=sha256(c['om'])),
                input_sha256={n:sha256(p) for n,p in c['inputs'].items()},
                historical_latency=c.get('historical'),formal_status='NOT_RUN',profiling_status='NOT_RUN'))
        finish(report,root)
        for label,command in (('npu-smi',['npu-smi','info']),('msprof-version',[args.msprof,'--version'])):
            with (root/(label+'.log')).open('w') as log:
                result=subprocess.run(command,env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT)
            report.setdefault('environment_commands',{})[label]=dict(command=command,returncode=result.returncode)
            if label=='npu-smi' and result.returncode: raise ValueError('npu-smi failed; same-device environment unverified')
        report['environment']['toolkit_versions']=[]
        for key in ('ASCEND_HOME_PATH','ASCEND_TOOLKIT_HOME'):
            if not env.get(key): continue
            for rel in ('version.cfg','version.info','compiler/version.info'):
                p=Path(env[key])/rel
                if p.is_file() and p.stat().st_size<1024*1024:
                    report['environment']['toolkit_versions'].append(dict(path=str(p.resolve()),sha256=sha256(p),text=p.read_text(errors='replace')))
        dynamic_runner=None
        if any(c.get('dynamic') for c in cases):
            from fp16_dynamic import runner as dynamic_build
            dynamic_runner,build=dynamic_build(root/'fp16-runner',env,getattr(args,'fp16_dynamic_build',None))
            report['fp16_dynamic_build']=build;watch[str(dynamic_runner)]=sha256(dynamic_runner)
        for c,out in zip(cases,report['cases']):
            folder=root/f"{c['model']}-c{c['gear']}";folder.mkdir()
            command=replay_command(c,folder/'formal',dynamic_runner if c.get('dynamic') else runner,args.device_id)
            out['formal_status']='RUNNING';out['formal_started_utc']=datetime.now(timezone.utc).isoformat()
            report['device_status']='RUNNING';finish(report,root)
            with (folder/'formal.log').open('w') as log:
                subprocess.run(command,env=env,cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True)
            out['formal_execution']=validate_outputs(c,folder/'formal');out['formal_latency']=timing(out['formal_execution'])
            out['own_golden_bitwise']='PASS';out['formal_status']='PASS';finish(report,root)
        report['device_status']='FORMAL_REPLAY_PASS';finish(report,root)
        profiler=shutil.which(args.msprof);atc=shutil.which(args.atc)
        if not profiler: raise ValueError('msprof missing; formal results retained, profiling NOT_RUN')
        for c,out in zip(cases,report['cases']):
            folder=root/f"{c['model']}-c{c['gear']}";nodes=[]
            pb=[p for p in c['payloads'] if p.name=='dynamo.pbtxt']
            if len(pb)==1:
                try: nodes=read_pbtxt(pb[0]);out['pre_atc_graph']=summarize(nodes,'pre_atc')
                except (ValueError,OSError) as error: out['pre_atc_graph']=dict(status='UNPARSED',error=str(error))
            if atc:
                target=folder/'om-graph.json';command=[atc,'--mode=1','--om='+str(c['om']),'--json='+str(target)]
                write_json(folder/'atc-inspect-command.json',command)
                with (folder/'atc-inspect.log').open('w') as log:
                    result=subprocess.run(command,env={**env,'ASCEND_WORK_PATH':str(folder)},cwd=folder,stdout=log,stderr=subprocess.STDOUT)
                try:
                    if result.returncode: raise ValueError('OM inspection failed')
                    nodes=read_om_json(target);out['compiled_graph']=summarize(nodes,'compiled_om')
                except (ValueError,OSError) as error: out['compiled_graph']=dict(status='UNAVAILABLE',error=str(error))
            graph,heads,linear=graph_regions(nodes) if nodes else ({},set(),set())
            out['region_mapping']=dict(head_names=sorted(heads),linear_names=sorted(linear),source='compiled_om' if out.get('compiled_graph',{}).get('status')=='PARSED' else 'pre_atc_or_missing')
            command=replay_command(c,folder/'profiled',dynamic_runner if c.get('dynamic') else runner,args.device_id)
            launch=[profiler,'--ascendcl=on','--runtime-api=on','--task-time=on','--aicpu=on','--ai-core=on',
                '--aic-mode=task-based','--aic-metrics=PipeUtilization','--output='+str(folder/'profiler'),'--application='+shlex.join(command)]
            write_json(folder/'msprof-command.json',launch)
            out['profiling_status']='RUNNING';finish(report,root)
            with (folder/'profile.log').open('w') as log:
                subprocess.run(launch,env=env,cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True)
            out['profiled_execution']=validate_outputs(c,folder/'profiled')
            out['profiling_status']='COMMAND_COMPLETED';out['profile']=collect(folder/'profiler',graph)
            out['attribution']=attribution(out['profile'],heads,linear);finish(report,root)
        report['status']='COLLECTED';report['root_cause']='UNPROVEN: compare mapped timelines; model differences confound quantization attribution'
    except Exception as error:
        report['status']='BLOCKED' if report['device_status']=='NOT_RUN' else 'FAIL'
        report['error']=f'{type(error).__name__}: {error}'
        if report.get('fp16_capture')=='RUNNING': report['fp16_capture']='FAIL'
        for c in report['cases']:
            for key in ('formal_status','profiling_status'):
                if c[key]=='RUNNING': c[key]='FAIL'
        raise
    finally:
        try:
            for p,digest in watch.items():
                if sha256(p)!=digest: raise ValueError('read-only source changed: '+p)
            if reuse: verify_reuse(reuse)
        except Exception as error:
            report['status']='FAIL';report['integrity_error']=str(error)
            raise
        finally: finish(report,root)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--w8-evidence',type=Path,required=True);p.add_argument('--search-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--fp16-deployment',type=Path)
    p.add_argument('--fp16-c16-replay',type=Path);p.add_argument('--fp16-c64-replay',type=Path)
    p.add_argument('--device-id',type=int,default=0);p.add_argument('--msprof',default='msprof');p.add_argument('--atc',default='atc')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--capture-fp16',action='store_true',help='explicitly run original FP16 Prefill/DebugDraftReplay to create new snapshots')
    p.add_argument('--fp16-c16-request',type=Path);p.add_argument('--fp16-c64-request',type=Path)
    p.add_argument('--fp16-dynamic-build',type=Path,help='reuse matching adapter build.json; otherwise build only the isolated C++ adapter')
    args=p.parse_args()
    if args.device_id<0: p.error('device must be nonnegative')
    if run(args) is False: raise SystemExit(2)


if __name__=='__main__': main()
