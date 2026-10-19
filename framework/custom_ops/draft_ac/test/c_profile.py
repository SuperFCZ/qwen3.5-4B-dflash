#!/usr/bin/env python3
"""Read-only frozen Draft attribution: no model load/export/OM compilation."""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from common import checked_file,record,sha256,write_json
from single_graph import load_reuse,verify_reuse
from c_profile_graph import read_pbtxt,read_om_json,summarize
from c_profile_trace import collect

MODES=('off','a','c','ac')
OUTPUTS=['draft_top1']+[f'd{i}_{kind}' for i in range(5) for kind in ('key','value')]


def load(path): return json.loads(Path(path).read_text())


def timing(report):
    t=report.get('timing',{})
    if (report.get('status')!='PASS' or report.get('cpu_fallback') is not False or
        any(report.get(k) is not True for k in ('guards','readonly','repeat')) or
        t.get('protocol')!='continuous-v1' or t.get('warmup')!=5 or t.get('repetitions')!=30 or
        t.get('timed_tail_checked') is not True): raise ValueError('runner correctness/timing contract differs')
    result=t['execute_sync']
    if len(result['samples_ms'])!=30 or any(not isinstance(result[k],(float,int)) or not math.isfinite(result[k]) or result[k]<0 for k in ('median_ms','p95_ms')):
        raise ValueError('invalid frozen timing samples')
    return result


def prepare_case(source,row,runner):
    folder=source/f"draft-{row['mode']}-c{row['gear']}"
    paths={key:checked_file(folder,value) for key,value in row['artifacts'].items()}
    air=load(paths['air']);dep=load(paths['deployment'])
    if air.get('status')!='PASS' or dep.get('status')!='PASS' or len(air['graphs'])!=1 or len(dep['graphs'])!=1:
        raise ValueError('missing accepted AIR/OM')
    graph=air['graphs'][0];compiled=dep['graphs'][0]
    if sha256(paths['air'])!=dep['air_manifest']['sha256'] or (paths['deployment'].parent/dep['air_manifest']['path']).resolve()!=paths['air']:
        raise ValueError('OM not bound to original AIR')
    if checked_file(paths['deployment'].parent,compiled['om'])!=paths['om']:
        raise ValueError('OM path/hash differs')
    if compiled['metadata']!=graph['metadata'] or compiled['runtime_input_abi']!=graph['runtime_input_abi'] or compiled['custom_op_audit']!=graph['custom_op_audit']:
        raise ValueError('AIR/OM audits differ')
    control=graph['metadata']['draft_ac']; abi=graph['metadata']['tensor_abi']
    if control['mode']!=row['mode'] or control['gear']!=row['gear'] or len(abi['inputs'])!=15 or [t['name'] for t in abi['outputs']]!=OUTPUTS:
        raise ValueError('frozen mode/gear/15-input/11-output ABI differs')
    expected={'DFlashGroupQuantLinear':26 if row['mode'] in ('a','ac') else 0,
              'DFlashDraftLmHeadTop1':1 if row['mode'] in ('c','ac') else 0,'DFlashDraftLmHeadTop1Audit':0}
    audit=graph['runtime_input_abi']['draft_ac']
    observed={r['ge_op_type']:r['ge_node_occurrences'] for r in graph['custom_op_audit']}
    if audit.get('status')!='PASS' or audit['counts']!=expected or any(observed.get(k,0)!=v for k,v in expected.items()) or observed.get('WeightQuantBatchMatmulV2',0)+expected['DFlashGroupQuantLinear']!=26:
        raise ValueError('missing original strict GE/Const audits')
    watch={str(p):sha256(p) for p in paths.values()}
    payloads=[checked_file(paths['air'].parent,p) for p in graph['payload_files']]
    for p in payloads: watch[str(p)]=sha256(p)
    inputs={name:folder/'input'/(name+'.bin') for name in row['inputs']}
    if list(inputs)!=[t['name'] for t in abi['inputs']]:
        # JSON object ordering is not an ABI; compare names, then use ABI order.
        if set(inputs)!={t['name'] for t in abi['inputs']}: raise ValueError('input inventory differs')
    for name,path in inputs.items():
        if sha256(path)!=row['inputs'][name]: raise ValueError('frozen input changed')
        watch[str(path)]=sha256(path)
    comparison=load(folder/'comparison.json')
    if comparison.get('status')!='PASS' or set(comparison['outputs'])!=set(OUTPUTS): raise ValueError('historical bitwise gate missing')
    candidate=folder/'candidate';native=folder/'native'
    for side,directory in (('custom',candidate),('native',native)):
        if row[side]!=load(directory/'execution.json'): raise ValueError('frozen execution report changed')
        timing(row[side])
        for t in abi['outputs']:
            name=t['name'];golden=comparison['outputs'][name]
            expected_hash=golden['actual_sha256' if side=='custom' else 'expected_sha256']
            if golden.get('equal') is not True or golden['actual_sha256']!=golden['expected_sha256']:
                raise ValueError('historical numerical comparison failed')
            for suffix in ('actual-0.bin','actual-1.bin','benchmark-last.bin','postcheck.bin'):
                p=directory/(name+'-'+suffix)
                if p.stat().st_size!=golden['bytes'] or sha256(p)!=expected_hash: raise ValueError('frozen output changed')
                watch[str(p)]=expected_hash
    command=load(candidate/'command.json')
    if len(command)!=5 or Path(command[0]).resolve()!=runner or Path(command[2]).resolve()!=paths['om'] or Path(command[3]).resolve()!=candidate:
        raise ValueError('frozen runner command differs')
    for name in ('inputs.tsv','outputs.tsv'):
        rows=abi['inputs' if name=='inputs.tsv' else 'outputs']
        expected_tsv=''.join(t['name']+' '+t['dtype']+' '+' '.join(map(str,t['shape']))+'\n' for t in rows)
        if (candidate/name).read_text()!=expected_tsv: raise ValueError('frozen runner TSV differs from ABI')
    for p in (folder/'comparison.json',candidate/'command.json',candidate/'execution.json',native/'execution.json',candidate/'inputs.tsv',candidate/'outputs.tsv'):
        watch[str(p)]=sha256(p)
    return dict(folder=folder,om=paths['om'],abi=abi,inputs=inputs,command=command,
                payloads=payloads,watch=watch,historical=timing(row['custom']),historical_native=timing(row['native']))


def check_outputs(case,output):
    execution=load(output/'execution.json');timing(execution)
    if execution.get('row_permutation_checked') is not False: raise ValueError('full Draft replay contract changed')
    for t in case['abi']['outputs']:
        name=t['name'];accepted=sha256(case['folder']/'native'/(name+'-actual-0.bin'))
        for suffix in ('actual-0.bin','actual-1.bin','benchmark-last.bin','postcheck.bin'):
            if sha256(output/(name+'-'+suffix))!=accepted: raise ValueError('profiled outputs differ from frozen Native: '+name)
    return execution


def render(report):
    lines=['# C 完整 Draft 性能归因', '', f"Profiling：{report['profiling_status']}；端到端：NOT_RUN。",
        '历史 median/p95 是未插桩数据；本次 profile 时间仅用于归因，不能替换历史 benchmark。', '',
        '| Gear | Mode | Frozen median / p95 ms | Profile 状态 |', '|---|---|---:|---|']
    for c in report['cases']:
        h=c['historical'];p95='未提供' if h.get('p95_ms') is None else f"{h['p95_ms']:.4f}"
        lines.append(f"| C{c['gear']} | {c['mode']} | {h['median_ms']:.4f} / {p95} | {c['profiling_status']} |")
    lines+=['','## 四模式对照','']
    for pair in report['pairs']:
        lines.append(f"- C{pair['gear']} {pair['candidate']} − {pair['baseline']}：历史 median 差 {pair['historical_delta_ms']:.4f} ms。")
        device=pair.get('profile_delta_us')
        if device:
            lines.append(f"  Profile 均值差（μs）：{json.dumps(device,ensure_ascii=False)}。这是插桩后区间占用分解，不是未插桩收益。")
        else: lines.append('  Device 时间线解释：NOT_RUN 或数据/Head 映射不足。')
    lines+=['','## Head 及转换明细','']
    for c in report['cases']:
        lines.extend([f"### C{c['gear']} / {c['mode']}",''])
        for stage in ('pre_atc_graph','compiled_graph'):
            g=c.get(stage,{})
            lines.append(f"- {stage}：{g.get('status','NOT_RUN')}；节点分类 {g.get('category_counts',{})}。")
        profile=c.get('profile',{})
        for table in profile.get('csv_tables',[]):
            lines.append(f"- CSV `{table['file']}`（全采集含加载/验证/warmup，不代表单次调用；不累加重复文件）：")
            for group in table['task_groups']:
                if group['label']!='other':
                    st=group['duration_us'];lines.append(f"  - `{group['op_name']}` / {group['op_type']} / {group['label']}：{st['count']} tasks；median {st['median']:.3f} μs，p95 {st['p95']:.3f} μs。")
        t=profile.get('selected_timeline')
        if t:
            lines.append('- 30 次测量窗口均值（μs）：'+json.dumps({k:v['mean'] for k,v in t['metrics'].items()},ensure_ascii=False))
        else: lines.append('- 测量窗口未可靠识别：不从所有 task 总和推算单次 Head 时间。')
        lines.append(f"- 描述符、权重格式、常量属性和邻接转换：见 `c-profile-report.json` cases 的图与原始 CSV fields。")
    lines+=['','## 已证实与未证实','',*['- '+x for x in report['confirmed']],*['- 未证实：'+x for x in report['unconfirmed']],
        '', '## 后续候选（尚未实现或测得收益）','',
        '- 若设备明细证明 Custom C 总时间高于 Native MatMul + Argmax，且额外转换不足以解释差值：分别评估 C 的 MatMul 数据流/权重布局，以及保留 Native MatMul、仅优化 Top1。',
        '- Native MatMul + Top1 必须消费最终 FP16 logits，保留饱和/舍入边界、NaN/±0、最小 ID tie-break；M15 全词表 logits 逻辑大小为 7,449,600 B，显式写回及读取、额外调度成本必须计入整图实测。',
        '- 只有同一 immutable-weight 整图下消除的已测耗时才构成收益假设依据；不从独立 runtime-weight 的 17.1/23.3 ms 外推。',
        '- Fused C 的 MatMul、Vector Top1 与跨核同步在单个 kernel 内；本工具不修改 Kernel，不能凭 Pipe 比例分割这些耗时。Pipe 计数可重叠，不能相加。',
        '- 只做性能归因：不修改 Kernel、Tiling、模型数学、Golden 或历史 evidence。']
    return '\n'.join(lines)+'\n'


def finish(report,root):
    report['pairs']=[]
    for gear in report['gears']:
        modes={c['mode']:c for c in report['cases'] if c['gear']==gear}
        for baseline,candidate in (('off','c'),('a','ac')):
            if baseline not in modes or candidate not in modes: continue
            b,c=modes[baseline],modes[candidate]
            pair=dict(gear=gear,baseline=baseline,candidate=candidate,historical_delta_ms=c['historical']['median_ms']-b['historical']['median_ms'],
                      root_cause='UNPROVEN')
            bt=b.get('profile',{}).get('selected_timeline');ct=c.get('profile',{}).get('selected_timeline')
            if bt and ct and bt['head_mapping']==ct['head_mapping']=='COMPLETE_BY_LABEL':
                pair['profile_delta_us']={k:ct['metrics'][k]['mean']-bt['metrics'][k]['mean'] for k in bt['metrics']}
                pair['profile_vs_frozen_delta_ms']=pair['profile_delta_us']['device_envelope_us']/1000-pair['historical_delta_ms']
                pair['root_cause']='REGION_TIMINGS_AVAILABLE; fused-kernel internals and unprofiled causality remain unproven'
            report['pairs'].append(pair)
    report['cross_pair_consistency']=[]
    for gear in report['gears']:
        pairs=[p for p in report['pairs'] if p['gear']==gear]
        if len(pairs)!=2: continue
        entry=dict(gear=gear,historical_C_cost_difference_ms=pairs[0]['historical_delta_ms']-pairs[1]['historical_delta_ms'],
                   interpretation='similar C increments are a hypothesis of shared bottleneck, not proof')
        if all('profile_delta_us' in p for p in pairs):
            entry['profile_head_delta_difference_us']=pairs[0]['profile_delta_us']['head_occupancy_us']-pairs[1]['profile_delta_us']['head_occupancy_us']
        report['cross_pair_consistency'].append(entry)
    write_json(root/'c-profile-report.json',report)
    (root/'c-profile-analysis.md').write_text(render(report))


def run(args):
    source=args.evidence.resolve();root=args.output_dir.resolve()
    old_args,reuse=load_reuse(source)
    if root.is_relative_to(source) or root.is_relative_to(Path(old_args.native_base).resolve()): raise ValueError('output must be outside original evidence and Native Base')
    suite=load(source/'suite.json');draft=load(source/'draft-suite.json')
    if suite.get('status')!='PASS' or suite.get('single_op_graph')!='PASS' or not (suite.get('draft_only')=='PASS' or suite.get('full_draft')=='PASS') or draft.get('status')!='PASS':
        raise ValueError('requires accepted 52+4+8 full-Draft evidence')
    from a4_contract import CASE_NAMES
    for name,count in (('a-probes.json',52),('c-probes.json',4)):
        item=load(source/name)
        if item.get('status')!='PASS' or len(item['cases'])!=count or any(c.get('status')!='PASS' for c in item['cases']): raise ValueError('historical single-op acceptance incomplete')
        actual=sorted(c['name'] for c in item['cases']) if name=='a-probes.json' else sorted((c['gear'],c['audit']) for c in item['cases'])
        expected=sorted(f'c{g}-{n}' for g in (16,64) for n in CASE_NAMES) if name=='a-probes.json' else sorted((g,a) for g in (16,64) for a in (False,True))
        if actual!=expected: raise ValueError('historical 52/4 case inventory differs')
    if sorted((r['gear'],r['mode']) for r in draft['cases'])!=sorted((g,m) for g in (16,64) for m in MODES) or any(r['status']!='PASS' for r in draft['cases']):
        raise ValueError('requires all eight accepted Draft modes')
    if os.environ.get('ASCEND310P_SIMULATION_ONLY')=='1': raise ValueError('simulation cannot provide profiling evidence')
    runner=Path(reuse['runner']);cases=[];watch={str(source/n):sha256(source/n) for n in ('suite.json','draft-suite.json','a-probes.json','c-probes.json')}
    for row in draft['cases']:
        if row['gear'] in args.gears:
            case=prepare_case(source,row,runner);case['row']=row;cases.append(case);watch.update(case['watch'])
    # All four modes of a gear must use identical frozen request bytes.
    for gear in args.gears:
        group=[c for c in cases if c['row']['gear']==gear]
        if len({json.dumps(c['row']['inputs'],sort_keys=True) for c in group})!=1: raise ValueError('mode inputs are not paired')
    root.mkdir(parents=True,exist_ok=False)
    report=dict(profiling_status='NOT_RUN',scope='frozen full Draft performance attribution only',source=str(source),gears=args.gears,cases=[],pairs=[],
                end_to_end='NOT_RUN',confirmed=['历史八组输出 bitwise PASS；本工具重新校验选中 OM、输入、输出和审计绑定。'],
                unconfirmed=['Native immutable Head 是否受益于离线排布/常量折叠，需要编译图与设备数据支持。',
                    'Runtime-weight 与 immutable-weight benchmark 不同，独立 Head 时间不能直接相减解释整图。',
                    '额外 TransData/Memcpy/Cast、C 内部 MatMul/Top1/同步，以及图调度各自贡献尚待设备证据。'],
                reuse=reuse,source_watch=watch,atc_inspection='mode=1 OM-to-JSON only; never compile or export a model')
    for case in cases:
        row=case['row']
        report['cases'].append(dict(gear=row['gear'],mode=row['mode'],historical=case['historical'],historical_native=case['historical_native'],
            profiling_status='NOT_RUN',compiled_graph={'status':'NOT_RUN'},profile={}))
    env=os.environ.copy();env.update(reuse['environment'])
    profiler=shutil.which(args.msprof);atc=shutil.which(args.atc)
    try:
        for case,out in zip(cases,report['cases']):
            row=case['row'];folder=root/f"c{row['gear']}-{row['mode']}";folder.mkdir()
            pb=[p for p in case['payloads'] if p.name=='dynamo.pbtxt']
            try:
                if len(pb)!=1: raise ValueError('no unique immutable AIR pbtxt')
                out['pre_atc_graph']=summarize(read_pbtxt(pb[0]),'pre_atc')
            except (ValueError,OSError) as error: out['pre_atc_graph']=dict(status='UNPARSED',error=str(error))
            finish(report,root)
            if args.prepare_only: continue
            if not profiler: raise ValueError('msprof not found; use the existing server CANN environment')
            if atc and not args.skip_om_dump:
                inspect_dir=folder/'inspect';inspect_dir.mkdir();target=inspect_dir/'om-graph.json'
                command=[atc,'--mode=1','--om='+str(case['om']),'--json='+str(target)]
                write_json(inspect_dir/'command.json',command)
                with (inspect_dir/'atc-inspect.log').open('w') as log:
                    result=subprocess.run(command,cwd=inspect_dir,env={**env,'ASCEND_WORK_PATH':str(inspect_dir)},stdout=log,stderr=subprocess.STDOUT)
                try:
                    if result.returncode: raise ValueError('OM inspection command failed; see log')
                    out['compiled_graph']=summarize(read_om_json(target),'compiled_om')
                except (ValueError,OSError) as error: out['compiled_graph']=dict(status='UNAVAILABLE',error=str(error))
            else: out['compiled_graph']=dict(status='NOT_RUN',reason='ATC unavailable or --skip-om-dump')
            replay=folder/'replay';replay.mkdir();inputs=replay/'inputs';inputs.mkdir()
            for name,path in case['inputs'].items(): (inputs/(name+'.bin')).symlink_to(path.resolve())
            for name in ('inputs.tsv','outputs.tsv'): shutil.copyfile(case['folder']/'candidate'/name,replay/name)
            command=[str(runner),case['command'][1],str(case['om']),str(replay),str(inputs)]
            write_json(replay/'command.json',command)
            profile_dir=folder/'profiler'
            launch=[profiler,'--ascendcl=on','--runtime-api=on','--task-time=on','--aicpu=on','--ai-core=on',
                    '--aic-mode=task-based','--aic-metrics='+args.metrics,'--output='+str(profile_dir),'--application='+shlex.join(command)]
            write_json(folder/'msprof-command.json',launch)
            out['msprof_binary']=dict(path=profiler,sha256=sha256(profiler));out['profiling_status']='RUNNING';finish(report,root)
            with (folder/'profile.log').open('w') as log: subprocess.run(launch,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            out['profiled_execution']=check_outputs(case,replay)
            out['profiled_host_execute_sync']=timing(out['profiled_execution'])
            out['profiling_status']='COMMAND_COMPLETED';out['numerical_match']=True
            graph=out['compiled_graph'] if out['compiled_graph'].get('status')=='PARSED' else out['pre_atc_graph']
            out['profile']=collect(profile_dir,graph)
            out['profiler_artifacts']=[record(p,folder) for p in sorted(profile_dir.rglob('*')) if p.is_file()]
            finish(report,root)
        for path,value in watch.items():
            if sha256(path)!=value: raise ValueError('original evidence changed during inspection/profile: '+path)
        verify_reuse(reuse)
        report['profiling_status']='NOT_RUN' if args.prepare_only else 'COLLECTED'
    except Exception as error:
        for entry in report['cases']:
            if entry['profiling_status']=='RUNNING': entry['profiling_status']='FAIL'
        report.update(profiling_status='FAIL',error=f'{type(error).__name__}: {error}')
        raise
    finally: finish(report,root)
    print('Report: '+str(root/'c-profile-report.json')+'; attribution only, end-to-end NOT_RUN')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--gears',type=int,nargs='+',choices=(16,64),default=[16])
    p.add_argument('--msprof',default='msprof');p.add_argument('--atc',default='atc')
    p.add_argument('--metrics',choices=('PipeUtilization','Memory','MemoryUB'),default='PipeUtilization')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--skip-om-dump',action='store_true')
    args=p.parse_args()
    if len(set(args.gears))!=len(args.gears):p.error('duplicate gears')
    run(args)
