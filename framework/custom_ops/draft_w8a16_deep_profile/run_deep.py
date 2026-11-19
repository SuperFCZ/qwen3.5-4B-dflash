"""One manual stage. No uninstrumented benchmark, model rebuild or deployment entrypoint."""
import argparse
from decimal import localcontext
from pathlib import Path
import traceback
from deep_common import HERE,admit,artifacts,c,collect,deep,recover,require,sources
from hotspots import analyze_case
from c_profile_graph import read_om_json,summarize
from signals import Graph
from gate_plan import node_binding

CAPTURES={f'profile-{mode}-c{gear}':(mode,gear) for mode in ('raw','off') for gear in ('16','64')}
PHASES=('admit',*CAPTURES,'analyze-raw','profile-paired-analysis')
STATE={phase:'DEEP_'+phase.upper().replace('-','_')+'_PASS' for phase in PHASES}


def deps(phase):
    if phase=='admit':return []
    if phase in CAPTURES:return ['admit']
    return ['admit']+[p for p,(mode,_) in CAPTURES.items() if phase=='profile-paired-analysis' or mode=='raw']


def table(report):
    return {key:value['time_per_execute']['mean_us'] for key,value in report['classes'].items() if value['observed']}


def asset_index(out,b):
    index={};head={};md=['# 编译 OM 的 Head 节点（只读，未运行模型）']
    for mode,variant in b['variants'].items():
        selected=variant['selected']
        for gear,case in variant['cases'].items():
            key=mode+'-c'+gear
            index[key]=dict(om=case['om'],deployment=case['deployment'],runner=selected['om']['runner'],
                opp=selected['opp']['install'],opp_gate=selected['kernel'],api=selected['api'],DSO=selected['manifest'],
                extension_pins=len(variant['extras']),embedded=case['atc_kernels'],baseline=case['baseline'],workspace=case['workspace'],weights=case['weights'])
            try:
                nodes=read_om_json(c.verify(case['dump']));net=Graph(nodes);anchors=summarize(nodes,'compiled_om')['head_anchors']
                names={n:[] for n in anchors if 'matmul' in net.nodes[n]['type'].lower()}
                rows=deep.paths_report(net,names,'head',{'rows':[]},[])
                for item in rows:
                    item.pop('timing');item.pop('csv')
                    attrs=net.nodes[item['name']]['attributes']
                    item['launch_attributes']={a['key']:a['value'] for a in attrs if isinstance(a,dict) and a.get('key') in ('tvm_blockdim','_kernelname','_kernel_bin_id')}
                head[key]=dict(graph=case['dump'],nodes=rows,status='GRAPH_BOUND' if rows else 'UNRESOLVED')
                for row in rows:md.append(f"- {key}: {row['name']}, Top1 paths={row['top1_paths']}, {row['weight_layout_finding']}.")
            except ValueError as exc:head[key]=dict(graph=case['dump'],status='UNRESOLVED',reason=str(exc))
    c.write(out/'asset-index.json',index);c.write(out/'head-om-evidence.json',head)
    local=c.read(HERE/'w16-ub-budget.json');control={}
    for gear,case in b['variants']['raw']['cases'].items():
        plan=case['plans']['16x2560x19456']['contract']
        control[gear]=dict(accepted_raw_cube=plan['cube_sha256'],local_configured_cube=local['cube_sha256'],
                           same_bytes=plan['cube_sha256']==local['cube_sha256'],accepted_raw_resources=plan['resources'],
                           candidate_formal_context='NOT_RUN',candidate_event_lifecycle='BLOCKED')
    c.write(out/'w16-baseline-crosscheck.json',control)
    (out/'head-graph-review.md').write_text('\n'.join(md)+'\nNative base/depth/DB、HBM 实际读字节不能由 launch blockDim 或描述符推算。\n')


def render(out,results,paired):
    md=['# Raw full Draft — measured profile', 'Calls 7–36 only. Task sums (µs/Execute) may overlap; never subtract them from formal wall latency.']
    head_md=['# 当前完整 OM 的 Head 观察','只取 calls 7..36。跨 GE 算子的 provisional sum 不是 wall time。']
    recommendation=['# 下一步（由本次采集生成）','Profile 用于提出单变量实验，不能证明可节省的墙钟时间。']
    for gear in ('16','64'):
        raw=results['raw',gear];off=results.get(('off',gear));base=raw['baseline']
        md+=['',f"## C{gear} · {raw['scope']}",f"Raw unprofiled median/P95 {base['median_ms']:.6f}/{base['p95_ms']:.6f} ms. Strict grade: {raw['strict_status']}.",
             f"Assigned timed tasks: {raw['timed_task_count']}. Strict warnings: {raw['strict_reasons']}.",
             '|Class|Raw tasks|Raw µs/Execute|Off µs/Execute|Raw − Off µs/Execute|','|---|---:|---:|---:|---:|']
        a=table(raw);b=table(off) if off else {}
        for k in sorted(set(raw['classes'])|set(b),key=lambda key:-a.get(key,0)):
            val=raw['classes'].get(k,{})
            comparable=k in a and k in b and val['counts_per_execute']==off['classes'][k]['counts_per_execute']
            md.append(f"|{k}|{val.get('tasks',{}).get('count','—')}|{a.get(k,'—')}|{b.get(k,'—')}|{round(a[k]-b[k],3) if comparable else '—'}|")
        md+=['','|Projection|Raw tasks|Mean/task µs|P95/task µs|Off µs/Execute|Raw − Off µs/Execute|Identification|','|---|---:|---:|---:|---:|---:|---|']
        for name,p in raw['projections'].items():
            s=p['timing']['tasks'];other=off['projections'][name] if off else None
            comparable=other and other['status']==p['status']=='BOUND_EVERY_CALL'
            left=other['timing']['time_per_execute']['mean_us'] if comparable else None
            md.append(f"|{name}|{s['count']}|{s['mean_us']}|{s['p95_us']}|{left if comparable else '—'}|{round(p['timing']['time_per_execute']['mean_us']-left,3) if comparable else '—'}|{p['status']}|")
        md+=['',f"Raw wait metrics: {raw['wait_summary']}; object/critical path UNRESOLVED."]
        if off:md.append(f"Off strict grade: {off['strict_status']}; Off waits: {off['wait_summary']}. Comparison uses the same ownership rule in both captures.")
        ctr=c.read(out/('raw-c'+gear)/'counters.json')
        for mode in (('raw','off') if paired else ('raw',)):
            head_folder=out/(mode+'-c'+gear);pipes=c.read(head_folder/'counters.json')
            head_md+=['',f'## {mode} C{gear} · '+results[mode,gear]['scope']]
            for item in c.read(head_folder/'head.json'):
                head_md+=['- '+item['name']+'; µs/Execute (observed task sum)='+str(item['timing']['time_per_execute']['mean_us'])+
                         '; '+str(item.get('weight_layout_finding','UNRESOLVED'))+'; Top1 path='+str(item.get('top1_paths',[])),
                         '  CSV descriptors: '+str(item['csv'])[:800]]
            memory={name:{k:v for k,v in values.items() if k in pipes['explicit_memory_read_fields']} for name,values in pipes['per_class'].items() if name.startswith('Head MatMul / ')}
            head_md.append('带原单位、唯一匹配的 Head 内存读字段（若含其它操作数不能等同于纯权重读取）: '+str({k:v for k,v in memory.items() if v} or 'UNAVAILABLE'))
            head_md.append('Native base/depth/DB、重复权重读取次数和实际 M 行复用率：UNRESOLVED。')
        md+=['',f"op_summary: {ctr['source']}; linked timed tasks {ctr['joined']}/{ctr['timed_tasks']}; counter headers: {ctr['columns']}; stall: {ctr['stall']}.",
             'Counter families (usable joined fields): '+str(ctr['families']),
             'Per-task counter medians by measured class (original units; no cross-pipe sums):']
        for name,cols in sorted(ctr['per_class'].items()):md.append(f"- {name}: "+', '.join(f"{k}={v['median_per_task']:.3g} (n={v['n']})" for k,v in cols.items()))
        if off:
            other=c.read(out/('off-c'+gear)/'counters.json')
            md.append(f"Fresh Off counter coverage {other['joined']}/{other['timed_tasks']}; median/task deltas Raw − Off (same class/task/core/unit; descriptive only):")
            for name,cols in sorted(ctr['per_class'].items()):
                common=sorted(set(cols)&set(other['per_class'].get(name,{})))
                if common:md.append('- '+name+': '+', '.join(f"{k}={cols[k]['median_per_task']-other['per_class'][name][k]['median_per_task']:.3g}" for k in common))
        compute={k:v for k,v in a.items() if k.startswith('Linear/') and k!='Linear/Unmapped' or k in ('Head MatMul','Attention','Attention graph candidate','Norm','Softmax')}
        winner=max(compute,key=compute.get) if compute else 'UNRESOLVED'
        recommendation+=['',f"**C{gear}**：Raw 正式 median {base['median_ms']:.6f} ms；{raw['scope']} 下，已识别计算类别按任务耗时和排首位的是 {winner}。",
                         f"Gate/Up、Down、Head µs/Execute：{a.get('Linear/gate_up','UNAVAILABLE')} / {a.get('Linear/down','UNAVAILABLE')} / {a.get('Head MatMul','UNAVAILABLE')}。",
                         f"WAIT union、与 GroupQuant 交集 µs/Execute：{raw['wait_summary']['wait_union_us']:.3f} / {raw['wait_summary']['intersection_us']:.3f}；等待对象和可消除比例未证实。",
                         f"Stall 计数器：{ctr['stall']}；正式窗口唯一匹配的 op_summary 任务 {ctr['joined']}/{ctr['timed_tasks']}。没有经证实的最大硬件 stall 关键路径。",
                         f"距离 20/13.33/13 ms：{base['median_ms']-20:.3f}/{base['median_ms']-13.33:.3f}/{base['median_ms']-13:.3f} ms。"]
        if off:
            for name in ('Linear/gate_up','Linear/down','Head MatMul','NOTIFY_WAIT'):
                if name in a and name in b and raw['classes'][name]['counts_per_execute']==off['classes'][name]['counts_per_execute']:
                    recommendation.append(f'- 新 profile {name}: Raw − Off = {a[name]-b[name]:+.3f} µs/Execute（同口径任务和，非 wall 差值）。')
    if not paired:md+=['','Fresh Off captures were not used. No paired profiling delta.']
    recommendation+=['','Off/Raw 类别差值和覆盖率见 profile-hotspots.md；不同归属质量或未分类任务会限制排序。' if paired else 'Off/Raw 新配对 Profile 差异仍 NOT_RUN。',
                     'W16：本地 SDK 容量实验可行，但 caller-owned W16 消费完成、内部队列/工作区复用和失败排空未闭合，继续 BLOCKED。保留 K256 / End / PIPE_ALL。',
                     'Head 可并行做隔离验证：同一 M15 全词表、真实 hidden/weight，先对 native MatMul→ArgMax 建立逐 logits 位型及 ID 对照，再只变一种布局/分块；特殊值和跨 tile tie、Guard/Readonly/Repeat 均通过后才谈新无插桩 A/B。',
                     '流水实验的第一步只验证一个真实 C16 Gate 的 B-bank 最后读取与排空，不计时；没有 SDK 完成协议前不启动双 W16。13 ms 不作预测。']
    (out/'profile-hotspots.md').write_text('\n'.join(md)+'\n');(out/'optimization-next-step.md').write_text('\n'.join(recommendation)+'\n')
    (out/'head-optimization-review.md').write_text('\n'.join(head_md)+'\n')


def main(args):
    phase=args.phase;root=args.root.resolve();old=args.gate8_root.resolve();archive=args.historical_off.resolve()
    out=c.fresh(root/phase,[old,archive])
    r=dict(phase=phase,status='BLOCKED',sources=sources(),predecessors={},folder=str(out),benchmark='NOT_RUN',launch_authorized=False,device='NOT_RUN')
    try:
        prev={}
        for name in deps(phase):
            file=root/name/'profile-report.json';entry=c.read(file);c.verify_tree(entry)
            require(entry['phase']==name and entry['status']==STATE[name] and entry['sources']==r['sources'] and entry['folder']==str(file.parent) and
                    not entry.get('error') and not entry.get('integrity_error') and entry['predecessors']==({} if name=='admit' else {'admit':r['predecessors']['admit']}),'invalid predecessor: '+name)
            prev[name]=entry;r['predecessors'][name]=c.identity(file)
        if phase=='admit':
            r.update(binding=admit(old,archive),gate8_root=str(old),historical_off=str(archive))
            asset_index(out,r['binding'])
        else:
            require(prev['admit']['gate8_root']==str(old) and prev['admit']['historical_off']==str(archive),'mixed input roots')
            binding=prev['admit']['binding']
            if phase in CAPTURES:
                require(args.allow_profile,'device capture requires explicit --allow-profile')
                mode,gear=CAPTURES[phase];b=binding['variants'][mode]
                r.update(mode=mode,gear=gear,model=b['cases'][gear]['om'],device='PROFILE_ATTEMPTED')
                collect(out,b,gear,r)
                kernels=b['cases'][gear]['atc_kernels']
                actual=[node_binding(node) for node in read_om_json(c.verify(r['graph'])) if node['type']=='DFlashGroupQuantLinear']
                require(len(actual)==len(kernels)==26 and {n['node'] for n in actual}==set(kernels) and
                        all(all(kernels[n['node']][k]==v for k,v in n.items()) for n in actual),'profile graph differs from admitted ATC kernels')
                t,tables,notes=recover(out/'profiler',r['runner_pid']);require(t['assignment_evidence']['real_event_count']>0,'no device tasks')
                c.write(out/'windows.json',t);r.update(windows=c.identity(out/'windows.json'),strict_status=t['device_assignment'],csv_sources=[x['file'] for x in tables],
                     other_timelines=notes,device='38_INSTRUMENTED_CALLS_AND_GOLDEN_VERIFIED')
                c.verify_opp(b['selected']['opp']);c.verify_tree(b)
            else:
                paired=phase=='profile-paired-analysis';results={};r['cases']={}
                for gear in ('16','64'):
                    modes=('off','raw') if paired else ('raw',)
                    use_host=any(prev['profile-'+m+'-c'+gear]['strict_status']!='DEVICE_STEPS_VERIFIED' for m in modes)
                    for mode in modes:
                        cap=prev['profile-'+mode+'-c'+gear];require(cap['mode']==mode and cap['gear']==gear and cap['model']==binding['variants'][mode]['cases'][gear]['om'],'wrong capture')
                        dest=out/(mode+'-c'+gear)
                        with localcontext() as ctx:
                            ctx.prec=64;result=analyze_case(dest,cap,binding,mode,gear,use_host)
                            if use_host and cap['strict_status']=='DEVICE_STEPS_VERIFIED':
                                analyze_case(out/(mode+'-c'+gear+'-strict'),cap,binding,mode,gear)
                        results[mode,gear]=result;r['cases'][dest.name]=c.identity(dest/'case-analysis.json')
                render(out,results,paired)
        require(r['sources']==sources(),'source drift');r['status']=STATE[phase]
    except Exception as exc:r['error']=repr(exc);(out/'traceback.txt').write_text(traceback.format_exc())
    finally:
        try:r['files']=artifacts(out);c.verify_tree(r);c.clean()
        except Exception as exc:r.update(status='BLOCKED',integrity_error=repr(exc))
        c.write(out/'profile-report.json',r)
    print(r['status'],r.get('error'),r.get('integrity_error'),out/'profile-report.json');return int(r['status']=='BLOCKED')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=PHASES);p.add_argument('--root',type=Path,required=True);p.add_argument('--allow-profile',action='store_true')
    prefix=Path('/home/w00949577/qwen35-dflash-run');p.add_argument('--gate8-root',type=Path,default=prefix/'gate-raw-atc8-Np20yu');p.add_argument('--historical-off',type=Path,default=prefix/'a-only-draft-profile-OC8pCW')
    raise SystemExit(main(p.parse_args()))
