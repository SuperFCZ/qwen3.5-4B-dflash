"""Attribution uses timeline intervals, never sums op_summary across the process."""
import csv
import json
from collections import Counter,defaultdict
from profile_common import c,require
import c_profile_graph as graph
import c_profile_trace as trace


def trace_pid(path,pid):
    raw=json.loads(path.read_text(),parse_float=str)
    events=raw['traceEvents'] if isinstance(raw,dict) else raw
    execute=[e for e in events if e.get('ph') in ('X','B') and trace.api_event(e)[0]=='execute']
    require(len(execute)==38 and all(str(pid) in (str(e.get('pid')),str(e.get('tid'))) for e in execute),
            'cannot link the 38 exported host Execute calls to the attested runner PID/main thread')


def recover(raw,pid):
    timelines=list(raw.rglob('msprof*.json'));require(timelines,'no exported msprof timeline')
    tables,meta=trace.csv_tables(raw);require(len(tables)==1 and tables[0]['rows'] and not tables[0]['rejected_rows'],
            'need a unique complete device op_summary export with Task Duration(us)')
    candidates=[];diagnostics=[]
    for path in timelines:
        try:
            parsed=trace.parse_timeline(path,meta,set())
            require(parsed['status']=='WINDOWS_IDENTIFIED',parsed.get('reason','unresolved windows'))
            trace_pid(path,pid);candidates.append(parsed)
        except (ValueError,KeyError) as err:diagnostics.append(dict(file=str(path),error=str(err)))
    require(len(candidates)==1,'need one runner-bound 38-call timeline: '+str(diagnostics))
    result=candidates[0];require(len(result['all_iterations'])==38 and len(result['iterations'])==30,'incomplete calls')
    for extra in result['unpaired_syncs']:
        # The frozen Session destructor synchronizes once after postcheck. Never pair it as a timed call.
        t=extra['relative_start_us']
        calls=result['all_iterations'];start=calls[0]['execute_api']['relative_start_us']
        end=calls[-1]['execute_api']['relative_start_us']+calls[-1]['host_window_us']
        require(not start<=t<end,'ambiguous extra Sync between first Execute and final paired Sync')
    for call in result['all_iterations']:
        for task in call['tasks']:
            args={trace.norm(k):v for k,v in task['args'].items()}
            stream=args.get('physicstreamid',args.get('streamid'))
            candidates=meta.get(('task',str(stream),str(args.get('taskid'))),[])
            same=[x for x in candidates if x['op_name']==task['name'] and x['op_type']==task['op_type']]
            names={str(v).strip() for x in same for k,v in x['fields'].items() if trace.norm(k)=='kernelname' and v}
            task['kernel_name']=next(iter(names)) if len(names)==1 else task['raw_name']
            task['kernel_basis']='op_summary kernel name' if len(names)==1 else 'timeline event name'
            task['csv_source']=dict(file=same[0]['file'],line=same[0]['line']) if len(same)==1 else None
    return result,tables,diagnostics


def category(kind,name,heads):
    key=trace.norm(kind)
    if 'dflashgroupquantlinear' in key:return 'GroupQuantLinear'
    if 'matmul' in key:return 'Head MatMul' if name in heads else 'Unmapped MatMul'
    if 'softmax' in key:return 'Softmax'
    if 'attention' in key:return 'Attention'
    if ('cache' in key and ('update' in key or 'scatter' in key or 'reshapeandcache' in key)):return 'KV Cache Update'
    if 'norm' in key:return 'Norm'
    if 'transdata' in key:return 'TransData'
    if 'memcpy' in key or key.startswith('cast'):return 'Memcpy / Cast'
    if key in ('add','addv2','sub','mul','div','silu','sigmoid','swiglu','gelu','pow','rsqrt','neg','exp','square'):return 'Elementwise'
    return 'Other / Unmapped'


def csvfile(path,rows,keys):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def summarize_rows(rows,field):
    groups=defaultdict(list)
    for row in rows:groups[row[field]].append(row['duration_us'])
    return sorted(({field:key,'task_count':len(values),'sum_us':sum(values),'mean_us':sum(values)/len(values),'max_us':max(values)}
                   for key,values in groups.items()),key=lambda row:-row['sum_us'])


def aggregate(folder,t,graph_path,baseline):
    graph_status='PARSED'
    try:nodes=graph.read_om_json(graph_path)
    except ValueError as err:nodes=[];graph_status='UNAVAILABLE: '+str(err)
    by_name=defaultdict(set)
    for node in nodes:by_name[node['name']].add(node['type'])
    try:heads=set(graph.summarize(nodes,'compiled_om')['head_anchors'])
    except ValueError:heads=set()
    evidence=t['assignment_evidence'];provisional=not evidence['excluded_iteration_markers'] and not evidence['unconfirmed_markers']
    scope='PROVISIONAL_HOST_ENCLOSURE' if provisional else t['device_assignment']
    rows=[];uncertain=Counter();counts=[];duplicates=[];nonmeasured=[]
    for call in t['all_iterations']:
        seen=Counter()
        for task in call['tasks']:
            args={trace.norm(k):v for k,v in task['args'].items()}
            ident=(args.get('modelid'),args.get('physicstreamid',args.get('streamid',task['tid'])),args.get('taskid'),args.get('batchid'),args.get('subtaskid'))
            if ident[2] is not None:seen[ident]+=1
        current=[]
        for task in call['tasks']:
            args={trace.norm(k):v for k,v in task['args'].items()};name=task['name'];kind=task['op_type']
            ident=(args.get('modelid'),args.get('physicstreamid',args.get('streamid',task['tid'])),args.get('taskid'),args.get('batchid'),args.get('subtaskid'))
            if seen[ident]>1:
                duplicates.append(dict(index=call['index'],event=task['event_index'],identity=ident));continue
            typ=by_name.get(name,set())
            basis='CSV/timeline type'
            if len(typ)==1 and (not kind or kind in typ):kind=next(iter(typ));basis='exact OM node name'
            elif typ and kind not in typ:kind='';basis='ambiguous graph/trace types'
            label=category(kind,name,heads)
            item=dict(call=call['index'],phase=call['phase'],event=task['event_index'],stream=ident[1],task_id=ident[2],
                      task_type=args.get('tasktype',''),kernel=task['kernel_name'],kernel_basis=task['kernel_basis'],op_name=name,
                      csv_line=(task['csv_source'] or {}).get('line'),op_type=kind,category=label,basis=basis,
                      attribution=scope,method=task['assignment']['method'],start_us=task['start_us'],duration_us=task['duration_us'])
            if call['phase']=='timed':
                current.append(item)
                if label in ('Unmapped MatMul','Other / Unmapped'):uncertain[basis+' / no confirmed operator class']+=1
            else:nonmeasured.append(item)
        if call['phase']=='timed':
            rows.extend(current);counts.append(dict(call=call['index'],host_window_us=call['host_window_us'],
                task_union_us=trace.union([(x['start_us'],x['start_us']+x['duration_us']) for x in current]),categories=summarize_rows(current,'category')))
    types=summarize_rows(rows,'category')
    kernel_rows=[r for r in rows if trace.norm(r['task_type']) in ('aicore','aicpu','aivector','mix','mixaic','mixaiv','aic','aiv')]
    top=summarize_rows(kernel_rows,'kernel')[:20]
    keys='call phase event stream task_id task_type kernel kernel_basis op_name csv_line op_type category basis attribution method start_us duration_us'.split()
    csvfile(folder/'operator-tasks.csv',rows+nonmeasured,keys);csvfile(folder/'operator-types.csv',types,['category','task_count','sum_us','mean_us','max_us'])
    reasons=Counter(x['reason'] for x in evidence['unresolved_tasks'])
    report=dict(attribution=scope,graph_status=graph_status,formal_baseline=baseline,measured_calls=counts,operator_types=types,top20_kernels=top,top20_device_tasks=summarize_rows(rows,'kernel')[:20],
                rankable_kernel_task_count=len(kernel_rows),measured_task_count=len(rows),classification_uncertain=dict(uncertain),duplicate_events=duplicates,
                step_unassigned_reasons=dict(reasons),outside_windows=len(evidence['outside_host_tasks']),provisional_measured_tasks=len(rows) if provisional else 0,
                rejected_step_reasons=evidence['remaining_failure_reasons'],boundary_crossing_observations=sum(len(x['boundary_crossing_tasks']) for x in t['all_iterations']),
                conclusion='Descriptive occupied intervals only. No causal savings inferred.')
    c.write(folder/'profile-report.json',report)
    md=['# A-only Draft profile',f'Attribution: {scope}. Formal unprofiled median {baseline["median_ms"]:.5f} ms, P95 {baseline["p95_ms"]:.5f} ms.',
        'Calls 7–36 only. Instrumented latency is not the official latency. Task-duration sums overlap and are not Draft wall time.',
        'Counts below count device tasks, not GE node invocations. Unassigned/duplicate events are excluded, not assigned zero cost.',
        '', '|Category|Tasks|Sum µs|Mean µs|Max µs|','|---|---:|---:|---:|---:|']
    md += ['|{category}|{task_count}|{sum_us:.3f}|{mean_us:.3f}|{max_us:.3f}|'.format(**row) for row in types]
    md+=['','Top kernels by total µs (explicit compute task type):']+[f'{i}. {row["kernel"]}: {row["sum_us"]:.3f}' for i,row in enumerate(top,1)]
    md += [f'Unresolved device assignments: {dict(reasons)}; ambiguous duplicate events excluded: {len(duplicates)}; unmapped classifications: {dict(uncertain)}.']
    (folder/'profile-hotspots.md').write_text('\n'.join(md)+'\n');return report
