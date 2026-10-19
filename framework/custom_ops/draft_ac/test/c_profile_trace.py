"""msprof parsing with explicit clock/overlap/iteration boundaries; no NPU imports."""
import collections
import csv
import json
import math
from pathlib import Path
import re
import statistics
from c_profile_graph import category


def norm(value): return re.sub('[^a-z0-9]','',str(value).lower())

def finite(value):
    try: result=float(value)
    except (TypeError,ValueError): return None
    return result if math.isfinite(result) and result>=0 else None


def stats(items):
    if not items: return None
    ordered=sorted(items)
    return dict(count=len(items),mean=statistics.mean(items),median=statistics.median(items),
                p95=ordered[max(0,math.ceil(.95*len(ordered))-1)],min=ordered[0],max=ordered[-1])


def union(intervals):
    total=0.;end=None
    for start,stop in sorted(intervals):
        if end is None or start>end: total+=stop-start;end=stop
        elif stop>end: total+=stop-end;end=stop
    return total


def csv_tables(root):
    tables=[];metadata=collections.defaultdict(list)
    for path in sorted(Path(root).rglob('op_summary*.csv')):
        rows=[]; rejected=0
        with path.open(encoding='utf-8-sig',newline='') as stream:
            reader=csv.DictReader(stream)
            for line,raw in enumerate(reader,2):
                row={norm(k):v for k,v in raw.items() if k}
                duration=finite(row.get('taskdurationus'))
                if duration is None: rejected+=1;continue
                entry=dict(file=str(path.relative_to(root)),line=line,op_name=row.get('opname',''),
                    op_type=row.get('optype',''),duration_us=duration,fields=raw)
                rows.append(entry)
                for key in (('name',entry['op_name']),('task',row.get('streamid'),row.get('taskid'))):
                    if all(v not in (None,'','N/A') for v in key): metadata[key].append(entry)
        tables.append(dict(file=str(path.relative_to(root)),rows=rows,rejected_rows=rejected))
    return tables,metadata


def classify(name,kind,fields,head_names):
    cat=category(kind or name)
    if cat=='custom_c': return 'custom_c'
    # Shape evidence must carry the complete vocabulary dimension, not just M15.
    shape=' '.join(str(v) for k,v in fields.items() if 'shape' in norm(k))
    vocab='248320' in re.findall(r'\d+',shape)
    matched=name in head_names or vocab
    if matched and cat in ('matmul','argmax','transdata','memcpy','cast','memset','view'):
        return 'head_'+cat
    if cat in ('matmul','argmax'): return 'unmapped_'+cat
    return cat if cat in ('transdata','memcpy','cast','memset') else 'other'


def event_label(event,metadata,head_names):
    args={norm(k):v for k,v in event.get('args',{}).items()}
    name=str(args.get('opname',event.get('name','')))
    matches=metadata.get(('name',name),[])
    if not matches and args.get('streamid') is not None and args.get('taskid') is not None:
        matches=metadata.get(('task',str(args['streamid']),str(args['taskid'])),[])
    identities={(r['op_name'],r['op_type'],classify(r['op_name'],r['op_type'],r['fields'],head_names)) for r in matches}
    if len(identities)==1:
        op,kind,label=next(iter(identities)); return label,op,kind
    kind=str(args.get('optype',''))
    return classify(name,kind,event.get('args',{}),head_names),name,kind


def parse_timeline(path,metadata,head_names):
    if Path(path).stat().st_size>512*1024*1024: raise ValueError('timeline exceeds 512 MiB analysis limit')
    data=json.loads(Path(path).read_text()); events=data.get('traceEvents',[]) if isinstance(data,dict) else data
    if not isinstance(events,list): raise ValueError('not Chrome trace events')
    hardware=set()
    for e in events:
        if e.get('ph')=='M' and e.get('name')=='process_name' and 'ascendhardware' in norm(e.get('args',{}).get('name','')):
            hardware.add(e.get('pid'))
    complete=[e for e in events if e.get('ph')=='X' and finite(e.get('ts')) is not None and finite(e.get('dur')) is not None]
    stacks=collections.defaultdict(list)
    for e in events:
        key=(e.get('pid'),e.get('tid'))
        if e.get('ph')=='B': stacks[key].append(e)
        elif e.get('ph')=='E' and stacks[key]:
            begin=stacks[key].pop();start=finite(begin.get('ts'));stop=finite(e.get('ts'))
            if start is not None and stop is not None and stop>=start:
                complete.append({**begin,'ph':'X','dur':stop-start})
    executes=sorted((e for e in complete if e.get('name')=='aclmdlExecuteAsync'),key=lambda e:float(e['ts']))
    syncs=sorted((e for e in complete if e.get('name') in ('aclrtSynchronizeStream','aclrtSynchronizeStreamWithTimeout')),key=lambda e:(float(e['ts']),-float(e['dur'])))
    result=dict(status='UNRESOLVED_WINDOWS',file=str(path),execute_count=len(executes),hardware_pids=sorted(hardware,key=str),
                clock_policy='Chrome trace ts/dur in us; never mix op_summary clocks',iterations=[])
    if len(executes)!=38 or not hardware: return result
    tasks=[e for e in complete if e.get('pid') in hardware]
    all_calls=[]
    for index,launch in enumerate(executes):
        start=float(launch['ts']);launch_end=start+float(launch['dur'])
        next_start=float(executes[index+1]['ts']) if index+1<len(executes) else math.inf
        following=[s for s in syncs if (s.get('pid'),s.get('tid'))==(launch.get('pid'),launch.get('tid'))
                   and float(s['ts'])>=launch_end and float(s['ts'])<next_start]
        if not following: return result
        sync=following[0];end=float(sync['ts'])+float(sync['dur'])
        if end>next_start: return result
        selected=[e for e in tasks if start<=float(e['ts']) and float(e['ts'])+float(e['dur'])<=end]
        if not selected: return result
        intervals=[];head=[];groups=collections.defaultdict(list);details=[]
        for e in selected:
            interval=(float(e['ts']),float(e['ts'])+float(e['dur']));intervals.append(interval)
            label,name,kind=event_label(e,metadata,head_names)
            if label=='custom_c' or label.startswith('head_'): head.append(interval)
            groups[label].append(float(e['dur']))
            details.append(dict(name=name,op_type=kind,label=label,stream=e.get('tid'),start_us=interval[0],duration_us=float(e['dur']),args=e.get('args',{})))
        occupancy=union(intervals);head_time=union(head);span=max(t[1] for t in intervals)-min(t[0] for t in intervals)
        all_calls.append(dict(index=index,host_window_us=end-start,host_launch_api_us=float(launch['dur']),host_sync_api_us=float(sync['dur']),
            device_envelope_us=span,device_occupancy_us=occupancy,head_occupancy_us=head_time,
            other_exclusive_occupancy_us=occupancy-head_time,device_gap_us=span-occupancy,
            task_duration_sums_us={k:sum(v) for k,v in groups.items()},tasks=details))
    measured=all_calls[7:37] # 2 correctness + 5 warmup, then 30 measured; last postcheck excluded.
    metrics=('host_window_us','host_launch_api_us','host_sync_api_us','device_envelope_us','device_occupancy_us',
             'head_occupancy_us','other_exclusive_occupancy_us','device_gap_us')
    result.update(status='WINDOWS_IDENTIFIED',selection='38 exact Execute/Sync pairs; zero-based measured calls 7..36',
                  metrics={k:stats([c[k] for c in measured]) for k in metrics},iterations=measured,
                  partition_note='Head union + other exclusive occupancy + gaps = device envelope per call; not causal decomposition. Sync overlaps kernels.')
    complete_head=True
    for call in measured:
        labels={t['label'] for t in call['tasks']}
        if not ('custom_c' in labels or {'head_matmul','head_argmax'}<=labels): complete_head=False
    result['head_mapping']='COMPLETE_BY_LABEL' if complete_head else 'INCOMPLETE'
    return result


def collect(root,graph):
    root=Path(root);names={n['name'] for n in graph.get('head_related_nodes',[])}
    tables,metadata=csv_tables(root)
    for table in tables:
        grouped=collections.defaultdict(list)
        for row in table['rows']:
            row['label']=classify(row['op_name'],row['op_type'],row['fields'],names)
            grouped[(row['op_name'],row['op_type'],row['label'])].append(row['duration_us'])
        table['task_groups']=[dict(op_name=k[0],op_type=k[1],label=k[2],duration_us=stats(v)) for k,v in grouped.items()]
    timelines=[]
    for path in sorted(root.rglob('msprof*.json')):
        try: timelines.append(parse_timeline(path,metadata,names))
        except (ValueError,OSError) as error: timelines.append(dict(status='UNPARSED',file=str(path),error=str(error)))
    valid=[t for t in timelines if t['status']=='WINDOWS_IDENTIFIED']
    selected=valid[0] if len(valid)==1 else None
    return dict(status='TIMELINE_AVAILABLE' if selected else 'PARTIAL_DATA' if tables or timelines else 'NOT_FOUND',
                csv_tables=tables,timelines=timelines,selected_timeline=selected,
                selection_note='No averaging across exports; multiple valid timelines remain ambiguous',
                limitation='Fused C internals (Matmul/Top1/cross-core sync) cannot be separated from one kernel duration; pipe counters may overlap')
