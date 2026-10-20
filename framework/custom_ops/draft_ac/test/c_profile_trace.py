"""msprof parsing with explicit clock/overlap/iteration boundaries; no NPU imports."""
import collections
import csv
import json
import math
from pathlib import Path
import re
import statistics
from decimal import Decimal, InvalidOperation
from c_profile_graph import category
from profile_device_assignment import assign
from profile_step_evidence import load as load_step_evidence


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
    if not matches and args.get('physicstreamid',args.get('streamid')) is not None and args.get('taskid') is not None:
        matches=metadata.get(('task',str(args.get('physicstreamid',args.get('streamid'))),str(args['taskid'])),[])
    identities={(r['op_name'],r['op_type'],classify(r['op_name'],r['op_type'],r['fields'],head_names)) for r in matches}
    if len(identities)==1:
        op,kind,label=next(iter(identities)); return label,op,kind
    kind=str(args.get('optype',''))
    return classify(name,kind,event.get('args',{}),head_names),name,kind



API_NAMES={'aclmdlexecuteasync':'execute','aclrtsynchronizestream':'sync','aclrtsynchronizestreamwithtimeout':'sync'}

def api_event(event):
    """Observed CANN9 AscendCL@ spelling + bare spelling; no fuzzy Execute match."""
    name=event.get('name','');name=name if isinstance(name,str) else ''
    normalized=name[len('AscendCL@'):] if name.startswith('AscendCL@') else name
    named=normalized.casefold() if isinstance(normalized,str) else ''
    args={norm(k):v for k,v in event.get('args',{}).items()}
    identified=str(args.get('id','')).casefold() if str(args.get('level','')).casefold()=='acl' else ''
    candidates={x for x in (named,identified) if x in API_NAMES}
    if len(candidates)>1 or (named in API_NAMES and identified and named!=identified):raise ValueError('conflicting AscendCL name and args.id: '+repr(name))
    canonical=next(iter(candidates)) if candidates else None
    return API_NAMES.get(canonical),canonical


def relative_clock(events):
    """Subtract a Decimal origin BEFORE float conversion (epoch us ~1.8e15)."""
    valid=[]
    for event in events:
        if event.get('ph') not in ('X','B','E'):continue
        try:
            value=Decimal(str(event['ts']))
            if value.is_finite() and value>=0:valid.append(value)
        except (KeyError,InvalidOperation):continue
    origin=min(valid) if valid else Decimal(0)
    converted=[]
    for e in events:
        item=dict(e, event_index=len(converted))
        try:
            value=Decimal(str(e['ts']))
            if value.is_finite() and value>=0:item.update(ts=float(value-origin),raw_timestamp_us=str(e['ts']))
        except (KeyError,InvalidOperation):pass
        converted.append(item)
    return converted,str(origin)


def task_shapes(event,metadata):
    args={norm(k):v for k,v in event.get('args',{}).items()}
    name=str(args.get('opname',event.get('name','')))
    rows=metadata.get(('name',name),[])
    if not rows and args.get('physicstreamid',args.get('streamid')) is not None and args.get('taskid') is not None:
        rows=metadata.get(('task',str(args.get('physicstreamid',args.get('streamid'))),str(args['taskid'])),[])
    shapes=[{k:v for k,v in row['fields'].items() if k and any(tag in norm(k) for tag in ('shape','format','datatype'))} for row in rows]
    empty=('', 'n/a', 'none', 'null', '[]', '-')
    shapes=[x for x in shapes if any(str(v).strip().lower() not in empty for k,v in x.items() if 'shape' in norm(k))]
    unique={json.dumps(x,sort_keys=True) for x in shapes}
    if len(unique)>1:return dict(status='AMBIGUOUS',fields={})
    if len(unique)==1:return dict(status='CSV',fields=shapes[0])
    fields={k:v for k,v in event.get('args',{}).items() if any(tag in norm(k) for tag in ('shape','format','datatype'))}
    return dict(status='TRACE' if fields else 'MISSING',fields=fields)

def parse_timeline(path,metadata,head_names):
    if Path(path).stat().st_size>512*1024*1024: raise ValueError('timeline exceeds 512 MiB analysis limit')
    # Preserve original decimal tokens (including trailing zeros) for Step checks.
    data=json.loads(Path(path).read_text(),parse_float=str); events=data.get('traceEvents',[]) if isinstance(data,dict) else data
    if not isinstance(events,list) or any(not isinstance(e,dict) for e in events): raise ValueError('not Chrome trace events')
    events,origin=relative_clock(events)
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
    api=[(e,*api_event(e)) for e in complete if e.get('pid') not in hardware]
    executes=sorted((e for e,kind,_ in api if kind=='execute'),key=lambda e:float(e['ts']))
    syncs=sorted((e for e,kind,_ in api if kind=='sync'),key=lambda e:(float(e['ts']),-float(e['dur'])))
    result=dict(status='UNRESOLVED_WINDOWS',file=str(path),execute_count=len(executes),hardware_pids=sorted(hardware,key=str),
                clock_policy='Decimal epoch origin subtracted before float; ts/dur in us; never mix CSV clocks',timestamp_origin_us=origin,
                api_event_names=dict(collections.Counter(e.get('name','') for e,kind,_ in api if kind)),
                acl_api_inventory=dict(collections.Counter(e.get('name','') for e,_,_ in api if str(e.get('args',{}).get('level','')).lower()=='acl' or 'ascendcl' in str(e.get('name','')).lower())),
                sync_count=len(syncs),iterations=[],all_iterations=[])
    if len(executes)!=38 or not hardware:
        result['reason']='requires exactly 38 public ExecuteAsync events and hardware process metadata';return result
    if len({(e.get('pid'),e.get('tid')) for e in executes})!=1:
        result['reason']='ExecuteAsync spans multiple host processes/threads';return result
    used_syncs=set()
    tasks=[e for e in complete if e.get('pid') in hardware]
    windows=[]
    for index,launch in enumerate(executes):
        start=float(launch['ts']);launch_end=start+float(launch['dur'])
        next_start=float(executes[index+1]['ts']) if index+1<len(executes) else math.inf
        following=[s for s in syncs if (s.get('pid'),s.get('tid'))==(launch.get('pid'),launch.get('tid'))
                   and float(s['ts'])>=launch_end and float(s['ts'])<next_start]
        if not following:
            result['reason']=f'no matching same-thread Sync after execution {index}';return result
        sync=following[0];used_syncs.add(id(sync));end=float(sync['ts'])+float(sync['dur'])
        if end>next_start: return result
        windows.append(dict(start=start,end=end,launch_end=launch_end,launch=launch,sync=sync))
    owners,proof,assignment,tasks=assign(events,tasks,windows,origin,load_step_evidence(Path(path).parent))
    result['assignment_evidence']=assignment
    all_calls=[]
    for index,w in enumerate(windows):
        start,end,launch,sync=w['start'],w['end'],w['launch'],w['sync']
        selected=[e for e in tasks if owners.get(e['event_index'])==index and proof[e['event_index']]['method']!='PRE_STEP_TRANSFER']
        transfers=[t for t in assignment['pre_step_transfers'] if t['call_index']==index]
        crossing=[dict(event_index=e['event_index'],name=e.get('name'),pid=e.get('pid'),tid=e.get('tid'),cat=e.get('cat'),args=e.get('args',{}),start_us=float(e['ts']),duration_us=float(e['dur']),owner=owners.get(e['event_index']),classification=proof[e['event_index']]['method'] if e['event_index'] in owners else 'UNRESOLVED_DEVICE_ASSIGNMENT') for e in tasks
                  if float(e['ts'])<end and float(e['ts'])+float(e['dur'])>start and
                  not (start<=float(e['ts']) and float(e['ts'])+float(e['dur'])<=end)]
        intervals=[];head=[];groups=collections.defaultdict(list);details=[]
        for e in selected:
            interval=(float(e['ts']),float(e['ts'])+float(e['dur']));intervals.append(interval)
            label,name,kind=event_label(e,metadata,head_names)
            if label=='custom_c' or label.startswith('head_'): head.append(interval)
            groups[label].append(float(e['dur']))
            details.append(dict(event_index=e['event_index'],raw_name=e.get('name'),pid=e.get('pid'),tid=e.get('tid'),cat=e.get('cat'),assignment=proof[e['event_index']],name=name,op_type=kind,label=label,stream=e.get('tid'),start_us=interval[0],duration_us=float(e['dur']),args=e.get('args',{}),shape_evidence=task_shapes(e,metadata)))
        occupancy=union(intervals);head_time=union(head);span=max(t[1] for t in intervals)-min(t[0] for t in intervals) if intervals else 0.
        all_calls.append(dict(index=index,phase='correctness_before' if index<2 else 'warmup' if index<7 else 'timed' if index<37 else 'postcheck',
            execute_api=dict(name=launch['name'],timestamp_us=launch.get('raw_timestamp_us'),relative_start_us=start,connection_id=launch.get('args',{}).get('connection_id')),
            sync_api=dict(name=sync['name'],timestamp_us=sync.get('raw_timestamp_us'),relative_start_us=float(sync['ts']),connection_id=sync.get('args',{}).get('connection_id')),
            pre_step_transfers=transfers,pre_step_transfer_count=len(transfers),pre_step_transfer_duration_us=sum(t['duration_us'] for t in transfers),
            device_window=assignment['device_windows'].get(str(index)),host_window_us=end-start,host_launch_api_us=float(launch['dur']),host_sync_api_us=float(sync['dur']),
            device_envelope_us=span,device_occupancy_us=occupancy,head_occupancy_us=head_time,
            other_exclusive_occupancy_us=occupancy-head_time,device_gap_us=span-occupancy,boundary_crossing_tasks=crossing,
            task_duration_sums_us={k:sum(v) for k,v in groups.items()},tasks=details))
    measured=all_calls[7:37] # 2 correctness + 5 warmup, then 30 measured; last postcheck excluded.
    metrics=('host_window_us','host_launch_api_us','host_sync_api_us','device_envelope_us','device_occupancy_us',
             'head_occupancy_us','other_exclusive_occupancy_us','device_gap_us')
    result.update(status='WINDOWS_IDENTIFIED',selection='38 exact Execute/Sync pairs; zero-based measured calls 7..36',
                  metrics={k:stats([c[k] for c in measured]) for k in metrics},iterations=measured,all_iterations=all_calls,
                  unpaired_syncs=[dict(name=e['name'],timestamp_us=e.get('raw_timestamp_us'),relative_start_us=float(e['ts']),pid=e.get('pid'),tid=e.get('tid')) for e in syncs if id(e) not in used_syncs],
                  partition_note='Head union + other exclusive occupancy + gaps = device envelope per call; not causal decomposition. Sync overlaps kernels.')
    result['device_assignment']=assignment['status']
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
