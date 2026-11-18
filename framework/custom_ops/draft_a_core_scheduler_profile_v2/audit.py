"""Offline-only Step and interval contracts. Times in the original trace clock (us)."""
from collections import Counter, defaultdict
from decimal import Decimal
import re
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'draft_ac/test'))
import c_profile_trace as trace
import profile_device_assignment as device
import profile_step_evidence as step_csv

PROVISIONAL='PROVISIONAL_HOST_ENCLOSURE'


def require(ok,message):
    if not ok:raise ValueError(message)


def complete(events):
    """Preserve decimal tokens; pair nested B/E on their own track only."""
    tracks=defaultdict(set);hardware=set();stacks=defaultdict(list);done=[]
    def add(event,start,duration,index):
        require(start.is_finite() and duration.is_finite() and duration>=0,'invalid interval')
        done.append(dict(event,event_index=index,raw_timestamp_us=str(event['ts']),
                         dur=str(event['dur']) if event.get('ph')=='X' else str(duration)))
    for index,event in enumerate(events):
        lane=(event.get('pid'),event.get('tid'));phase=event.get('ph')
        if phase=='M':
            name=event.get('args',{}).get('name','')
            if event.get('name')=='process_name' and 'ascendhardware' in trace.norm(name):hardware.add(lane[0])
            if event.get('name')=='thread_name':tracks[lane].add(name)
        elif phase=='X':add(event,Decimal(str(event['ts'])),Decimal(str(event['dur'])),index)
        elif phase=='B':stacks[lane].append((index,event))
        elif phase=='E' and stacks[lane]:
            idx,begin=stacks[lane].pop();start=Decimal(str(begin['ts']))
            add(begin,start,Decimal(str(event['ts']))-start,idx)
    return done,tracks,hardware


def bounds(event):
    start=Decimal(event['raw_timestamp_us']);return start,start+Decimal(event['dur'])


def host_windows(done,hardware,frozen,pid):
    calls=frozen['all_iterations']
    require(len(calls)==38 and frozen['execute_count']==38,'need 38 verified Execute/Sync pairs')
    api=defaultdict(list)
    for event in done:
        if event['pid'] not in hardware:
            kind,_=trace.api_event(event)
            if kind:api[kind,Decimal(event['raw_timestamp_us'])].append(event)
    windows=[];lane=None
    for i,call in enumerate(calls):
        require(call['index']==i,'call order drift')
        launches=api['execute',Decimal(call['execute_api']['timestamp_us'])]
        require(len(launches)==1,'ambiguous Execute')
        launch=launches[0];current=(launch['pid'],launch['tid'])
        require(str(pid) in map(str,current) and (lane is None or lane==current),'runner lane differs')
        lane=current
        syncs=[e for e in api['sync',Decimal(call['sync_api']['timestamp_us'])] if (e['pid'],e['tid'])==lane]
        require(len(syncs)==1,'ambiguous paired Sync')
        start,launch_end=bounds(launch);sync_start,end=bounds(syncs[0])
        require(start<=launch_end<=sync_start<=end and (not windows or windows[-1][1]<=start),'overlapping or reordered Host calls')
        windows.append((start,end))
    return windows


def partition(done,tracks,hardware):
    markers=[];unconfirmed=[];tasks=[]
    for event in done:
        if event['pid'] not in hardware:continue
        marker=device.step_marker(event,tracks)
        if marker:markers.append(dict(event,**marker))
        elif re.fullmatch(r'Iteration \d+',str(event.get('name',''))) or event.get('cat')=='Iteration Time':unconfirmed.append(event)
        else:tasks.append(event)
    return sorted(markers,key=lambda e:(Decimal(e['raw_timestamp_us']),e['event_index'])),unconfirmed,tasks


def step_audit(markers,tracks,windows,csv_evidence,frozen):
    evidence=frozen['assignment_evidence'];old={m['event_index']:m for m in evidence['excluded_iteration_markers']}
    require(set(old)=={m['event_index'] for m in markers},'Step inventory drift')
    rows=[]
    for marker in markers:
        idx=marker['event_index'];mid=marker['model_id'];sid=marker['iteration_id']
        check=device.step_time_check(marker);prior=old[idx]['time_consistency']
        require(all(check.get(k)==prior.get(k) for k in ('status','raw_start_us','raw_end_us','signed_error_us','allowed_error_max_us')),
                'strict Decimal check differs from frozen report')
        endpoints=step_csv.corroborate(csv_evidence,mid,sid,check)
        for record in endpoints['matches']+endpoints['unsupported_rows']:
            fields={trace.norm(k):v for k,v in record['fields'].items()}
            for field,source in [('iterationstartus','raw_start_us'),('starttimeus','raw_start_us'),('endtimeus','raw_end_us')]:
                if field in fields:
                    try:delta=Decimal(fields[field])-Decimal(str(check[source]))
                    except ArithmeticError:delta=None
                    record['time_field_comparison'][field]=dict(raw_csv=fields[field],raw_json=check[source],
                        csv_minus_json_us=str(delta) if delta is not None and delta.is_finite() else None)
        if not endpoints['matches'] and not endpoints['unsupported_rows']:
            endpoints['reason']='no CSV row with matching Iteration ID in the same export' if csv_evidence['tables'] else 'no step_trace CSV in the same export directory'
        start=Decimal(check['raw_start_us'])
        rows.append(dict(id=sid,model=mid,event_index=idx,pid=marker['pid'],tid=marker['tid'],track=sorted(tracks[marker['pid'],marker['tid']]),
                         precision=check,csv=endpoints,start_in_host=[i for i,(lo,hi) in enumerate(windows) if lo<=start<hi],
                         strict_host_call=old[idx].get('host_call_index')))
    ids=[r['id'] for r in rows];seen=Counter(ids)
    first=next((row for row in rows if row['precision']['status']=='REJECTED' and row['csv']['status']!='EXACT_ENDPOINTS_CORROBORATED'),None)
    errors=evidence['errors']
    # Keep ID gaps bounded even if malformed captures contain very distant IDs.
    unique=sorted(seen)
    gaps=[(lo+1,hi-1) for lo,hi in zip(unique,unique[1:]) if hi>lo+1]
    return dict(strict_status=frozen['device_assignment'],marker_count=len(ids),ids=ids,
                missing_id_ranges_inclusive=gaps,
                expected_count=38,edge_id_policy='Only internal gaps identifiable without a bound first/last iteration ID',
                duplicates={i:n for i,n in seen.items() if n>1},unconfirmed=evidence['unconfirmed_markers'],
                missing_strict_host_calls=sorted(set(range(38))-{r['strict_host_call'] for r in rows}),
                first_blocking_step=first,first_mapping_error=errors[0] if errors else None,errors=errors,
                csv_tables=[{k:v for k,v in table.items() if k!='rows'} for table in csv_evidence['tables']],steps=rows)


def enclose(tasks,windows):
    rows=[];groups=defaultdict(list)
    for event in tasks:
        start,end=bounds(event)
        hits=[i for i,(lo,hi) in enumerate(windows) if lo<=start and end<=hi]
        intersect=[i for i,(lo,hi) in enumerate(windows) if start<hi and end>lo]
        args=device.args(event);stream=args.get('physicstreamid',args.get('streamid',event['tid']))
        identity=[event['pid'],args.get('modelid'),stream,args.get('taskid'),args.get('batchid'),args.get('subtaskid')]
        state=('ENCLOSED' if len(hits)==1 else 'AMBIGUOUS_BOUNDARY' if hits else 'CROSSING' if intersect else 'OUTSIDE')
        row=dict(event_index=event['event_index'],name=event.get('name'),pid=event['pid'],tid=event['tid'],identity=identity,
                 start_us=str(start),end_us=str(end),duration_us=float(end-start),state=state,candidates=hits,intersections=intersect,call=None)
        if state=='ENCLOSED':
            if identity[3] in (None,'','N/A','-'):row['state']='MISSING_TASK_ID'
            else:groups[(hits[0],*map(str,identity))].append(row)
        rows.append(row)
    for group in groups.values():
        for row in group:
            if len(group)>1:row['state']='DUPLICATE_ID'
            else:row.update(call=row['candidates'][0],state=PROVISIONAL)
    return rows
