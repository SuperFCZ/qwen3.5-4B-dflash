"""Observed CANN9 Step Trace attribution, independent of Host wall-time clipping."""
from collections import defaultdict, Counter
from decimal import Decimal, InvalidOperation
import re


def args(event):
    return {re.sub('[^a-z0-9]','',str(k).lower()):v for k,v in event.get('args',{}).items()}


def raw(event):
    return {k:event[k] for k in ('event_index','name','ph','pid','tid','cat','ts','dur','raw_timestamp_us','args') if k in event}


def step_marker(event, threads):
    """Require all three independent observed identifiers, not a name heuristic."""
    name=re.fullmatch(r'Iteration (\d+)',str(event.get('name','')))
    names=threads.get((event.get('pid'),event.get('tid')),set())
    model=[re.fullmatch(r'Step Trace\(Model ID:(\d+)\)',n) for n in names]
    model=[m.group(1) for m in model if m]
    a=args(event)
    if name and event.get('cat')=='Iteration Time' and len(names)==1 and len(model)==1 and str(a.get('iterationid'))==name.group(1) and not a.get('tasktype'):
        return dict(model_id=model[0],iteration_id=int(name.group(1)))
    return None


def assign(events, hardware_events, windows, origin):
    threads=defaultdict(set)
    for e in events:
        if e.get('ph')=='M' and e.get('name')=='thread_name':threads[e.get('pid'),e.get('tid')].add(str(e.get('args',{}).get('name','')))
    metadata=[e for e in events if e.get('ph')=='M' and e.get('name') in ('process_name','thread_name')]
    markers=[];tasks=[];quarantine=[]
    for e in hardware_events:
        marker=step_marker(e,threads)
        if marker:markers.append(dict(event=e,**marker))
        elif re.fullmatch(r'Iteration \d+',str(e.get('name',''))) or e.get('cat')=='Iteration Time':quarantine.append(raw(e))
        else:tasks.append(e)
    result=dict(rule='CANN9 Step Trace start anchored to unique Host Execute/Sync; full task interval contained in same-model device iteration',
                metadata=metadata,excluded_iteration_markers=[],unconfirmed_markers=quarantine,
                unresolved_tasks=[],outside_host_tasks=[],errors=[],status='UNRESOLVED_DEVICE_ASSIGNMENT')
    owners={};proof={};bounds={};marker_by_call={}
    # A repeated connection_id/Task Id describes a graph task, not a call ID.
    result['task_connection_ids']=dict(Counter(str(args(t).get('connectionid')) for t in tasks))
    markers.sort(key=lambda m:float(m['event']['ts']))
    if markers:
        for m in markers:
            e=m['event'];info={**raw(e),**{k:m[k] for k in ('model_id','iteration_id')}}
            result['excluded_iteration_markers'].append(info)
            try:
                start=Decimal(e['raw_timestamp_us']);stop=Decimal(str(args(e)['iterationend']))
                # Check only the declared timestamp's decimal serialization precision.
                # This is NOT a task assignment tolerance; intervals are never expanded.
                quantum=Decimal(1).scaleb(stop.as_tuple().exponent)
                if (start+Decimal(str(e['dur']))).quantize(quantum)!=stop or stop<=start:raise ValueError('inconsistent Iteration End/duration')
                begin=float(start-Decimal(origin));end=float(stop-Decimal(origin))
            except (KeyError,InvalidOperation,ValueError) as error:
                result['errors'].append(dict(marker=info,reason=str(error)));continue
            hits=[i for i,w in enumerate(windows) if w['launch_end']<=begin<w['end']]
            info.update(start_us=begin,end_us=end,host_candidates=hits)
            if len(hits)!=1:result['errors'].append(dict(marker=info,reason='Step start has no unique Host call'));continue
            call=hits[0]
            if call in bounds:result['errors'].append(dict(marker=info,reason='multiple Steps anchored to one Host call'));continue
            bounds[call]=(begin,end);marker_by_call[call]=m
            info['host_call_index']=call
        identities={(m['event']['pid'],m['event']['tid'],m['model_id']) for m in markers}
        ids=[m['iteration_id'] for m in markers]
        if len(markers)!=38 or len(bounds)!=38 or len(identities)!=1 or len(set(ids))!=38 or any(b!=a+1 for a,b in zip(ids,ids[1:])):
            result['errors'].append(dict(reason='requires 38 unique consecutive Steps, one model/track, one per Host call'))
        ordered=[bounds[i] for i in sorted(bounds)]
        if any(b[0]<a[1] for a,b in zip(ordered,ordered[1:])):result['errors'].append(dict(reason='device Steps overlap or reorder'))
        if not result['errors'] and not quarantine:
            for e in tasks:
                a=args(e);start=float(e['ts']);end=start+float(e['dur'])
                hits=[i for i,(lo,hi) in bounds.items() if lo<=start and end<=hi and str(a.get('modelid'))==marker_by_call[i]['model_id'] and e.get('pid')==marker_by_call[i]['event']['pid']]
                if len(hits)==1 and a.get('tasktype') and a.get('taskid') is not None and a.get('physicstreamid',a.get('streamid')) is not None:
                    i=hits[0];owners[e['event_index']]=i
                    proof[e['event_index']]=dict(method='MODEL_TASK_WITHIN_ANCHORED_STEP',iteration_id=marker_by_call[i]['iteration_id'],model_id=a['modelid'],task_id=a['taskid'],physical_stream_id=a.get('physicstreamid',a.get('streamid')))
                else:
                    item=dict(event=raw(e),reason='not a unique same-model task fully inside one device Step',device_candidates=hits)
                    if any(start<hi and end>lo for lo,hi in bounds.values()) or a.get('modelid') is not None:result['unresolved_tasks'].append(item)
                    else:result['outside_host_tasks'].append(item)
            # Duplicated Task identity within a single model iteration is ambiguous.
            keys=defaultdict(list)
            for e in tasks:
                uid=e['event_index']
                if uid in owners:
                    a=args(e);keys[owners[uid],e['pid'],str(a.get('modelid')),str(a.get('physicstreamid',a.get('streamid'))),str(a.get('taskid')),str(a.get('batchid')),str(a.get('subtaskid'))].append(e)
            for duplicate in keys.values():
                if len(duplicate)>1:
                    for e in duplicate:
                        owners.pop(e['event_index']);proof.pop(e['event_index'])
                        result['unresolved_tasks'].append(dict(event=raw(e),reason='duplicate task identity within device iteration'))
            if len(set(owners.values()))!=38:result['errors'].append(dict(reason='one or more device Steps have no uniquely identified tasks'))
            if not result['unresolved_tasks'] and not result['errors']:result['status']='DEVICE_STEPS_VERIFIED'
    else:
        result['errors'].append(dict(reason='no verified CANN9 Step Trace; Host enclosure is provisional only'))
        # Backward-compatible partial statistics for legacy timelines. Never claim
        # complete attribution based on this; crossing events have no owner.
        for e in tasks:
            start=float(e['ts']);end=start+float(e['dur'])
            hits=[i for i,w in enumerate(windows) if w['start']<=start and end<=w['end']]
            if len(hits)==1:
                owners[e['event_index']]=hits[0];proof[e['event_index']]=dict(method='PROVISIONAL_HOST_ENCLOSURE')
            elif any(start<w['end'] and end>w['start'] for w in windows):result['unresolved_tasks'].append(dict(event=raw(e),reason='Host boundary crossing without device Step'))
            else:result['outside_host_tasks'].append(dict(event=raw(e),reason='outside all Host windows'))
    if markers and (result['errors'] or quarantine):
        owners.clear();proof.clear()
        result['unresolved_tasks']=[dict(event=raw(e),reason='invalid Step/Host mapping') for e in tasks]
    result['assigned_task_count']=len(owners);result['real_event_count']=len(tasks)
    result['device_windows']={str(i):dict(start_us=lo,end_us=hi) for i,(lo,hi) in bounds.items()}
    return owners,proof,result,tasks
