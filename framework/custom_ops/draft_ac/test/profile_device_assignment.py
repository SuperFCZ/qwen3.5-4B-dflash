"""Observed CANN9 Step Trace attribution, independent of Host wall-time clipping."""
from collections import defaultdict, Counter
from decimal import Decimal, InvalidOperation, localcontext
import re
from profile_step_evidence import corroborate


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


def step_time_check(event):
    """Independent decimal rounding of S, E, D: |D-(E-S)| <= (qS+qE+qD)/2.

    This checks serialization consistency only. It never expands or corrects
    the recorded device interval used to assign tasks. JSON numeric tokens must
    arrive as strings, before any binary float conversion.
    """
    check=dict(status='REJECTED',raw_start_us=event.get('raw_timestamp_us'),
               raw_end_us=args(event).get('iterationend'),raw_duration_us=str(event.get('dur')))
    try:
        start=Decimal(check['raw_start_us']);stop=Decimal(str(check['raw_end_us']))
        duration=Decimal(check['raw_duration_us'])
        if not all(x.is_finite() for x in (start,stop,duration)) or start<0 or stop<=start or duration<0:
            raise ValueError('invalid Step timestamp/duration')
        # Enough decimal digits for cancellation of the epoch and the finest
        # source field; do not depend on the process-wide Decimal precision.
        precision=max(50,max(x.adjusted() for x in (start,stop,duration))-min(x.as_tuple().exponent for x in (start,stop,duration))+8)
        with localcontext() as context:
            context.prec=precision
            quanta=[Decimal(1).scaleb(x.as_tuple().exponent) for x in (start,stop,duration)]
            elapsed=stop-start;error=duration-elapsed;bound=sum(quanta)/2
            check.update(resolution_us={k:str(q) for k,q in zip(('start','end','duration'),quanta)},
                         endpoint_elapsed_us=str(elapsed),signed_error_us=str(error),absolute_error_us=str(abs(error)),
                         allowed_error_min_us=str(-bound),allowed_error_max_us=str(bound),
                         endpoint_rounding_bound_us=str((quanta[0]+quanta[1])/2),
                         duration_rounding_bound_us=str(quanta[2]/2),
                         bound_rule='(start_resolution + end_resolution + duration_resolution) / 2; independent decimal rounding')
            if abs(error)>bound:raise ValueError('inconsistent Iteration End/duration: exceeds serialized precision bound')
        check['status']='CONSISTENT_WITH_SERIALIZED_PRECISION'
    except (KeyError,TypeError,InvalidOperation,ValueError) as error:
        check['reason']=str(error)
    return check


def time_check_summary(checks):
    valid=[c for c in checks if 'signed_error_us' in c]
    def extent(key):
        values=[Decimal(c[key]) for c in valid]
        return dict(min=str(min(values)),max=str(max(values))) if values else None
    return dict(marker_count=len(checks),consistent_count=sum(c['status']=='CONSISTENT_WITH_SERIALIZED_PRECISION' for c in checks),
                rejected_count=sum(c['status']=='REJECTED' for c in checks),signed_error_us=extent('signed_error_us'),
                absolute_error_us=extent('absolute_error_us'),allowed_positive_bound_us=extent('allowed_error_max_us'))


def pre_step_transfers(tasks, owners, proof, bounds, markers, windows, result):
    """Recognize an exact repeated model transfer cohort in the inter-Step gap.

    Count alone is insufficient: every transfer must match a unique Execute
    connection_id, Model/physical stream cohort and nonoverlapping stream peers.
    Task IDs may change across calls; they remain unique within each call.
    """
    streams=defaultdict(list)
    for e in tasks:
        a=args(e);stream=a.get('physicstreamid',a.get('streamid'))
        if stream is not None:streams[e['pid'],str(stream)].append(e)
    peers={}
    for stream in streams.values():
        stream.sort(key=lambda e:float(e['ts']))
        for j,e in enumerate(stream):peers[e['event_index']]=(stream[j-1] if j else None,stream[j+1] if j+1<len(stream) else None)
    cohorts=defaultdict(list)
    connections=defaultdict(list)
    for i,w in enumerate(windows):
        connection=args(w['launch']).get('connectionid')
        if connection is not None:connections[str(connection)].append(i)
    for e in tasks:
        a=args(e);uid=e['event_index']
        if uid in owners or str(a.get('tasktype','')).upper()!='MEMCPY_ASYNC':continue
        start=float(e['ts']);end=start+float(e['dur'])
        candidates=[i for i,(lo,_) in bounds.items() if windows[i]['start']<=start and end<=min(lo,windows[i]['end'])
                    and (i==0 or start>=bounds[i-1][1]) and str(a.get('modelid'))==markers[i]['model_id'] and e['pid']==markers[i]['event']['pid']]
        before,after=peers.get(uid,(None,None))
        neighbors_ok=(before is None or float(before['ts'])+float(before['dur'])<=start) and (after is None or end<=float(after['ts']))
        identity=(e['pid'],str(a.get('modelid')),str(a.get('physicstreamid',a.get('streamid'))),str(a.get('batchid')),str(a.get('subtaskid')),e.get('name'))
        ids_ok=all(re.fullmatch(r'\d+',str(v)) for v in (a.get('modelid'),a.get('taskid'),a.get('physicstreamid',a.get('streamid'))))
        connection_hits=connections.get(str(a.get('connectionid')),[])
        evidence=dict(event=raw(e),host_candidates=candidates,execute_connection_candidates=connection_hits,stream_previous=raw(before) if before else None,stream_next=raw(after) if after else None,
                      stream_nonoverlap=neighbors_ok,identity_valid=bool(ids_ok),identity=list(identity),status='UNPROVEN_PRE_STEP_TRANSFER')
        result['pre_step_candidates'].append(evidence)
        if len(candidates)==1 and connection_hits==candidates and neighbors_ok and ids_ok:
            i=candidates[0];lo=bounds[i][0]
            current=[t for t in tasks if owners.get(t['event_index'])==i]
            previous=[t for t in tasks if i>0 and owners.get(t['event_index'])==i-1]
            evidence['step_first_task']=raw(min(current,key=lambda t:float(t['ts']))) if current else None
            evidence['previous_step_last_task']=raw(max(previous,key=lambda t:float(t['ts'])+float(t['dur']))) if previous else None
            evidence.update(call_index=i,start_us=start,end_us=end,duration_us=float(e['dur']),
                            gap_to_step_start_us=lo-end,previous_step_end_us=bounds[i-1][1] if i else None,
                            step_start_us=lo,host_start_us=windows[i]['start'],host_end_us=windows[i]['end'])
            cohorts[identity].append((e,evidence))
    for cohort in cohorts.values():
        indices=[v['call_index'] for _,v in cohort]
        for _,item in cohort:item['cohort_count']=len(cohort);item['cohort_call_indices']=sorted(indices)
        if len(cohort)!=38 or sorted(indices)!=list(range(38)):continue
        for e,v in cohort:
            uid=e['event_index'];i=v['call_index'];a=args(e)
            owners[uid]=i
            proof[uid]=dict(method='PRE_STEP_TRANSFER',basis='unique matching Execute connection_id; same model/physical stream cohort once per each of 38 Host calls; full transfer in inter-Step gap; stream neighbors nonoverlapping',
                           connection_id=a['connectionid'],iteration_id=markers[i]['iteration_id'],model_id=a['modelid'],task_id=a['taskid'],physical_stream_id=a.get('physicstreamid',a.get('streamid')))
            v['status']='PRE_STEP_TRANSFER';v['assignment']=proof[uid]
            result['pre_step_transfers'].append(v)
    assigned={x['event']['event_index'] for x in result['pre_step_transfers']}
    result['unresolved_tasks']=[x for x in result['unresolved_tasks'] if x['event']['event_index'] not in assigned]


def assign(events, hardware_events, windows, origin, step_evidence=None):
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
                unresolved_tasks=[],outside_host_tasks=[],errors=[],metadata_warnings=[],step_csv_evidence=step_evidence or dict(status='MISSING',tables=[]),step_time_checks=[],pre_step_transfers=[],pre_step_candidates=[],status='UNRESOLVED_DEVICE_ASSIGNMENT')
    owners={};proof={};bounds={};marker_by_call={}
    # A repeated connection_id/Task Id describes a graph task, not a call ID.
    result['task_connection_ids']=dict(Counter(str(args(t).get('connectionid')) for t in tasks))
    markers.sort(key=lambda m:float(m['event']['ts']))
    if markers:
        for m in markers:
            e=m['event'];info={**raw(e),**{k:m[k] for k in ('model_id','iteration_id')}}
            result['excluded_iteration_markers'].append(info)
            check=step_time_check(e)
            if check['status']=='REJECTED' and 'signed_error_us' in check:check['metadata_status']='DURATION_METADATA_INCONSISTENT'
            result['step_time_checks'].append(dict(iteration_id=m['iteration_id'],model_id=m['model_id'],event_index=e['event_index'],**check))
            info['time_consistency']=check
            if check['status']=='REJECTED':
                if 'signed_error_us' in check:
                    check['metadata_status']='DURATION_METADATA_INCONSISTENT'
                    endpoint=corroborate(result['step_csv_evidence'],m['model_id'],m['iteration_id'],check)
                    info['endpoint_evidence']=endpoint
                    result['metadata_warnings'].append(dict(iteration_id=m['iteration_id'],status='DURATION_METADATA_INCONSISTENT',time_consistency=check,endpoint_evidence=endpoint))
                    if endpoint['status']!='EXACT_ENDPOINTS_CORROBORATED':
                        result['errors'].append(dict(marker=info,reason='DURATION_METADATA_INCONSISTENT: endpoints not independently corroborated; '+check['reason']));continue
                else:
                    result['errors'].append(dict(marker=info,reason=check['reason']));continue
            with localcontext() as context:
                context.prec=50
                begin=float(Decimal(check['raw_start_us'])-Decimal(origin))
                end=float(Decimal(str(check['raw_end_us']))-Decimal(origin))
            hits=[i for i,w in enumerate(windows) if w['start']<=begin<w['end']]
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
            pre_step_transfers(tasks,owners,proof,bounds,marker_by_call,windows,result)
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
                        for candidate in result['pre_step_candidates']:
                            if candidate['event']['event_index']==e['event_index']:candidate['status']='REJECTED_DUPLICATE_TASK'
                        result['unresolved_tasks'].append(dict(event=raw(e),reason='duplicate task identity within device iteration'))
            if len(set(owners.values()))!=38:result['errors'].append(dict(reason='one or more device Steps have no uniquely identified tasks'))
            if not result['unresolved_tasks'] and not result['errors']:result['status']='DURATION_METADATA_INCONSISTENT' if result['metadata_warnings'] else 'DEVICE_STEPS_VERIFIED'
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
        owners.clear();proof.clear();result['outside_host_tasks']=[]
        for candidate in result['pre_step_candidates']:
            if candidate['status']=='PRE_STEP_TRANSFER':candidate['status']='REVOKED_STEP_MAPPING'
        result['unresolved_tasks']=[dict(event=raw(e),reason='invalid Step/Host mapping') for e in tasks]
    result['pre_step_transfers']=[t for t in result['pre_step_transfers'] if t['event']['event_index'] in owners]
    result['pre_step_summary']=dict(count=len(result['pre_step_transfers']),duration_sum_us=sum(t['duration_us'] for t in result['pre_step_transfers']),excluded_from_step_kernel_time=True)
    result['step_time_summary']=time_check_summary(result['step_time_checks'])
    result['remaining_failure_reasons']=dict(Counter([x['reason'] for x in result['errors']]+[x['reason'] for x in result['unresolved_tasks']]+[x['status'] for x in result['metadata_warnings']]+['unconfirmed Iteration marker']*len(quarantine)))
    result['assigned_task_count']=len(owners);result['real_event_count']=len(tasks)
    result['event_accounting']=dict(step_tasks=len(owners)-len(result['pre_step_transfers']),pre_step_transfers=len(result['pre_step_transfers']),unresolved=len(result['unresolved_tasks']),outside=len(result['outside_host_tasks']))
    result['device_windows']={str(i):dict(start_us=lo,end_us=hi) for i,(lo,hi) in bounds.items()}
    return owners,proof,result,tasks
