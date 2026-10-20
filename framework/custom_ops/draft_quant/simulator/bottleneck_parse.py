"""Conservative offline parsers. Unknown schemas never imply timing semantics."""
from collections import Counter,defaultdict
import csv
from decimal import Decimal,InvalidOperation
import json
from pathlib import Path
import re

DOC='https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_simulator_performance_data.md'
INSTR=('instr','addr','pipe','call_count','cycles','running_time(us)','detail')
CODE=('code','call_count','cycles','running_time(us)')


def decimal(value):
    try:result=Decimal(str(value))
    except InvalidOperation:raise ValueError('invalid numeric field: '+str(value))
    if not result.is_finite() or result<0:raise ValueError('negative/nonfinite metric')
    return result


def pc(value,trace=False):
    # Receiver evidence: CSV addr is decimal, Trace args.pc_addr is hexadecimal.
    # No relocation/base-address inference is allowed.
    if not isinstance(value,str):return None
    if re.fullmatch(r'0[xX][0-9a-fA-F]+',value):return int(value,16)
    if not trace and re.fullmatch(r'[0-9]+',value):return int(value,10)
    return None


def csv_table(path,kind):
    with Path(path).open(newline='',encoding='utf-8-sig') as stream:
        reader=csv.reader(stream);headers=next(reader,None)
        if not headers or len(headers)!=len(set(headers)):raise ValueError('missing/duplicate CSV header')
        rows=[]
        for line,row in enumerate(reader,2):
            if len(row)!=len(headers):raise ValueError(f'CSV row width differs at {path}:{line}')
            rows.append(dict(zip(headers,row)))
    expected=INSTR if kind=='instructions' else CODE
    known=set(headers)==set(expected)
    if known:
        for row in rows:
            count=decimal(row['call_count']);decimal(row['cycles']);decimal(row['running_time(us)'])
            if count!=count.to_integral_value():raise ValueError('fractional instruction count')
    return dict(status='DOCUMENTED_CSV_SCHEMA' if known else 'UNSUPPORTED_CSV_SCHEMA',
                schema_reference=DOC,headers=headers,row_count=len(rows),samples=rows[:3]),rows if known else []


def trace_schema(path):
    data=json.loads(Path(path).read_text(),parse_float=Decimal)
    events=data.get('traceEvents') if isinstance(data,dict) else data
    if not isinstance(events,list):raise ValueError('not a Trace Event list')
    metadata=[];samples=[];seen=set();fields=Counter();phases=Counter();args_keys=Counter();names=Counter();sync_samples={}
    for event in events:
        if not isinstance(event,dict):raise ValueError('non-object trace event')
        fields.update(event.keys());phases[str(event.get('ph'))]+=1
        args=event.get('args',{})
        if isinstance(args,dict):args_keys.update(args.keys())
        if event.get('ph')=='M':
            if len(metadata)<100 and event not in metadata:metadata.append(event)
        else:
            name=str(event.get('name',''));names[name]+=1
            for term in ('wait','barrier','set_flag','pipe'):
                if term in name.lower() and len(sync_samples.setdefault(term,[]))<2:sync_samples[term].append(event)
            signature=(str(event.get('ph')),str(event.get('pid')),str(event.get('tid')),tuple(sorted(args)) if isinstance(args,dict) else ())
            if signature not in seen and len(samples)<32:seen.add(signature);samples.append(event)
    # Retain full fields/args, bounded by event samples; never choose a clock or
    # infer dependency edges from adjacency, source order, or thread labels.
    return dict(status='TRACE_SEMANTICS_UNRESOLVED',event_count=len(events),event_fields=dict(fields),
        phases=dict(phases),argument_fields=dict(args_keys),metadata=metadata,samples=samples,
        event_names=dict(names),lexical_sync_samples=sync_samples,lexical_sample_semantics='Name matches only, not verified waits or dependency edges',
        top_level_fields=list(data) if isinstance(data,dict) else ['<event-list>'],
        top_level_metadata={k:v for k,v in data.items() if k!='traceEvents'} if isinstance(data,dict) else {},
        simulator_elapsed_cycles=None,pipe_active_cycles=None,synchronization_wait_fraction=None,
        reason='Receiver trace clock, PC correlation and dependency/wait interval fields require verification; displayTimeUnit alone is not a cycle clock.')


def symbolizer_frames(text,addresses):
    result={};current=None;lines=[]
    def finish():
        if current is None:return
        if current in result:raise ValueError('duplicate symbolizer address')
        frames=[]
        if len(lines)%2:raise ValueError('unexpected addr2line function/location pairs')
        for function,location in zip(lines[::2],lines[1::2]):
            match=re.fullmatch(r'(.+):(\d+)(?: \(discriminator \d+\))?',location)
            if match and match[1]!='??' and int(match[2])>0:
                frames.append(dict(function=function,file=match[1],line=int(match[2])))
        result[current]=frames
    for line in text.splitlines():
        if re.fullmatch(r'0[xX][0-9a-fA-F]+',line.strip()):
            finish();current=int(line.strip(),16);lines=[]
        elif current is not None:lines.append(line)
        elif line.strip():raise ValueError('unexpected addr2line preamble')
    finish()
    if set(result)!=set(addresses):raise ValueError('symbolizer PC set differs; no offset/rebase inference')
    return result


def source_category(frames,source,focus):
    categories=set();verified=[]
    for frame in frames:
        path=Path(frame['file'])
        if not path.is_absolute() or path.resolve()!=source.resolve():continue
        verified.append(frame)
        for category,lines in focus.items():
            if frame['line'] in lines:categories.add(category)
    return (next(iter(categories)) if len(categories)==1 else 'UNMAPPED'),verified


def instruction_summary(rows,mapping,source,focus):
    groups=defaultdict(lambda:dict(call_count=0,instruction_cycles=Decimal(0),static_rows=0))
    pipes=defaultdict(lambda:dict(call_count=0,instruction_cycles=Decimal(0)))
    details=[]
    for row in rows:
        address=pc(row['addr']);frames=mapping.get(address,[]) if address is not None else []
        category,verified=source_category(frames,source,focus)
        count=int(decimal(row['call_count']));cycles=decimal(row['cycles'])
        groups[category]['call_count']+=count;groups[category]['instruction_cycles']+=cycles;groups[category]['static_rows']+=1
        pipe=row['pipe'];pipes[pipe]['call_count']+=count;pipes[pipe]['instruction_cycles']+=cycles
        details.append(dict(**row,category=category,dwarf_frames=frames,verified_source_frames=verified,
            mapping_reason='unique verified source category' if category!='UNMAPPED' else 'missing/ambiguous PC, DWARF or verified source line'))
    return dict(by_source_operation=dict(groups),by_reported_pipe=dict(pipes),instructions=details,
        count_semantics='Dynamic instruction calls, not C++ API invocations; do not add code CSV counts to instruction CSV counts.',
        cycle_semantics='Sum of documented instruction execution cycles; overlap is possible. NOT pipe active time, exposed wait, kernel span or critical path.',
        cpp_api_call_count='UNRESOLVED',k_tile_assignment='UNRESOLVED_NOT_INFERRED_FROM_ORDER')


def union_length(intervals):
    """Generic exact interval arithmetic; used only after a clock is verified."""
    merged=[]
    for a,b in sorted(intervals):
        if b<a:raise ValueError('negative interval')
        if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
        else:merged.append([a,b])
    return sum((b-a for a,b in merged),Decimal(0))



def copy_sites(source):
    lines=source.read_text().splitlines();sites=[]
    roles={'CopyRawTile(qBanks[(1 - bank)': 'RAW_NEXT_K_COPY',
           'CopyRawTile(qBanks, sBanks': 'RAW_INITIAL_COPY',
           'CopyRawTile(qLocal, sLocal': 'SERIAL_CURRENT_COPY'}
    for i,line in enumerate(lines):
        for prefix,role in roles.items():
            if line.strip().startswith(prefix):
                end=next((j for j in range(i,len(lines)) if ';' in lines[j]),None)
                if end is not None:sites.append(dict(role=role,first_line=i+1,last_line=end+1,source='\n'.join(lines[i:end+1])))
    return sites


def copy_role(frames,sites):
    roles={site['role'] for frame in frames for site in sites if site['first_line']<=frame['line']<=site['last_line']}
    return next(iter(roles)) if len(roles)==1 else 'UNRESOLVED'


def trace_analysis(path,rows,mapping,source,focus):
    data=json.loads(Path(path).read_text(),parse_float=Decimal)
    if not isinstance(data,dict) or not {'displayTimeUnit','profilingType','schemaVersion','traceEvents'}<=data.keys():
        return dict(status='UNSUPPORTED_TRACE_SCHEMA',kernel_elapsed_cycles=None)
    events=data['traceEvents'];process={};conflicts=[]
    for e in events:
        if e.get('ph')=='M' and e.get('name')=='process_name':
            pid=e.get('pid');name=e.get('args',{}).get('name')
            if pid in process and process[pid]!=name:conflicts.append(str(pid))
            process[pid]=name
    csv_counts=Counter()
    for r in rows:
        address=pc(r['addr'])
        if address is not None:csv_counts[(address,r['instr'],r['pipe'])]+=int(decimal(r['call_count']))
    matched=[];unresolved=[];trace_counts=Counter();zero=0;sites=copy_sites(source)
    for i,e in enumerate(events):
        if e.get('ph')!='X':continue
        args=e.get('args',{});address=pc(args.get('pc_addr',''),trace=True) if isinstance(args,dict) else None
        key=(address,e.get('name'),process.get(e.get('pid')))
        if key not in csv_counts:
            unresolved.append(dict(event=i,pc=address,name=e.get('name'),pipe=key[2]));continue
        start=decimal(e['ts']);duration=decimal(e['dur']);end=start+duration
        category,verified=source_category(mapping.get(address,[]),source,focus)
        matched.append(dict(event=i,pc=f'0x{address:x}',instr=e['name'],pipe=key[2],start=start,end=end,
                            category=category,source_frames=verified,profiler_code_hint=args.get('code'),
                            detail=args.get('detail'),copy_role=copy_role(verified,sites),k_tile='UNRESOLVED'))
        trace_counts[key]+=1;zero+=duration==0
    mismatch=[dict(pc=f'0x{k[0]:x}',instr=k[1],pipe=k[2],csv=v,trace=trace_counts[k]) for k,v in csv_counts.items() if v!=trace_counts[k]]
    groups=defaultdict(list);source_groups=defaultdict(list)
    for e in matched:
        groups[e['pipe']].append((e['start'],e['end']));source_groups[e['category']].append((e['start'],e['end']))
    all_intervals=[(e['start'],e['end']) for e in matched]
    span=max((b for a,b in all_intervals),default=Decimal(0))-min((a for a,b in all_intervals),default=Decimal(0))
    def stats(intervals):
        total=sum((b-a for a,b in intervals),Decimal(0));union=union_length(intervals)
        return dict(event_count=len(intervals),interval_sum=total,interval_union=union,overlap_excess=total-union)
    overlaps={}
    if 'MTE2' in groups:
        for pipe in groups:
            if pipe!='MTE2':
                overlaps['MTE2 & '+pipe]=union_length(groups['MTE2'])+union_length(groups[pipe])-union_length(groups['MTE2']+groups[pipe])
    next_copies=[(e['start'],e['end']) for e in matched if e['copy_role']=='RAW_NEXT_K_COPY' and e['pipe']=='MTE2']
    compute=[interval for pipe,intervals in groups.items() if pipe in ('CUBE','VEC','VECTOR') for interval in intervals]
    prefetch=dict(status='OBSERVED_SOURCE_PC_INTERVALS' if next_copies else 'UNRESOLVED',
        next_copy_events=len(next_copies),next_copy_interval_union=union_length(next_copies),
        next_copy_compute_intersection=union_length(next_copies)+union_length(compute)-union_length(next_copies+compute),
        source_sites=sites,interpretation='Source-verified next-copy call site only. No per-K numbering or proof of reduced exposed wait.')
    coherent=bool(rows and matched) and not unresolved and not mismatch and not conflicts
    return dict(status='PC_INSTRUCTION_COUNTS_MATCHED' if coherent else 'UNRESOLVED_TRACE_CORRELATION',
        clock=dict(unit='raw Chrome trace ts/dur units; not cycles or hardware wall-time',
                   displayTimeUnit=data['displayTimeUnit'],profilingType=data['profilingType'],schemaVersion=data['schemaVersion']),
        trace_instruction_span=span,interval_union=union_length(all_intervals),by_pipe={k:stats(v) for k,v in groups.items()},
        by_source_operation={k:stats(v) for k,v in source_groups.items()},mte2_pairwise_intersections=overlaps,
        prefetch_callsite_overlap=prefetch,zero_duration_events=zero,unresolved_events=unresolved,count_mismatches=mismatch,metadata_conflicts=conflicts,
        intervals=matched,interval_semantics='Observed X-event occupancy, not proof of useful pipe activity. Zero-width events can have nonzero CSV cycles. Pairwise intersections must not be summed.',
        kernel_elapsed_cycles=None,kernel_critical_path='UNRESOLVED_NO_DEPENDENCY_GRAPH',
        synchronization_wait_fraction='UNRESOLVED_NO_VERIFIED_WAIT_DEPENDENCIES',
        k_tile_assignment='UNRESOLVED_NOT_INFERRED_FROM_ORDER')
