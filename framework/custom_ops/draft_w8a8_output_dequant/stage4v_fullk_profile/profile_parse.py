"""Keep task/core/file/stream/unit provenance; no overlapping-pipe time sums."""
import csv
import json
import math
import re
import statistics
from profile_evidence import identity,verify
OP='dflashw8a8fullkchannel'


def norm(x):return ''.join(c.lower() for c in str(x) if c.isalnum())


def field(row,names):
 values={str(v) for k,v in row.items() if k and norm(k) in names and v not in ('',None,'N/A','NA','-')}
 return next(iter(values)) if len(values)==1 else None


def ids(row):
 return {k:field(row,names) for k,names in dict(device={'deviceid','device'},stream={'streamid'},task={'taskid'},core={'coreid','aicoreid'},context={'contextid'},pid={'pid','processid'}).items()}


def names(row):return {k:v for k,v in row.items() if k and norm(k) in ('opname','optype','kernelname','taskname') and isinstance(v,str)}


def matched(row):return any(OP in norm(v) for v in names(row).values())


def kind(path):
 name=path.name.lower()
 return 'op_summary' if name.startswith('op_summary') else 'task_time' if name.startswith('task_time') else 'other_csv'


def metric(key,value):
 k=norm(key);group=next((x for x in ('mte1','mte2','mte3','scalar','vec','mac','cube') if x in k),None)
 if not group and ('duration' in k or k.startswith('aicoretime')):group='duration'
 if any(x in k for x in ('bandwidth','memory','bytes','ddr','l0','l1','ubread','ubwrite')):group='memory'
 if not group:return None
 unit_match=re.search(r'[\(\[]([^\)\]]+)[\)\]]',key)
 unit=unit_match.group(1).strip() if unit_match else '%' if isinstance(value,str) and value.strip().endswith('%') else 'UNAVAILABLE'
 try:number=float(str(value).strip().rstrip('%'))
 except (ValueError,TypeError):number=None
 if number is not None and (not math.isfinite(number) or number<0):number=None
 eligible=number is not None and unit!='UNAVAILABLE'
 if ('time' in k or group=='duration') and unit not in ('us','µs','μs','ns','ms','s'):eligible=False
 return dict(category=group,raw=value,value=number,unit=unit,
  statistic_eligible=eligible,interpretation='RAW_COUNTER_NOT_CRITICAL_PATH_CONTRIBUTION')


def session(path,root):
 parts=path.relative_to(root).parts
 for i,x in enumerate(parts):
  if x.startswith('PROF_'):return '/'.join(parts[:i+1])
 return '.'


def task_key(row):
 x=row['identifiers']
 if any(x[k] is None for k in ('device','stream','task')):return None
 return (row['session'],x['device'],x['stream'],x['task'],x['context'],x['pid'])


def collect(root):
 root=root.resolve(strict=True);files=[];candidates=[];skipped=[];timeline=[]
 for path in sorted(root.rglob('*')):
  if not path.is_file():continue
  if root not in path.resolve().parents:raise ValueError('profiler artifact escapes root')
  f=identity(path);files.append(f)
  if path.suffix.lower()=='.csv':
   if f['bytes']>64*1024*1024:skipped.append(dict(file=f,reason='CSV_SIZE_LIMIT'));continue
   try:
    with path.open(encoding='utf-8-sig',newline='') as stream:
     reader=csv.DictReader(stream)
     if not reader.fieldnames or len(set(reader.fieldnames))!=len(reader.fieldnames):raise ValueError('missing/duplicate CSV headers')
     for record_number,raw in enumerate(reader,1):
      if None in raw:raise ValueError('CSV has extra unnamed values')
      ident=ids(raw);tt=field(raw,{'tasktype','optypeid'})
      rt='CORE_TAGGED' if ident['core'] is not None else 'TASK_TAGGED' if ident['task'] is not None else 'UNRESOLVED_AGGREGATE_OR_TASK'
      count=field(raw,{'count','taskcount','opcount'})
      if count is not None:
       try:
        if float(count)>1:rt='AGGREGATE'
       except ValueError:rt='UNRESOLVED_AGGREGATE_OR_TASK'
      metrics={k:m for k,v in raw.items() if (m:=metric(k,v)) is not None}
      candidates.append(dict(file=f,csv_record_number=record_number,physical_line_end=reader.line_num,file_kind=kind(path),session=session(path,root),
       row_type=rt,task_type=tt or 'UNAVAILABLE',identifiers=ident,names=names(raw),raw_record=raw,metrics=metrics,
       operator_binding='DIRECT_NAME' if matched(raw) else 'UNRESOLVED',duplicate_of=None))
   except (ValueError,UnicodeError,csv.Error) as x:skipped.append(dict(file=f,reason=str(x)));candidates=[r for r in candidates if r['file']!=f]
  elif path.suffix.lower()=='.json' and path.name.startswith('msprof_'):
   if f['bytes']>128*1024*1024:skipped.append(dict(file=f,reason='TIMELINE_SIZE_LIMIT'));continue
   try:
    obj=json.loads(path.read_text());events=obj.get('traceEvents',[]) if isinstance(obj,dict) else obj
    if not isinstance(events,list):raise ValueError('unknown timeline schema')
    # displayTimeUnit is a display preference, not proof of timestamp encoding.
    unit=obj.get('timeUnit') if isinstance(obj,dict) else None
    for index,event in enumerate(events):
     if not isinstance(event,dict) or event.get('ph')!='X':continue
     args=event.get('args') if isinstance(event.get('args'),dict) else {}
     if OP not in norm(event.get('name','')) and not matched(args):continue
     timeline.append(dict(file=f,event_index=index,raw_event=event,identifiers=ids(args),
      task_type=field(args,{'tasktype'}) or 'UNAVAILABLE',timestamp_unit=unit or 'UNAVAILABLE_NO_EXPLICIT_TIMESTAMP_UNIT',
      measured_phase_attribution='UNAVAILABLE_NO_SOURCE_OR_PC_MAPPING'))
   except (ValueError,UnicodeError) as x:skipped.append(dict(file=f,reason=str(x)))
  verify(f)
 # Unnamed task_time rows can only join a unique named op_summary task within a session.
 index={}
 for row in candidates:
  key=task_key(row)
  if key is not None and row['file_kind']=='op_summary' and row['operator_binding']=='DIRECT_NAME':
   index.setdefault(key,{})[json.dumps(row['raw_record'],sort_keys=True)]=row
 for row in candidates:
  if row['operator_binding']=='UNRESOLVED' and row['file_kind']=='task_time' and not row['names']:
   anchors=index.get(task_key(row),{})
   if len(anchors)==1:
    anchor=next(iter(anchors.values()));row['operator_binding']='UNIQUE_TASK_STREAM_DEVICE_JOIN';row['joined_to']=dict(file=anchor['file'],csv_record_number=anchor['csv_record_number'])
 # Keep unbound task_time rows in a separate diagnostic list; never include them in statistics.
 rows=[r for r in candidates if r['operator_binding']!='UNRESOLVED'];unbound=[r for r in candidates if r['file_kind']=='task_time' and r['operator_binding']=='UNRESOLVED']
 seen={};groups={}
 for row in rows:
  signature=json.dumps([row['session'],row['file_kind'],row['raw_record']],sort_keys=True)
  origin=dict(file=row['file'],csv_record_number=row['csv_record_number'])
  row['duplicate_of']=seen.get(signature);seen.setdefault(signature,origin)
  key=json.dumps([row['file']['path'],row['file_kind'],row['row_type'],row['task_type'],row['names'],
   {k:v for k,v in row['identifiers'].items() if k!='task'},sorted(row['raw_record']),{k:m['unit'] for k,m in row['metrics'].items()}],sort_keys=True)
  group=groups.setdefault(key,dict(file=row['file'],file_kind=row['file_kind'],row_type=row['row_type'],task_type=row['task_type'],names=row['names'],
   identifiers={k:v for k,v in row['identifiers'].items() if k!='task'},records=[],duplicate_records=[],metrics={}))
  group['records'].append(row['csv_record_number'])
  if row['duplicate_of']:group['duplicate_records'].append(row['csv_record_number']);continue
  # No core/task/aggregate/stream/unit mixing and no unidentified hardware-task summaries.
  eligible=row['row_type'] in ('TASK_TAGGED','CORE_TAGGED') and norm(row['task_type']) in ('aicore','aic','aivector','aiv') and task_key(row) is not None
  if eligible:
   for k,m in row['metrics'].items():
    if not m['statistic_eligible']:continue
    s=group['metrics'].setdefault(k,dict(category=m['category'],unit=m['unit'],samples=[],records=[]));s['samples'].append(m['value']);s['records'].append(row['csv_record_number'])
 for g in groups.values():
  for s in g['metrics'].values():s.update(count=len(s['samples']),median=statistics.median(s['samples']),min=min(s['samples']),max=max(s['samples']))
 availability={category:'AVAILABLE_IDENTIFIED_ROWS' if any(m['category']==category for g in groups.values() for m in g['metrics'].values()) else 'UNAVAILABLE' for category in ('duration','cube','mac','mte1','mte2','mte3','vec','scalar','memory')}
 tasks={task_key(r) for r in rows if r['file_kind']=='op_summary' and r['row_type']=='TASK_TAGGED' and norm(r['task_type']) in ('aicore','aic') and task_key(r) is not None and not r['duplicate_of']}
 core_available=any(r['row_type']=='CORE_TAGGED' for r in rows)
 return dict(status='PROFILE_ROWS_PARSED' if rows or timeline else 'METRICS_UNAVAILABLE',files=files,rows=rows,groups=list(groups.values()),timeline=timeline,unbound_task_time=unbound,skipped=skipped,
  availability=availability,primary_device_tasks=len(tasks),expected_runner_executions=2,
  task_coverage='TWO_IDENTIFIED_AICORE_TASKS' if len(tasks)==2 else 'UNRESOLVED_NOT_EXACTLY_TWO_IDENTIFIED_TASKS',
  core_tail='RAW_CORE_ROWS_REQUIRE_SAME_TASK_REVIEW' if core_available else 'UNAVAILABLE_NO_CORE_ROWS',
  causal_phase_times={k:'UNAVAILABLE_NO_PC_SOURCE_MAPPING' for k in ('ND_TO_NZ','VDEQ16','NZ_TO_ND','End_sync','Guard_Arena')},
  pipe_sum='FORBIDDEN_OVERLAPPING_ACTIVITIES',benchmark_latency=False,duplicate_policy='FLAG_EXACT_RECORDS_EXCLUDE_FROM_SAME_GROUP_STATISTICS_NEVER_MERGE_TABLE_TYPES')
