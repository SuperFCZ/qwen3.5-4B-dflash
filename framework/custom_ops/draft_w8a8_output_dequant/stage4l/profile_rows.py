"""Keep CSV provenance/type/task/core/unit evidence separate; never pool rows."""
import csv,re
from pathlib import Path
from l_common import read,verify
from profile_metrics_k import collect as raw_collect,norm

def review(raw):
 output=[];tables={};seen={}
 for row in raw['counter_rows']:
  key=row['file']['path']
  if key not in tables:
   p=verify(row['file'])
   with p.open(encoding='utf-8-sig',newline='') as f:tables[key]=list(csv.DictReader(f))
   verify(row['file'])
  record=tables[key][row['line']-2]
  ids={key:value for key,value in record.items() if key and any(s in norm(key) for s in ('task','core','stream','device','optype','opname'))}
  cores={key:value for key,value in ids.items() if 'coreid' in norm(key) and value not in ('','N/A',None)}
  tasks={key:value for key,value in ids.items() if 'taskid' in norm(key) and value not in ('','N/A',None)}
  row_type='CORE_TAGGED' if cores else ('TASK_TAGGED' if tasks else 'UNRESOLVED_AGGREGATE_OR_TASK')
  units={key:('%' if '%' in key or 'ratio' in norm(key) and '%' in str(value) else (re.search(r'\(([^)]+)\)',key).group(1) if re.search(r'\(([^)]+)\)',key) else 'UNRESOLVED')) for key,value in row['fields'].items()}
  signature=(tuple(sorted((key,str(value)) for key,value in record.items() if key)))
  previous=seen.get(signature);origin=dict(file=row['file'],csv_record_number=row['line']-1)
  output.append(dict(**origin,file_kind=Path(key).name,row_type=row_type,identifiers=ids,units=units,raw_record=record,
   duplicate_of=previous,aggregation='NOT_PERFORMED'))
  seen.setdefault(signature,origin)
 return dict(status='ROW_PROVENANCE_REVIEWED_NO_AGGREGATION',rows=output,
  limitations='Unknown type/unit stays UNRESOLVED. Duplicate-looking rows are flagged, not merged. Task/Core/file types are not pooled; no inferred critical path or causal time split.')

def collect(root,op):
 raw=raw_collect(root,op);raw['row_review']=review(raw);return raw

def review_historical(report):
 p=read(verify(report));result={}
 for name,raw in p['raw_profile'].items():
  result[name]=review(raw)
 return dict(report=report,variants=result,status='HISTORICAL_CSV_SOURCE_REVIEW_ONLY',historical_latency_used_as_new_samples=False)
