"""Read-only cross-artifact Step evidence. Never infer Start from End-Duration."""
import csv
import hashlib
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
import re


def norm(value):return re.sub('[^a-z0-9]','',str(value).lower())


def load(directory):
    tables=[]
    for path in sorted(Path(directory).glob('step_trace*.csv')):
        if path.is_symlink():raise ValueError('symlink step_trace CSV is not accepted')
        item=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=[],status='READ')
        try:
            with path.open(encoding='utf-8-sig',newline='') as stream:
                reader=csv.DictReader(stream);item['headers']=reader.fieldnames
                if not reader.fieldnames or len({norm(k) for k in reader.fieldnames})!=len(reader.fieldnames):raise ValueError('missing/duplicate CSV headers')
                for line,row in enumerate(reader,2):
                    if None in row or any(v is None for v in row.values()):raise ValueError('malformed CSV row')
                    item['rows'].append(dict(line=line,fields=row))
        except (OSError,ValueError,csv.Error,UnicodeError) as error:item.update(status='UNPARSED',error=str(error))
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('step_trace CSV changed during read')
        tables.append(item)
    return dict(status='AVAILABLE' if tables else 'MISSING',tables=tables,
                limitation='Separate exported fields can corroborate endpoints independently of Duration, but are not an independent hardware capture.')


def corroborate(evidence,model_id,iteration_id,check):
    matches=[];unsupported=[]
    unreadable=[dict(path=t['path'],status=t['status'],error=t.get('error')) for t in evidence.get('tables',[]) if t['status']!='READ']
    # Explicit units only; unknown headers remain raw evidence, never guessed.
    for table in evidence.get('tables',[]):
        if table['status']!='READ':continue
        for row in table['rows']:
            fields={norm(k):v.strip() for k,v in row['fields'].items()}
            if fields.get('iterationid')!=str(iteration_id):continue
            record=dict(path=table['path'],sha256=table['sha256'],line=row['line'],fields=row['fields'])
            comparisons={}
            for field,key in (('iterationendus','raw_end_us'),('iterationtimeus','raw_duration_us'),('fpstartus','raw_start_us')):
                if field not in fields:continue
                try:
                    with localcontext() as context:
                        context.prec=50
                        a,b=Decimal(fields[field]),Decimal(str(check[key]))
                        comparisons[field]=dict(raw_csv=fields[field],raw_json=check[key],csv_minus_json_us=str(a-b) if a.is_finite() and b.is_finite() else None)
                except InvalidOperation:comparisons[field]=dict(raw_csv=fields[field],raw_json=check[key],csv_minus_json_us=None)
            record['time_field_comparison']=comparisons
            if fields.get('modelid')!=str(model_id):
                unsupported.append(dict(**record,reason='missing/different explicit Model ID'));continue
            starts=[fields[k] for k in ('iterationstartus','starttimeus') if k in fields]
            ends=[fields[k] for k in ('iterationendus','endtimeus') if k in fields]
            if len(starts)!=1 or len(ends)!=1:
                unsupported.append(dict(**record,reason='no unique explicit Start/End in microseconds; never derive Start from Duration'));continue
            try:
                values=[Decimal(x) for x in (starts[0],ends[0])]
                equal=all(v.is_finite() for v in values) and values==[Decimal(check['raw_start_us']),Decimal(str(check['raw_end_us']))]
            except InvalidOperation:equal=False
            matches.append(dict(**record,start_us=starts[0],end_us=ends[0],exact_endpoint_match=equal))
    return dict(status='EXACT_ENDPOINTS_CORROBORATED' if len(matches)==1 and matches[0]['exact_endpoint_match'] and not unsupported and not unreadable else 'UNPROVEN_ENDPOINTS',
                matches=matches,unsupported_rows=unsupported,unreadable_tables=unreadable,root_cause='UNPROVEN',
                rule='one same-export CSV row with explicit Model/Iteration ID and exact independently stored Start/End(us); no duration-derived endpoint, no offset/clock fit')
