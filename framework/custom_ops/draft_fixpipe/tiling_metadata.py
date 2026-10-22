"""Bounded structural review of 310P Matmul repository JSON, not capability proof."""
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

MAX_METADATA_BYTES=32*1024*1024
MAX_NODES=2_000_000
MAX_DEPTH=64
MAX_DOCUMENTS=32768
MAX_DOCUMENT_EVIDENCE_BYTES=256*1024*1024
MAX_FIELD_SAMPLES=128  # per category; exact total counts are retained


def is_repository(relative):
    parts=[p.lower() for p in relative.parts]
    return (len(parts)>=4 and 'opp' in parts and
            parts[-3:-1]==['tiling','ascend310p'] and
            bool(re.fullmatch(r'repository_ascend310p(?:3)?_\w*matmul\w*\.json',parts[-1])))


def normalized(value):
    return re.sub(r'[^a-z0-9]','',value.lower())


def category(key,value):
    """Classify observed fields, never hardware acceptance/rejection."""
    name=normalized(key)
    text=normalized(value) if isinstance(value,str) else ''
    if any(token in name or token in text for token in
           ('fixpipe','antiquant','dequant','quantmode','quantpre','deqscale','scaledfp32',
            'setquant','f322f32','f322f16','deqf16','enpartialsum')) or 'quant' in name or 'scale' in name:
        return 'conversion_related'
    if any(token in name for token in ('support','dtype','datatype','format','layout','soc','product','chip','accum','atomic')):
        return 'type_layout_support_related'
    if any(token in name for token in ('tiling','basem','basen','basek','singlecore','deptha1','depthb1','dbl0','blockdim','splitk')) or name in ('m','n','k'):
        return 'tiling_related'
    return None


def location(parts):
    pointer=('' if not parts else '/')+'/'.join(str(p).replace('~','~0').replace('/','~1') for p in parts)
    if len(pointer)<=2048:return dict(json_pointer=pointer)
    return dict(json_pointer_prefix=pointer[:2048],json_pointer_chars=len(pointer),
                json_pointer_sha256=hashlib.sha256(pointer.encode()).hexdigest(),
                locator_truncated=True)


def field_value(value):
    if isinstance(value,dict):return dict(type='object',member_count=len(value))
    if isinstance(value,list):
        result=dict(type='array',length=len(value))
        # Preserve actual compact shape/stride vectors, not just their length.
        if len(value)<=32 and all(type(v) in (int,bool,type(None)) or isinstance(v,str) and len(v)<=128 for v in value):
            result['value']=value
        return result
    if isinstance(value,Decimal):return dict(type='number',decimal_literal=str(value))
    if isinstance(value,str) and len(value)>512:
        return dict(type='string',prefix=value[:512],characters=len(value),
                    sha256=hashlib.sha256(value.encode()).hexdigest(),value_truncated=True)
    return dict(type=type(value).__name__,value=value)


def inspect(raw,source,document_sink=None):
    """Validate the complete JSON stream; infer framing only after reaching EOF.

    Node/depth/document budgets apply across the whole file. Global report
    samples and per-document samples are independently explicit; every document
    is traversed even after the global sample cap. The optional sink streams
    per-document observations to an isolated evidence file.
    """
    result=dict(status='UNPARSED',source=source,limit_bytes=MAX_METADATA_BYTES,
        node_limit=MAX_NODES,depth_limit=MAX_DEPTH,document_limit=MAX_DOCUMENTS,
        sample_limit_per_category=MAX_FIELD_SAMPLES,inspected_nodes=0,field_count=0,
        unmatched_field_count=0,unmatched_samples=[],documents=[],validated_document_count=0,
        document_numbering='document_number is 1-based; JSON Pointer is relative to that document',
        format=dict(status='UNRESOLVED',detected='UNRESOLVED',basis='Full-stream document boundaries, never a previous parser exception'),
        matches={k:dict(count=0,fields=[]) for k in
                 ('conversion_related','type_layout_support_related','tiling_related')},
        compatibility='UNRESOLVED',classification='UNCLASSIFIED_JSON_CONTENT',
        unparsed=['Entire input pending structural parse'],
        semantic_unreviewed=['SDK support semantics/chip dispatch','Scale encoding and FP32 destination',
                             'FP16 x INT8 Cube acceptance','20-group on-chip accumulator lifetime',
                             'Embedded serialized configurations inside JSON strings are not recursively parsed'])
    if document_sink is None:result['document_details']=[]
    def pairs(items):
        out={}
        for key,value in items:
            if key in out:raise ValueError('duplicate JSON key: '+key[:160])
            out[key]=value
        return out
    def constant(value):raise ValueError('nonstandard JSON number: '+value)
    def children(path,value):
        entries=value.items() if isinstance(value,dict) else enumerate(value)
        for key,child in entries:yield (*path,key),child
    def observe(record,path,key,value,origin='field'):
        kind=category(key,value)
        if origin=='field':
            result['field_count']+=1;record['field_count']+=1
        if kind:
            row=dict(document_number=record['document_number'],**location(path),origin=origin,observed=field_value(value))
            if origin=='field':row.update(field=key[:512],field_truncated=len(key)>512)
            for target in (result['matches'][kind],record['matches'][kind]):
                target['count']+=1
                if len(target['fields'])<MAX_FIELD_SAMPLES:target['fields'].append(row)
        elif origin=='field':
            row=dict(document_number=record['document_number'],**location(path),observed=field_value(value))
            for target in (result,record):
                target['unmatched_field_count']+=1
                if len(target['unmatched_samples'])<MAX_FIELD_SAMPLES:target['unmatched_samples'].append(row)
    def cap_flags(target):
        for group in target['matches'].values():group['samples_truncated']=group['count']>len(group['fields'])
        target['unmatched_samples_truncated']=target['unmatched_field_count']>len(target['unmatched_samples'])
    text='';cursor=byte_cursor=tracked=0;line=column=1;current_document=1
    def position(index):
        nonlocal tracked,byte_cursor,line,column
        segment=text[tracked:index];byte_cursor+=len(segment.encode('utf-8'))
        newlines=segment.count('\n')
        if newlines:line+=newlines;column=len(segment)-segment.rfind('\n')
        else:column+=len(segment)
        tracked=index
        return dict(char=index,byte=byte_cursor,line=line,column=column)
    try:
        if len(raw)>MAX_METADATA_BYTES:raise ValueError('metadata exceeds 32 MiB ceiling')
        if source['bytes']!=len(raw) or source['sha256']!=hashlib.sha256(raw).hexdigest():
            raise ValueError('metadata bytes differ from bound source')
        text=raw.decode('utf-8')
        decoder=json.JSONDecoder(object_pairs_hook=pairs,parse_float=Decimal,parse_constant=constant)
        whitespace=re.compile(r'[ \t\r\n]*')
        jsonl=True;previous_end_line=0
        while True:
            cursor=whitespace.match(text,cursor).end()
            if cursor==len(text):
                position(cursor);break
            current_document=result['validated_document_count']+1
            start=position(cursor)
            result['first_unreviewed_document']=dict(document_number=current_document,start=start)
            if current_document>MAX_DOCUMENTS:raise ValueError('document count limit exceeded')
            tree,end=decoder.raw_decode(text,cursor)
            if not isinstance(tree,(dict,list)):raise ValueError('expected repository object or array')
            stop=position(end)
            if current_document==1:result['top_level']=field_value(tree)
            jsonl &= start['line']==stop['line'] and start['line']==previous_end_line+1
            previous_end_line=stop['line']
            document_bytes=raw[start['byte']:stop['byte']]
            record=dict(document_number=current_document,start=start,end_exclusive=stop,
                sha256=hashlib.sha256(document_bytes).hexdigest(),top_level=field_value(tree),
                inspected_nodes=0,field_count=0,unmatched_field_count=0,unmatched_samples=[],
                matches={k:dict(count=0,fields=[]) for k in result['matches']})
            stack=[iter([((),tree)])]
            while stack:
                try:path,value=next(stack[-1])
                except StopIteration:stack.pop();continue
                result['inspected_nodes']+=1;record['inspected_nodes']+=1
                if result['inspected_nodes']>MAX_NODES or len(path)>MAX_DEPTH:
                    result['first_unreviewed_location']=dict(document_number=current_document,**location(path))
                    raise ValueError('metadata traversal node/depth limit exceeded')
                if path:
                    if isinstance(path[-1],str):observe(record,path,path[-1],value)
                    else:observe(record,path,'',value,'array_value')
                if isinstance(value,dict):
                    stack.append(children(path,value))
                elif isinstance(value,list):stack.append(children(path,value))
            cap_flags(record);record['status']='DOCUMENT_STRUCTURALLY_VALIDATED'
            if document_sink is None:result['document_details'].append(record)
            else:document_sink(record)
            result['documents'].append({k:record[k] for k in
                ('document_number','start','end_exclusive','sha256','top_level','inspected_nodes','field_count','status')})
            result['validated_document_count']+=1;cursor=end
        if not result['validated_document_count']:raise ValueError('empty repository')
        # Blank lines / pretty-printed records / same-line concatenation form a
        # general JSON document sequence, not JSONL. Trailing single LF is legal.
        tail=text[result['documents'][-1]['end_exclusive']['char']:]
        jsonl &= tail.count('\n')<=1
        detected=('SINGLE_JSON_DOCUMENT' if result['validated_document_count']==1 else
                  'JSONL' if jsonl else 'JSON_DOCUMENT_SEQUENCE')
        result['format'].update(status='VERIFIED_FROM_COMPLETE_STREAM',detected=detected,
            validated_documents=result['validated_document_count'],consumed_bytes=byte_cursor,
            framing='Objects/arrays separated by JSON whitespace or directly concatenated; no comments/RS/BOM recovery')
        result.pop('first_unreviewed_document',None)
        result.update(status='STRUCTURAL_REVIEW_COMPLETE',unparsed=[])
        counts={k:v['count'] for k,v in result['matches'].items()}
        if counts['conversion_related'] or counts['type_layout_support_related']:
            result['classification']='CAPABILITY_RELATED_FIELDS_REQUIRE_SOURCE_REVIEW'
        elif counts['tiling_related']:result['classification']='TILING_FIELDS_OBSERVED_NO_CAPABILITY_FIELD_HITS'
        result['classification_basis']=dict(field_counts=counts,
            rule='Actual parsed fields/values in all documents; filename and an Extra data exception are not format or capability evidence.')
    except (ValueError,UnicodeError,RecursionError) as e:
        result.update(status='INCOMPLETE_METADATA_REVIEW',error=f'{type(e).__name__}: {e}',
            failure_document_number=current_document if text else None,
            unparsed=['Current document or remaining stream not verified; prefix documents cannot establish complete format/review'])
        if isinstance(e,json.JSONDecodeError):
            result['parse_error']=dict(char=e.pos,line=e.lineno,column=e.colno)
        elif isinstance(e,UnicodeDecodeError):
            result['parse_error']=dict(byte=e.start,reason=e.reason,document_number='UNRESOLVED_BEFORE_UTF8_DECODE')
    cap_flags(result)
    return result


def inspect_to_file(raw,source,records_path):
    """Bound the per-document evidence file as well as the input/parser budgets."""
    size=0
    with Path(records_path).open('xb') as stream:
        def sink(record):
            nonlocal size
            payload=(json.dumps(record,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode()
            if size+len(payload)>MAX_DOCUMENT_EVIDENCE_BYTES:
                raise ValueError('per-document evidence byte budget exceeded')
            stream.write(payload);size+=len(payload)
        result=inspect(raw,source,sink)
    result['document_evidence_bytes']=size
    result['document_evidence_limit_bytes']=MAX_DOCUMENT_EVIDENCE_BYTES
    result['document_evidence_scope']='One line per completely validated document; sample caps explicit, all documents still parsed/traversed. Partial file on failure.'
    return result
