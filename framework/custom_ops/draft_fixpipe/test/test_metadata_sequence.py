"""Strict whole-stream validation and historical-snapshot-only replay."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import tiling_metadata as meta
import metadata_only as offline
from aq_common import identity,write


def inspect(raw):
    return meta.inspect(raw,dict(path='/not-opened/repository.json',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))


@pytest.mark.parametrize('raw,kind,count',[
    (b'{"baseM":16}', 'SINGLE_JSON_DOCUMENT',1),
    (b'{\n"baseM":16\n}\n', 'SINGLE_JSON_DOCUMENT',1),
    (b'{"baseM":16}\n{"baseM":32}\n', 'JSONL',2),
    (b' {"baseM":16}\r\n\t{"baseN":64}\r\n','JSONL',2),
    (b'{\n"baseM":16\n}\n{"baseN":64}','JSON_DOCUMENT_SEQUENCE',2),
    (b'{"baseM":16}{"baseN":64}','JSON_DOCUMENT_SEQUENCE',2),
    (b'{"baseM":16} {"baseN":64}','JSON_DOCUMENT_SEQUENCE',2),
    (b'\n{"baseM":16}\n\n{"baseN":64}\n','JSON_DOCUMENT_SEQUENCE',2),
])
def test_format_determined_from_complete_boundaries(raw,kind,count):
    r=inspect(raw)
    assert r['status']=='STRUCTURAL_REVIEW_COMPLETE'
    assert r['format']['detected']==kind and r['format']['consumed_bytes']==len(raw)
    assert r['validated_document_count']==count and len(r['document_details'])==count
    assert [x['document_number'] for x in r['documents']]==list(range(1,count+1))
    for doc in r['documents']:
        start,end=doc['start']['byte'],doc['end_exclusive']['byte']
        assert hashlib.sha256(raw[start:end]).hexdigest()==doc['sha256']


def test_each_record_retains_values_pointers_and_unicode_byte_locations():
    raw='{"说明":"中文","A_dtype":"float16","A_shape":[32,48,13,16,16]}\n{"baseN":64,"mad_dtype":"float32"}\n'.encode()
    r=inspect(raw);a,b=r['document_details']
    assert b['start']['byte']==raw.index(b'{"baseN"') and b['start']['byte']>b['start']['char']
    assert b['start']['line']==2
    fields=a['matches']['type_layout_support_related']['fields']+b['matches']['type_layout_support_related']['fields']
    assert [(f['document_number'],f['json_pointer'],f['observed']['value']) for f in fields]==[
        (1,'/A_dtype','float16'),(2,'/mad_dtype','float32')]
    shape=next(x for x in a['unmatched_samples'] if x['json_pointer']=='/A_shape')
    assert shape['observed']['value']==[32,48,13,16,16]


def test_reported_first_record_boundary_and_invalid_record_beyond_sample_cap():
    first=b'{"shape":{"A_dtype":"float16","B_dtype":"float16","mad_dtype":"float32"},"note":"'
    first+=b'x'*(1155-len(first)-2)+b'"}'
    r=inspect(first+b'\n{"baseM":16}\n')
    assert r['format']['detected']=='JSONL'
    assert r['documents'][1]['start']==dict(char=1156,byte=1156,line=2,column=1)
    raw=b'{"baseM":16}\n'*1000+b'{"baseM":'
    r=inspect(raw)
    assert r['validated_document_count']==1000 and r['failure_document_number']==1001
    assert r['status']=='INCOMPLETE_METADATA_REVIEW' and r['format']['detected']=='UNRESOLVED'


@pytest.mark.parametrize('bad',[
    b'{"baseM":', b'{"baseM":16,"baseM":32}',b'{"value":NaN}',b'{"v":Infinity}',
    b'not-json',b'{"x":1,}',b'42',b'\x1e{}',b'{"bad":"\xff"}',
])
def test_bad_later_record_or_utf8_never_claims_complete(bad):
    r=inspect(b'{"baseM":16}\n'+bad)
    assert r['status']=='INCOMPLETE_METADATA_REVIEW' and r['unparsed']
    assert r['format']['detected']=='UNRESOLVED'
    assert r['compatibility']=='UNRESOLVED'
    if b'\xff' not in bad:assert r['validated_document_count']==1 and r['failure_document_number']==2


def test_aggregate_node_depth_and_document_budgets():
    with patch.object(meta,'MAX_NODES',3):r=inspect(b'{"baseM":16}\n{"baseN":64}\n')
    assert r['status']=='INCOMPLETE_METADATA_REVIEW' and r['failure_document_number']==2
    assert r['inspected_nodes']==4
    with patch.object(meta,'MAX_DEPTH',1):r=inspect(b'{"baseM":16}\n{"x":{"baseN":64}}')
    assert r['status']=='INCOMPLETE_METADATA_REVIEW' and r['first_unreviewed_location']['document_number']==2
    with patch.object(meta,'MAX_DOCUMENTS',2):r=inspect(b'{}\n{}\n{}\n')
    assert r['status']=='INCOMPLETE_METADATA_REVIEW' and r['validated_document_count']==2


def test_global_sample_cap_does_not_skip_later_record_validation_or_fields(tmp_path):
    raw=b'{"baseM":16}\n{"baseM":32}\n{"baseM":64}\n'
    with patch.object(meta,'MAX_FIELD_SAMPLES',1):r=inspect(raw)
    assert r['status']=='STRUCTURAL_REVIEW_COMPLETE' and r['matches']['tiling_related']['count']==3
    assert r['matches']['tiling_related']['samples_truncated']
    assert [d['matches']['tiling_related']['fields'][0]['observed']['value'] for d in r['document_details']]==[16,32,64]
    p=tmp_path/'records.jsonl';source=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    r=meta.inspect_to_file(raw,source,p)
    assert len(p.read_text().splitlines())==3 and 'document_details' not in r
    with patch.object(meta,'MAX_DOCUMENT_EVIDENCE_BYTES',1):r=meta.inspect_to_file(raw,source,tmp_path/'limited.jsonl')
    assert r['status']=='INCOMPLETE_METADATA_REVIEW' and r['validated_document_count']==0


def history(tmp_path,count=848):
    root=tmp_path/'history';root.mkdir();snap=root/'fixpipe-sources';snap.mkdir()
    cann=tmp_path/'sdk-not-mounted';files=[]
    for i in range(count):
        if i==0:name='version.info';raw=b'Version=9.0.0\n';role='version'
        elif i==1:name='include/ascendc/dav_m200/fixpipe.h';raw=b'// Fixpipe GetTensorC\n';role='source'
        elif i==2:
            name='opp/built-in/data/tiling/Ascend310P/repository_Ascend310P3_matmul.json'
            raw=b'{"shape":{"A_dtype":"float16","B_dtype":"float16","mad_dtype":"float32"},"baseM":16}\n{"baseM":32}\n';role='tiling_metadata'
        else:name=f'include/helper{i}.h';raw=b'// source\n';role='source'
        p=snap/f'{i}-{Path(name).name}';p.write_bytes(raw);item=identity(p)
        source=dict(item,path=str(cann/name))
        f=dict(source=source,snapshot=item,role=role,hits=[],excerpts=[],frontend_hints=[])
        if role=='tiling_metadata':
            old_meta=root/'repository-metadata-0.json'
            write(old_meta,dict(status='INCOMPLETE_METADATA_REVIEW',source=source,error='JSONDecodeError: Extra data'))
            f.update(metadata_report=identity(old_meta),metadata_status='INCOMPLETE_METADATA_REVIEW',content_classification='UNCLASSIFIED_JSON_CONTENT')
        files.append(f)
    scan=dict(status='INCOMPLETE',snapshotted_files=count,incomplete=[{'path':files[2]['source']['path'],'reason':'METADATA_REVIEW_INCOMPLETE'}])
    write(root/'fixpipe-scan.json',scan)
    old=dict(status='BLOCKED_AUDIT',soc='Ascend310P3',cann_root=str(cann),files=files,scan=scan,scan_file=identity(root/'fixpipe-scan.json'))
    report=root/'group128-capability.json';write(report,old)
    return report,old


def test_metadata_only_reuses_all_848_snapshots_without_live_sdk_or_rescan(tmp_path):
    path,old=history(tmp_path);before={str(p):identity(p) for p in path.parent.rglob('*') if p.is_file()}
    args=SimpleNamespace(report=path,output_dir=tmp_path/'new');original=Path.open
    def guard(p,*a,**kw):
        assert Path(old['cann_root']) not in p.parents,'live SDK must not be opened'
        return original(p,*a,**kw)
    with patch.object(offline.audit,'collect_fix',side_effect=AssertionError('no scan')),patch.object(Path,'open',guard):
        assert offline.run(args)==0
    r=json.loads((args.output_dir/'group128-capability.json').read_text())
    assert r['status']=='METADATA_REPARSED_SDK_REVIEW_INCOMPLETE'
    assert r['snapshot_verification']['count']==848 and r['sdk_scan']=='NOT_RESUMED'
    assert r['historical_scan']==old['scan'] and r['scan']['status']=='NOT_RESUMED'
    assert r['scan']['reused_snapshot_count']==848 and r['metadata_results'][0]['format']['detected']=='JSONL'
    assert r['metadata_results'][0]['validated_documents']==2
    assert all(v['compatibility']=='UNRESOLVED' for v in r['paths'].values())
    assert r['npu']==r['cann_compile']==r['simulator']=='NOT_RUN'
    assert before=={str(p):identity(p) for p in path.parent.rglob('*') if p.is_file()}
    assert not Path(old['cann_root']).exists()
    priority=json.loads((args.output_dir/'priority-evidence.json').read_text())
    assert set(priority['questions'])==set(offline.audit.DECISION_PATTERNS)
    assert priority['historical_scan']['resumed'] is False


@pytest.mark.parametrize('failure',['tamper','missing','source_binding'])
def test_history_integrity_failures_block_offline_reparse(tmp_path,failure):
    path,old=history(tmp_path,3)
    if failure=='tamper':Path(old['files'][1]['snapshot']['path']).write_text('changed')
    elif failure=='missing':Path(old['files'][1]['snapshot']['path']).unlink()
    else:
        old['files'][1]['source']['sha256']='0'*64;write(path,old)
    args=SimpleNamespace(report=path,output_dir=tmp_path/'new')
    assert offline.run(args)==1
    r=json.loads((args.output_dir/'group128-capability.json').read_text())
    assert r['status']=='BLOCKED_OFFLINE_REPARSE'
    assert not list(args.output_dir.glob('repository-documents-*'))


def test_offline_output_cannot_overwrite_or_nest_in_history(tmp_path):
    path,_=history(tmp_path,3)
    for out in (path.parent,path.parent/'new'):
        with pytest.raises(ValueError,match='overlaps'):
            offline.run(SimpleNamespace(report=path,output_dir=out))
