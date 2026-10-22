#!/usr/bin/env python3
"""Read-only installed-SDK evidence for AntiQuant Input vs FixPipe Output paths.

Never imports SDK Python implementations (TIK files are read as text). Never
compiles, launches a device operator, runs a simulator, or changes a Golden.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'draft_antiquant'))
from aq_common import identity,verify,write,fresh
from aq_sdk import reviewed as reviewed_input
from fixpipe_math import counterexamples,traffic
from tiling_metadata import MAX_METADATA_BYTES,is_repository,inspect_to_file as inspect_metadata

TERMS=re.compile(r'fix_?pipe|FixPipe|Fixpipe|deq[Ss]cale|deqScalar|quantPre|QuantMode|'
                 r'SetAntiQuantVector|MatmulType|Src[AB]?T|'
                 r'SetQuantVector|SetDequantType|GetTensorC|enAtomic|enPartialSum|'
                 r'F322F16|F322F32|DEQF16|deqscale|dequant|api_check_support|get_soc_name')
COMMON={
    'chip_dispatch':'Actual Ascend310P3/dav_2002 CANN9 branch and supported specialization, not another chip API.',
    'cube_input':'FP16 A + raw INT8 B Cube input acceptance; identify any internal B conversion. Quantizing A to INT8 is W8A8, not this W8A16 contract.',
    'dtype_pair':'Exact L0C input and scaled destination dtype combination; separate FP32->FP16 from scaled FP32->FP32.',
    'buffers':'CO1/L0C, any FixPipe buffer/A1/UB/GM route, ownership, capacity and lifetimes.',
    'layout_stride':'NZ/ND source/destination, N16/block alignment, stride units/ranges, M16 N64 validity.',
    'scale_encoding':'N64 vector location/type/length/alignment, uint64/float bitfields if applicable, exponent/mantissa truncation and conversion rules.',
    'per_group_update':'Different N64 scale for each K128 partial output; no stale parameter cache; required completion fences.',
    'scaled_fp32':'Multiply partial result by scale and return/store FP32 without intermediate FP16 rounding/saturation.',
    'cross_group_accumulator':'Preserve separate FP32 accumulator across 20 scaled groups; clear/reuse L0C, scheduler/atomic ordering and final single FP16 rounding.',
    'rounding_special_values':'Actual scale multiply/output rounding, overflow, subnormal, NaN and signed-zero behavior; Golden cannot be modified.',
}
FRONTENDS={
    'ascendc_explicit':dict(label='Ascend C explicit Fixpipe',extra='Resolve exact overload/parameter structure and chip guards, not FixpipeParamsV220 by name.'),
    'tik':dict(label='TIK fixpipe',extra='Inspect Python product checks and lowered intrinsic. TIK support does not establish Ascend C overload support.'),
    'matmul_sdk':dict(label='Matmul SDK output conversion',extra='Trace GetTensorC/SetQuantVector/SetDequantType to actual CO1 transfer; do not confuse input AntiQuant with output quantization.'),
}


def frontend_hints(path,text):
    """Search hints only; never capability classification."""
    parts=[x.lower() for x in path.parts];out=[]
    if 'tik' in parts or path.suffix=='.py' and 'fixpipe' in text:out.append('tik')
    if any('matmul' in x for x in parts):out.append('matmul_sdk')
    if any('ascendc' in x or 'tikcfw' in x or 'fixpipe' in x for x in parts) and path.suffix in ('.h','.hpp','.cpp'):
        out.append('ascendc_explicit')
    return out or ['UNCLASSIFIED_SOURCE']


MAX_SOURCE_BYTES=16*1024*1024
SOURCE_SUFFIXES={'.h','.hpp','.cpp','.cc','.py','.ini'}
DECISION_PATTERNS={
    'fp16_int8_cube_input':r'SetAntiQuant|MatmulType|Src[AB]?T|Src[AB]?Type|int8_t|DT_INT8',
    'n64_scale_and_scaled_fp32':r'Fixpipe|fixpipe|SetQuantVector|SetDequantType|GetTensorC|quantPre|deqScale|F322F32|DEQF16',
    'twenty_groups_onchip_fp32':r'enPartialSum|GetTensorC|Iterate|CO1|L0C|SetAntiQuant|SetQuantVector',
}


def priority_index(report):
    """Index existing snapshots only; no new SDK paths or compatibility inference."""
    questions={name:dict(status='UNRESOLVED',candidate_count=0,candidates=[],sample_limit=64)
               for name in DECISION_PATTERNS}
    unreviewed=[]
    def rank(pair):
        path=pair[1]['source']['path'].lower()
        return (0 if any(p in path for p in ('dav_m200','dav_2002','ascend310p')) else
                1 if 'fixpipe' in path else 2 if 'matmul' in path or '/tik/' in path else 3,pair[0])
    for index,f in sorted(enumerate(report['files']),key=rank):
        path=f['source']['path'];parts=Path(path).parts
        architectures=[p for p in parts if p.startswith('dav_')]
        if architectures and not any(p in ('dav_m200','dav_2002') for p in architectures):
            unreviewed.append(dict(source_index=index,reason='OTHER_ARCHITECTURE_NOT_USED_AS_310P_PROOF'))
            continue
        if f['role']=='tiling_metadata':
            unreviewed.append(dict(source_index=index,reason='STRUCTURAL_METADATA_ONLY_NOT_HARDWARE_SEMANTICS',
                                  metadata_report=f['metadata_report'],classification=f['content_classification']))
            continue
        matched=False
        for name,pattern in DECISION_PATTERNS.items():
            excerpts=[e for e in f.get('excerpts',[]) if re.search(pattern,e['text'])]
            if not excerpts:continue
            matched=True;q=questions[name];q['candidate_count']+=1
            if len(q['candidates'])<q['sample_limit']:
                q['candidates'].append(dict(source_index=index,source=f['source'],snapshot=f['snapshot'],
                    architecture_hint=architectures or ['COMMON_SOURCE_CHIP_DISPATCH_REQUIRES_REVIEW'],
                    excerpts=excerpts[:4],excerpts_truncated=len(excerpts)>4))
            else:unreviewed.append(dict(source_index=index,question=name,reason='PRIORITY_INDEX_SAMPLE_LIMIT_FULL_SNAPSHOT_AVAILABLE'))
        if not matched:unreviewed.append(dict(source_index=index,reason='NO_DECISION_TERM_IN_COLLECTED_EXCERPTS_FULL_SNAPSHOT_AVAILABLE'))
    for q in questions.values():q['samples_truncated']=q['candidate_count']>len(q['candidates'])
    return dict(status='SOURCE_REVIEW_REQUIRED',questions=questions,unreviewed_evidence=unreviewed,
        unreviewed_repositories=report.get('scan',{}).get('unreviewed_repositories',[]),
        missing_hardware_review=list(questions),
        rule='Index of source candidates, not proof. Any explicit unsupported verdict needs actual chip/type branch and source citation; never infer from filename, field presence or another architecture.')


def scan_rule(relative):
    """Path-only admission, before resolve/stat/open. No source-family pruning."""
    name=relative.name.lower();suffix=relative.suffix.lower()
    if name in ('version.info','version.cfg') or suffix in SOURCE_SUFFIXES:
        return True,'SOURCE_OR_VERSION'
    if suffix!='.json':return False,'NON_SOURCE_SUFFIX'
    parts=[re.sub(r'[^a-z0-9]','',p.lower()) for p in relative.parts]
    stem=re.sub(r'[^a-z0-9]','',relative.stem.lower())
    if 'tiling' in parts and relative.name.lower().startswith('repository_') and 'matmul' in stem and not is_repository(relative):
        return False,'NON_TARGET_MATMUL_REPOSITORY_JSON'
    # Explicit operator/feature names can occur in generic configuration files
    # nested under an op directory. Do not exclude those based on directory alone.
    if any(any(token in part for token in ('fixpipe','matmul','quant','dequant')) for part in parts):
        return True,'RELATED_OPERATOR_JSON'
    if 'fusionstrategy' in parts:return False,'UNRELATED_FUSION_STRATEGY_JSON'
    if 'l2cache' in stem:return False,'UNRELATED_L2_CACHE_JSON'
    # Aggregate op registries and platform/product capability lists may contain
    # relevant restrictions without naming FixPipe in their filenames.
    registries=('opsinfo','opinfo','opsconfig','opsupport','opssupport','supportedops',
                'supportedsoc','supportlist','socinfo','socconfig','platforminfo',
                'platformconfig','productinfo','productconfig','chipinfo','chipconfig',
                'capability','capabilities')
    if any(token in stem for token in registries):return True,'SUPPORT_REGISTRY_JSON'
    if any(p in ('opinfocfg','opsinfo','opinfo','opsconfig','opsupport','opssupport') for p in parts[:-1]):
        return True,'OP_REGISTRY_DIRECTORY_JSON'
    if any(p in ('platformconfig','socconfig','productconfig','chipconfig') for p in parts[:-1]):
        return True,'PLATFORM_CONFIG_JSON'
    return False,'JSON_OUTSIDE_DECLARED_SCOPE'


def collect_fix(cann,root,scan=None,files=None):
    """Retain partial evidence on failure; oversized admitted sources still block."""
    if scan is None:scan={}
    if files is None:files=[]
    scan.update(status='RUNNING',policy='fixpipe-sdk-path-filter-v2',
        max_source_bytes=MAX_SOURCE_BYTES,max_repository_json_bytes=MAX_METADATA_BYTES,
        metadata_files=0,enumerated_files=0,admitted_files=0,
        read_files=0,snapshotted_files=0,skipped_files=0,skipped_by_reason={},
        skip_sample_limit_per_reason=20,incomplete=[],unreviewed_repositories=[],excluded_directory_names=['__pycache__','.git'],
        scope='All .h/.hpp/.cpp/.cc/.py/.ini and version files; JSON admitted by explicit operator/registry/platform paths. Only 310P Matmul repository JSON gets a separate 32 MiB structural review. Discovery, not full SDK capability proof.')
    folder=root/'fixpipe-sources';seen=set();current=None;phase='create_snapshot_directory'
    def skip(reason,path):
        scan['skipped_files']+=1
        entry=scan['skipped_by_reason'].setdefault(reason,dict(count=0,samples=[],samples_truncated=False))
        entry['count']+=1
        if len(entry['samples'])<scan['skip_sample_limit_per_reason']:entry['samples'].append(str(path))
        else:entry['samples_truncated']=True
        if reason=='NON_TARGET_MATMUL_REPOSITORY_JSON':
            scan['unreviewed_repositories'].append(dict(path=str(path),reason=reason,content='NOT_READ',identity='NOT_HASHED'))
    def error(e):
        nonlocal current,phase
        current=e.filename;phase='enumerate_directory';raise e
    try:
        folder.mkdir()
        for base,dirs,names in os.walk(cann,followlinks=False,onerror=error):
            dirs[:]=sorted(d for d in dirs if d not in ('__pycache__','.git'))
            for name in sorted(names):
                p=Path(base)/name;current=str(p);phase='path_filter'
                scan['enumerated_files']+=1
                admitted,reason=scan_rule(p.relative_to(cann))
                if not admitted:
                    skip(reason,p);continue
                scan['admitted_files']+=1;phase='resolve'
                actual=p.resolve(strict=True)
                if cann not in actual.parents:raise ValueError('SDK file escapes selected root: '+str(p))
                if actual in seen:
                    skip('DUPLICATE_RESOLVED_SOURCE',p);continue
                seen.add(actual);phase='stat';size=actual.stat().st_size
                repository=is_repository(p.relative_to(cann))
                limit=MAX_METADATA_BYTES if repository else MAX_SOURCE_BYTES
                if size>limit:
                    # Streaming SHA256 is evidence only; do not parse/copy the
                    # oversized source or silently count it as reviewed.
                    issue=dict(path=str(p),resolved_path=str(actual),bytes=size,
                        reason='METADATA_SIZE_LIMIT_EXCEEDED' if repository else 'SOURCE_SIZE_LIMIT_EXCEEDED',admission_rule=reason,
                        limit_bytes=limit,review='NOT_PARSED_OR_SNAPSHOTTED')
                    scan['incomplete'].append(issue);phase='hash_oversized_source'
                    issue['source']=identity(actual)
                    phase='size_limit'
                    raise ValueError(f'oversized SDK source: {p} ({size} bytes; limit {limit})')
                phase='read'
                with actual.open('rb') as stream:raw=stream.read(limit+1)
                if len(raw)>limit:
                    scan['incomplete'].append(dict(path=str(p),resolved_path=str(actual),
                        bytes_at_stat=size,observed_bytes_at_least=len(raw),limit_bytes=limit,
                        reason='SOURCE_GREW_BEYOND_LIMIT',review='NOT_PARSED_OR_SNAPSHOTTED'))
                    raise ValueError('SDK source grew beyond read limit: '+str(p))
                scan['read_files']+=1
                if repository:
                    phase='repository_metadata'
                    source=identity(actual)
                    if source['bytes']!=len(raw) or source['sha256']!=hashlib.sha256(raw).hexdigest():
                        raise ValueError('repository changed during read: '+str(p))
                    snapshot=folder/(str(len(files))+'-'+name);snapshot.write_bytes(raw)
                    records_path=root/('repository-documents-'+str(scan['metadata_files'])+'.jsonl')
                    metadata=inspect_metadata(raw,source,records_path)
                    metadata['document_records']=identity(records_path)
                    metadata_path=root/('repository-metadata-'+str(scan['metadata_files'])+'.json')
                    write(metadata_path,metadata)
                    files.append(dict(source=source,snapshot=identity(snapshot),role='tiling_metadata',
                        admission_rule=reason,frontend_hints=['matmul_sdk'],hits=[],excerpts=[],
                        metadata_report=identity(metadata_path),document_records=identity(records_path),metadata_status=metadata['status'],
                        content_classification=metadata['classification']))
                    scan['metadata_files']+=1;scan['snapshotted_files']=len(files)
                    verify(source)
                    if metadata['status']!='STRUCTURAL_REVIEW_COMPLETE':
                        scan['incomplete'].append(dict(path=str(p),bytes=size,source=source,
                            reason='METADATA_REVIEW_INCOMPLETE',metadata_report=identity(metadata_path)))
                        raise ValueError('repository metadata incomplete: '+str(metadata.get('error')))
                    continue
                text=raw.decode('utf-8',errors='replace');lines=text.splitlines()
                hits=[i for i,line in enumerate(lines,1) if TERMS.search(line)]
                special=name in ('version.info','version.cfg','Ascend310P3.ini')
                family=any('fixpipe' in s.lower() or 'matmul' in s.lower() for s in p.relative_to(cann).parts)
                if not (hits or special or family or p.suffix.lower()=='.json'):
                    skip('NO_RELEVANT_CONTENT_OR_FAMILY',p);continue
                phase='snapshot'
                source=identity(actual)
                # Bind the text used for excerpts to the file being recorded,
                # including if a source changed after the bounded read.
                if source['bytes']!=len(raw) or source['sha256']!=hashlib.sha256(raw).hexdigest():
                    raise ValueError('SDK source changed during read: '+str(p))
                snapshot=folder/(str(len(files))+'-'+name);snapshot.write_bytes(raw)
                windows=[]
                for i in hits:
                    lo,hi=max(1,i-12),min(len(lines),i+30)
                    if windows and lo<=windows[-1][1]+1:windows[-1][1]=max(hi,windows[-1][1])
                    else:windows.append([lo,hi])
                files.append(dict(source=source,snapshot=identity(snapshot),frontend_hints=frontend_hints(p,text),
                    admission_rule=reason,hits=hits,
                    role='version' if name in ('version.info','version.cfg') else 'platform' if special else 'source',
                    excerpts=[dict(start=a,end=b,text='\n'.join(f'{i}: {lines[i-1]}' for i in range(a,b+1))) for a,b in windows]))
                scan['snapshotted_files']=len(files);verify(source)
        phase='verify_snapshots'
        for item in files:
            verify(item['source']);verify(item['snapshot'])
            if 'metadata_report' in item:verify(item['metadata_report'])
            if 'document_records' in item:verify(item['document_records'])
        scan['status']='COMPLETE_WITHIN_DECLARED_SCOPE'
        return files
    except Exception as e:
        scan.update(status='INCOMPLETE',error=dict(path=str(current) if current else None,
            phase=phase,type=type(e).__name__,message=str(e)))
        raise
    finally:
        write(root/'fixpipe-scan.json',scan)


def verify_input(path,cann):
    doc=json.loads(path.read_text())
    if doc.get('cann_root')!=str(cann) or doc.get('status') not in ('SDK_REVIEW_REQUIRED','BLOCKED_MISSING_API'):
        raise ValueError('AntiQuant audit incomplete or from a different SDK')
    for f in doc['files']:verify(f['source']);verify(f['snapshot'])
    for f in doc['experiment_sources']:verify(f)
    for t in doc['tools'].values():verify(t['file'])
    for lib in doc['alternative_libraries']:verify(lib['file'])
    versions=[verify(f['snapshot']).read_text(errors='replace') for f in doc['files'] if f['role']=='version']
    if not any(re.search(r'(?<![\d.])9\.0\.0(?![\d.])',v) for v in versions):raise ValueError('CANN 9.0.0 not established')
    return doc


def focused_input(files,cann):
    """Reuse this scan; no second SDK tree walk or new library/tool discovery."""
    versions=[];api=[]
    for index,f in enumerate(files):
        if f['role']=='tiling_metadata':continue
        verify(f['source']);text=verify(f['snapshot']).read_text(errors='replace')
        if f['role']=='version':versions.append(dict(source=f['source'],snapshot=f['snapshot'],text=text))
        if 'SetAntiQuantVector' in text:api.append(dict(source_index=index,source=f['source'],snapshot=f['snapshot']))
    if not any(re.search(r'(?<![\d.])9\.0\.0(?![\d.])',v['text']) for v in versions):
        raise ValueError('CANN 9.0.0 not established by collected version files')
    return dict(status='FOCUSED_SOURCE_EVIDENCE_ONLY',cann_root=str(cann),versions=versions,
        api_candidates=api,compatibility='UNRESOLVED',
        unreviewed=['Actual mixed-type/chip template branch','Group128 refresh and FP32 partial sums',
                    'Compiler/toolchain instantiation','Native V3/Nz runtime support'],
        scope='Same collected source snapshots; not a standalone AntiQuant build authorization or a complete SDK audit')


def conclusions(input_doc,review=None):
    source_only='SOURCE_REVIEWED_K128_ONLY' if review else 'UNRESOLVED'
    return {
        'AntiQuant Input Path':dict(compatibility=source_only,
            interface_discovery=input_doc['status'],k128='NOT_RUN_IN_THIS_AUDIT',
            group128_twenty_groups='UNRESOLVED',golden_bitwise='UNRESOLVED',
            explanation='Input scaling can preserve pre-Cube FP16 weight rounding only if actual SDK behavior is verified; K128 source review cannot prove per-group refresh/partial sums.'),
        'FixPipe Output Path':dict(compatibility='UNRESOLVED',frontends={
            key:dict(label=v['label'],compatibility='UNRESOLVED',frontend_requirement=v['extra'],
                     claims={k:dict(status='UNRESOLVED',requirement=text,finding='',citations=[]) for k,text in COMMON.items()})
            for key,v in FRONTENDS.items()},
            universal_bitwise_equivalence='DISPROVEN_FOR_IDEAL_OUTPUT_SCALE_FACTORIZATION_BY_CPU_COUNTEREXAMPLE',
            frozen_workload_bitwise='UNRESOLVED',
            activation_quantized_variant='W8A8_DIFFERENT_NUMERICAL_SCHEME_NOT_ACCEPTED',
            explanation='Output scaling does not by itself legalize FP16 A / raw INT8 B Cube inputs, preserve RN16(q*s), or keep a scaled FP32 accumulator. Each requires separate evidence.')}


def markdown(r):
    text=['# Group128 input/output scale feasibility audit','', 'Status: '+r['status'],
          'Hardware compile / NPU / Simulator / performance: NOT_RUN. No Kernel/Golden changed.','']
    scan=r.get('scan',{})
    text += ['## SDK scan coverage','', 'Scan: '+scan.get('status','NOT_RUN'),
             f'Enumerated: {scan.get("enumerated_files",0)}; snapshots: {scan.get("snapshotted_files",0)}; skipped: {scan.get("skipped_files",0)}.']
    if 'reused_snapshot_count' in scan:
        text += [f'Reused verified historical snapshots: {scan["reused_snapshot_count"]}; no new SDK scan. Historical state: {r["historical_scan"]["status"]}.']
    text += [f'- {reason}: {entry["count"]}' for reason,entry in scan.get('skipped_by_reason',{}).items()]
    text += [f'- Incomplete: {item["path"]}: {item["reason"]}, bytes={item.get("bytes",item.get("observed_bytes_at_least"))}' for item in scan.get('incomplete',[])]
    text += ['Completion means the declared scan scope only, not verified hardware support.','']
    for f in r['files']:
        if f['role']=='tiling_metadata':
            text += [f'- Repository metadata: {f["source"]["path"]}; bytes={f["source"]["bytes"]}; SHA256={f["source"]["sha256"]}; '
                     f'{f["metadata_status"]}; {f["content_classification"]}.']
    text += ['', 'Three decision questions and unreviewed evidence: priority-evidence.json. All unresolved pending source review.','']
    for name,route in r.get('paths',{}).items():
        text += ['## '+name,'','Compatibility: '+route['compatibility'],route['explanation'],'']
        for frontend,d in route.get('frontends',{}).items():
            text += ['### '+d['label'],'',d['frontend_requirement'],
                     '| Contract | Status | Required evidence |','| --- | --- | --- |']
            text += [f'| {k} | {v["status"]} | {v["requirement"]} |' for k,v in d['claims'].items()]
            text += ['']
    text += ['## CPU arithmetic, not device observations','']
    for c in r['arithmetic']['cases']:
        result=c['result'];text += [f'- {c["name"]}: baseline {result["baseline_bits"]} vs alternative {result["alternative_bits"]}.']
    t=r['traffic'];text += ['', '## Conditional memory/reduction cost','',
        f'- 20 group partial FP32 writes: {t["scaled_fp32_partial_spill"]["gm_write_bytes"]:,} B; equal read bytes.',
        f'- With on-chip accumulator, extra partial GM traffic: {t["scaled_fp32_partial_spill"]["extra_gm_bytes"]:,} B (47.5 MiB).',
        f'- If accumulator also lives in GM under the stated schedule: {t["gm_accumulator_variant"]["extra_gm_bytes"]:,} B (95 MiB).',
        f'- {t["logical_reduction_additions"]:,} extra logical FP32 reduction additions; launch/actual instruction count UNRESOLVED.',
        '- Final FP16 output write is common to all paths, excluded from the extra-byte figures.',
        '- CO1-to-on-chip scaled FP32 return could avoid partial GM traffic only if this exact SDK/chip supports it.',
        '- No bandwidth/latency conversion. Whether traffic/synchronization negates savings remains UNRESOLVED.',
        '', 'Reference targets (user-reported, not measured here): W8 Off 2.542 ms, W8 Raw 2.332 ms, Custom FP16 1.449 ms.',
        '', 'API/source hits are discovery only. Inspect actual chip branches, parameter bitfields and downstream intrinsics; no automatic SUPPORTED result.',r.get('error','')]
    return '\n'.join(text)+'\n'


def run(args):
    cann=args.cann_root.resolve(strict=True)
    protected=[cann]+[p.parent for p in (args.antiquant_audit,args.antiquant_review) if p]
    root=fresh(args.output_dir,protected)
    report=dict(status='RUNNING',cann_root=str(cann),soc='Ascend310P3',architecture_target='dav_2002',
        cann_compile='NOT_RUN',npu='NOT_RUN',simulator='NOT_RUN',performance='NOT_RUN',full_gate='NOT_RUN',
        arithmetic=counterexamples(),traffic=traffic(),files=[],scan={'status':'NOT_RUN'},
        sources=[identity(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts],
        interpretation='Source discovery and CPU arithmetic only; not permission to replace W8 math or change Golden.')
    report['paths']=conclusions({'status':'NOT_RUN'})
    write(root/'group128-capability.json',report)
    try:
        # One scoped scan. Keep the original evidence verifier when reusing an
        # explicit AntiQuant audit, otherwise bind input-path observations to
        # these same snapshots/version files rather than walking the SDK again.
        collect_fix(cann,root,report['scan'],report['files'])
        path=args.antiquant_audit
        review=None
        if path is None:
            doc=focused_input(report['files'],cann)
            write(root/'input-path-evidence.json',doc)
            report['input_path_evidence']=identity(root/'input-path-evidence.json')
        else:
            doc=verify_input(path,cann)
            report['antiquant_audit']=identity(path)
        if args.antiquant_review:
            _,review=reviewed_input(path,args.antiquant_review)
            report['antiquant_review']=identity(args.antiquant_review)
        report['paths']=conclusions(doc,review)
        if path is not None:verify_input(path,cann)
        report['status']='EVIDENCE_COLLECTED_COMPATIBILITY_UNRESOLVED'
    except Exception as e:report.update(status='BLOCKED_AUDIT',error=f'{type(e).__name__}: {e}')
    finally:
        try:
            for f in report['files']:
                verify(f['source']);verify(f['snapshot'])
                if 'metadata_report' in f:verify(f['metadata_report'])
                if 'document_records' in f:verify(f['document_records'])
            if report['scan']['status']!='NOT_RUN':report['scan_file']=identity(root/'fixpipe-scan.json')
            for item in report['sources']:verify(item)
            for key in ('antiquant_audit','antiquant_review','input_path_evidence'):
                if key in report:verify(report[key])
        except Exception as e:report.update(status='BLOCKED_AUDIT',integrity_error=str(e))
        write(root/'priority-evidence.json',priority_index(report))
        report['priority_evidence']=identity(root/'priority-evidence.json')
        write(root/'group128-capability.json',report)
        (root/'group128-capability.md').write_text(markdown(report))
        if report['status']=='EVIDENCE_COLLECTED_COMPATIBILITY_UNRESOLVED':
            # Independent template, bound to this completed evidence file.
            # No consumer can use it to authorize AntiQuant compilation.
            write(root/'fixpipe-review-template.json',dict(schema='fixpipe-group128-source-review-v1',
                audit=identity(root/'group128-capability.json'),cann_root=str(cann),reviewer='',
                frontends=report['paths']['FixPipe Output Path']['frontends'],
                caveat='Record unsupported combinations explicitly. Cite source path/SHA256/line range/verbatim text. Never infer all three frontend capabilities from one API.',
                numerical_adoption='UNRESOLVED_REQUIRES_SEPARATE_DEVICE_AND_GOLDEN_VALIDATION'))
    print(report['status']+': '+str(root/'group128-capability.json'))
    return int(report['status']=='BLOCKED_AUDIT')


def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cann-root',type=Path,default=Path('/usr/local/Ascend/cann-9.0.0'))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--antiquant-audit',type=Path)
    p.add_argument('--antiquant-review',type=Path)
    a=p.parse_args()
    if a.antiquant_review and not a.antiquant_audit:p.error('--antiquant-review requires existing --antiquant-audit')
    return a


if __name__=='__main__':raise SystemExit(run(parse()))
