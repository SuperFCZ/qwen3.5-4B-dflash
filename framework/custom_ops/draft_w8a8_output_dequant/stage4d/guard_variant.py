"""The sole experimental Kernel edit is deletion of the group guard conditional."""
import difflib,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
FROZEN=HERE.parent/'stage4b'
KERNEL='op_kernel/d_flash_w8_a8_full_gate.cpp'
SCAN='''        if(!CheckGuard(scratchRaw.Get<uint32_t>(),td.localMemSize)||
           !CheckGuard(p1Raw.Get<uint32_t>(),2048)||!CheckGuard(p0FinalRaw.Get<uint32_t>(),2048)||
           !CheckGuard(castRaw.Get<uint32_t>(),4096)||!CheckGuard(sumRaw.Get<uint32_t>(),4096))return;
'''
POLICIES={'A':'BASELINE_GROUP_GUARDS','B':'DIAGNOSTIC_ONLY_WEAKER_GUARD_COVERAGE'}

def digest(data):return hashlib.sha256(data).hexdigest()

def transform(original,variant):
    if variant not in POLICIES:raise ValueError('unknown variant')
    text=original.decode('utf-8')
    if text.count(SCAN)!=1:raise ValueError('group scan must occur exactly once; no adaptive patching')
    start=text.index(SCAN)
    before='SetFlag<HardEvent::V_S>(groupReady);WaitFlag<HardEvent::V_S>(groupReady);\n'
    if not text[:start].endswith(before) or not text[start+len(SCAN):].startswith('    }\n    // All modes completed twenty Matmuls'):
        raise ValueError('guard span is not the audited group-loop tail')
    return original if variant=='A' else (text[:start]+text[start+len(SCAN):]).encode()


def materialize(root,identity,write):
    root.mkdir();original=(FROZEN/KERNEL).read_bytes();variants={}
    originals=[FROZEN/'operator.json']+[p for folder in ('op_host','op_kernel') for p in sorted((FROZEN/folder).iterdir()) if p.is_file()]
    for variant in POLICIES:
        target=root/variant;target.mkdir();files=[]
        for p in originals:
            rel=p.relative_to(FROZEN);out=target/rel;out.parent.mkdir(parents=True,exist_ok=True)
            payload=transform(original,variant) if str(rel)==KERNEL else p.read_bytes()
            out.write_bytes(payload);files.append(identity(out))
        variants[variant]=dict(source_dir=str(target),files=files,kernel=identity(target/KERNEL),safety_protocol=POLICIES[variant])
    a=(root/'A'/KERNEL).read_bytes();b=(root/'B'/KERNEL).read_bytes()
    if a!=original or b!=transform(a,'B'):raise ValueError('source transformation mismatch')
    for p in originals:
        rel=p.relative_to(FROZEN)
        if str(rel)!=KERNEL and not (p.read_bytes()==(root/'A'/rel).read_bytes()==(root/'B'/rel).read_bytes()):raise ValueError('non-Kernel source differs')
    diff=''.join(difflib.unified_diff(a.decode().splitlines(True),b.decode().splitlines(True),fromfile='A/'+KERNEL,tofile='B/'+KERNEL))
    (root/'group-guard-only.diff').write_text(diff)
    audit=dict(status='EXACT_GROUP_SCAN_ONLY_SOURCE_DIFF',originals=[identity(p) for p in originals],variants=variants,
        diff=identity(root/'group-guard-only.diff'),removed_text_sha256=digest(SCAN.encode()),removed_check_calls=5,
        group_scan_success_path_getvalue_calls_removed=304*20*5*64*2,
        synchronized_group_tail_retained=True,host_and_operator_json_identical=True,
        interpretation='Source-level success-path call count, not measured device instructions or pure wall-time attribution',
        adoption='NOT_ELIGIBLE_FOR_PRODUCTION; B has weaker detection coverage')
    write(root/'source-audit.json',audit);return audit


def verify_audit(audit,verify):
    expected_paths=[FROZEN/'operator.json']+[p for sub in ('op_host','op_kernel') for p in sorted((FROZEN/sub).iterdir()) if p.is_file()]
    if audit.get('status')!='EXACT_GROUP_SCAN_ONLY_SOURCE_DIFF' or set(audit.get('variants',{}))!={'A','B'} or len(audit['originals'])!=len(expected_paths) or {x['path'] for x in audit['originals']}!={str(p.resolve()) for p in expected_paths}:raise ValueError('source audit inventory differs')
    for item in audit['originals']:verify(item)
    a=verify(audit['variants']['A']['kernel']).read_bytes();b=verify(audit['variants']['B']['kernel']).read_bytes()
    if a!=(FROZEN/KERNEL).read_bytes() or b!=transform(a,'B'):raise ValueError('Kernel source difference changed')
    for variant,entry in audit['variants'].items():
        if entry['safety_protocol']!=POLICIES[variant]:raise ValueError('coverage label differs')
        wanted={str((Path(entry['source_dir'])/p.relative_to(FROZEN)).resolve()) for p in expected_paths}
        if len(entry['files'])!=len(wanted) or {x['path'] for x in entry['files']}!=wanted:raise ValueError('variant source inventory differs')
        for item in entry['files']:verify(item)
        for p in audit['originals']:
            original=verify(p);rel=original.relative_to(FROZEN)
            expected=transform(a,variant) if str(rel)==KERNEL else original.read_bytes()
            if (Path(entry['source_dir'])/rel).read_bytes()!=expected:raise ValueError('materialized source drift')
    actual=verify(audit['diff']).read_text()
    expected_diff=''.join(difflib.unified_diff(a.decode().splitlines(True),b.decode().splitlines(True),fromfile='A/'+KERNEL,tofile='B/'+KERNEL))
    if actual!=expected_diff or audit['removed_text_sha256']!=digest(SCAN.encode()) or audit['group_scan_success_path_getvalue_calls_removed']!=3891200:raise ValueError('audit diff/count metadata differs')
