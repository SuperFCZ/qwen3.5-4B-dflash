"""Complete read-only source audit; no torch/NPU import until all entries match."""
import json
from pathlib import Path
import re
from common import ROOT, sha256, write_json


def audit_source_lock(root, historical):
    root = Path(root).resolve()
    lock = root/'SOURCE_LOCK.json'
    report = dict(status='FAIL', scope='source identity only; GE/ATC/NPU NOT_RUN',
                  files=[], historical=historical, identity_diff={})
    if lock.is_symlink() or not lock.is_file():
        report['error'] = 'SOURCE_LOCK missing or symlink'
        return report
    try:
        payload = json.loads(lock.read_text())
        if not isinstance(payload, dict): raise ValueError('SOURCE_LOCK must be an object')
    except (ValueError, OSError) as error:
        report['error'] = str(error)
        return report

    def visit(value, location='$'):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.endswith('_file') and isinstance(item, str) and isinstance(value.get(key[:-5]+'_sha256'), str):
                    expected = value[key[:-5]+'_sha256']
                    entry = dict(declaration=location+'.'+key, path=item, expected_sha256=expected, status='FAIL')
                    path = root/item
                    try:
                        if path.is_symlink(): raise ValueError('SYMLINK')
                        resolved = path.resolve()
                        if resolved == root or root not in resolved.parents: raise ValueError('OUTSIDE_ROOT')
                        if not resolved.is_file(): raise ValueError('MISSING_FILE')
                        if not re.fullmatch('[0-9a-f]{64}', expected): raise ValueError('INVALID_SHA256')
                        entry['actual_sha256'] = sha256(resolved)
                        if entry['actual_sha256'] != expected: raise ValueError('HASH_MISMATCH')
                        entry['status'] = 'PASS'
                    except (OSError, ValueError) as error:
                        entry['error'] = str(error)
                    report['files'].append(entry)
                visit(item, location+'.'+key)
        elif isinstance(value, list):
            for i, item in enumerate(value): visit(item, f'{location}[{i}]')
    visit(payload)
    entries = report['files']
    report['declaration_count'] = len(entries)  # Same duplicate counting as the locked factory.
    report['unique_file_count'] = len({e['path'] for e in entries})
    report['failed_declaration_count'] = sum(e['status'] != 'PASS' for e in entries)
    report['current'] = dict(path=str(lock), sha256=sha256(lock), verified_file_count=len(entries))
    if not isinstance(historical, dict):
        report['identity_diff'] = {'historical': dict(expected='quant_source_lock object', actual=historical)}
    else:
        report['identity_diff'] = {k: dict(historical=historical.get(k), current=report['current'].get(k))
            for k in historical.keys() | report['current'].keys() if historical.get(k) != report['current'].get(k)}
    if len(entries) < 10: report['error'] = 'SOURCE_LOCK contains too few source pairs'
    elif not report['failed_declaration_count'] and not report['identity_diff']: report['status'] = 'PASS'
    return report


def require_source_lock(native, output, *, native_root=None):
    selected = Path(native_root).resolve() if native_root is not None else ROOT
    report = audit_source_lock(selected, native['metadata'].get('quant_source_lock'))
    try:
        if report['status'] != 'PASS':
            raise ValueError('source files or historical quant_source_lock identity differ')
        # Do not replace or weaken the original factory verifier. Run it too.
        from qwen35_dflash.ascend310p.quant_factory import _verify_quant_source_lock
        if native_root is not None:
            from native_base import active
            boundary=active()
            if boundary is None or boundary.base!=selected:
                raise RuntimeError('historical source verification requires an isolated Native boundary')
            verified=boundary.before_factory(native)
        else:
            verified = _verify_quant_source_lock()
        if verified != report['current']:
            raise ValueError('factory verifier identity differs from full audit')
        report['factory_verifier'] = 'PASS'
        # Extensions are recorded separately, never passed off as locked Native code.
        base = ROOT/'framework/python/qwen35_dflash/ascend310p'
        report['diagnostic_extensions'] = [dict(path=str(p.relative_to(ROOT)), sha256=sha256(p))
            for p in sorted(base.glob('draft_ac_*.py'))]
    except Exception as error:
        report.update(status='FAIL', error=f'{type(error).__name__}: {error}')
    write_json(output, report)
    if report['status'] != 'PASS':
        raise RuntimeError(f'complete source lock preflight failed; see {output}')
    return report
