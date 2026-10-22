"""Collect installed SDK evidence. API names are discovery, never support proof.

Build is gated by a separate, source-cited review of the collected CANN files.
No import of torch/torch_npu, no device initialization, no old OPP loading.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from aq_common import HERE, identity, verify, write, fresh, experiment_sources

PATTERN = re.compile(r'SetAntiQuantVector|AntiQuant|anti_?quant|antiquant|WeightQuantBatchMatmul(?:V3|Nz)|'
                     r'GetMatmulType|MatmulType|enPartialSum|Src[AB]?T.*(?:int8|half)|Ascend310P3')
REQUIRED = {
    'chip': 'Ascend310P3 / dav_2002 (SDK may use dav_m200), actual 9.0.0 branch',
    'mixed_types': 'A=GM ND half; B=GM NZ int8_t transpose=true; C=GM ND half',
    'nz_layout': 'B physical [4,4,16,32], logical [64,128], exact signed INT8 NZ inverse',
    'accumulator': 'mixed half/int8 specialization uses FP32 L0C, final FP16 output',
    'vector_contract': 'LocalTensor<half> VECIN offset[64]/scale[64], 32B alignment, 128B each',
    'parameter_lifetime': 'PIPE_ALL after GM->UB; both vectors live through IterateAll + End',
    'tiling': 'MatmulApiTiling GM/NZ/DT_INT8 transpose=true + GM/ND/DT_FLOAT16, base 16/64/128',
    'workspace_contract': '256B user vectors + (UB-256-8192) SDK local workspace; internal allocations fit remaining 8192B',
}
LATER = ('group128_scale_refresh_fp32_partial_sum', 'aclnnWeightQuantBatchMatmulV3',
         'aclnnWeightQuantBatchMatmulNz')


def collect(cann, output):
    cann = Path(cann).resolve(strict=True)
    root = fresh(output, (cann,))
    report = dict(status='SDK_REVIEW_REQUIRED', cann_root=str(cann), soc='Ascend310P3',
                  cann_compile='NOT_RUN', npu='NOT_RUN', k256='NOT_RUN', k2560='NOT_RUN',
                  full_gate='NOT_RUN', files=[], tools={}, alternative_libraries=[],
                  experiment_sources=experiment_sources(),
                  rule='Presence of a declaration/header/symbol is NOT chip/dtype/group support proof.')
    try:
        (root / 'sources').mkdir()
        seen = set()
        # Scan headers/implementation and platform/version metadata, not binaries.
        # Resolve internal SDK symlinks but never follow an escaped SDK file.
        def scan_error(error):
            raise error
        for base, dirs, names in os.walk(cann, followlinks=False, onerror=scan_error):
            dirs[:] = sorted(d for d in dirs if d not in ('node_modules', '__pycache__', '.git'))
            for name in sorted(names):
                p = Path(base) / name
                normalized_path=re.sub(r'[^a-z0-9]','',str(p.relative_to(cann)).lower())
                op_metadata=p.suffix.lower() in ('.json','.ini','.py') and any(
                    token in normalized_path for token in ('weightquantbatchmatmulv3','weightquantbatchmatmulnz'))
                if not (p.suffix.lower() in ('.h', '.hpp', '.cpp', '.cc', '.ini') or
                        name in ('version.info', 'version.cfg') or op_metadata):
                    continue
                resolved = p.resolve(strict=True)
                if cann not in resolved.parents:
                    raise ValueError('SDK source escapes selected root: ' + str(p))
                if resolved in seen:
                    continue
                seen.add(resolved)
                if resolved.stat().st_size > 16 * 1024 * 1024:
                    raise ValueError('unexpected SDK source size: ' + str(p))
                text = resolved.read_text(errors='replace')
                version = name in ('version.info', 'version.cfg')
                platform = name == 'Ascend310P3.ini'
                hits = [i for i, line in enumerate(text.splitlines(), 1) if PATTERN.search(line)]
                # Include the complete Matmul helper family, not just methods
                # containing the public API name (e.g. Src/Dst/AccType traits).
                family = any('matmul' in part.lower() or 'anti_quant' in part.lower()
                             for part in p.relative_to(cann).parts)
                if not (hits or version or platform or family or op_metadata):
                    continue
                record = identity(resolved)
                snapshot = root / 'sources' / (str(len(report['files'])) + '-' + name)
                shutil.copyfile(resolved, snapshot)
                lines = text.splitlines()
                intervals = [(max(1, n-12), min(len(lines), n+30)) for n in hits]
                merged = []
                for lo, hi in intervals:
                    if merged and lo <= merged[-1][1] + 1:
                        merged[-1][1] = max(hi, merged[-1][1])
                    else:
                        merged.append([lo, hi])
                report['files'].append(dict(source=record, snapshot=identity(snapshot),
                    role='version' if version else 'platform' if platform else 'source',
                    hits=hits, excerpts=[dict(start=lo, end=hi,
                        text='\n'.join(f'{i}: {lines[i-1]}' for i in range(lo, hi+1))) for lo, hi in merged]))
        versions = [verify(f['snapshot']).read_text(errors='replace') for f in report['files'] if f['role']=='version']
        if not any(re.search(r'(?<![\d.])9\.0\.0(?![\d.])', v) for v in versions):
            raise ValueError('selected SDK does not identify CANN 9.0.0')
        api = [f for f in report['files'] if 'SetAntiQuantVector' in verify(f['snapshot']).read_text(errors='replace')]
        if not api:
            report.update(status='BLOCKED_MISSING_API', reason='No SetAntiQuantVector declaration/implementation in selected SDK')
        for tool in ('bisheng', 'ccec', 'msopgen'):
            located = shutil.which(tool)
            if located:
                result = subprocess.run([located, '--version'], capture_output=True, text=True, timeout=30)
                report['tools'][tool] = dict(file=identity(located), returncode=result.returncode,
                                            stdout=result.stdout, stderr=result.stderr)
        # Merely inventory native API libraries; declarations/symbols cannot prove 310P3 support.
        for name in ('libopapi.so', 'libnnopbase.so'):
            paths = {p.resolve() for p in cann.rglob(name) if p.is_file()}
            for p in sorted(paths):
                if cann not in p.parents:
                    raise ValueError('SDK library escapes selected root')
                item = dict(file=identity(p), support='UNRESOLVED')
                nm = shutil.which('nm')
                if nm:
                    r = subprocess.run([nm, '-D', '--defined-only', str(p)], capture_output=True, text=True, timeout=60)
                    item.update(nm_returncode=r.returncode, symbols=[line for line in r.stdout.splitlines()
                        if re.search(r'aclnnWeightQuantBatchMatmul(?:V3|Nz)', line)], stderr=r.stderr)
                report['alternative_libraries'].append(item)
        for f in report['files']:
            verify(f['source']); verify(f['snapshot'])
    except Exception as e:
        report.update(status='BLOCKED_AUDIT', error=f'{type(e).__name__}: {e}')
    path = root / 'sdk-capability.json'
    write(path, report)
    template = dict(audit=identity(path), reviewer='', decision='UNRESOLVED',
        claims={k: dict(status='UNRESOLVED', requirement=v, finding='', citations=[])
                for k,v in REQUIRED.items()},
        later={k:dict(status='UNRESOLVED', finding='', citations=[]) for k in LATER},
        citation_format={'source_path':'exact source.path in audit', 'sha256':'source.sha256',
                         'start':1, 'end':2, 'quote':'verbatim source lines without line-number prefixes'})
    write(root / 'review-template.json', template)
    (root / 'sdk-capability.md').write_text(
        '# Installed CANN AntiQuant capability evidence\n\nStatus: ' + report['status'] + '\n\n' +
        '\n'.join('- '+k+': '+v for k,v in REQUIRED.items()) +
        '\n\nSource declarations are discovery only. Inspect implementation/chip/template restrictions and fill a separate source-cited review. '
        'Do not mark group128 partial-sum or V3/Nz supported from interface names alone. '
        'K128 compile, NPU, K256/K2560 and full Gate: NOT_RUN.\n')
    return report, path


def reviewed(audit_path, review_path):
    """Fail closed before msopgen/build/device. No automatic token-to-support inference."""
    audit = json.loads(Path(audit_path).read_text())
    review = json.loads(Path(review_path).read_text())
    if review.get('audit') != identity(audit_path) or audit.get('status') != 'SDK_REVIEW_REQUIRED':
        raise ValueError('review does not bind a complete SDK audit')
    if not review.get('reviewer', '').strip():
        raise ValueError('source reviewer identity required')
    for item in audit['experiment_sources']:
        verify(item)
    files = {f['source']['path']: f for f in audit['files']}
    for f in files.values():
        verify(f['source']); verify(f['snapshot'])
    for item in audit['tools'].values(): verify(item['file'])
    for item in audit['alternative_libraries']: verify(item['file'])
    if set(review.get('claims', {})) != set(REQUIRED):
        raise ValueError('incomplete K128 contract review')
    if set(review.get('later', {})) != set(LATER):
        raise ValueError('missing separate group128/V3/Nz investigation status')
    for key, claim in {**review['claims'], **review['later']}.items():
        if claim.get('status') not in ('SUPPORTED', 'UNSUPPORTED', 'UNRESOLVED'):
            raise ValueError('invalid claim status')
        if claim['status']=='UNRESOLVED':
            continue
        if not claim.get('finding', '').strip() or not claim.get('citations'):
            raise ValueError('source-cited findings required: '+key)
        for cite in claim['citations']:
            f = files.get(cite.get('source_path'))
            if not f or f['source']['sha256'] != cite.get('sha256'):
                raise ValueError('citation not bound to SDK snapshot')
            lines = verify(f['snapshot']).read_text(errors='replace').splitlines()
            lo,hi = cite.get('start'),cite.get('end')
            if type(lo) is not int or type(hi) is not int or not 1<=lo<=hi<=len(lines):
                raise ValueError('invalid citation lines')
            if not cite.get('quote') or cite['quote'] != '\n'.join(lines[lo-1:hi]):
                raise ValueError('citation text changed')
    unsupported = [k for k,c in review['claims'].items() if c['status']=='UNSUPPORTED']
    if unsupported:
        raise ValueError('BLOCKED_UNSUPPORTED: '+','.join(unsupported))
    unresolved = [k for k,c in review['claims'].items() if c['status']=='UNRESOLVED']
    if unresolved:
        raise ValueError('SDK_REVIEW_REQUIRED: '+','.join(unresolved))
    if review.get('decision') != 'K128_SOURCE_REVIEWED':
        raise ValueError('K128 source review incomplete')
    return audit, review


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cann-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();r,path=collect(a.cann_root,a.output_dir)
    print(r['status']+': '+str(path));raise SystemExit(1 if r['status'].startswith('BLOCKED') else 0)
