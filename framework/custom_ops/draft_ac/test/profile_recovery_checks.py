"""Read-only recovery checks. No model execution and no acceptance promotion."""
import csv
import json
import math
import re
from pathlib import Path
import sqlite3
import statistics
from common import sha256


def snapshot(root, *, profiler=False):
    root=Path(root).resolve();items={};cache={}
    if not root.is_dir():raise ValueError('missing evidence directory: '+str(root))
    for path in sorted(root.rglob('*')):
        if path.is_symlink() and (profiler or path.is_dir()):raise ValueError('unexpected evidence symlink: '+str(path))
        if not path.is_file():continue
        resolved=path.resolve();before=resolved.stat();key=(str(resolved),before.st_size,before.st_mtime_ns)
        digest=cache.get(key)
        if digest is None:digest=sha256(resolved);cache[key]=digest
        after=resolved.stat()
        if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):
            raise ValueError('evidence changed while hashing: '+str(path))
        items[str(path.relative_to(root))]=dict(path=str(resolved),bytes=after.st_size,sha256=digest)
    return items


def profiler_inventory(root, expected_count):
    files=snapshot(root,profiler=True)
    if len(files)!=expected_count:raise ValueError(f'profiler file inventory differs: expected {expected_count}, found {len(files)}')
    formats={};usable=0
    for rel,record in files.items():
        path=Path(root)/rel
        try:
            if path.suffix=='.json':
                if record['bytes']>512*1024*1024:raise ValueError('JSON exceeds validation memory limit')
                with path.open() as stream:json.load(stream)
                formats[rel]='JSON_PARSED'
                usable+=path.name.startswith('msprof')
            elif path.suffix=='.csv':
                with path.open(encoding='utf-8-sig',newline='') as stream:
                    rows=csv.reader(stream,strict=True);header=next(rows,None)
                    if not header:raise ValueError('CSV has no header')
                    if path.name.startswith('op_summary') and not {'opname','optype','taskdurationus'}<={re.sub('[^a-z0-9]','',h.lower()) for h in header}:
                        raise ValueError('op_summary name/type/duration-us columns missing')
                    count=0
                    for row in rows:
                        if row and len(row)!=len(header):raise ValueError('CSV row width differs')
                        count+=bool(row)
                    if path.name.startswith('op_summary') and not count:raise ValueError('op_summary has no kernel rows')
                formats[rel]='CSV_PARSED';usable+=path.name.startswith('op_summary')
            elif path.suffix in ('.db','.sqlite','.sqlite3') and record['bytes']:
                connection=sqlite3.connect(path.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
                try:
                    if connection.execute('PRAGMA quick_check').fetchall()!=[('ok',)]:raise ValueError('SQLite quick_check failed')
                finally:connection.close()
                formats[rel]='SQLITE_QUICK_CHECK'
            else:formats[rel]='OPAQUE_BYTES_SNAPSHOTTED_NOT_DECODED'
        except Exception as error:raise ValueError('profiler artifact invalid: '+str(path)+': '+str(error)) from error
    if not usable:raise ValueError('no existing msprof timeline/op_summary export; cannot validate a completed capture')
    return dict(files=files,format_checks=formats,file_count=len(files),
                integrity_scope='053c25a did not persist capture-time profiler hashes. These are recovery-time hashes plus structural checks, not proof against prior undetectable byte changes.')


def timing_check(execution):
    t=execution['timing'];values=t['execute_sync'];samples=values['samples_ms']
    if (t.get('protocol')!='continuous-v1' or t.get('warmup')!=5 or t.get('repetitions')!=30 or
        t.get('timed_tail_checked') is not True or len(samples)!=30 or any(not math.isfinite(x) or x<=0 for x in samples)):
        raise ValueError('saved timing is not complete 5/30 continuous-v1')
    expected={'median_ms':statistics.median(samples),'p95_ms':sorted(samples)[28]}
    for key,value in expected.items():
        # Timing::Write serializes nine significant digits; this is serialization
        # validation, NOT a tolerance on any tensor/Token correctness comparison.
        if not math.isclose(values[key],value,rel_tol=1e-8,abs_tol=1e-8):raise ValueError('saved timing statistic differs: '+key)


def tensor_metrics(a,b):
    import numpy as np
    aa=a.view('<f2').astype('float64');bb=b.view('<f2').astype('float64')
    finite=np.isfinite(aa)&np.isfinite(bb);error=np.abs(aa[finite]-bb[finite]);n=int(a.size)
    return dict(elements=n,bit_mismatches=int(np.count_nonzero(a!=b)),mismatch_ratio=float(np.count_nonzero(a!=b)/n) if n else 0.,
        actual_nonfinite=int(np.count_nonzero(~np.isfinite(aa))),reference_nonfinite=int(np.count_nonzero(~np.isfinite(bb))),
        nonfinite_pairs=int(n-np.count_nonzero(finite)),finite_pairs=int(error.size),
        max_abs=float(error.max()) if error.size else None,
        rmse=float(np.sqrt(np.sum(error*error,dtype='float64')/error.size)) if error.size else None,
        p99_abs=float(np.partition(error,math.ceil(.99*error.size)-1)[math.ceil(.99*error.size)-1]) if error.size else None)


def output_metrics(case,folder,observation):
    """Verify saved boundary output bytes against embedded diagnostic metrics."""
    import numpy as np
    start=case['controls']['start_position'];valid=case['controls']['valid_rows'];gear=case['gear']
    for spec in case['abi']['outputs']:
        name=spec['name'];path=folder/(name+'-'+observation['observation']);ref=case['golden'][name]
        size=math.prod(spec['shape'])*(8 if name=='draft_top1' else 2)
        if not path.is_file() or path.stat().st_size!=size:raise ValueError('saved output missing/truncated: '+str(path))
        expected=observation['outputs'][name]
        if name=='draft_top1':
            a=np.memmap(path,dtype='<i8',mode='r');b=np.memmap(ref,dtype='<i8',mode='r')
            count=int(np.count_nonzero(a!=b))
            if expected['actual_ids']!=a.tolist() or expected['mismatches']!=count or expected['bitwise_equal']!=(count==0):
                raise ValueError('saved Token bytes differ from observation')
            continue
        a=np.memmap(path,dtype='<u2',mode='r',shape=tuple(spec['shape']))
        b=np.memmap(ref,dtype='<u2',mode='r',shape=tuple(spec['shape']))
        x=np.memmap(case['inputs'][name],dtype='<u2',mode='r',shape=tuple(spec['shape']))
        regions={'all':(a,b),'valid_prefix':(a[:,:,:start+valid,:],b[:,:,:start+valid,:]),
            'written_valid':(a[:,:,start:start+valid,:],b[:,:,start:start+valid,:]),
            'written_padding':(a[:,:,start+valid:start+gear,:],b[:,:,start+valid:start+gear,:]),
            'untouched_prefix_vs_input':(a[:,:,:start,:],x[:,:,:start,:]),
            'untouched_tail_vs_input':(a[:,:,start+gear:,:],x[:,:,start+gear:,:])}
        for region,(left,right) in regions.items():
            for key,actual in tensor_metrics(left,right).items():
                wanted=expected[region][key]
                equal=actual==wanted
                if key=='rmse' and actual is not None and wanted is not None:
                    equal=math.isclose(actual,wanted,rel_tol=1e-12,abs_tol=1e-15)
                if not equal:raise ValueError(f'saved FP16 output metrics differ: {path.name}/{region}/{key}')
