"""Source-only bounded adapter and read-only incomplete-trace inspection."""
import csv
import difflib
import hashlib
import json
from pathlib import Path

from simulator_prepare import replace_once, focus_lines

SHAPE=(16,2560,19456)
TILES=304


def tile_number(value):
    if type(value) is not int or not 0 <= value < TILES:
        raise ValueError('N tile must be an integer 0..303')
    return value


def bounded_source(original, tile):
    tile_number(tile)
    # Both insertions precede all per-tile computation. Removing these exact
    # insertions must reproduce the accepted source, including every K loop,
    # barrier, flag, failure cleanup and final Matmul End.
    owner = f"""    // SIMULATOR ONLY: full Gate ABI, one original cyclic owner.
    if (globalM != 16U || globalK != 2560U || globalN != 19456U || tileK != 256U) return;
    if (blockIdx != {tile}U % blockNum) return;
"""
    select = f"""        // SIMULATOR ONLY: skip other N tiles; preserve the full K body.
        if (nTile != {tile}U) continue;
"""
    first='    if (blockNum == 0 || blockIdx >= blockNum || blockIdx >= nTiles) return;\n'
    loop='    for (uint32_t nTile = blockIdx; nTile < nTiles; nTile += blockNum) {\n'
    modified=replace_once(original,first,first+owner)
    modified=replace_once(modified,loop,loop+select)
    if modified.replace(owner,'',1).replace(select,'',1)!=original:
        raise ValueError('bounded patch changed computational body')
    return modified,owner,select


def patch_kernel(path, tile):
    original=path.read_text();modified,owner,select=bounded_source(original,tile)
    path.write_text(modified)
    digest=lambda text:hashlib.sha256(text.encode()).hexdigest()
    return dict(policy='simulator-single-n-tile-full-k-v1',n_tile=tile,columns=[tile*64,(tile+1)*64],
        logical_shape=list(SHAPE),tile_n=64,tile_k=256,k_iterations=10,group128_count=20,
        owner_rule=f'{tile} % original blockNum',original_sha256=digest(original),
        bounded_sha256=digest(modified),insertions=[owner,select],
        reversible_source_check=True,full_gate_correctness='NOT_RUN',
        diff=''.join(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),
            fromfile='accepted/kernel.cpp',tofile='isolated/bounded/kernel.cpp')),
        focus_lines=focus_lines(path))


def schedule(tile, blocks, raw):
    """CPU source-policy model, not a simulator execution claim."""
    tile_number(tile)
    if type(blocks) is not int or not 1 <= blocks <= TILES:raise ValueError('invalid block count')
    return [dict(core=tile%blocks,n_tile=tile,k_begin=k*256,bank=k%2 if raw else 0,
                 prefetch_k=(k+1)*256 if raw and k<9 else None) for k in range(10)]


def compare_tile(actual, golden, tile):
    import numpy as np
    tile_number(tile)
    size=16*19456*2
    if len(actual)!=size or len(golden)!=size:raise ValueError('full ABI output byte size mismatch')
    a=np.frombuffer(actual,dtype='<u2').reshape(16,19456)
    b=np.frombuffer(golden,dtype='<u2').reshape(16,19456)
    lo,hi=tile*64,(tile+1)*64
    mismatch=a[:,lo:hi]!=b[:,lo:hi]
    # The borrowed single-call runner poisons every output byte with 0x7f.
    # Check all skipped columns as well as original buffer guards.
    untouched=bool(np.all(a[:,:lo]==0x7f7f) and np.all(a[:,hi:]==0x7f7f))
    where=np.argwhere(mismatch)
    first=None
    if len(where):
        row,col=map(int,where[0]);col+=lo
        first=dict(row=row,column=col,actual_bits=f'0x{a[row,col]:04x}',expected_bits=f'0x{b[row,col]:04x}')
    return dict(scope='only selected N64 columns, all 16 rows and complete K2560',
                n_tile=tile,elements=1024,bit_mismatches=int(mismatch.sum()),
                selected_tile_bitwise_equal=not bool(mismatch.any()),
                skipped_output_poison_intact=untouched,first_difference=first,
                full_gate_correctness='NOT_RUN')


def inspect_partial(report_path, output, identify):
    """Never write into or treat the interrupted run as completed evidence."""
    report_path=Path(report_path).resolve();root=report_path.parent
    report=json.loads(report_path.read_text())
    if report.get('status')!='BLOCKED' or report.get('phase')!='gate-off':
        raise ValueError('expected interrupted gate-off report')
    smoke=report.get('runs',{}).get('off',{}).get('smoke',{})
    if not isinstance(smoke,dict) or smoke.get('simulator_output_bitwise_match') is not True or smoke.get('profile_kernel_identity',{}).get('status')!='MATCHED' or smoke.get('artifacts',{}).get('status')!='COLLECTED':
        raise ValueError('prior completed Off smoke evidence missing')
    folder=root/'off/gate'
    execution=folder/'input/execution.json'
    execution_data=json.loads(execution.read_text())
    if execution_data.get('status')!='RUNNING':raise ValueError('partial Gate no longer RUNNING; inspect new evidence explicitly')
    profile=folder/'profile'
    if not profile.is_dir():raise ValueError('missing interrupted profile')
    entries=[];selected=[]
    for path in sorted(profile.rglob('*')):
        if path.is_symlink():raise ValueError('partial profile contains symlink: '+str(path))
        if not path.is_file():continue
        entries.append(dict(path=str(path.relative_to(profile)),bytes=path.stat().st_size))
        if path.name=='trace.json' or (path.suffix=='.csv' and ('code_exe' in path.name or 'instr_exe' in path.name)):
            item=identify(path)
            try:
                if path.suffix=='.json':
                    data=json.loads(path.read_text());events=data.get('traceEvents',[]) if isinstance(data,dict) else data
                    if not isinstance(events,list):raise ValueError('trace events not a list')
                    item.update(parse_status='PARSED_PARTIAL',event_count=len(events),
                        event_names=sorted({str(e.get('name','')) for e in events if isinstance(e,dict)})[:80])
                else:
                    with path.open(newline='',encoding='utf-8-sig') as f:
                        reader=csv.reader(f);header=next(reader,None);samples=[];count=0
                        for row in reader:
                            count+=1
                            if len(samples)<3:samples.append([v[:256] for v in row])
                    item.update(parse_status='PARSED_PARTIAL',headers=header,rows=count,samples=samples)
            except (ValueError,UnicodeError,csv.Error) as error:item.update(parse_status='UNPARSED_PARTIAL',error=str(error)[:1000])
            selected.append(item)
    if not selected:raise ValueError('no partial Trace/CSV found')
    result=dict(status='INCOMPLETE_INTERRUPTED_READ_ONLY',prior_report=identify(report_path),
        prior_execution=identify(execution),prior_execution_status=execution_data['status'],
        profile_bytes=sum(e['bytes'] for e in entries),files=entries,artifacts=selected,
        completion='NOT_COMPLETED',correctness='NOT_RUN',
        interpretation='Partial files establish available formats only; no completed tile/count or numerical claim.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result
