"""Reuse four frozen Stage3B snapshots. No SDK scan or inferred runtime PASS."""
from pathlib import Path
from local_common import identity,verify,write

RANGES={
 'quant_processor_datacopy.h':[(1,260)],
 'copy_cube_out_datacopy.h':[(1,136),(163,200),(245,327)],
 'kernel_operator_data_copy_impl.h':[(542,572),(1050,1105)],
 'copy_cube_out_datacopy_wrapper.h':[(290,360)],
}
NEEDED={
 'quant_processor_datacopy.h':('SetQuantVector','QuantMode_t::VDEQF16','DeqScale::VDEQ16','CopyQuantTensor'),
 'copy_cube_out_datacopy.h':('CopyOutNZ2NZ','CopyOutNZ2ND','CopyCo12Co2WithQuant','CopyCo12Co2WithoutQuant'),
 'kernel_operator_data_copy_impl.h':('CRMODE_DEQSCALE_VDEQ16','copy_matrix_cc_to_ubuf','set_deqscale'),
 'copy_cube_out_datacopy_wrapper.h':('Muls<',),
}


def review(static,root):
    root.mkdir();seen=set();files=[];lines=[]
    for record in static['files']:
        name=Path(record['relative']).name
        if name not in RANGES or name in seen:raise ValueError('unexpected/duplicate static source')
        seen.add(name)
        src=record['source'];snap=record['snapshot'];verify(src);raw=verify(snap).read_bytes()
        if any(src[k]!=snap[k] for k in ('sha256','bytes')):raise ValueError('SDK snapshot identity differs')
        text=raw.decode('utf-8');body=text.splitlines()
        if any(t not in text for t in NEEDED[name]):raise ValueError('expected bounded source route absent; review UNRESOLVED')
        excerpts=[dict(start=lo,end=min(hi,len(body)),text='\n'.join(f'{i}: {body[i-1]}' for i in range(lo,min(hi,len(body))+1))) for lo,hi in RANGES[name]]
        files.append(dict(source=src,snapshot=snap,excerpts=excerpts))
        lines += [f"SOURCE: {src['path']}",f"SHA256: {src['sha256']}",*[e['text'] for e in excerpts],'']
    if seen!=set(RANGES):raise ValueError('incomplete fixed SDK snapshots')
    result=dict(status='STATIC_BRANCH_EVIDENCE_LOCAL_COMBINATION_UNRESOLVED',files=files,
        findings=dict(local_nz='Provided lines 107-136 call CopyCo12Co2WithQuant; complete template/tiling selection remains a compile probe',
            local_nd='Provided lines 163-200 call CopyCo12Co2WithoutQuant; do not substitute Local ND',
            scale='INT8/half SetQuantVector -> VDEQF16 -> VDEQ16; CopyQuantTensor supplies local address; lifetime/offset body preserved for review',
            instruction='dav_m200 s32->half branch uses copy_matrix_cc_to_ubuf with CRMODE_DEQSCALE_VDEQ16; not standalone Vector Cast+Mul',
            layout='Muls in SDK NZ/ND wrapper is separate layout handling; probe uses bit-preserving aligned DMA instead'),
        local_template='UNRESOLVED_UNTIL_CANN_COMPILE',host_tiling='NOT_RUN',npu_scale_layout_safety='NOT_RUN',
        source_review='NOT_A_SOURCE_REVIEWED_CERTIFICATE',fallback='NONE')
    write(root/'local-review.json',result);(root/'local-excerpts.txt').write_text('\n'.join(lines))
    return result
