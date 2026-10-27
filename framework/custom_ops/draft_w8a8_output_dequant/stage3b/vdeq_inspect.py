"""Four fixed CANN files only; no scanning or inferred hardware support."""
from pathlib import Path
from s3b_common import identity,verify,write

FILES={
 'impl/adv_api/detail/matmul/stage/copy_cube_out/quant/quant_processor_datacopy.h':[(28,90),(113,158)],
 'impl/adv_api/detail/matmul/stage/copy_cube_out/copy_cube_out_datacopy.h':[(107,136),(163,229),(245,327)],
 'impl/basic_api/dav_m200/kernel_operator_data_copy_impl.h':[(542,707)],
 'impl/adv_api/detail/matmul/stage/copy_cube_out/copy_cube_out_datacopy_wrapper.h':[(290,360)],
}


def inspect(cann,out,prior):
    asc=(cann/'aarch64-linux/asc').resolve(strict=True);out.mkdir();rows=[];text=[]
    indexed={f['source']['path']:f['source'] for f in prior.get('files',[])}
    for rel,ranges in FILES.items():
        path=asc/rel;row=dict(relative=rel,status='UNRESOLVED');rows.append(row)
        try:
            real=path.resolve(strict=True)
            if asc not in real.parents:raise ValueError('SDK path escapes asc root')
            meta=identity(real)
            if meta['bytes']>16*1024*1024:raise ValueError('fixed source exceeds 16 MiB')
            if str(real) in indexed:verify(indexed[str(real)]);row['prior_binding']='MATCHED'
            else:row['prior_binding']='NEW_LIVE_SOURCE_IDENTITY_ONLY'
            raw=real.read_bytes();verify(meta);lines=raw.decode('utf-8').splitlines()
            excerpts=[]
            for lo,hi in ranges:
                excerpts.append(dict(start=lo,end=min(hi,len(lines)),text='\n'.join(f'{i}: {lines[i-1]}' for i in range(lo,min(hi,len(lines))+1))))
            snap=out/(str(len(rows))+'-'+path.name);snap.write_bytes(raw)
            row.update(status='SOURCE_COLLECTED_NOT_REVIEWED',source=meta,snapshot=identity(snap),excerpts=excerpts,
                symbol_lines={word:[i for i,line in enumerate(lines,1) if word in line] for word in
                  ('VDEQ16','VDEQF16','copy_matrix_cc_to_ubuf','set_deqscale','CRMODE_','CopyCo12Co2WithoutQuant','Muls<')})
            text += ['FILE: '+str(real),'SHA256: '+meta['sha256'],*[x['text'] for x in excerpts],'']
        except (OSError,ValueError,UnicodeError) as ex:row['error']=f'{type(ex).__name__}: {ex}'
    result=dict(status='READ_ONLY_STATIC_EVIDENCE_ONLY',files=rows,
        low_level_intrinsic_review='UNRESOLVED_UNTIL_ACTUAL_SOURCE_BRANCH_REVIEW',
        known_route='Stage2A/2B: INT32 L0C -> enhanced DataCopy VDEQ16 -> FP16 UB/GM; not a standalone Vector Cast+Mul or 910B Fixpipe specialization',
        layout_distinction='Muls(...,1) in NZ/ND wrappers is layout handling, not the coefficient dequantization',
        stage3b2='DESIGN_ONLY; Local ND vs NZ GetTensorC quantization behavior must be checked separately',npu='NOT_RUN',performance='NOT_RUN')
    write(out/'vdeq-static.json',result);(out/'vdeq-excerpts.txt').write_text('\n'.join(text))
    return result
