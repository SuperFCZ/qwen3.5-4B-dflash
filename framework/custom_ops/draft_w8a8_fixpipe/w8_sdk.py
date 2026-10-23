"""Fixed-file CANN preflight. No recursive SDK discovery and no inferred support."""
import json
from pathlib import Path
import re
import shutil
import subprocess
from w8_common import HERE,identity,verify,write,sources

ASC_FILES=(
 'include/adv_api/matmul/matmul_config.h',
 'impl/adv_api/detail/matmul/matmul_impl_base.h',
 'impl/adv_api/detail/matmul/utils/matmul_utils.h',
 'impl/adv_api/detail/matmul/feature_trait/matmul_chip_cap.h',
 'impl/adv_api/detail/matmul/stage/copy_cube_out/quant/quant_processor_using_ub.h',
 'impl/adv_api/detail/matmul/stage/copy_cube_out/quant/quant_processor_intf.h',
 'impl/adv_api/detail/matmul/stage/copy_cube_out/quant/quant_processor_fixpipe.h',
 'impl/adv_api/detail/matmul/stage/copy_cube_out/copy_cube_out_using_ub.h',
 'impl/basic_api/dav_m200/kernel_operator_data_copy_impl.h',
 'impl/basic_api/dav_m200/kernel_operator_vec_vconv_impl.h',
)
COMMON_CLAIMS={
 'chip':'Actual 310P3/dav_2002 CANN9 branch; integer Matmul and output conversion',
 'integer_cube':'GM INT8 x INT8, GetMmDstType<int8_t>=int32_t and actual Mmad path; no weight-to-half dequant',
 'tensor_quant':'SetDequantType(TENSOR) paired with SetQuantVector(GlobalTensor<uint64_t>), N64 column scaling',
 'host_c_type':'Prove reviewed Host SetCType for half Kernel output, and raw int32 output; no copied example assumption',
 'encoding':'Exact uint64 float field, flags/sign/zero offset, storage precision and N64 contiguous layout',
 'conversion':'Effective scale precision, coefficient truncation, product rounding, FP16 RNE, overflow and subnormal behavior',
 'buffers':'GM quant parameter alignment/lifetime, SDK UB workspace and L1/L0 capacities for M16 K128 N64',
}


def preflight(cann,root,extra=()):
    report=dict(status='UNRESOLVED',cann_root=str(cann),soc='Ascend310P3',sources=sources(),files=[],missing=[],
        cann_compile='NOT_RUN',npu='NOT_RUN',performance='NOT_RUN',
        tools={},scope='Fixed source/ABI paths only. Presence is not support; uint64 encoding is not guessed.')
    folder=root/'sdk-sources';folder.mkdir();seen=set()
    paths=[cann/'version.info',cann/'version.cfg',cann/'aarch64-linux/asc/version.info',
           cann/'include/experiment/tiling/tiling_api.h',cann/'include/tiling/tiling_api.h']
    paths += [cann/'aarch64-linux/asc'/p for p in ASC_FILES]
    paths += [cann/p for p in extra]
    try:
        for p in paths:
            if not p.is_file():report['missing'].append(str(p));continue
            actual=p.resolve(strict=True)
            if cann not in actual.parents:raise ValueError('SDK path escapes chosen root')
            if actual in seen:continue
            seen.add(actual)
            if actual.stat().st_size>16*1024*1024:raise ValueError('source exceeds 16 MiB: '+str(p))
            source=identity(actual);snapshot=folder/(str(len(report['files']))+'-'+actual.name)
            snapshot.write_bytes(actual.read_bytes())
            if identity(snapshot)['sha256']!=source['sha256']:raise ValueError('SDK changed during snapshot')
            report['files'].append(dict(source=source,snapshot=identity(snapshot)))
        versions=[verify(f['snapshot']).read_text(errors='replace') for f in report['files'] if Path(f['source']['path']).name in ('version.info','version.cfg')]
        if not any(re.search(r'(?<![\d.])9\.0\.0(?![\d.])',v) for v in versions):raise ValueError('CANN 9.0.0 version not verified')
        for f in report['files']:verify(f['source']);verify(f['snapshot'])
        for tool in ('msopgen','bisheng','ccec','cmake','c++'):
            p=shutil.which(tool)
            if p:
                r=subprocess.run([p,'--version'],capture_output=True,text=True,timeout=30)
                report['tools'][tool]=dict(file=identity(p),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
        report['collection']='FIXED_FILES_COLLECTED'
    except Exception as e:report.update(status='BLOCKED_PREFLIGHT',error=f'{type(e).__name__}: {e}')
    path=root/'preflight.json';write(path,report)
    template=dict(preflight=identity(path),reviewer='',status='UNRESOLVED',
        common={k:dict(status='UNRESOLVED',finding=v,citations=[]) for k,v in COMMON_CLAIMS.items()},
        layouts={k:dict(status='UNRESOLVED',finding=v,citations=[]) for k,v in {
            'nd':'B GM ND [128,64], non-transposed; int8 raw and tensor-dequant output',
            'nz':'B GM INT8 NZ [4,4,16,32], logical [64,128], transpose=true; separate from ND acceptance'}.items()},
        settings=dict(host_scaled_c_dtype=None,codec={k:None for k in (
            'float_field','constant_u64','stored_mantissa_bits','effective_mantissa_bits','coefficient_rounding',
            'product_rounding','overflow','subnormal','sign','vector_storage')}),
        citation_format=dict(path='actual source path',sha256='actual SHA256',start=1,end=2,quote='verbatim source lines'))
    write(root/'review-template.json',template)
    return report


def reviewed(path,review_path,layout):
    p=json.loads(path.read_text());r=json.loads(review_path.read_text())
    if p.get('status')!='UNRESOLVED' or p.get('collection')!='FIXED_FILES_COLLECTED' or r.get('preflight')!=identity(path):
        raise ValueError('preflight identity/integrity incomplete')
    for item in p['sources']:verify(item)
    files={f['source']['path']:f for f in p['files']}
    for f in files.values():verify(f['source']);verify(f['snapshot'])
    for item in p['tools'].values():verify(item['file'])
    if set(r.get('common',{}))!=set(COMMON_CLAIMS) or set(r.get('layouts',{}))!={'nd','nz'}:
        raise ValueError('incomplete review schema')
    required={**r['common'],layout:r['layouts'][layout]}
    unsupported=[];unresolved=[]
    for key,claim in required.items():
        status=claim.get('status')
        if status=='UNRESOLVED':unresolved.append(key);continue
        if status not in ('SUPPORTED','UNSUPPORTED') or not claim.get('finding') or not claim.get('citations'):
            raise ValueError('source citations required: '+key)
        for c in claim['citations']:
            f=files.get(c.get('path'))
            if not f or c.get('sha256')!=f['source']['sha256']:raise ValueError('citation identity differs')
            lines=verify(f['snapshot']).read_text(errors='replace').splitlines();lo,hi=c.get('start'),c.get('end')
            if type(lo) is not int or type(hi) is not int or not 1<=lo<=hi<=len(lines):raise ValueError('invalid citation line range')
            if c.get('quote')!='\n'.join(lines[lo-1:hi]):raise ValueError('citation quote differs')
        if status=='UNSUPPORTED':unsupported.append(key)
    if unsupported:raise ValueError('BLOCKED_UNSUPPORTED: '+','.join(unsupported))
    if unresolved:raise ValueError('UNRESOLVED: '+','.join(unresolved))
    if r.get('status')!='SOURCE_REVIEWED' or not r.get('reviewer'):raise ValueError('source review not completed')
    from w8_reference import check_codec
    settings=r['settings'];check_codec(settings['codec'])
    if set(settings)!={'host_scaled_c_dtype','codec'} or settings['host_scaled_c_dtype'] not in ('DT_INT32','DT_FLOAT16'):
        raise ValueError('reviewed Host C type required; not inferred from ABI')
    return p,r
