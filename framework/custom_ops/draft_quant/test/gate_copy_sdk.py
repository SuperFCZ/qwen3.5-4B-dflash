"""Read the receiver CANN9/dav_m200 SDK before the isolated Gate copy build."""
import argparse
from pathlib import Path
import re
from a2_common import sha256,write_json


def inspect(root,experiment="copy"):
    root=Path(root).resolve()
    versions=[]
    for name in ('version.info','version.cfg'):
        for path in sorted(root.rglob(name)):
            if path.stat().st_size>65536:continue
            text=path.read_text(errors='replace')
            versions.append(dict(path=str(path),sha256=sha256(path),text=text))
    if not any(re.search(r'(?<![0-9])9\.0\.0(?![0-9])',v['text']) for v in versions):
        raise ValueError('cannot verify CANN 9.0.0 under CANN_ROOT; inspect version.info/version.cfg')
    structs=[];implementations=[]
    search={root}|{(root/p).resolve() for p in ('include/ascendc','compiler/tikcpp/tikcfw','aarch64-linux/include/ascendc','x86_64-linux/include/ascendc') if (root/p).is_dir()}
    for path in sorted({p.resolve() for base in search for p in base.rglob('kernel_struct_data_copy.h')}):
        text=path.read_text(errors='replace')
        if 'DataCopyParams' in text and all(field in text for field in ('blockCount','blockLen','srcStride','dstStride')):
            structs.append(dict(path=str(path),sha256=sha256(path)))
    for path in sorted({p.resolve() for base in search for p in base.rglob('kernel_operator_data_copy_impl.h')}):
        if 'dav_m200' not in path.parts:continue
        text=path.read_text(errors='replace')
        if 'copy_gm_to_ubuf' in text and all(field in text for field in ('blockCount','blockLen','srcStride','dstStride')):
            implementations.append(dict(path=str(path),sha256=sha256(path)))
    if not structs or not implementations:raise ValueError('CANN9 dav_m200 basic GM->UB DataCopy implementation/parameter fields not found; do not substitute another architecture')
    result=dict(experiment=experiment,status='SOURCE_INSPECTED',cann_root=str(root),versions=versions,parameters=structs,dav_m200=implementations,
                units='Basic DataCopy blockLen/srcStride/dstStride are 32B blocks; dstStride=0; no DataCopyPad',
                q_params=[8,64,19392,0],scale_params=[2,4,1212,0],cann_compile='NOT_RUN',npu_correctness='NOT_RUN',
                documentation='https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0103.html')
    if experiment=="prefetch":
        result.update(q_params=None,scale_params=None,raw_copy_calls_per_tile=10,
            event_api=["TPipe::AllocEventID<MTE2_S>","TPipe::AllocEventID<MTE2_V>","SetFlag/WaitFlag", "V_S", "PipeBarrier<PIPE_ALL>", "TPipe::ReleaseEventID"],
            event_headers=[dict(path=str(p),sha256=sha256(p)) for p in sorted({p.resolve() for base in search for p in base.rglob('kernel_tpipe*.h')})],
            synchronization="same reserved-event branch as accepted Down; target compilation and hardware correctness still required")
    return result



if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cann-root',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--experiment',choices=('copy','prefetch'),default='copy')
    a=p.parse_args()
    try:r=inspect(a.cann_root,a.experiment)
    except Exception as error:write_json(a.report,dict(status='FAIL',error=str(error),npu_correctness='NOT_RUN'));raise
    write_json(a.report,r)
