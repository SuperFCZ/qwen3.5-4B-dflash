"""Bounded read-only device mapping evidence; never create/destroy a vNPU."""
import argparse,json,os,shutil,subprocess
from pathlib import Path
from platform_common import c


def main(a):
    root=c.fresh(a.output_dir);tool=shutil.which('npu-smi')
    if not tool:raise ValueError('npu-smi unavailable')
    r=dict(status='READ_ONLY_DEVICE_EVIDENCE',tool=c.identity(tool),logical_device=0,
        requested_physical_mapping={'npu_id':a.npu_id,'chip_id':a.chip_id},mapping_verification='OPERATOR_MUST_CONFIRM_AGAINST_INFO_LIST',
        virtualization='UNRESOLVED_NOT_INFERRED_FROM_CORE_COUNT',environment={k:os.environ.get(k) for k in ('ASCEND_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES','ASCEND_DEVICE_ID','ASCEND_VNPU_SPECS','ASCEND_RUNTIME_OPTIONS')},queries=[])
    for label,args in [('list',['info','-l']),('health',['info']),('help',['info','-h']),('vnpu',['info','-t','info-vnpu','-i',str(a.npu_id),'-c',str(a.chip_id)])]:
        out=root/(label+'.stdout');err=root/(label+'.stderr')
        with out.open('wb') as f,err.open('wb') as g:rc=subprocess.run([tool,*args],stdout=f,stderr=g).returncode
        r['queries'].append(dict(label=label,argv=[tool,*args],returncode=rc,stdout=c.identity(out),stderr=c.identity(err)))
    r['device_nodes']=[dict(path=str(p),major=os.major(p.stat().st_rdev),minor=os.minor(p.stat().st_rdev)) for p in sorted(Path('/dev').glob('davinci*'))]
    mdev=Path('/sys/bus/mdev/devices');r['visible_mdev_entries']=sorted(p.name for p in mdev.iterdir()) if mdev.is_dir() else None
    c.verify_tree(r);c.write(root/'report.json',r);print(json.dumps(r,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--npu-id',type=int,required=True);p.add_argument('--chip-id',type=int,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if a.npu_id<0 or a.chip_id<0:p.error('nonnegative physical identifiers required')
    main(a)
