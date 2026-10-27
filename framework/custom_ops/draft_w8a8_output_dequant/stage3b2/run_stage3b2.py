#!/usr/bin/env python3
"""Single K128 Local NZ output probe. Never reruns prior operators or fuses groups."""
import argparse,json,os
from pathlib import Path
from local_common import *
from local_review import review
from build_local import build,installed
from run_stage2b import command


def smoke(args,root,r):
    if args.local_build_report is None:raise ValueError('--local-build-report required')
    b=json.loads(args.local_build_report.read_text());verify_tree(b)
    if b.get('status')!='STAGE3B2_BUILT_NPU_NOT_RUN' or any(b.get(k)!=r[k] for k in ('settings','sources','evidence','device')):
        raise ValueError('Local NZ build/evidence/config binding differs')
    if installed(Path(b['opp']))!=b['installed']:raise ValueError('Local NZ private OPP changed')
    r.update(local_build_report=identity(args.local_build_report),cann_compile=b['cann_compile'],installation=b['installation'],cases=[])
    for entry in r['evidence']['cases']:
        folder=root/'execute'/entry['name'];folder.mkdir(parents=True)
        case=core_case(json.loads(verify(entry['case']).read_text()))
        for name,item in case['files'].items():(folder/name).symlink_to(verify(item))
        row=dict(name=entry['name'],prior=entry,status='NOT_RUN');r['cases'].append(row)
        r.update(phase='npu_'+entry['name'],npu='RUNNING');write(root/'report.json',r)
        command([verify(b['runner']),args.device,folder],folder/'runner.log',common.env(args.cann_root,Path(b['opp'])),r,entry['name'],300)
        row['execution']=identity(folder/'execution.json');check_execution(json.loads(verify(row['execution']).read_text()),args.device)
        text=(folder/'runner.log').read_text()
        if 'HostC=VECIN_NZ_DT_FLOAT16 dequant=TENSOR' not in text or 'single=16/64/128 base=16/64/128' not in text or 'localOutput=2048 reserve=8192' not in text:
            raise ValueError('actual Local NZ tiling resource plan missing')
        row['outputs']=[identity(folder/f'actual-{i}.bin') for i in range(2)]
        row['comparison']=compare(entry,[verify(x).read_bytes() for x in row['outputs']])
        row['status']='PASS' if row['comparison']['status']=='CANDIDATE_BITWISE_MATCH' else 'FAIL_NUMERICAL'
        write(root/'report.json',r)
        if row['status']!='PASS':raise ValueError('Local output mismatch; stop, no fallback or Golden change')
    if installed(Path(b['opp']))!=b['installed']:raise ValueError('Local NZ OPP changed during run')
    r.update(status='STAGE3B2_LOCAL_NZ_MATCH',phase='complete',npu='MATCHED_SIX_K128_CASES_ONLY',
        scale_observation='matched original GM and candidate; unscaled/first-channel-broadcast witnesses distinguished',
        fusion='NOT_IMPLEMENTED')


def main(args):
    protected=[args.cann_root,args.stage3b1_root,args.stage2c_report.parent,args.stage2b_root]
    if args.local_build_report:protected.append(args.local_build_report.parent)
    root=fresh(args.output_dir,protected)
    r=dict(stage='3B-2-single-group',status='NOT_RUN',phase='evidence',settings=SETTINGS,device=args.device,sources=sources(),
        cann_compile='NOT_RUN',installation='NOT_RUN',npu='NOT_RUN',performance='NOT_RUN',fusion='DESIGN_ONLY',
        k2560='NOT_RUN',full_gate='NOT_RUN',prior_kernel_rerun=False)
    try:
        r['evidence']=evidence(args);r['local_review']=review(r['evidence']['static'],root/'source-review')
        write(root/'report.json',r)
        if args.stage=='prepare':
            r['scale_controls']={x['name']:scale_controls(json.loads(verify(x['case']).read_text())) for x in r['evidence']['cases']}
            for x in r['evidence']['cases']:compare(x,[verify(o).read_bytes() for o in x['outputs']])
            r.update(status='STAGE3B2_PREPARED_NPU_NOT_RUN',phase='complete')
        elif args.stage=='build':build(args,root,r)
        else:smoke(args,root,r)
    except Exception as error:
        r.update(status='FAILED',error=f'{type(error).__name__}: {error}')
        for key in ('cann_compile','installation','npu'):
            if r[key]=='RUNNING':r[key]='FAILED_OR_INCOMPLETE'
    finally:
        try:
            for item in r['sources']:verify(item)
            if 'evidence' in r:verify_tree(r['evidence'])
            if 'local_review' in r:verify_tree(r['local_review'])
        except Exception as error:r.update(status='FAILED',integrity_error=str(error))
        write(root/'report.json',r)
    print(r['status']+': '+str(root/'report.json'));return int(r['status']=='FAILED')


def parse():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','build','smoke'))
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--local-build-report',type=Path)
    p.add_argument('--stage3b1-root',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-stage3b-20261027-024844'))
    p.add_argument('--stage2b-root',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-stage2b-20261026-100533'))
    for name in ('build','basic','zero','nonpower'):p.add_argument('--'+name+'-report',type=Path)
    p.add_argument('--stage2c-report',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-stage2c-20261026-104943/report.json'))
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--cann-root',type=Path,default=Path(os.environ.get('CANN_ROOT','/usr/local/Ascend/cann-9.0.0')))
    args=p.parse_args();args.cann_root=args.cann_root.resolve()
    for name in ('build','basic','zero','nonpower'):
        if getattr(args,name+'_report') is None:setattr(args,name+'_report',args.stage2b_root/name/'report.json')
    if args.device<0:p.error('negative device')
    return args

if __name__=='__main__':raise SystemExit(main(parse()))
