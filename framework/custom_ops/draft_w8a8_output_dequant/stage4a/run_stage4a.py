#!/usr/bin/env python3
"""Eight real N64 slots: serial, multiple slots/core, and capped parallel correctness."""
import argparse,json,os
from pathlib import Path
from tile_common import *
from build_tiles import build,installed
from run_stage2b import command


def smoke(args,root,r,case):
    if args.multi_build_report is None:raise ValueError('--multi-build-report required')
    b=json.loads(args.multi_build_report.read_text());verify_tree(b)
    if b.get('status')!='STAGE4A_BUILT_NPU_NOT_RUN' or any(b.get(k)!=r[k] for k in ('settings','sources','evidence','device')):raise ValueError('multi-tile build/evidence mismatch')
    if installed(Path(b['opp']))!=b['installed']:raise ValueError('private OPP changed')
    d=verify_case(case);r.update(multi_build_report=identity(args.multi_build_report),cann_compile=b['cann_compile'],installation=b['installation'],runs=[])
    modes=[21,0]+[g+1 for g in args.diagnostic_groups]
    limits=[1];seen_cores=None
    index=0
    while index<len(limits):
        limit=limits[index];index+=1
        for mode in modes:
            folder=root/'execute'/f'limit-{limit}'/f'mode-{mode}';folder.mkdir(parents=True)
            for name in ('a_i8.bin','b_i8.bin','quant_u64_doc_candidate.bin'):(folder/name).symlink_to(verify(case['files'][name]))
            entry=dict(core_limit=limit,mode=mode,status='RUNNING');r['runs'].append(entry)
            r.update(phase=f'limit{limit}_mode{mode}',npu='RUNNING');write(root/'report.json',r)
            command([verify(b['runner']),args.device,folder,mode,limit],folder/'runner.log',common.env(args.cann_root,Path(b['opp'])),r,f'l{limit}_m{mode}',600)
            entry['execution']=identity(folder/'execution.json');ex=json.loads(verify(entry['execution']).read_text());check_execution(ex,args.device,mode,limit)
            plan=plan_from_log((folder/'runner.log').read_text(),mode,limit);entry['plan']=plan
            if ex['workspace_bytes']<plan['total']:raise ValueError('ACLNN workspace smaller than declared system plus private arenas')
            entry['resource_admission']=identity(folder/'resource-admission.json')
            admission=json.loads(verify(entry['resource_admission']).read_text())
            if admission!=dict(status='ADMITTED_BEFORE_EXECUTION',workspace_bytes=ex['workspace_bytes'],declared_total=plan['total'],blocks=plan['blocks'],cores=plan['cores']):raise ValueError('pre-execution resource admission differs from actual plan')
            if seen_cores is None:seen_cores=plan['cores']
            elif seen_cores!=plan['cores']:raise ValueError('platform core count changed during experiment')
            entry['outputs']=[identity(folder/f'actual-{i}.bin') for i in range(2)]
            entry['comparison']=compare(case,d,mode,plan['blocks'],[verify(x).read_bytes() for x in entry['outputs']])
            entry['status']=entry['comparison']['status'];write(root/'report.json',r)
            if entry['status']!='BITWISE_MATCH':raise ValueError('slot/layout/numerical mismatch; stop without fallback')
        if limit==1:
            if seen_cores<2:raise ValueError('BLOCKED_NO_MULTICORE: serial result retained; cannot grant Stage4A PASS')
            limits.append(2) # ensures multiple slots per core even on >=8-core devices
            if min(8,seen_cores)!=2:limits.append(0)
    # Every numerical mode must also agree between serial and all parallel schedules.
    for mode in [0]+[g+1 for g in args.diagnostic_groups]:
        outs=[verify(e['outputs'][0]).read_bytes() for e in r['runs'] if e['mode']==mode]
        if len(outs)!=len(limits) or any(x!=outs[0] for x in outs):raise ValueError('serial/parallel numerical outputs differ')
    if installed(Path(b['opp']))!=b['installed']:raise ValueError('OPP changed during execution')
    verify_case(case)
    r.update(status='STAGE4A_MULTI_TILE_MATCH',phase='complete',npu='EIGHT_REAL_TILES_SERIAL_AND_MULTICORE_MATCHED',
        available_aicores=seen_cores,core_limits_tested=limits,output_modes_tested=modes,
        summary=dict(runs=len(r['runs']),matched_runs=sum(e['status']=='BITWISE_MATCH' for e in r['runs']),repetitions=2,tiles=8),
        stage4b='NOT_RUN',performance='NOT_RUN')


def main(args):
    protected=[args.stage3b4_root,args.stage3b3_root,args.stage3b2_root,args.stage3b1_root,args.stage3a_root,args.stage2b_root,args.stage2c_report.parent,args.cann_root]
    if args.multi_build_report:protected.append(args.multi_build_report.parent)
    root=fresh(args.output_dir,protected)
    r=dict(stage='4A',status='NOT_RUN',phase='evidence',device=args.device,settings=SETTINGS,sources=sources(),
        cann_compile='NOT_RUN',installation='NOT_RUN',npu='NOT_RUN',performance='NOT_RUN',stage4b='NOT_RUN',draft='NOT_RUN',
        diagnostic_groups=args.diagnostic_groups,prior_kernel_rerun=False)
    try:
        r['evidence']=evidence(args);case=prepare(root/'data',r['evidence']);r['case']=identity(root/'data/case.json')
        verify_case(case);write(root/'report.json',r)
        if args.stage=='prepare':r.update(status='STAGE4A_PREPARED_NPU_NOT_RUN',phase='complete')
        elif args.stage=='build':build(args,root,r)
        else:smoke(args,root,r,case)
    except Exception as error:
        r.update(status='FAILED',error=f'{type(error).__name__}: {error}')
        for key in ('cann_compile','installation','npu'):
            if r[key]=='RUNNING':r[key]='FAILED_OR_INCOMPLETE'
        for entry in r.get('runs',[]):
            if entry['status']=='RUNNING':entry['status']='FAILED_OR_INCOMPLETE'
    finally:
        try:
            for item in r['sources']:verify(item)
            if 'evidence' in r:verify_tree(r['evidence'])
            if 'case' in r:verify_case(json.loads(verify(r['case']).read_text()))
        except Exception as error:r.update(status='FAILED',integrity_error=str(error))
        if r['status']=='FAILED':r['failure_artifacts']=[identity(p) for p in sorted((root/'execute').rglob('*')) if p.is_file() and p.name in ('runner.log','execution.json','resource-admission.json','actual-0.bin','actual-1.bin')]
        write(root/'report.json',r)
    print(r['status']+': '+str(root/'report.json'));return int(r['status']=='FAILED')


def parse():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','build','smoke'))
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--multi-build-report',type=Path)
    p.add_argument('--diagnostic-groups',type=int,nargs='*',default=[])
    for name,default in [('stage3b4','w8a8-twenty-group-20261027-055348'),('stage3b3','w8a8-two-group-20261027-042044'),
                         ('stage3b2','w8a8-local-nz-20261027-031856'),('stage3b1','w8a8-stage3b-20261027-024844'),
                         ('stage3a','w8a8-stage3a-20261027-011523'),('stage2b','w8a8-stage2b-20261026-100533')]:
        p.add_argument('--'+name+'-root',type=Path,default=Path('/home/w00949577/qwen35-dflash-run')/default)
    for name in ('build','basic','zero','nonpower'):p.add_argument('--'+name+'-report',type=Path)
    p.add_argument('--stage2c-report',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-stage2c-20261026-104943/report.json'))
    p.add_argument('--device',type=int,default=0);p.add_argument('--cann-root',type=Path,default=Path(os.environ.get('CANN_ROOT','/usr/local/Ascend/cann-9.0.0')))
    args=p.parse_args();args.cann_root=args.cann_root.resolve()
    for name in ('build','basic','zero','nonpower'):
        if getattr(args,name+'_report') is None:setattr(args,name+'_report',args.stage2b_root/name/'report.json')
    if args.device<0:p.error('negative device')
    if len(set(args.diagnostic_groups))!=len(args.diagnostic_groups) or any(g not in (0,1,9,10,18,19) for g in args.diagnostic_groups):p.error('diagnostic groups must be unique members of 0,1,9,10,18,19')
    return args

if __name__=='__main__':raise SystemExit(main(parse()))
