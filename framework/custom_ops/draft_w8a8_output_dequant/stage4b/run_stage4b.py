#!/usr/bin/env python3
"""Full 304-tile correctness: automatic core limit then an independent small-core control."""
import argparse,json,os
from pathlib import Path
from full_common import *
from full_reference import *
from build_full import build,installed
from run_stage2b import command

def admission_expected(ex,plan,host_only):
    return dict(status='ADMITTED_BEFORE_EXECUTION',workspace_bytes=ex['workspace_bytes'],declared_total=plan['total'],
                blocks=plan['blocks'],cores=plan['cores'],core_policy=plan['core_policy'],host_proof_checked=not host_only)


def validate_host_probe(path,build_id,r):
    if path is None:raise ValueError('--host-probe-report required before normal Smoke')
    hp=json.loads(path.read_text());verify_tree(hp)
    if hp.get('status')!='STAGE4B_HOST_TILING_ONLY' or hp.get('phase')!='complete' or hp.get('host_only') is not True or hp.get('npu')!='NOT_RUN' or hp.get('multi_build_report')!=build_id or any(hp.get(k)!=r[k] for k in ('settings','sources','evidence','device','prepare_report','case')):raise ValueError('Host-only proof must bind this Build/source/device/evidence')
    if len(hp.get('runs',[]))!=1:raise ValueError('one Host-only query required')
    row=hp['runs'][0];ex=json.loads(verify(row['execution']).read_text())
    if row.get('status')!='HOST_TILING_ONLY' or row.get('mode')!=21 or row.get('core_limit')!=1 or ex.get('status')!='HOST_TILING_ONLY' or ex.get('host_only') is not True or ex.get('npu')!='NOT_RUN' or ex.get('device')!=r['device'] or ex.get('output_mode')!=21 or ex.get('core_limit')!=1:raise ValueError('Host-only query scope differs')
    plan=plan_from_log(verify(hp['commands']['l1_m21']['log_identity']).read_text(),21,1)
    if row.get('plan')!=plan or hp.get('core_policy')!=plan['core_policy'] or json.loads(verify(row['resource_admission']).read_text())!=admission_expected(ex,plan,True):raise ValueError('Host-only policy/resource proof differs')
    verify_case(json.loads(verify(hp['case']).read_text()))
    return plan['core_policy']


def smoke(args,root,r,case):
    host_only=getattr(args,'host_only',False)
    if args.multi_build_report is None:raise ValueError('--multi-build-report required')
    b=json.loads(args.multi_build_report.read_text());verify_tree(b)
    if b.get('status')!='STAGE4B_BUILT_NPU_NOT_RUN' or any(b.get(k)!=r[k] for k in ('settings','sources','evidence','device','prepare_report','case')):raise ValueError('full-Gate build/preparation/evidence mismatch')
    if installed(Path(b['opp']))!=b['installed']:raise ValueError('private OPP changed')
    d=verify_case(case);r.update(multi_build_report=identity(args.multi_build_report),cann_compile=b['cann_compile'],installation=b['installation'],runs=[])
    expected_policy=None
    if not host_only:
        expected_policy=validate_host_probe(getattr(args,'host_probe_report',None),r['multi_build_report'],r)
        r['host_probe_report']=identity(args.host_probe_report);r['expected_core_policy']=expected_policy
    modes=[21] if host_only else [21,0]+[g+1 for g in args.diagnostic_groups]
    if not host_only and expected_policy['selected']<2:raise ValueError('BLOCKED_NO_MULTICORE: full-Gate multi-core acceptance unavailable')
    limits=[1] if host_only else [0,2 if min(304,expected_policy['selected'])>2 else 1]
    seen_cores=None
    index=0
    while index<len(limits):
        limit=limits[index];index+=1
        for mode in modes:
            folder=root/'execute'/f'limit-{limit}'/f'mode-{mode}';folder.mkdir(parents=True)
            for name in ('a_i8.bin','b_i8.bin','quant_u64_doc_candidate.bin'):(folder/name).symlink_to(verify(case['files'][name]))
            entry=dict(core_limit=limit,mode=mode,status='RUNNING');r['runs'].append(entry)
            r.update(phase=f'limit{limit}_mode{mode}',npu='NOT_RUN' if host_only else 'RUNNING');write(root/'report.json',r)
            argv=[verify(b['runner']),args.device,folder,mode,limit]+(['--host-only'] if host_only else [])
            environment=common.env(args.cann_root,Path(b['opp']));environment.pop('DFLASH_STAGE4B_EXPECTED_CORE_POLICY',None)
            if expected_policy:environment['DFLASH_STAGE4B_EXPECTED_CORE_POLICY']=','.join(str(expected_policy[k]) for k in ('aic','aiv','all','selected','arch'))
            command(argv,folder/'runner.log',environment,r,f'l{limit}_m{mode}',600)
            entry['execution']=identity(folder/'execution.json');ex=json.loads(verify(entry['execution']).read_text())
            if host_only:
                if ex.get('status')!='HOST_TILING_ONLY' or ex.get('host_only') is not True or ex.get('npu')!='NOT_RUN' or ex.get('device')!=args.device or ex.get('output_mode')!=21 or ex.get('core_limit')!=1:raise ValueError('invalid Host-only diagnostic report')
            else:check_execution(ex,args.device,mode,limit)
            plan=plan_from_log((folder/'runner.log').read_text(),mode,limit);entry['plan']=plan
            if ex['workspace_bytes']<plan['total']:raise ValueError('ACLNN workspace smaller than declared system plus private arenas')
            entry['resource_admission']=identity(folder/'resource-admission.json')
            admission=json.loads(verify(entry['resource_admission']).read_text())
            if admission!=admission_expected(ex,plan,host_only):raise ValueError('pre-execution resource admission differs from actual plan')
            if host_only:
                entry['status']='HOST_TILING_ONLY'
                r.update(status='STAGE4B_HOST_TILING_ONLY',phase='complete',npu='NOT_RUN',core_policy=plan['core_policy'])
                return
            if plan['core_policy']!=expected_policy:raise ValueError('Core policy differs from Host-only admission')
            if seen_cores is None:seen_cores=plan['cores']
            elif seen_cores!=plan['cores']:raise ValueError('platform core count changed during experiment')
            entry['outputs']=[identity(folder/f'actual-{i}.bin') for i in range(2)]
            entry['comparison']=compare(case,d,mode,plan['blocks'],[verify(x).read_bytes() for x in entry['outputs']])
            entry['status']=entry['comparison']['status'];write(root/'report.json',r)
            if entry['status']!='BITWISE_MATCH':raise ValueError('slot/layout/numerical mismatch; stop without fallback')
    # Every numerical mode must also agree between serial and all parallel schedules.
    for mode in [0]+[g+1 for g in args.diagnostic_groups]:
        outs=[verify(e['outputs'][0]).read_bytes() for e in r['runs'] if e['mode']==mode]
        if len(outs)!=len(limits) or any(x!=outs[0] for x in outs):raise ValueError('serial/parallel numerical outputs differ')
    if installed(Path(b['opp']))!=b['installed']:raise ValueError('OPP changed during execution')
    verify_case(case,numerical=False)
    r.update(status='STAGE4B_FULL_GATE_MATCH',phase='complete',npu='ALL_304_TILES_AUTO_AND_CONTROL_MATCHED',
        scheduling_core_limit=seen_cores,core_policy=expected_policy,core_limits_tested=limits,output_modes_tested=modes,
        summary=dict(runs=len(r['runs']),matched_runs=sum(e['status']=='BITWISE_MATCH' for e in r['runs']),repetitions=2,tiles=304),
        stage4c='NOT_RUN',performance='NOT_RUN')


def prepared(path,r):
    if path is None:raise ValueError('completed --prepare-report required')
    p=json.loads(path.read_text());verify_tree(p)
    if p.get('status')!='STAGE4B_PREPARED_NPU_NOT_RUN' or p.get('phase')!='complete' or p.get('npu')!='NOT_RUN' or any(p.get(k)!=r[k] for k in ('settings','sources','evidence','device')):raise ValueError('CPU preparation/source/evidence mismatch')
    case=json.loads(verify(p['case']).read_text());verify_case(case)
    r.update(prepare_report=identity(path),case=p['case'])
    return case


def main(args):
    if args.stage=='smoke' and args.multi_build_report and args.prepare_report is None:
        built=json.loads(args.multi_build_report.read_text());args.prepare_report=verify(built['prepare_report'])
    protected=[args.stage4a_root,args.stage3b4_root,args.stage3b3_root,args.stage3b2_root,args.stage3b1_root,args.stage3a_root,args.stage2b_root,args.stage2c_report.parent,args.cann_root]
    protected += [p.parent for p in (args.prepare_report,args.multi_build_report,args.host_probe_report) if p]
    root=fresh(args.output_dir,protected)
    r=dict(stage='4B',status='NOT_RUN',phase='evidence',device=args.device,settings=SETTINGS,sources=sources(),
        cann_compile='NOT_RUN',installation='NOT_RUN',npu='NOT_RUN',performance='NOT_RUN',stage4c='NOT_RUN',draft='NOT_RUN',
        diagnostic_groups=args.diagnostic_groups,host_only=args.host_only,prior_kernel_rerun=False)
    try:
        r['evidence']=evidence(args)
        if args.stage=='prepare':
            r['phase']='cpu_304_tile_preparation';write(root/'report.json',r)
            case=prepare(root/'data',r['evidence']);r['case']=identity(root/'data/case.json')
            verify_case(case,numerical=False)
            r.update(status='STAGE4B_PREPARED_NPU_NOT_RUN',phase='complete',validated_tiles=304,validated_nz_groups=6080,
                cross_scheme_accuracy=case['w8a16_reference'],native_comparison=case['native_comparison'])
        else:
            r['phase']='cpu_preparation_admission';case=prepared(args.prepare_report,r);write(root/'report.json',r)
            if args.stage=='build':build(args,root,r)
            else:smoke(args,root,r,case)
    except Exception as error:
        r.update(status='FAILED',error=f'{type(error).__name__}: {error}')
        if isinstance(error,ReferenceFailure):r['error_detail']=error.detail
        for key in ('cann_compile','installation','npu'):
            if r[key]=='RUNNING':r[key]='FAILED_OR_INCOMPLETE'
        for entry in r.get('runs',[]):
            if entry['status']=='RUNNING':entry['status']='FAILED_OR_INCOMPLETE'
    finally:
        try:
            for item in r['sources']:verify(item)
            if 'evidence' in r:verify_tree(r['evidence'])
            if 'case' in r:verify_case(json.loads(verify(r['case']).read_text()),numerical=False)
        except Exception as error:r.update(status='FAILED',integrity_error=str(error))
        if r['status']=='FAILED':r['failure_artifacts']=[identity(p) for p in sorted((root/'execute').rglob('*')) if p.is_file() and p.name in ('runner.log','execution.json','resource-admission.json','actual-0.bin','actual-1.bin')]
        if 'case' in r and 'integrity_error' not in r:
            c=json.loads(verify(r['case']).read_text())
            accuracy=dict(status='DIAGNOSTIC_ONLY',candidate_status=r['status'],reference=c['files']['w8a16_cpu_fp16.bin'],
                arithmetic=c['w8a16_reference']['arithmetic'],cpu_candidate_vs_w8a16=c['w8a16_reference']['accuracy'],
                native=c['native_comparison'],npu_runs=[dict(core_limit=e['core_limit'],status=e['status'],outputs=e['outputs'],
                    metrics=e['comparison']['cross_scheme_accuracy']['npu_vs_w8a16_cpu']) for e in r.get('runs',[])
                    if e['mode']==0 and 'comparison' in e])
            write(root/'accuracy-report.json',accuracy);r['accuracy_report']=identity(root/'accuracy-report.json')
        write(root/'report.json',r)
    print(r['status']+': '+str(root/'report.json'));return int(r['status']=='FAILED')


def parse():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','build','smoke'))
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--prepare-report',type=Path);p.add_argument('--multi-build-report',type=Path)
    p.add_argument('--diagnostic-groups',type=int,nargs='*',default=[])
    p.add_argument('--host-only',action='store_true');p.add_argument('--host-probe-report',type=Path)
    for name,default in [('stage4a','w8a8-core-policy-20261027-083223'),('stage3b4','w8a8-twenty-group-20261027-055348'),
                         ('stage3b3','w8a8-two-group-20261027-042044'),('stage3b2','w8a8-local-nz-20261027-031856'),
                         ('stage3b1','w8a8-stage3b-20261027-024844'),('stage3a','w8a8-stage3a-20261027-011523'),('stage2b','w8a8-stage2b-20261026-100533')]:
        p.add_argument('--'+name+'-root',type=Path,default=Path('/home/w00949577/qwen35-dflash-run')/default)
    for name in ('build','basic','zero','nonpower'):p.add_argument('--'+name+'-report',type=Path)
    p.add_argument('--stage2c-report',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-stage2c-20261026-104943/report.json'))
    p.add_argument('--device',type=int,default=0);p.add_argument('--cann-root',type=Path,default=Path(os.environ.get('CANN_ROOT','/usr/local/Ascend/cann-9.0.0')))
    args=p.parse_args();args.cann_root=args.cann_root.resolve()
    for name in ('build','basic','zero','nonpower'):
        if getattr(args,name+'_report') is None:setattr(args,name+'_report',args.stage2b_root/name/'report.json')
    if args.device<0:p.error('negative device')
    if args.stage=='build' and args.prepare_report is None:p.error('build requires successful --prepare-report')
    if args.stage=='prepare' and args.prepare_report:p.error('prepare creates its own references')
    if args.host_only and (args.stage!='smoke' or args.diagnostic_groups):p.error('--host-only requires smoke without partial diagnostics')
    if args.host_probe_report and (args.stage!='smoke' or args.host_only):p.error('--host-probe-report is for normal Smoke only')
    if len(set(args.diagnostic_groups))!=len(args.diagnostic_groups) or any(g not in (0,19) for g in args.diagnostic_groups):p.error('optional diagnostic groups are 0 and 19 only')
    return args

if __name__=='__main__':raise SystemExit(main(parse()))
