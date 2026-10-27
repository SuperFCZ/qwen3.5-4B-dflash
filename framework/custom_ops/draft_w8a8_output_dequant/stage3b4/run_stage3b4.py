#!/usr/bin/env python3
"""Serial twenty-group fused correctness probe; all partial modes are mandatory."""
import argparse,json,os
from pathlib import Path
from twenty_common import *
from build_twenty import build,installed
from run_stage2b import command


def smoke(args,root,r,cases):
    if args.fused_build_report is None:raise ValueError('--fused-build-report required')
    b=json.loads(args.fused_build_report.read_text());verify_tree(b)
    if b.get('status')!='STAGE3B4_BUILT_NPU_NOT_RUN' or any(b.get(k)!=r[k] for k in ('settings','sources','evidence','device')):
        raise ValueError('fused build/evidence/config binding differs')
    if installed(Path(b['opp']))!=b['installed']:raise ValueError('fused OPP changed')
    r.update(fused_build_report=identity(args.fused_build_report),cann_compile=b['cann_compile'],installation=b['installation'],
        cases=[dict(name=json.loads(verify(item).read_text())['name'],case=item,status='NOT_RUN',
                    modes=[dict(mode=m,status='NOT_RUN') for m in MODES]) for item in cases])
    if tuple(c['name'] for c in r['cases'])!=CASE_NAMES:raise ValueError('all four cases required; no partial-suite PASS')
    for entry in r['cases']:
        case=json.loads(verify(entry['case']).read_text());verify_case(case);entry['status']='RUNNING'
        saved={}
        for row in entry['modes']:
            mode=row['mode']
            folder=root/'execute'/case['name']/f'mode-{mode}';folder.mkdir(parents=True)
            for name in ('a_i8.bin','b_i8.bin','quant_u64_doc_candidate.bin'):(folder/name).symlink_to(verify(case['files'][name]))
            row['status']='RUNNING'
            r.update(phase=f"{case['name']}_mode{mode}",npu='RUNNING');write(root/'report.json',r)
            command([verify(b['runner']),args.device,folder,mode],folder/'runner.log',common.env(args.cann_root,Path(b['opp'])),r,case['name']+f'_mode{mode}',300)
            row['execution']=identity(folder/'execution.json');check_execution(json.loads(verify(row['execution']).read_text()),args.device,mode)
            text=(folder/'runner.log').read_text()
            if f'STAGE3B4_OUTPUT_MODE={mode}' not in text or 'groups=20 M=16 K=2560 groupK=128 N=64' not in text or 'scratch=239104' not in text or 'userUB=12288 ubGuards=2560 reserve=8192' not in text or 'single=16/64/128 base=16/64/128' not in text:
                raise ValueError('actual twenty-group resource plan missing')
            row['outputs']=[identity(folder/f'actual-{i}.bin') for i in range(2)]
            saved[mode]=[verify(x).read_bytes() for x in row['outputs']]
            row['comparison']=compare(case,mode,saved[mode]);row['status']=row['comparison']['status'];write(root/'report.json',r)
            if row['status']!='BITWISE_MATCH':
                entry['status']='FAIL_NUMERICAL'
                raise ValueError('partial/final mismatch; no next mode/case, no fallback')
        # Also derive the final reference from THIS kernel's separately observed partials.
        consistency=[]
        for repeat in range(2):
            total=np.zeros((16,64),'<f4')
            for mode in range(1,21):total=np.add(total,np.frombuffer(saved[mode][repeat],'<f2').reshape(16,64).astype('<f4'),dtype=np.float32)
            result=metrics(total.astype('<f2'),np.frombuffer(saved[0][repeat],'<f2').reshape(16,64));consistency.append(result)
            if not result['bitwise_equal']:raise ValueError('final differs from independently observed ordered partials')
        entry.update(status='PASS',partial_final_consistency=consistency,checked_partials=20,
                     ub_guards='ENFORCED_BEFORE_ALL_SUCCESSFUL_OUTPUTS')
    swapped=[next(c for c in r['cases'] if c['name']==name) for name in ('synthetic_distinct','synthetic_rotated_scale')]
    targets=[next(m for m in c['modes'] if m['mode']==0)['outputs'][0] for c in swapped]
    r['scale_swap_observation']=metrics(np.fromfile(verify(targets[0]),'<f2'),np.fromfile(verify(targets[1]),'<f2'))
    if r['scale_swap_observation']['bit_mismatches']==0:raise ValueError('scale swap did not change final output')
    if installed(Path(b['opp']))!=b['installed']:raise ValueError('fused OPP changed during run')
    r.update(status='STAGE3B4_20_GROUP_FUSED_MATCH',phase='complete',npu='MATCHED_REAL_AND_SYNTHETIC_TWENTY_GROUPS_ONLY',
             lifecycle='TWENTY_GROUP_REUSE_MATCHED_SELECTED_CASES',full_n19456='NOT_RUN')


def main(args):
    protected=[args.stage3b3_root,args.cann_root,args.stage3b2_root,args.stage3b1_root,args.stage3a_root,args.stage2c_report.parent,args.stage2b_root]
    if args.fused_build_report:protected.append(args.fused_build_report.parent)
    root=fresh(args.output_dir,protected)
    r=dict(stage='3B-4',status='NOT_RUN',phase='evidence',settings=SETTINGS,device=args.device,sources=sources(),
        cann_compile='NOT_RUN',installation='NOT_RUN',npu='NOT_RUN',performance='NOT_RUN',full_n19456='NOT_RUN',full_gate='NOT_RUN',prior_kernel_rerun=False)
    try:
        r['evidence']=evidence(args);r['lifecycle']='same validated Stage3B3 sequence; twenty-group reuse NOT_RUN until smoke'
        r['cpu_cases']=prepare(root/'data',r['evidence']);write(root/'report.json',r)
        if args.stage=='prepare':r.update(status='STAGE3B4_PREPARED_NPU_NOT_RUN',phase='complete')
        elif args.stage=='build':build(args,root,r)
        else:smoke(args,root,r,r['cpu_cases'])
    except Exception as error:
        r.update(status='FAILED',error=f'{type(error).__name__}: {error}')
        for key in ('cann_compile','installation','npu'):
            if r[key]=='RUNNING':r[key]='FAILED_OR_INCOMPLETE'
        for case in r.get('cases',[]):
            if case['status']=='RUNNING':case['status']='FAILED_OR_INCOMPLETE'
            for mode in case['modes']:
                if mode['status']=='RUNNING':mode['status']='FAILED_OR_INCOMPLETE'
    finally:
        try:
            for item in r['sources']:verify(item)
            if 'evidence' in r:verify_tree(r['evidence'])
            for item in r.get('cpu_cases',[]):verify_case(json.loads(verify(item).read_text()))
        except Exception as error:r.update(status='FAILED',integrity_error=str(error))
        modes=[m for c in r.get('cases',[]) for m in c['modes']]
        r['summary']=dict(expected_cases=4,expected_modes=84,repeats_per_mode=2,
            passed_cases=sum(c['status']=='PASS' for c in r.get('cases',[])),
            matched_modes=sum(m['status']=='BITWISE_MATCH' for m in modes),
            failed_modes=sum(m['status'] in ('FAIL_NUMERICAL','FAILED_OR_INCOMPLETE') for m in modes),
            not_run_modes=84-sum(m['status']!='NOT_RUN' for m in modes))
        if r['status']=='FAILED':
            r['failure_artifacts']=[identity(p) for p in sorted((root/'execute').rglob('*'))
                if p.is_file() and p.name in ('actual-0.bin','actual-1.bin','execution.json','runner.log')]
        write(root/'report.json',r)
    print(r['status']+': '+str(root/'report.json'));return int(r['status']=='FAILED')


def parse():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','build','smoke'))
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--fused-build-report',type=Path)
    p.add_argument('--stage3b3-root',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-two-group-20261027-042044'))
    p.add_argument('--stage3b2-root',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-local-nz-20261027-031856'))
    p.add_argument('--stage3b1-root',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-stage3b-20261027-024844'))
    p.add_argument('--stage3a-root',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-stage3a-20261027-011523'))
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
