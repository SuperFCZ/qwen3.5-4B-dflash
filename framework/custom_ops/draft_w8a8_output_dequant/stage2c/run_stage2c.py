#!/usr/bin/env python3
"""Reuse Stage2B build for one real group0/N64 smoke. No build/install/timing."""
import argparse
import json
import os
from pathlib import Path
from s2c_evidence import HERE,common,stage2b,identity,verify,write,fresh,verify_tree,CODEC,sources,validated_stage2b,frozen_source
from s2c_reference import prepare,verify_case,compare


def markdown(r):
    lines=['# Stage 2C: real layer-0 Gate/Up, K[0:128], N[0:64]','',r['status'],'',
      'Reuses Stage2B Kernel/OPP/Runner. No rebuild, Stage1 execution or performance measurement.',
      'The W8A16 reference is a deterministic CPU K128 partial calculation, not a slice of the full-K Native Golden.','']
    c=r.get('comparison')
    if c:
        lines += ['| Reference vs NPU repeat 0 | Bit mismatches | Max ULP | Max abs | RMSE | Relative L2 | Cosine |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for label,key in [('candidate (strict gate)','npu_vs_candidate'),('ideal W8A8','npu_vs_ideal_w8a8'),('W8A16 K128 CPU','npu_vs_w8a16_k128')]:
            m=c[key][0];lines.append('| '+label+' | '+' | '.join(str(m[k]) for k in ('bit_mismatches','max_ulp','max_abs','rmse','relative_l2_error','cosine'))+' |')
        error=c['scale_error']['effective_vs_combined_fp32']
        lines += ['',f"Scale effective vs combined FP32: max_abs={error['absolute']['max']}, RMSE={error['rmse']}, relative_L2={error['relative_l2_error']}; full distribution and 64 channels in report.json."]
    if r.get('error'):lines+=['','Error: '+r['error']]
    lines+=['','Acceptance requires exact candidate bits and Guard/Readonly/Poison/Repeat. Other reference errors are diagnostic only.',
            'Codec remains EXPERIMENTAL_CODEC_CANDIDATE. Full Gate, K2560/20-group fusion, Draft and performance: NOT_RUN.']
    return '\n'.join(lines)+'\n'


def execute(args,root,r,built,case):
    folder=root/'execute';folder.mkdir()
    for name in common.FILES:(folder/name).symlink_to(verify(case['files'][name]))
    r.update(phase='npu_group0',npu='RUNNING');write(root/'report.json',r)
    stage2b.command([verify(built['runner']),args.device,folder],folder/'runner.log',common.env(args.cann_root,Path(built['opp'])),r,'real_k128',300)
    r['execution']=identity(folder/'execution.json');ex=json.loads(verify(r['execution']).read_text());common.check_execution(ex,args.device)
    log=(folder/'runner.log').read_text()
    if 'STAGE2B_PLAN arch=2002 M=16 K=128 N=64 blockDim=1 HostC=DT_FLOAT16 dequant=TENSOR' not in log or 'single=16/64/128 base=16/64/128' not in log:raise ValueError('accepted Stage2B Host plan not observed')
    r['outputs']=[identity(folder/f'actual-{i}.bin') for i in range(2)]
    r['comparison']=compare(case,[verify(f).read_bytes() for f in r['outputs']])
    write(root/'report.json',r)
    if r['comparison']['status']!='CANDIDATE_BITWISE_MATCH':raise ValueError('real K128 candidate reference mismatch; no Golden adjustment/tolerance/fallback')
    r.update(status='STAGE2C_REAL_K128_MATCH',phase='complete',npu='MATCHED_SINGLE_REAL_K128_ONLY',
             host_tiling_runtime='REUSED_STAGE2B_PLAN_ACCEPTED_FOR_THIS_RUN')


def main(args):
    protected=[args.cann_root,args.stage2b_root,args.a51_evidence.parent,args.prefetch_evidence.parent]
    protected += [p.parent for p in (args.build_report,args.basic_report,args.zero_report,args.nonpower_report)]
    root=fresh(args.output_dir,protected)
    r=dict(stage='2C',status='NOT_RUN',phase='stage2b_evidence',device=args.device,codec_status=CODEC,sources=sources(),
       shape=[16,128,64],original_shape=[16,2560,19456],layer=0,projection='gate_up',group=0,k_range=[0,128],n_range=[0,64],
       cann_compile='NOT_RUN_REUSE_ONLY',installation='NOT_RUN_REUSE_ONLY',host_tiling_runtime='NOT_RUN',npu='NOT_RUN',
       stage1_rerun=False,stage2b_rerun=False,performance='NOT_RUN',full_gate='NOT_RUN',k2560_fusion='NOT_RUN',draft='NOT_RUN')
    try:
        built,r['stage2b_evidence']=validated_stage2b(args)
        r['phase']='frozen_identity';write(root/'report.json',r)
        r['frozen_evidence']=frozen_source(args.a51_evidence,args.prefetch_evidence)
        r['phase']='cpu_references';write(root/'report.json',r)
        case=prepare(root/'data',r['frozen_evidence']);r['case']=identity(root/'data/case.json')
        verify_case(case)
        r['activation_quantization']=case['activation'];r['scale_error']=case['scale_error']
        if args.prepare_only:r.update(status='CPU_REFERENCES_READY_NPU_NOT_RUN',phase='complete')
        else:execute(args,root,r,built,case)
        if stage2b.installed(Path(built['opp']))!=built['installed']:raise ValueError('reused Stage2B OPP changed during run')
        verify_tree(built);verify_case(case)
    except Exception as e:
        r.update(status='FAILED',error=f'{type(e).__name__}: {e}')
        if r['npu']=='RUNNING':r['npu']='FAILED_OR_INCOMPLETE'
    finally:
        try:
            for f in r['sources']:verify(f)
            for key in ('stage2b_evidence','frozen_evidence'):
                if key in r:verify_tree(r[key])
            if 'case' in r:verify_tree(json.loads(verify(r['case']).read_text()))
        except Exception as e:r.update(status='FAILED',integrity_error=str(e))
        write(root/'report.json',r);(root/'analysis.md').write_text(markdown(r))
    print(r['status']+': '+str(root/'report.json'))
    return int(r['status']=='FAILED')


def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage2b-root',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-stage2b-20261026-100533'))
    for key in ('build','basic','zero','nonpower'):p.add_argument('--'+key+'-report',type=Path)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');p.add_argument('--device',type=int,default=0)
    p.add_argument('--cann-root',type=Path,default=Path(os.environ.get('CANN_ROOT','/usr/local/Ascend/cann-9.0.0')))
    p.add_argument('--a51-evidence',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/a51-fallback-20261012-063342/suite.json'))
    p.add_argument('--prefetch-evidence',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/gate-raw-prefetch-reverse-20261020-025853/suite.json'))
    args=p.parse_args();args.cann_root=args.cann_root.resolve()
    for key in ('build','basic','zero','nonpower'):
        if getattr(args,key+'_report') is None:setattr(args,key+'_report',args.stage2b_root/key/'report.json')
    if args.device<0:p.error('negative device')
    return args

if __name__=='__main__':raise SystemExit(main(parse()))
