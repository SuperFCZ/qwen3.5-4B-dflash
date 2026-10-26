#!/usr/bin/env python3
"""Twenty independent K128 calls and CPU aggregation, not a fused K2560 kernel."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from s3a_reference import HERE,e2c,common,identity,verify,verify_tree,write,prepare,verify_prepared,effect_report,accumulate_saved,fp32_metrics,metrics
from s3a_evidence import baseline,sources
from native_semantics import audit
from w8_common import fresh


def native_compare(native,arrays):
    if native['native_semantics_status']!='VERIFIED':return None
    for item in native.get('evidence',[])+native.get('generator_sources',[]):verify(item)
    path=verify(native['golden'])
    if native['golden']['bytes']!=622592:raise ValueError('Native full-K output size changed')
    golden=np.fromfile(path,'<f2').reshape(16,19456)[:,:64].copy()
    if not np.isfinite(golden).all():raise ValueError('nonfinite frozen Native Golden')
    return dict(scope='full K2560 W8A16 Native first N64 columns; cross-scheme diagnostic, never per-group Golden',
                identity=native['golden'],comparisons={name:metrics(golden,value) for name,value in arrays.items()})


def execute_groups(args,root,r,built,prepared):
    r['groups']=[dict(group_index=g,name=f'group-{g:02d}',status='NOT_RUN',case=prepared['cases'][g]) for g in range(20)]
    for entry in r['groups']:
        g=entry['group_index'];folder=root/'execute'/entry['name'];folder.mkdir(parents=True)
        case=json.loads(verify(entry['case']).read_text())
        for name in common.FILES:(folder/name).symlink_to(verify(case['files'][name]))
        entry['status']='RUNNING';r.update(phase=entry['name'],npu='RUNNING');write(root/'report.json',r)
        try:
            e2c.stage2b.command([verify(built['runner']),args.device,folder],folder/'runner.log',common.env(args.cann_root,Path(built['opp'])),r,entry['name'],300)
            entry['execution']=identity(folder/'execution.json');common.check_execution(json.loads(verify(entry['execution']).read_text()),args.device)
            log=(folder/'runner.log').read_text()
            if 'STAGE2B_PLAN arch=2002 M=16 K=128 N=64 blockDim=1 HostC=DT_FLOAT16 dequant=TENSOR' not in log or 'single=16/64/128 base=16/64/128' not in log:raise ValueError('Stage2B plan missing')
            entry['outputs']=[identity(folder/f'actual-{i}.bin') for i in range(2)]
            entry['comparison']=e2c.reference.compare(case,[verify(v).read_bytes() for v in entry['outputs']])
            entry['status']='PASS' if entry['comparison']['status']=='CANDIDATE_BITWISE_MATCH' else 'FAIL_NUMERICAL'
            # Numerical failures are recorded and remaining independent groups still run.
        except Exception as error:
            entry.update(status='FAIL_EXECUTION',error=f'{type(error).__name__}: {error}')
            for pending in r['groups'][g+1:]:pending['status']='NOT_RUN_AFTER_EXECUTION_FAILURE'
            write(root/'report.json',r);raise
        write(root/'report.json',r)
    r['npu']='ALL_20_GROUPS_EXECUTED'


def aggregate(root,r,refs):
    sums,halves=accumulate_saved(r['groups']);outputs=[]
    for i,(s,h) in enumerate(zip(sums,halves)):
        fp=root/f'npu-partials-host-sum-{i}.fp32.bin';hp=root/f'npu-partials-host-sum-{i}.fp16.bin'
        s.tofile(fp);h.tofile(hp);outputs.append(dict(fp32=identity(fp),fp16=identity(hp)))
    comparisons=[dict(fp32_vs_partial_reference=fp32_metrics(refs['partial_f32'],s),
                      fp16_vs_partial_reference=metrics(refs['partial_f16'],h)) for s,h in zip(sums,halves)]
    repeated=sums[0].tobytes()==sums[1].tobytes() and halves[0].tobytes()==halves[1].tobytes()
    matched=repeated and all(s.tobytes()==refs['partial_f32'].tobytes() and h.tobytes()==refs['partial_f16'].tobytes() for s,h in zip(sums,halves))
    r['aggregation']=dict(status='MATCH' if matched else 'FAIL_NUMERICAL',location='CPU from saved NPU FP16 outputs',
        order=list(range(20)),arithmetic='FP16 partial -> FP32 sequential additions -> one final FP16',repeat_equal=repeated,
        outputs=outputs,comparisons=comparisons,
        npu_aggregate_vs_ideal=metrics(refs['ideal_f16'],halves[0]),
        npu_aggregate_vs_effective=metrics(refs['effective_f16'],halves[0]),
        npu_aggregate_vs_w8a16_cpu=metrics(refs['w8a16_f16'],halves[0]))
    return halves


def report_text(r):
    lines=['# Stage 3A: K2560/N64 correctness via 20 K128 calls','',r['status'],'',
      'No fused K2560 Kernel. Final aggregation runs on CPU from saved NPU FP16 partials. No timing or performance conclusion.',
      '',f"Native semantics: {r.get('native_semantics_status','UNRESOLVED')}; comparison: {r.get('native_comparison_status','SKIPPED')}.",'',
      '| Group | K range | Result | Candidate bit mismatches (two repeats) |','| --- | --- | --- | --- |']
    for row in r.get('groups',[]):
        bits=[v['bit_mismatches'] for v in row.get('comparison',{}).get('candidate_reference',[])]
        g=row['group_index'];lines.append(f'| {g} | {128*g}:{128*(g+1)} | {row["status"]} | {bits} |')
    effects=r.get('cpu_error_decomposition',{})
    for name in ('scale_encoding','group_fp16_rounding','ideal_w8a8_vs_w8a16','fp16_partial_scheme_vs_w8a16'):
        if name in effects:lines+=['',name+': '+json.dumps(effects[name],ensure_ascii=False)]
    if r.get('error'):lines+=['','Error: '+r['error']]
    if 'native' in r:lines+=['','Native audit: '+r['native']['reason'],r['native'].get('error','')]
    lines+=['','All error components compare independent references; relative norms are not additive.',
             'Native differences are diagnostic only and do not change group bitwise acceptance. Stage 1/2 suites and Native OM are never rerun.']
    return '\n'.join(lines)+'\n'


def main(args):
    protected=[args.stage2c_report.parent,args.stage2b_root,args.cann_root,args.a51_evidence.parent,args.prefetch_evidence.parent]
    protected += [p.parent for p in (args.build_report,args.basic_report,args.zero_report,args.nonpower_report)]
    if args.cpu_report:protected.append(args.cpu_report.parent)
    root=fresh(args.output_dir,protected)
    r=dict(status='NOT_RUN',stage='3A',phase='evidence',shape=[16,2560,64],group_count=20,device=args.device,sources=sources(),
      codec_status=common.CODEC,npu='NOT_RUN',performance='NOT_RUN',kernel_build='NOT_RUN_REUSED_STAGE2B',installation='NOT_RUN_REUSED_STAGE2B',
      fused_kernel='NOT_IMPLEMENTED',full_gate='NOT_RUN',draft='NOT_RUN',native_semantics_status='UNRESOLVED',native_comparison_status='SKIPPED',
      groups=[dict(group_index=g,status='NOT_RUN') for g in range(20)])
    try:
        built,r['baseline'],frozen=baseline(args)
        r['phase']='cpu_reference';write(root/'report.json',r)
        if args.cpu_report:
            cpu=json.loads(args.cpu_report.read_text())
            if cpu.get('status')!='CPU_20_GROUP_REFERENCES_READY_NPU_NOT_RUN' or cpu.get('baseline')!=r['baseline'] or cpu.get('sources')!=r['sources']:
                raise ValueError('CPU-only report does not bind this source/input/build baseline')
            for item in cpu['sources']:verify(item)
            r['cpu_report']=identity(args.cpu_report);r['cpu_manifest']=cpu['cpu_manifest']
            prepared=json.loads(verify(r['cpu_manifest']).read_text())
        else:
            prepared=prepare(root/'data',r['baseline']['input_files']);r['cpu_manifest']=identity(root/'data/references.json')
        refs=verify_prepared(prepared)
        old_case=json.loads(verify(r['baseline']['case']).read_text());new0=json.loads(verify(prepared['cases'][0]).read_text())
        for name in common.FILES:
            if any(new0['files'][name][key]!=old_case['files'][name][key] for key in ('bytes','sha256')):raise ValueError('group0 differs from accepted Stage2C')
        r['cpu_error_decomposition']=effect_report(refs);r['activation']=prepared['activation'];r['scale_error']=prepared['scale_error']
        r['groups']=[dict(group_index=g,status='NOT_RUN',case=prepared['cases'][g]) for g in range(20)]
        r['phase']='optional_native_semantics';write(root/'report.json',r)
        # Native audit never controls whether candidate groups run.
        try:r['native']=audit(args,r['baseline']['input_files'],frozen)
        except Exception as ex:r['native']=dict(native_semantics_status='UNRESOLVED',native_comparison_status='SKIPPED',
            reason='NATIVE_COMPARISON_SKIPPED_SEMANTICS_UNRESOLVED',error=f'{type(ex).__name__}: {ex}',checks={'audit':{'status':'UNKNOWN','detail':str(ex)}})
        r['native_semantics_status']=r['native']['native_semantics_status'];write(root/'native-semantics.json',r['native'])
        if args.prepare_only:
            r.update(status='CPU_20_GROUP_REFERENCES_READY_NPU_NOT_RUN',phase='complete')
        else:
            execute_groups(args,root,r,built,prepared)
            halves=aggregate(root,r,refs)
            if r['native_semantics_status']=='VERIFIED':
                try:
                    r['native_comparison']=native_compare(r['native'],dict(npu_partial_sum=halves[0],ideal_w8a8=refs['ideal_f16'],
                        effective_w8a8=refs['effective_f16'],cpu_partial_sum=refs['partial_f16'],cpu_w8a16=refs['w8a16_f16']))
                    r['native_comparison_status']='COMPLETED';r['native']['native_comparison_status']='COMPLETED'
                    r['native']['reason']='full-K cross-scheme diagnostic completed after 20-group accumulation'
                except OSError as error:
                    r['native_semantics_status']='UNRESOLVED';r['native'].update(native_semantics_status='UNRESOLVED',reason='NATIVE_COMPARISON_SKIPPED_SEMANTICS_UNRESOLVED',error=str(error))
                except ValueError as error:
                    r['native_semantics_status']='MISMATCH';r['native'].update(native_semantics_status='MISMATCH',reason='NATIVE_SEMANTICS_MISMATCH',error=str(error))
            ok=all(c['status']=='PASS' for c in r['groups']) and r['aggregation']['status']=='MATCH'
            r.update(status='STAGE3A_20_GROUP_MATCH' if ok else 'FAIL_NUMERICAL',phase='complete')
        verify_prepared(prepared);verify_tree(built)
        if e2c.stage2b.installed(Path(built['opp']))!=built['installed']:raise ValueError('Stage2B OPP changed')
    except Exception as ex:
        r.update(status='FAILED',error=f'{type(ex).__name__}: {ex}')
        if r['npu']=='RUNNING':r['npu']='FAILED_OR_INCOMPLETE'
    finally:
        try:
            for item in r['sources']:verify(item)
            if 'baseline' in r:verify_tree(r['baseline'])
            if 'cpu_manifest' in r:verify_tree(json.loads(verify(r['cpu_manifest']).read_text()))
        except Exception as ex:r.update(status='FAILED',integrity_error=str(ex))
        r['summary']=dict(passed=sum(c['status']=='PASS' for c in r.get('groups',[])),
          failed=sum(c['status'].startswith('FAIL') for c in r.get('groups',[])),
          not_run=sum(c['status'].startswith('NOT_RUN') for c in r.get('groups',[])))
        if 'native' in r:write(root/'native-semantics.json',r['native'])
        write(root/'report.json',r);(root/'analysis.md').write_text(report_text(r))
    print(r['status']+': '+str(root/'report.json'))
    return int(r['status'] in ('FAILED','FAIL_NUMERICAL'))


def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage2c-report',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-stage2c-20261026-104943/report.json'))
    p.add_argument('--stage2b-root',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-stage2b-20261026-100533'))
    for name in ('build','basic','zero','nonpower'):p.add_argument('--'+name+'-report',type=Path)
    p.add_argument('--cpu-report',type=Path)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');p.add_argument('--device',type=int,default=0)
    p.add_argument('--cann-root',type=Path,default=Path(os.environ.get('CANN_ROOT','/usr/local/Ascend/cann-9.0.0')))
    p.add_argument('--a51-evidence',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/a51-fallback-20261012-063342/suite.json'))
    p.add_argument('--prefetch-evidence',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/gate-raw-prefetch-reverse-20261020-025853/suite.json'))
    args=p.parse_args();args.cann_root=args.cann_root.resolve()
    for name in ('build','basic','zero','nonpower'):
        if getattr(args,name+'_report') is None:setattr(args,name+'_report',args.stage2b_root/name/'report.json')
    if args.device<0:p.error('negative device')
    if args.cpu_report and args.prepare_only:p.error('--cpu-report is for reusing CPU-only evidence during NPU validation')
    return args

if __name__=='__main__':raise SystemExit(main(parse()))
