"""Bind the frozen FP16 failure and completed independent Raw NPU evidence."""
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from vdeq_common import raw,raw_run,read,verify,identity,KEY_DOTS


def load(frozen_root,raw_root,cann):
 original=raw.diag.load(frozen_root);report=read(raw_root/'bounded/report.json')
 if report['status']!='RAW_INT32_NPU_MATCH' or report['phase']!='complete' or report['npu']!='RAW_INT32_NPU_MATCH':raise ValueError('completed Raw NPU report required')
 host=read(verify(report['host_report']));build=read(verify(report['build_report']))
 args=SimpleNamespace(frozen_root=frozen_root,raw_host_report=verify(report['evidence']['raw_host']),cann_root=cann,device=report['device'])
 proof,a,b,oracle=raw.evidence(args)
 raw_run.verify_gate(host,args,proof,'FORMAL_HOST_FEASIBLE');raw_run.verify_gate(build,args,proof,'BUILT_NPU_NOT_RUN')
 if report['sources']!=raw.sources() or report['evidence']!=proof or build['host_report']!=report['host_report'] or any(report[k]!=host[k] for k in ('artifacts','workspace_bytes','formal_plan')):raise ValueError('Raw phase/source/artifact binding changed')
 raw_run.verify_artifacts(report['artifacts'],proof)
 if len(report['outputs'])!=2:raise ValueError('two Raw outputs required')
 outputs=[verify(x).read_bytes() for x in report['outputs']]
 comparison=raw.compare(oracle,outputs)
 if comparison['status']!='MATCH' or comparison!=report['integer_comparison']:raise ValueError('Raw integer evidence mismatch')
 for name,expected in [('a_i8.bin',a),('b_i8.bin',b),('oracle_i32.bin',oracle.astype('<i4').tobytes()),('oracle_i64.bin',oracle.astype('<i8').tobytes())]:
  if verify(report['data'][name]).read_bytes()!=expected:raise ValueError('Raw input/oracle mismatch '+name)
 receiver={Path(x['path']).name:verify(x) for x in report['receiver_files']}
 ex=read(receiver['execution.json']);ad=read(receiver['resource-admission.json'])
 expected=dict(status='RAW_INT32_EXECUTION_MATCH',npu='COMPLETED',device=args.device,workspace_bytes=report['workspace_bytes'],shape=[16,2560,256],elements=4096,repetitions=2,guards=True,readonly=True,poison=True,poison_patterns=[127,128],repeat_equal=True,cpu_fallback=False,benchmark='NOT_RUN')
 if ex!=expected:raise ValueError('Raw safety protocol differs')
 pl=raw_run.plan(receiver['runner.log'].read_text(),proof['t1'],proof['policy'])
 if pl!=report['formal_plan'] or ad!=dict(status='ADMITTED_BEFORE_EXECUTION',workspace_bytes=report['workspace_bytes'],declared_total=pl['total'],blocks=1,cores=proof['policy']['selected'],core_policy=proof['policy'],host_proof_checked=True):raise ValueError('Raw formal plan/admission mismatch')
 for i,item in enumerate(report['outputs']):
  if identity(receiver[f'actual-{i}.bin'])!=item:raise ValueError('Raw output not bound to execution')
 dots=np.frombuffer(outputs[0],'<i4').reshape(16,256).astype(np.int64)
 if not np.array_equal(dots,original['oracle']):raise ValueError('Raw and original independent dot disagree')
 if any(int(dots[pos])!=value for pos,value in KEY_DOTS.items()):raise ValueError('six frozen key dots differ from reported experiment')
 raw_standalone=read(args.raw_host_report);fp_host=read(verify(original['identities']['host']))
 return dict(original=original,raw=report,raw_host=host,raw_standalone=raw_standalone,fp_host=fp_host,dots=dots,
             identities=dict(frozen=original['identities'],raw_bounded=identity(raw_root/'bounded/report.json'),raw_host=report['host_report'],raw_build=report['build_report'],raw_standalone=identity(args.raw_host_report),raw_outputs=report['outputs']))
