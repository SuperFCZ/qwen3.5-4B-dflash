"""Read-only frozen failed-run binding. No normal Stage4V state changes."""
import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4v'))
import v_common as v
import reference_v as ref
from protocol_v import verify_build,plan
identity,verify,read,write,fresh,verify_tree=v.identity,v.verify,v.read,v.write,v.fresh,v.verify_tree


def sources():return [identity(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]


def load(root):
 root=Path(root).resolve(strict=True);report=read(root/'bounded/report.json')
 if report.get('status')!='BLOCKED':raise ValueError('frozen failed Bounded report required; no normal admission')
 for name,status in [('prepare','STAGE4V_CPU_PREPARED_QUALITY_UNRESOLVED'),('host','STAGE4V_FULL_K_HOST_FEASIBLE'),('build','STAGE4V_BUILT_NPU_NOT_RUN')]:
  old=read(verify(report[name+'_report']))
  if old.get('status')!=status or old.get('phase')!='complete' or any(old.get(k)!=report.get(k) for k in ('sources','evidence','device','policy')):raise ValueError('upstream evidence mismatch '+name)
 build=read(verify(report['build_report']));verify_build(build,report)
 case=read(verify(report['case']));maps=ref.verify_case(case)
 # Discover only explicitly bound failure outputs; do not glob the server tree.
 found=[]
 for item in report.get('failure_diagnostics',[]):
  diag=read(verify(item))
  if (diag.get('mode'),diag.get('limit'),diag.get('tile'))==(0,0,0):
   for row in diag['outputs']:
    if Path(row['file']['path']).name=='actual-0.bin':found.append(row['file'])
 if len(found)!=1:raise ValueError('missing/ambiguous bound Tile0 actual-0 identity')
 actual=verify(found[0]);folder=actual.parent
 if root not in folder.parents:raise ValueError('failed output outside frozen run')
 expected_file=folder/'expected-0.bin';expected_id=identity(expected_file)
 execution_id=identity(folder/'execution.json');read(verify(execution_id))
 log_id=identity(folder/'runner.log');admission_id=identity(folder/'resource-admission.json')
 pl=plan(verify(log_id).read_text(),0,0,0,report['t1'],report['evidence']['formal_core_policy'])
 admission=read(verify(admission_id))
 if admission.get('status')!='ADMITTED_BEFORE_EXECUTION' or admission.get('blocks')!=pl['blocks'] or admission.get('declared_total')!=pl['total'] or admission.get('workspace_bytes',0)<pl['total'] or admission.get('core_policy')!=pl['core_policy']:raise ValueError('frozen resource admission invalid')
 a=maps['a_i8.bin'];b=maps['b_i8.bin'][0]
 oracle=a.astype(np.int64)@b.astype(np.int64)
 dots=maps['dot_i32.bin'][:,:256].astype(np.int64)
 if not np.array_equal(oracle,dots):raise ValueError('stored CPU INT32 dot differs from independent INT64 oracle')
 if np.any(np.abs(oracle)>41290240):raise ValueError('INT32 safe bound exceeded')
 words=maps['quant_u64_doc_candidate.bin'][0].copy();effective=maps['effective_f32.bin'][:256].copy()
 decoded=(words&np.uint64(0xffffffff)).astype('<u4').view('<f4')
 if decoded.tobytes()!=effective.tobytes():raise ValueError('effective Scale does not match word bits')
 baseline=(dots.astype('<f4')*effective).astype('<f2')
 expected=np.full((16,19456),0x7f7f,'<u2');expected[:,:256]=baseline.view('<u2')
 if baseline.tobytes()!=maps['golden_fp16.bin'][:,:256].copy().tobytes():raise ValueError('original CPU model/Golden mismatch')
 if verify(expected_id).read_bytes()!=expected.tobytes():raise ValueError('expected-0 changed, wrong stride or scope')
 raw=actual.read_bytes()
 if len(raw)!=622592:raise ValueError('full ND output size must be622592 bytes, not a compact prefix')
 y=np.frombuffer(raw,'<u2').reshape(16,19456)
 if not np.all(y[:,256:]==0x7f7f):raise ValueError('inactive columns changed')
 ids=dict(bounded=identity(root/'bounded/report.json'),prepare=report['prepare_report'],host=report['host_report'],build=report['build_report'],case=report['case'],actual=found[0],expected=expected_id,execution=execution_id,log=log_id,admission=admission_id,
  files={n:case['files'][n] for n in ('dot_i32.bin','effective_f32.bin','quant_u64_doc_candidate.bin','a_i8.bin','b_i8.bin','golden_fp16.bin')})
 return dict(identities=ids,frozen=report,case=case,dots=dots,words=words,actual=y[:,:256].copy(),expected=baseline.view('<u2').copy(),oracle=oracle,
  readonly_and_ub_guard='NOT_REATTESTED_BY_OFFLINE_ANALYSIS; original log and failed status preserved')
