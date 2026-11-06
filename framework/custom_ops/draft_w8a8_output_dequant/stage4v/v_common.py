"""Independent full-K scheme; read-only historical evidence and no inherited Golden."""
import sys,difflib
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4r'))
import run_stage4r as rr
q=rr.q;p=q.p;identity,verify,read,write,fresh,verify_tree=q.identity,q.verify,q.read,q.write,q.fresh,q.verify_tree
common=q.common
NAME='DFlashW8A8FullKChannel';SNAKE='d_flash_w8_a8_full_k_channel'
COVERAGE='FULL_K_TILE_384_WORD_THREE_BOUNDARIES'
POLICY=dict(output='BITWISE_FP16_FP32_DOT_TIMES_EFFECTIVE_SCALE',weight_scale='FP64_MAX_ABS_OVER_K_DIV127_ZERO_SCALE1',weight_round='RNE_CLIP_NEG127_POS127',combined='RN32(FP64(scale_A)*FP64(scale_W))',codec='MASK13_BIT46_EXPERIMENTAL_CANDIDATE_FULL_K_NOT_DEVICE_VERIFIED',quality_go='UNRESOLVED_NO_PREDECLARED_THRESHOLD',performance_ratio_min=1.03)

def sources():
 rows={x['path']:x for x in rr.sources()}
 for f in HERE.rglob('*'):
  if f.is_file() and '__pycache__' not in f.parts and '.pytest_cache' not in f.parts:rows[str(f.resolve())]=identity(f)
 return [rows[k] for k in sorted(rows)]

def evidence(args):
 e=q.evidence(args);root=args.stage4r_root
 br=read(root/'benchmark/benchmark-report.json')
 current=dict(sources=rr.sources(),evidence=e,device=args.device,candidate='N256',policy=rr.POLICY,machine=br['machine'])
 rr.prior(root/'benchmark/benchmark-report.json','benchmark',current)
 prep=rr.prior(verify(br['prepare_report']),'prepare',current)
 build=rr.prior(verify(br['build_report']),'build',current)
 current.update(prepare_report=br['prepare_report'],build_report=br['build_report'])
 data=read(verify(prep['reference']))
 smoke=rr.validate_smoke(verify(br['smoke_report']),current,build,data)
 rr.verify_binding(build['native']['header_binding'],args.cann_root)
 if build['prepare_report']!=br['prepare_report'] or build['w8_runner_build']['evidence']!=e['variants']['N256'] or build['w8_runner_build']['linked_libraries']['LIB_custom']!=e['variants']['N256']['installed']['api']:raise ValueError('Stage4R build/data binding changed')
 import numpy as np
 reference=np.fromfile(verify(data['files']['reference_f64.bin']),'<f8').reshape(16,19456)
 if [(x['round'],x['position'],x['scheme']) for x in br['runs']]!=[(i,j,n) for i,o in enumerate((('FP16','W8'),('W8','FP16'),('FP16','W8'))) for j,n in enumerate(o)]:raise ValueError('Stage4R paired coverage differs')
 for row in br['runs']:
  if row['scheme']=='W8':q.validate_run(row,e,args.device,True)
  elif row['scheme']=='FP16':
   ex=read(verify(row['execution']));rr.fc.check_execution(ex,'fp16',args.device,True)
   if ex.get('independent_cpu_reference') is not True or ex.get('poison_verified') is not True or ex.get('timed_log_stable') is not True or ex.get('verify_calls')!=4 or len(row['outputs'])!=4:raise ValueError('Stage4R native evidence incomplete')
   for out in row['outputs']:
    raw=verify(out).read_bytes()
    if raw!=verify(smoke['frozen_native']).read_bytes():raise ValueError('Stage4R native frozen result differs')
    rr.check(reference,np.frombuffer(raw,'<f2').reshape(16,19456))
   if {k:q.statistics_ms(ex['timing'][k]['samples_ms'],30) for k in ('execute_sync','prepare')}!=row['timing']:raise ValueError('historical native timing evidence changed')
  else:raise ValueError('unknown historical scheme')
 return dict(qproof=e,baseline=e['variants']['N256'],native_build=build['native'],native_reference=prep['reference'],native_frozen=smoke['frozen_native'],stage4r_benchmark=identity(root/'benchmark/benchmark-report.json'),stage4r_smoke=br['smoke_report'],preflight=e['variants']['N256']['preflight'],formal_core_policy=e['variants']['N256']['policy'])

def environment(proof,cann):
 e=q.common.env(cann,Path(proof['opp']))
 for k in ('DFLASH_STAGE4P_EXPECTED_CORE_POLICY','DFLASH_STAGE4J_EXPECTED_CORE_POLICY'):e.pop(k,None)
 e['DFLASH_STAGE4V_EXPECTED_CORE_POLICY']=','.join(str(proof['policy'][k]) for k in ('aic','aiv','all','selected','arch'))
 return e

def immutable(e):
 verify_tree(e);q.immutable(e['qproof'])


def contract_header(t1):
 return p.contract_header(t1).replace('S4P','S4V').replace('STAGE4P','STAGE4V')


def materialize(dest,fault=False):
 dest.mkdir();rows=[]
 for f in [HERE/'operator.json']+[f for sub in ('op_host','op_kernel') for f in (HERE/sub).iterdir() if f.is_file()]:
  out=dest/f.relative_to(HERE);out.parent.mkdir(parents=True,exist_ok=True)
  raw=f.read_bytes()
  if f.parent.name=='op_kernel':raw=b'#define STAGE4V_FAULT '+str(int(fault)).encode()+b'\n'+raw
  out.write_bytes(raw);rows.append(dict(source=identity(f),generated=identity(out)))
 baseline=HERE.parent/'stage4p/op_kernel/d_flash_w8_a8_n256_gate.cpp'
 target=(dest/'op_kernel'/ (SNAKE+'.cpp')).read_text()
 diff=''.join(difflib.unified_diff(baseline.read_text().splitlines(True),target.splitlines(True),fromfile='Stage4P_Group128',tofile='Stage4V_FullK_NEW_SEMANTICS'))
 (dest/'arithmetic-source.diff').write_text(diff)
 a=dict(fault=fault,coverage=COVERAGE,files=rows,baseline_kernel=identity(baseline),diff=identity(dest/'arithmetic-source.diff'),status='EXACT_FULL_K_SOURCE_PLUS_FAULT_DEFINE')
 write(dest/'source-audit.json',a);return a


def verify_audit(a):
 expected={str(f.resolve()) for f in [HERE/'operator.json']+[f for sub in ('op_host','op_kernel') for f in (HERE/sub).iterdir() if f.is_file()]}
 if {r['source']['path'] for r in a['files']}!=expected or len(a['files'])!=len(expected) or a['coverage']!=COVERAGE:raise ValueError('source audit scope drift')
 baseline=verify(a['baseline_kernel']);verify(a['diff'])
 for row in a['files']:
  f=verify(row['source']);raw=f.read_bytes()
  if f.parent.name=='op_kernel':raw=b'#define STAGE4V_FAULT '+str(int(a['fault'])).encode()+b'\n'+raw
  if verify(row['generated']).read_bytes()!=raw:raise ValueError('candidate source changed')
  if f.parent.name=='op_kernel':
   diff=''.join(difflib.unified_diff(baseline.read_text().splitlines(True),raw.decode().splitlines(True),fromfile='Stage4P_Group128',tofile='Stage4V_FullK_NEW_SEMANTICS'))
   if verify(a['diff']).read_text()!=diff:raise ValueError('source difference record changed')
