"""Read-only accepted GM-output evidence and strict Local-output comparisons."""
import json
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage3b'))
import s3b_common as reduction
from s3b_common import common,e2c,identity,verify,verify_tree,write,fresh,metrics
reference=e2c.reference
r2c=reduction.s3a.r2c
NAME='DFlashW8A8LocalDequant'
SNAKE='d_flash_w8_a8_local_dequant'
SETTINGS=dict(op=NAME,shape=[16,128,64],base=[16,64,128],block_dim=1,
    A='GM_INT8_ND',B='GM_INT8_ND_NON_TRANSPOSED',C='VECIN_FP16_NZ',ABI_output='GM_FP16_ND',
    dequant='TENSOR',codec=common.CODEC,nz_shape=[4,16,16],explicit_ub=2048,reserve_ub=8192,
    sdk_scratch='floor32(actual_UB - 2048 - 8192)',layout_copy='64 aligned 32-byte DataCopy blocks',
    fusion='NOT_IMPLEMENTED',performance='NOT_RUN')


def sources():
    files=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
    files += [HERE.parent/'stage3b/s3b_common.py',HERE.parent/'stage2c/s2c_evidence.py',
              HERE.parent/'stage2b/run_stage2b.py',common.FIX/'compile_raw.py']
    return [identity(p) for p in sorted(set(files))]


def evidence(args):
    # Revalidate completed Stage3B-1 without invoking its Runner or changing any report.
    rp=args.stage3b1_root/'smoke/report.json';r=json.loads(rp.read_text());verify_tree(r)
    if r.get('status')!='STAGE3B1_REDUCTION_MATCH' or r.get('phase')!='complete' or r.get('device')!=args.device or r.get('settings')!=reduction.SETTINGS:
        raise ValueError('accepted same-device Stage3B-1 reduction required')
    case=json.loads(verify(r['case']).read_text());reduction.check_execution(json.loads(verify(r['execution']).read_text()),args.device)
    compared=reduction.compare(case,[verify(x).read_bytes() for x in r['outputs']])
    if compared!=r['comparison'] or compared['status']!='REDUCTION_BITWISE_MATCH':raise ValueError('Stage3B-1 numerical evidence changed')
    build=json.loads(verify(r['build_report']).read_text());verify_tree(build)
    if build.get('status')!='STAGE3B1_BUILT_NPU_NOT_RUN' or build.get('settings')!=r['settings'] or build.get('stage3a_evidence')!=r['stage3a_evidence']:
        raise ValueError('Stage3B-1 build binding differs')
    # Existing Stage2B validator binds all GM output tests, tool/source identities and private OPP.
    _,proof=e2c.validated_stage2b(args)
    real=json.loads(args.stage2c_report.read_text());verify_tree(real)
    scope=dict(stage='2C',status='STAGE2C_REAL_K128_MATCH',phase='complete',shape=[16,128,64],original_shape=[16,2560,19456],
        npu='MATCHED_SINGLE_REAL_K128_ONLY',layer=0,projection='gate_up',group=0,k_range=[0,128],n_range=[0,64],
        device=args.device,codec_status=common.CODEC,performance='NOT_RUN')
    if any(real.get(k)!=v for k,v in scope.items()) or real.get('stage2b_evidence')!=proof or real.get('sources')!=e2c.sources():
        raise ValueError('accepted Stage2C K128/N64 evidence required')
    common.check_execution(json.loads(verify(real['execution']).read_text()),args.device)
    selected=[]
    for label in ('basic','nonpower'):
        old=json.loads(getattr(args,label+'_report').read_text())
        for row in old['cases']:
            selected.append(dict(name=row['name'],case=row['case'],outputs=row['outputs']))
    real_case=json.loads(verify(real['case']).read_text())
    frozen=real['frozen_evidence']
    if frozen['identity']['name']!='layer-0-gate_up' or any(frozen['files'][name]!=item or frozen['identity']['capture_files'][name]!=item for name,item in real_case['source_files'].items()):
        raise ValueError('Stage2C frozen Capture binding differs')
    if r2c.compare(real_case,[verify(x).read_bytes() for x in real['outputs']])!=real['comparison']:
        raise ValueError('Stage2C numerical evidence changed')
    selected.append(dict(name='real_gate_up',case=real['case'],outputs=real['outputs']))
    for entry in selected:
        c=core_case(json.loads(verify(entry['case']).read_text()));outs=[verify(x).read_bytes() for x in entry['outputs']]
        if reference.compare(c,outs)['status']!='CANDIDATE_BITWISE_MATCH':raise ValueError('original GM-output Golden changed')
    if len(selected)!=6 or len({x['name'] for x in selected})!=6:raise ValueError('complete basic/nonpower/real fixture set required')
    pf=proof['upstream']['stage1']['preflight']
    if pf!=r['stage3a_evidence']['preflight']:raise ValueError('SDK provenance differs across accepted stages')
    return dict(reduction_report=identity(rp),stage2b=proof,real_report=identity(args.stage2c_report),
        static=build['vdeq_static'],preflight=pf,cases=selected,earlier_runners_executed=False)


def core_case(case):
    if set(case['files'])!=set(common.FILES):r2c.verify_case(case)
    core=dict(name=case['name'],files={name:case['files'][name] for name in common.FILES})
    reference.verify_case(core)
    return core


def scale_controls(case):
    case=core_case(case)
    acc,scale,effective,ideal,golden=reference.verify_case(case)
    unscaled=common.stage2a.convert(acc,np.ones(64,'<f4'))
    scalar=common.stage2a.convert(acc,np.full(64,effective[0],'<f4'))
    return dict(vs_unscaled=metrics(unscaled,golden),vs_first_channel_broadcast=metrics(scalar,golden),
                distinct_effective_scales=int(len(np.unique(effective))))


def compare(entry,outputs):
    case=core_case(json.loads(verify(entry['case']).read_text()))
    result=reference.compare(case,outputs)
    old=[verify(x).read_bytes() for x in entry['outputs']]
    if len(old)!=2 or old[0]!=old[1]:raise ValueError('original GM repeats differ')
    result['vs_accepted_gm']=[metrics(np.frombuffer(old[0],'<f2'),np.frombuffer(x,'<f2')) for x in outputs]
    result['scale_controls']=scale_controls(case)
    if any(not v['bitwise_equal'] for v in result['vs_accepted_gm']):result['status']='FAIL_NUMERICAL'
    # Observability is mandatory for dedicated scale witnesses, not just Scale=1.
    controls=result['scale_controls']
    if entry['name'] in ('scale_half','per_channel_powers','per_channel_blocks') and controls['vs_unscaled']['bit_mismatches']==0:
        raise ValueError('Scale fixture cannot distinguish ignored Scale')
    if entry['name'] in ('per_channel_powers','per_channel_blocks') and (controls['distinct_effective_scales']<2 or controls['vs_first_channel_broadcast']['bit_mismatches']==0):
        raise ValueError('Scale fixture cannot distinguish scalar from per-channel dequant')
    return result


def check_execution(ex,device):
    if ex.get('op')!=NAME:raise ValueError('wrong local probe operator')
    # Same public ABI/safety checks; adapt the expected operator name only.
    old=dict(ex);old['op']=common.NAME;common.check_execution(old,device)


def nz_to_nd_model(bits):
    if bits.shape!=(4,16,16):raise ValueError('fixed aligned Local NZ shape required')
    return bits.transpose(1,0,2).reshape(16,64).copy()
