"""Read-only Stage2B and frozen Gate identities. No Stage1 execution or build."""
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
S2B=HERE.parent/'stage2b'
sys.path.insert(0,str(S2B))
import s2b_common as common
import s2b_reference as reference
import run_stage2b as stage2b
from s2b_common import identity,verify,write,fresh,verify_tree,CODEC
sys.path.insert(0,str(common.QUANT/'fp16_compare'))
from fp16_gate_evidence import accepted

SETTINGS=dict(op='DFlashW8A8OutputDequant',shape=[16,128,64],A='INT8_ND',B='INT8_ND_NON_TRANSPOSED',O='FP16_ND',
    scale='UINT64_ND_64',HostC='DT_FLOAT16',dequant='TENSOR',block_dim=1,base=[16,64,128],
    codec='doc_mask13_bit46',rounding_model='FP32_product_then_FP16_RNE')


def sources():
    paths=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
    paths += [common.QUANT/'fp16_compare'/name for name in ('fp16_gate_evidence.py','fp16_gate_common.py')]
    paths += [common.FIX/'w8_reference.py',common.FIX/'w8_common.py',common.FIX/'w8_run.py']
    return [identity(p) for p in sorted(set(paths))]


def validated_stage2b(args):
    b=json.loads(args.build_report.read_text());verify_tree(b)
    if b.get('status')!='STAGE2B_BUILT_NPU_NOT_RUN' or b.get('phase')!='complete' or b.get('settings')!=SETTINGS or b.get('codec_status')!=CODEC:
        raise ValueError('requires accepted Stage2B build/config')
    if b.get('sources')!=common.sources():raise ValueError('Stage2B sources changed since accepted build')
    if b.get('device')!=args.device:raise ValueError('Stage2B build device differs')
    pf=json.loads(verify(b['upstream']['stage1']['preflight']).read_text())
    if Path(pf['cann_root']).resolve()!=args.cann_root:raise ValueError('SDK root differs from accepted Stage2B environment')
    verify_tree(pf)
    if stage2b.installed(Path(b['opp']))!=b['installed']:raise ValueError('Stage2B OPP identity changed')
    build_id=identity(args.build_report);proofs={};case_evidence=[]
    for group,path in (('basic',args.basic_report),('zero',args.zero_report),('nonpower',args.nonpower_report)):
        r=json.loads(path.read_text());verify_tree(r)
        if r.get('status')!='STAGE2B_'+group.upper()+'_MATCH' or r.get('phase')!='complete' or r.get('group')!=group:
            raise ValueError('Stage2B group proof missing: '+group)
        if r.get('build_report')!=build_id or r.get('device')!=args.device or r.get('upstream')!=b['upstream'] or r.get('settings')!=SETTINGS or r.get('sources')!=b['sources'] or r.get('codec_status')!=CODEC or r.get('performance')!='NOT_RUN':
            raise ValueError('Stage2B group binding differs: '+group)
        if group!='basic' and r.get('basic_proof')!=identity(args.basic_report):raise ValueError('advanced proof refers to a different basic run')
        names=common.GROUPS[group]
        if len(r.get('cases',[]))!=len(names) or {c['name'] for c in r['cases']}!=set(names):raise ValueError('incomplete Stage2B group: '+group)
        for entry in r['cases']:
            case=json.loads(verify(entry['case']).read_text())
            if case['name']!=entry['name'] or case['group']!=group:raise ValueError('Stage2B case scope mismatch')
            reference.verify_origin(case,b['upstream'])
            common.check_execution(json.loads(verify(entry['execution']).read_text()),args.device)
            comparison=reference.compare(case,[verify(f).read_bytes() for f in entry['outputs']])
            if comparison!=entry['comparison'] or comparison['status']!='CANDIDATE_BITWISE_MATCH':raise ValueError('Stage2B saved numerical evidence changed')
            case_evidence.append(dict(group=group,name=entry['name'],case=entry['case'],execution=entry['execution'],outputs=entry['outputs']))
        proofs[group]=identity(path)
    return b,dict(build=build_id,groups=proofs,runner=b['runner'],installed=b['installed'],settings=b['settings'],
                  sources=b['sources'],upstream=b['upstream'],case_evidence=case_evidence,stage1_rerun=False,stage2b_rerun=False)


def frozen_source(a51,pair):
    # Existing strict validator binds Capture/NZ/Scale/Native full-K Golden and accepted W8 OPP.
    # It only reads evidence; it never launches the old W8 or Raw Runner.
    src=accepted(a51,pair)
    return dict(files=src['files'],identity=src['identity'],watched=src['watched'],full_k_golden=src['golden'],
        scope='C16 layer-0-gate_up; select K[0:128], N[0:64]; full-K Golden is identity-only')
