"""Stage2C proof is required; Native files are explicitly optional diagnostics."""
import json
from pathlib import Path
from s3a_reference import HERE,e2c,r2c,common,identity,verify,verify_tree


def sources():
    paths=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
    paths += [e2c.common.OPS/'draft_ac/test'/n for n in ('c_profile_graph.py','profile_pbtxt.py')]
    return [identity(p) for p in sorted(paths)]


def baseline(args):
    built,bproof=e2c.validated_stage2b(args)
    r=json.loads(args.stage2c_report.read_text())
    required=dict(stage='2C',status='STAGE2C_REAL_K128_MATCH',phase='complete',shape=[16,128,64],original_shape=[16,2560,19456],
        npu='MATCHED_SINGLE_REAL_K128_ONLY',layer=0,projection='gate_up',group=0,k_range=[0,128],n_range=[0,64],device=args.device,codec_status=e2c.CODEC,performance='NOT_RUN')
    if any(r.get(k)!=v for k,v in required.items()) or r.get('stage2b_evidence')!=bproof or r.get('sources')!=e2c.sources():raise ValueError('Stage2C proof scope/source/build differs')
    for item in r['sources']:verify(item)
    case=json.loads(verify(r['case']).read_text());r2c.verify_case(case)
    common.check_execution(json.loads(verify(r['execution']).read_text()),args.device)
    result=r2c.compare(case,[verify(f).read_bytes() for f in r['outputs']])
    if result!=r['comparison'] or result['status']!='CANDIDATE_BITWISE_MATCH':raise ValueError('Stage2C numerical evidence changed')
    f=r['frozen_evidence'];files=case['source_files']
    if f['identity']['name']!='layer-0-gate_up' or any(f['files'][name]!=item or f['identity']['capture_files'][name]!=item for name,item in files.items()):raise ValueError('Stage2C frozen Capture identity differs')
    verify_tree(files)
    # Do not recursively validate Native/eager artifacts here. They are not inputs to the 20-group candidate.
    return built,dict(stage2c=identity(args.stage2c_report),case=r['case'],execution=r['execution'],outputs=r['outputs'],
        stage2b=bproof,sources=r['sources'],input_files=files),f
