"""Independent Down evidence. No mutation of Stage4W or any accepted OPP."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
sys.path.insert(0,str(BASE/'stage4w_draft_mlp'))
import real_mlp_common as w
import run_real_gate as wg
from raw_probe_runtime_identity import verify_runtime,trace
identity,verify,read,write,verify_tree=w.identity,w.verify,w.read,w.write,w.verify_tree
p,q,common=w.pe.nz.p,w.pe.nz.q,w.pe.nz.common


def sources():
    return [identity(f) for f in sorted(HERE.rglob('*')) if f.is_file() and '__pycache__' not in f.parts]


def bind(root,device):
    prep_path=root/'prepare/prepare-report.json';gp=root/'gate-smoke/report.json';mp=root/'mlp-smoke/report.json'
    prep=w.prepared(prep_path);g=read(gp);m=read(mp);verify_tree(g);verify_tree(m)
    if g.get('status')!=wg.STATES['smoke'] or m.get('status')!='REAL_MLP_BOUNDARIES_RECORDED_W8_DOWN_NOT_RUN':raise ValueError('real W Gate/MLP smoke prerequisites')
    for r in (g,m):
        if r.get('error') or r.get('integrity_error') or r['sources']!=w.sources() or r['npu']!='COMPLETED' or r['prepare_report']!=identity(prep_path):raise ValueError('W source/preparation/NPU evidence drift')
    if g['device']!=device or m['environment']['device']!=device or m['gate_smoke_report']!=identity(gp):raise ValueError('W device/Gate chain differs')
    wg.pair.verify_build(g['runtime'],g['evidence'],Path(g['cann_root']))
    expected=[x['id'] for x in prep['cases']]
    if [x['case'] for x in g['runs']]!=expected or [x['case'] for x in m['cases']]!=expected or len(expected)!=4:raise ValueError('four real sample inventory required')
    for case,row,mcase in zip(prep['cases'],g['runs'],m['cases']):
        if row['validation']!=wg.validate(row,case,g['evidence'],device):raise ValueError('historical real Gate safety changed')
        for scheme,outputs in mcase['boundaries'].items():
            if len(outputs)!=2:raise ValueError('two real MLP repeats required')
            for key in outputs[0]:
                if verify(outputs[0][key]).read_bytes()!=verify(outputs[1][key]).read_bytes():raise ValueError('MLP repeat changed')
        mixed=mcase['boundaries']['MIXED_BNZ_GATE_W8A16_DOWN'][0]
        if verify(mixed['gate_up']).read_bytes()!=verify(row['outputs'][0]).read_bytes():raise ValueError('mixed path did not consume accepted real Gate output')
        if mixed['multiply']['bytes']!=16*9728*2 or mcase['context_rows']!=case['capture']['context_rows']:raise ValueError('real mixed Down input scope')
    if sorted(x['context_rows'] for x in m['cases'])!=[16,16,64,64] or len({x['boundaries']['MIXED_BNZ_GATE_W8A16_DOWN'][0]['multiply']['sha256'] for x in m['cases']})!=4:raise ValueError('distinct C16/C64 inputs required')
    return dict(prepare=identity(prep_path),gate=identity(gp),mlp=identity(mp),
        formal_core_policy=g['evidence']['bnz']['evidence']['baseline']['policy'],
        baseline=g['evidence']['bnz']['evidence']['baseline'],cann_root=g['cann_root'],machine=g['machine'])


def sdk_layout(cann):
    record=BASE/'stage4v_fullk_b_nz/sdk-review.json';rows=[]
    for entry in read(record)['sources']:
        f=identity(cann/entry['relative_path'])
        if cann not in Path(f['path']).parents or f['sha256']!=entry['sha256']:raise ValueError('B-NZ SDK identity mismatch: '+entry['relative_path'])
        rows.append(f)
    return dict(review=identity(record),files=rows,status='FIXED_LAYOUT_SDK_IDENTITIES_MATCHED')


def env(cann,opp,policy,t1,limit=0):
    value=common.env(cann,Path(opp))
    value['DFLASH_STAGE4X_EXPECTED_CORE_POLICY']=','.join(str(policy[k]) for k in ('aic','aiv','all','selected','arch'))
    system=t1['system_workspace_bytes'];blocks=limit or min(10,policy['selected'])
    value['FULLK_EXPLORE_WORKSPACE_BYTES']=str(system+blocks*((system+511)//512*512+1024))
    return value


def verify_build(r):
    from build_x import installed
    from x_transform import verify_repository
    verify_repository();verify_tree(r)
    for kind,value in r['variants'].items():
        if value['installed']!=installed(Path(value['opp']),kind):raise ValueError('private installed OPP changed')
        for dep in value['dependencies'].values():
            verify_runtime(dep)
            for f in dep['private_opp_libraries']:
                if trace(f['requested_path'])!={k:f[k] for k in ('requested_path','real_path','symlink_chain')}:raise ValueError('private library alias changed')
