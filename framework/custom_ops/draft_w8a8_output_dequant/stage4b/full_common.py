"""Frozen Stage4A admission and full Gate configuration. No historical device calls."""
import json,re
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4a'))
import tile_common as old
import run_stage4a as old_run
from tile_common import common,identity,verify,verify_tree,write,fresh,metrics
NAME='DFlashW8A8FullGate';SNAKE='d_flash_w8_a8_full_gate'
M,K,N,TILES=16,2560,19456,304
HISTORY=old.TILES
SETTINGS=dict(op=NAME,A=[20,16,128],B=[304,20,128,64],scale=[304,20,64],output=[16,19456],
    tile_n=64,groups=20,base=[16,64,128],scratch=239104,user_ub=12288,ub_guards=2560,reserve=8192,
    core_policy=old.SETTINGS['core_policy'],workspace=old.SETTINGS['workspace'],workspace_cap=old.SETTINGS['workspace_cap'],
    scheduling='tile=GetBlockIdx()+i*GetBlockNum(); clear FP32 sum per tile',performance='NOT_RUN')
SPECS={'a_i8.bin':('i1',(20,16,128)), 'b_i8.bin':('i1',(304,20,128,64)),
    'quant_u64_doc_candidate.bin':('<u8',(304,20,64)), 'scale_f32.bin':('<f4',(304,20,64)),
    'scale_effective_f32.bin':('<f4',(304,20,64)), 'partials_fp16.bin':('<f2',(304,20,16,64)),
    'sum_fp32.bin':('<f4',(16,19456)), 'golden_fp16.bin':('<f2',(16,19456)),
    'w8a16_cpu_fp16.bin':('<f2',(16,19456))}


def sources():
    paths=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
    paths += [HERE.parent/'stage4a/tile_common.py',HERE.parent/'stage4a/run_stage4a.py',HERE.parent/'stage2b/run_stage2b.py',common.FIX/'compile_raw.py']
    return [identity(p) for p in sorted(set(paths))]


def evidence(args):
    upstream=old.evidence(args)
    path=args.stage4a_root/'smoke/report.json';r=json.loads(path.read_text());verify_tree(r)
    if r.get('status')!='STAGE4A_MULTI_TILE_MATCH' or r.get('phase')!='complete' or r.get('device')!=args.device or r.get('settings')!=old.SETTINGS or r.get('sources')!=old.sources() or r.get('evidence')!=upstream or r.get('performance')!='NOT_RUN':raise ValueError('unchanged accepted Stage4A required')
    b=json.loads(verify(r['multi_build_report']).read_text());verify_tree(b)
    if b.get('status')!='STAGE4A_BUILT_NPU_NOT_RUN' or any(b.get(k)!=r[k] for k in ('evidence','sources','settings','device')):raise ValueError('Stage4A build binding differs')
    policy=old_run.validate_host_probe(verify(r['host_probe_report']),r['multi_build_report'],r)
    if r.get('core_policy')!=policy or r.get('scheduling_core_limit')!=policy['selected'] or policy['selected']<2:raise ValueError('Stage4A multicore policy not accepted')
    case=json.loads(verify(r['case']).read_text());d=old.verify_case(case)
    limits=[1,2]+([0] if min(8,policy['selected'])!=2 else [])
    modes=[21,0]+[g+1 for g in r.get('diagnostic_groups',[])]
    if [(x['core_limit'],x['mode']) for x in r.get('runs',[])]!=[(limit,mode) for limit in limits for mode in modes]:raise ValueError('Stage4A run coverage incomplete')
    finals=[]
    for row in r['runs']:
        limit,mode=row['core_limit'],row['mode'];ex=json.loads(verify(row['execution']).read_text());old.check_execution(ex,args.device,mode,limit)
        log=r['commands'][f'l{limit}_m{mode}']['log_identity'];plan=old.plan_from_log(verify(log).read_text(),mode,limit)
        if row.get('plan')!=plan or plan['core_policy']!=policy or json.loads(verify(row['resource_admission']).read_text())!=old_run.admission_expected(ex,plan,False):raise ValueError('Stage4A resource proof differs')
        outputs=[verify(x).read_bytes() for x in row['outputs']];result=old.compare(case,d,mode,plan['blocks'],outputs)
        if row.get('status')!='BITWISE_MATCH' or result!=row['comparison'] or result['status']!='BITWISE_MATCH':raise ValueError('Stage4A numerical evidence differs')
        if mode==0:finals.extend(row['outputs'])
    if any(verify(x).read_bytes()!=verify(case['files']['golden_fp16.bin']).read_bytes() for x in finals):raise ValueError('Stage4A final outputs do not match bound Golden')
    return dict(stage4a_report=identity(path),stage4a_build=r['multi_build_report'],source=case['source'],preflight=upstream['preflight'],
        history=dict(tiles=HISTORY,case=r['case'],files=case['files'],npu_outputs=finals,core_policy=policy),
        earlier_runners_executed=False,native_comparison=dict(status='SKIPPED',reason='Native Full-K semantics not re-audited; CPU W8A16 diagnostic is independent'))


def schedule(cores,limit):
    if type(cores) is not int or cores<1 or type(limit) is not int or limit not in range(305):raise ValueError('invalid Core limit')
    blocks=min(304,cores) if limit==0 else limit
    if blocks>cores:raise ValueError('more blocks than selected scheduling limit')
    return blocks,[list(range(core,304,blocks)) for core in range(blocks)]


def plan_from_log(text,mode,limit):
    probes=set(re.findall(r'STAGE4B_CORE_PROBE aic=(\d+) aiv=(\d+) all=(\d+)',text))
    policies=set(re.findall(r'STAGE4B_CORE_POLICY aic=(\d+) aiv=(\d+) all=(\d+) selected=(\d+)',text))
    arches=set(re.findall(r'STAGE4B_ARCH_PROBE arch=(\S+)',text))
    if len(probes)!=1 or len(policies)!=1 or arches!={'2002'}:raise ValueError('missing/inconsistent dav_2002 Core policy')
    raw=tuple(map(int,probes.pop()));policy=tuple(map(int,policies.pop()))
    if raw!=policy[:3] or not all(raw) or raw[0]!=raw[1] or raw[2]>raw[0] or policy[3]!=raw[2]:raise ValueError('invalid Core policy; no fallback')
    rows=set(re.findall(r'STAGE4B_SCHEDULE coreLimit=(\d+) blocks=(\d+) cores=(\d+) system=(\d+) arena=(\d+) stride=(\d+) user=(\d+) total=(\d+) mode=(\d+)',text))
    if len(rows)!=1:raise ValueError('missing/ambiguous workspace plan')
    p=dict(zip(('limit','blocks','cores','system','arena','stride','user','total','mode'),map(int,rows.pop())))
    p['core_policy']=dict(zip(('aic','aiv','all','selected','arch'),(*policy,2002)))
    blocks,owners=schedule(p['cores'],limit)
    if p['cores']!=policy[3] or p['limit']!=limit or p['mode']!=mode or p['blocks']!=blocks or p['system']<1 or p['arena']!=(p['system']+511)//512*512 or p['stride']!=p['arena']+1024 or p['user']!=blocks*p['stride'] or p['total']!=p['system']+p['user'] or p['total']>SETTINGS['workspace_cap']:raise ValueError('workspace/Core plan differs')
    if 'scratch=239104' not in text or 'userUB=12288 ubGuards=2560 reserve=8192' not in text or 'single=16/64/128 base=16/64/128' not in text:raise ValueError('resource/Tiling validation missing')
    p['owners']=owners;p['arena_intervals']=[[i*p['stride']+512,i*p['stride']+512+p['arena']] for i in range(blocks)]
    return p


def check_execution(ex,device,mode,limit):
    if ex.get('op')!=NAME or ex.get('shape')!=[16,2560,19456] or ex.get('tile_count')!=304 or ex.get('group_count')!=20 or ex.get('output_mode')!=mode or ex.get('core_limit')!=limit or ex.get('resource_admission')!='BEFORE_EXECUTION' or ex.get('ub_guard_protocol')!='UB_AND_CORE_ARENA_POISON_ON_FAILURE':raise ValueError('Stage4B execution scope differs')
    t=dict(ex);t['op']=common.NAME;t['shape']=[16,128,64];common.check_execution(t,device)
