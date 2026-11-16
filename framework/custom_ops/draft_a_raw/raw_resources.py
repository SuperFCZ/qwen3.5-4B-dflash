"""Strict serialized Host-log blocks; never infer a plan from a different Op."""
import json
import raw_common as c
from a3_launch import launch_evidence
from gate_prefetch_contract import match_plans

PREFIXES=['LAUNCH','GATE_PREFETCH','GATE_COPY','SCALE','K_TILE','CUBE']


def collect(log,out,gear,cfg):
    out.mkdir();blocks=[];current=[]
    for line in log.read_text(errors='replace').splitlines():
        if 'DFLASH_GROUP_QUANT_' not in line:continue
        part=line[line.index('DFLASH_GROUP_QUANT_'):]
        key=part.split(' ',1)[0][len('DFLASH_GROUP_QUANT_'):]
        if key not in PREFIXES:continue
        if key!=PREFIXES[len(current)]:raise ValueError('interleaved/incomplete Host resource log; no guessed pairing')
        current.append(part)
        if len(current)==6:blocks.append(current);current=[]
    if current or not blocks:raise ValueError('complete formal Host plans not emitted')
    result={}
    expected={(gear,12800,2560),(gear+16,2560,2048),(16,2560,4096),(16,4096,2560),(16,2560,19456),(16,9728,2560)}
    for i,lines in enumerate(blocks):
        row=json.loads(lines[0].split(' ',1)[1]);shape=tuple(row[k] for k in ('m','k','n'))
        if shape not in expected:raise ValueError('unexpected full-Draft projection shape')
        if json.loads(lines[2].split(' ',1)[1])!={'version':1,'mode':'loop','selected':'loop'}:raise ValueError('2D copy changed')
        p=out/f'block-{i}.log';p.write_text('\n'.join(lines)+'\n')
        plan=launch_evidence(p,shape,cfg['core_limit'],row['system_workspace_bytes'],cfg['dequant_mode'],cfg['pipeline_mode'],cfg['launch_version'],cfg['kv_m80_mode'],cfg['scale_mode'],cfg['k_tile'],cfg['k_tile_policy'],cfg['gate_prefetch_mode'])
        key=','.join(map(str,shape))
        if key in result and result[key]!=plan:raise ValueError('same-shape Tiling drift')
        result[key]=plan
    if set(result)!={','.join(map(str,s)) for s in expected}:raise ValueError('incomplete actual Host shape coverage')
    return dict(source=c.identity(log),plans=result,scope='FORMAL_HOST_LOG_OP_WORKSPACE_REQUESTS; MODEL_ACTUAL_ALLOCATION_IS_RUNNER_QUERY')


def compare(before,after):
    if set(before['plans'])!=set(after['plans']):raise ValueError('shape coverage changed')
    for shape,old in before['plans'].items():
        new=after['plans'][shape]
        if shape=='16,2560,19456':match_plans(old,new)
        else:
            # The global configured mode label changes, but must be disabled
            # for every non-Gate shape; all other plan fields remain identical.
            if old['gate_prefetch']!={'version':1,'mode':'off','enabled':False} or new['gate_prefetch']!={'version':1,'mode':'raw','enabled':False}:raise ValueError('non-Gate prefetch enabled')
            if {k:v for k,v in old.items() if k!='gate_prefetch'}!={k:v for k,v in new.items() if k!='gate_prefetch'}:raise ValueError('non-Gate plan changed')


def accepted_gate(plan,evidence,variant):
    key='baseline_launch' if variant=='B' else 'candidate_launch'
    reference=evidence['pairs']['forward']['comparison']['cases'][0][key]
    if plan!=reference:raise ValueError('actual Gate plan differs from accepted five-layer configuration')
