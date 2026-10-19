"""Opt-in original FP16 DebugDraftReplay capture, never synthetic hidden states."""
import json
import os
from pathlib import Path
import shlex
import subprocess

from common import sha256, write_json
from fp16_evidence import deployment, inventory, legacy_file, load, replay


def absolute_record(item):
    path=Path(item['path'])
    if not path.is_absolute() or not path.is_file() or sha256(path)!=item['sha256']:
        raise ValueError('historical absolute path/SHA256 differs: '+str(path))
    return path.resolve()


def capture_request(base, request_path, gear):
    """Accept original tools/profile_om.py or debug_draft_om.py request.json."""
    path=Path(request_path).resolve();r=load(path)
    dep=absolute_record(r['deployment_manifest']);runner=absolute_record(r['runner'])
    if dep!=base['path']: raise ValueError('capture request belongs to another Deployment')
    ids=r['prompt_token_ids'];c=base['graph']['metadata']['incremental_contract']
    if (not isinstance(ids,list) or not ids or any(type(i) is not int or not 0<=i<c['vocab_size'] for i in ids)
        or len(ids)+16>c['capacity'] or (16 if (len(ids)%64 or 64)<=16 else 64)!=gear):
        raise ValueError('original FP16 prompt is invalid or selects wrong context gear')
    pad=r.get('pad_token_id',0)
    if type(pad) is not int or not 0<=pad<c['vocab_size']: raise ValueError('invalid original padding token')
    plan=path.parent/'plan.txt'
    lines=plan.read_text().splitlines();dep_data=load(dep)
    expected=[c['abi'],f"capacity {c['capacity']} {c['vocab_size']}",f"draft_prefill_policy {c['draft_prefill_policy']}"]
    watch={str(p):sha256(p) for p in (path,runner,plan,dep)}
    # Verify every plan graph against original Deployment and immutable files;
    # no rewriting old manifests or weakening the production plan validator.
    present=[shlex.split(line)[1] for line in lines if line.startswith('graph ')]
    if present not in (['target_prefill','target_verify','draft'],['target_prefill','target_decode','target_verify','draft']):
        raise ValueError('historical plan role inventory differs')
    air_graphs={g['name']:g for g in base['air']['graphs']}
    for name in present:
        gs=[g for g in dep_data['graphs'] if g['name']==name]
        if len(gs)!=1: raise ValueError('missing plan graph: '+name)
        g=gs[0]; original=air_graphs[name]
        if any(g.get(k)!=original.get(k) for k in ('metadata','runtime_input_abi','custom_op_audit')):
            raise ValueError('Target/Draft plan AIR identity mismatch: '+name)
        om,audit=legacy_file(dep.parent,g['om']);watch[str(om)]=audit['sha256']
        expected.append(f'graph {name} {json.dumps(str(om),ensure_ascii=False)} {g["om"]["sha256"]}')
        if name=='draft' and not base['dynamic']:
            item=g['static_gear_oms'][1]['om'];other,audit=legacy_file(dep.parent,item);watch[str(other)]=audit['sha256']
            expected.append(f'static_gear64 {json.dumps(str(other))} {item["sha256"]}')
        for direction,marker in (('inputs','I'),('outputs','O')):
            for t in g['metadata']['tensor_abi'][direction]:
                expected.append(' '.join(map(str,[marker,t['name'],t['dtype'],len(t['shape']),*t['shape']])))
        if g.get('constant_inputs'): raise ValueError('FP16 capture plan unexpectedly has runtime constant inputs')
        expected.append('end')
    expected.append('done')
    if lines!=expected: raise ValueError('original capture plan differs from hash-bound Deployment/ABI')
    return dict(request=str(path),runner=runner,plan=plan,ids=ids,pad=pad,
                eos=r.get('eos_token_ids',[248044]),watch=watch,gear=gear)


def inspect(search_root, selected=None, requests=None):
    deps,replays=inventory(search_root);rows=[]
    request_candidates=[]
    for directory,dirs,files in os.walk(search_root,followlinks=False):
        depth=len(Path(directory).relative_to(search_root).parts)
        dirs[:]=[d for d in dirs if depth<5 and d not in ('.git','weights','profiler','checkpoint','checkpoints')]
        for name in ('request.json','profile-request.json'):
            p=Path(directory)/name
            if name not in files or p.stat().st_size>4*1024*1024: continue
            try:
                r=load(p)
                if isinstance(r,dict) and isinstance(r.get('deployment_manifest'),dict):
                    request_candidates.append((p,r))
            except (ValueError,OSError): pass
    for path in ([selected] if selected else deps):
        row=dict(path=str(path),status='BLOCKED',device_gear_query='NOT_RUN')
        try:
            base=deployment(path)
            row.update(manifest_status='PASS',shape_mode='dynamic' if base['dynamic'] else 'static',
                       file_audit=base['file_audit'],om_sha256=base['graph']['om']['sha256'],gears=[16,64],
                       source_watch=base['watch'],replays={},capture_requests={})
            row['possible_original_requests']=[dict(path=str(p),status='UNVERIFIED_NOT_SELECTED',
                plan_exists=(p.parent/'plan.txt').is_file(),prompt_tokens=len(r['prompt_token_ids']) if isinstance(r.get('prompt_token_ids'),list) else None)
                for p,r in request_candidates if r['deployment_manifest'].get('sha256')==sha256(base['path'])]
            for gear in (16,64):
                matches=[p for p in replays if load(p).get('draft_om_sha256')==base['graph']['om']['sha256']
                         and load(p).get('kv_output_audit',{}).get('context_rows')==gear]
                if len(matches)==1:
                    replay(base,matches[0],gear);row['replays'][str(gear)]=dict(status='PASS',path=str(matches[0]))
                else: row['replays'][str(gear)]=dict(status='BLOCKED',reason=f'{len(matches)} matching FP16 frozen replay(s); need exactly one')
                if requests and requests.get(gear):
                    req=capture_request(base,requests[gear],gear)
                    row['capture_requests'][str(gear)]=dict(status='VALIDATED_NOT_RUN',request=req['request'],watch=req['watch'])
                else: row['capture_requests'][str(gear)]=dict(status='MISSING',reason=f'need original FP16 --fp16-c{gear}-request (request.json/profile-request.json with adjacent plan.txt)')
            row['status']='READY_FOR_REPLAY' if all(x['status']=='PASS' for x in row['replays'].values()) else 'BLOCKED_MISSING_FP16_REPLAY'
        except (ValueError,KeyError,OSError,TypeError) as error: row['error']=str(error)
        rows.append(row)
    return dict(status='CHECKED_NOT_DEVICE_VALIDATED',candidates=rows,
                selection='explicit --fp16-deployment required when Chunk/MTP are both present',
                capture='NOT_RUN',formal_latency='NOT_RUN',profiling='NOT_RUN')


def capture(base, requests, output, device, env):
    checked={g:capture_request(base,requests[g],g) for g in (16,64)}
    if checked[16]['runner']!=checked[64]['runner']: raise ValueError('C16/C64 capture runners differ')
    output.mkdir();write_json(output/'requests.json',{str(g):{**r,'runner':str(r['runner']),'plan':str(r['plan'])} for g,r in checked.items()})
    results={}
    try:
        for gear,req in checked.items():
            for p,h in {**base['watch'],**req['watch']}.items():
                if sha256(p)!=h: raise ValueError('capture source changed before execution: '+p)
            report=output/f'c{gear}.json'
            command=[str(req['runner']),'--model-kind','chunk','--mode','dflash','--model',str(req['plan']),
                '--model-sha256',sha256(req['plan']),'--output',str(report),'--prompt-token-ids',','.join(map(str,req['ids'])),
                '--eos-token-ids',','.join(map(str,req['eos'])),'--pad-token-id',str(req['pad']),
                '--device-id',str(device),'--max-draft-tokens','15','--debug-draft-replay','20','--debug-draft-workspace','private']
            write_json(output/f'c{gear}-command.json',command)
            with (output/f'c{gear}.log').open('w') as log:
                subprocess.run(command,cwd=output,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            actual=load(report)
            lines=(Path(actual['input_directory'])/'contract.txt').read_text().splitlines()
            prompt_index=3+int(lines[3].startswith('static64 '))
            contract=base['graph']['metadata']['incremental_contract']
            if (actual.get('device_id')!=device or list(map(int,lines[prompt_index].split()))!=req['ids'] or
                list(map(int,lines[2].split()))!=[contract['capacity'],req['pad'],15]):
                raise ValueError('captured request/device differs from original FP16 prompt contract')
            replay(base,report,gear)  # Includes all FP16 input/output SHA, repeat, full KV and readonly gates.
            results[gear]=report
    finally:
        for req in checked.values():
            for p,h in {**base['watch'],**req['watch']}.items():
                if sha256(p)!=h: raise ValueError('original FP16 artifact changed: '+p)
    write_json(output/'capture.json',dict(status='PASS',scope='FP16 frozen replay only; performance NOT_RUN',reports={str(g):str(p) for g,p in results.items()}))
    return results
